"""Model clients. Agents depend on the small LLM interface below, not on a vendor SDK.

That lets the same graph run against OpenAI or a deterministic MockLLM (tests,
offline demos). Each call is *stateless*: the model only sees what the calling
node explicitly puts in `system` and `user`. It cannot see the graph state.

Two call styles:
  complete(role, system, user, schema)  -> one prompt in, one answer out
  chat(role, messages, tools)           -> one turn of a tool-calling conversation; the
                                           reply is text OR a list of tool calls to run

Production concerns handled here (so agents don't have to):
  - timeouts and retries with exponential backoff + jitter, only for transient errors
  - structured output: pass a JSON Schema and the API enforces it (strict mode)
  - usage accounting: every call returns token counts, attempts and latency
  - output limits + finish_reason checks: truncated, filtered or refused output is an
    error, never silently treated as a complete answer
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Protocol

import telemetry

log = logging.getLogger("llm")


class LLMError(RuntimeError):
    """Raised when a model call fails for good. Nodes catch it and record an error."""


# Output-token cap per role (max_completion_tokens). Bounds cost and latency of a single
# call; generous enough that a normal answer finishes with finish_reason="stop".
MAX_OUTPUT_TOKENS = {"researcher": 1500, "reviewer": 800, "finalizer": 1200, "judge": 800}
DEFAULT_MAX_OUTPUT_TOKENS = 1000


def check_finish(role: str, finish_reason: str | None, refusal: str | None, limit: int,
                 allowed: tuple[str, ...] = ("stop",)) -> None:
    """Turn every non-normal ending into an LLMError.

    finish_reason is set by OpenAI's servers (the SDK only parses it) and says why
    token generation stopped:
      stop            -> the model emitted its end token (or hit a stop sequence): OK
      length          -> cut off at max_completion_tokens / context window: text is incomplete
      content_filter  -> moderation removed or stopped the output
      tool_calls      -> the model asked us to run tools: OK only for chat() turns that offer tools
    refusal: structured-output refusal -- content is null, refusal holds the reason.
    """
    if refusal:
        raise LLMError(f"Model refused the {role} request: {refusal}")
    if finish_reason == "length":
        raise LLMError(f"{role} output was truncated at the {limit}-token limit "
                       f"(finish_reason=length); refusing to use an incomplete answer.")
    if finish_reason == "content_filter":
        raise LLMError(f"{role} output was blocked by the content filter (finish_reason=content_filter).")
    if finish_reason is not None and finish_reason not in allowed:
        raise LLMError(f"Unexpected finish_reason={finish_reason!r} for {role}.")


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 1
    latency_ms: int = 0
    finish_reason: str = "stop"
    tool_calls: list[dict] = field(default_factory=list)  # [{"id", "name", "arguments" (JSON string)}]
    raw: dict | None = None  # the provider's full response body (id, finish_reason, usage, ...)


class LLM(Protocol):
    name: str

    def build_request(self, role: str, system: str, user: str, schema: dict | None = None) -> dict: ...

    def complete(self, role: str, system: str, user: str, schema: dict | None = None) -> LLMResponse: ...

    def build_chat_request(self, role: str, messages: list[dict], tools: list[dict],
                           tool_choice: str = "auto") -> dict: ...

    def chat(self, role: str, messages: list[dict], tools: list[dict], tool_choice: str = "auto") -> LLMResponse: ...


def assistant_message(resp: LLMResponse) -> dict:
    """The assistant turn to append to the conversation, in OpenAI's message format."""
    msg: dict = {"role": "assistant", "content": resp.text or None}
    if resp.tool_calls:
        msg["tool_calls"] = [{"id": c["id"], "type": "function",
                              "function": {"name": c["name"], "arguments": c["arguments"]}}
                             for c in resp.tool_calls]
    return msg


def transcript(messages: list[dict]) -> str:
    """Readable rendering of a conversation (everything after the system prompt), for traces."""
    parts = []
    for m in messages:
        if m["role"] == "assistant":
            body = m.get("content") or ""
            for c in m.get("tool_calls") or []:
                body += f"\n-> {c['function']['name']}({c['function']['arguments']})  [id={c['id']}]"
            parts.append(f"[assistant]{body if body.startswith(chr(10)) else ' ' + body}")
        elif m["role"] == "tool":
            parts.append(f"[tool result for {m['tool_call_id']}]\n{m['content']}")
        else:
            parts.append(f"[{m['role']}]\n{m['content']}")
    return "\n\n".join(parts)


def describe_tool_calls(calls: list[dict]) -> str:
    return "\n".join(f"-> {c['name']}({c['arguments']})  [id={c['id']}]" for c in calls)


def _is_transient(exc: Exception) -> bool:
    """Retry rate limits, timeouts, connection drops and 5xx. Never retry 4xx (bad key, bad request)."""
    try:
        import openai
    except ImportError:
        return False
    if isinstance(exc, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                        openai.InternalServerError)):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


class OpenAILLM:
    def __init__(self, model: str, timeout_s: float = 30, max_retries: int = 3, base_delay_s: float = 1.0):
        from openai import OpenAI

        # SDK retries off: we retry ourselves so attempts are visible in traces and tests.
        self.client = OpenAI(timeout=timeout_s, max_retries=0)  # reads OPENAI_API_KEY
        self.model = model
        self.name = f"openai:{model}"
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s

    def build_request(self, role: str, system: str, user: str, schema: dict | None = None) -> dict:
        """The exact JSON body POSTed to https://api.openai.com/v1/chat/completions."""
        request = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.2,
            "max_completion_tokens": MAX_OUTPUT_TOKENS.get(role, DEFAULT_MAX_OUTPUT_TOKENS),
        }
        if schema:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": f"{role}_output", "schema": schema, "strict": True},
            }
        return request

    def build_chat_request(self, role: str, messages: list[dict], tools: list[dict],
                           tool_choice: str = "auto") -> dict:
        """Request body for a tool-calling turn: the whole conversation so far + the tool menu."""
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_completion_tokens": MAX_OUTPUT_TOKENS.get(role, DEFAULT_MAX_OUTPUT_TOKENS),
            "tools": tools,
            "tool_choice": tool_choice,   # "auto": model decides; "none": must answer in text
        }

    def _send(self, role: str, request: dict):
        """POST with retries for transient errors. Returns (response, attempts, start time)."""
        start = time.perf_counter()
        for attempt in range(1, self.max_retries + 2):
            try:
                return self.client.chat.completions.create(**request), attempt, start
            except Exception as exc:
                if attempt > self.max_retries or not _is_transient(exc):
                    raise LLMError(f"OpenAI call for {role} failed after {attempt} attempt(s): "
                                   f"{type(exc).__name__}: {exc}") from exc
                delay = self.base_delay_s * 2 ** (attempt - 1) * (0.5 + random.random())
                log.warning("%s attempt %d failed (%s); retrying in %.1fs", role, attempt, type(exc).__name__, delay)
                # Retries are invisible in the final response; record each one on the model-call span.
                telemetry.event("retry", {"app.attempt": attempt, "error.type": type(exc).__name__,
                                          "app.error": str(exc)[:300], "app.delay_s": round(delay, 2)})
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _response(self, resp, attempt: int, start: float, text: str, tool_calls: list[dict]) -> LLMResponse:
        usage = resp.usage
        return LLMResponse(
            text=text, model=self.model, attempts=attempt, tool_calls=tool_calls,
            finish_reason=resp.choices[0].finish_reason or "stop",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=round((time.perf_counter() - start) * 1000),
            raw=resp.model_dump(exclude_unset=True) if hasattr(resp, "model_dump") else None,
        )

    def complete(self, role: str, system: str, user: str, schema: dict | None = None) -> LLMResponse:
        request = self.build_request(role, system, user, schema)
        resp, attempt, start = self._send(role, request)
        # Not retried: a truncated/filtered/refused answer would very likely repeat.
        choice = resp.choices[0]
        check_finish(role, getattr(choice, "finish_reason", None),
                     getattr(choice.message, "refusal", None), request["max_completion_tokens"])
        text = choice.message.content or ""
        if not text.strip():
            raise LLMError(f"OpenAI returned an empty response for {role}.")
        return self._response(resp, attempt, start, text, [])

    def chat(self, role: str, messages: list[dict], tools: list[dict], tool_choice: str = "auto") -> LLMResponse:
        """One turn of a tool-calling conversation.

        The model either answers in text (finish_reason="stop") or asks us to run tools
        (finish_reason="tool_calls"). It never runs anything itself.
        """
        request = self.build_chat_request(role, messages, tools, tool_choice)
        resp, attempt, start = self._send(role, request)
        choice = resp.choices[0]
        check_finish(role, getattr(choice, "finish_reason", None), getattr(choice.message, "refusal", None),
                     request["max_completion_tokens"], allowed=("stop", "tool_calls"))
        calls = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                 for tc in (choice.message.tool_calls or [])]
        text = choice.message.content or ""
        if not text.strip() and not calls:
            raise LLMError(f"OpenAI returned neither text nor tool calls for {role}.")
        return self._response(resp, attempt, start, text, calls)


class MockLLM:
    """Deterministic stand-in for a real model. Output is labelled [MOCK].

    The mock researcher builds its findings only from the sources it was given,
    so it never invents sources.

    review_verdicts: optional list consumed one per reviewer call. Each item is a bool
    (approved?) or an (approved, score) tuple, e.g. [(False, 3), (False, 5)].
    Default: approve. Default scores: 8 when approved, 4 when rejected.
    """

    name = "mock"

    def __init__(self, review_verdicts: list | None = None):
        self.review_verdicts = list(review_verdicts or [])
        self.calls: list[dict] = []  # lets tests inspect exactly what each agent saw

    def build_request(self, role: str, system: str, user: str, schema: dict | None = None) -> dict:
        # Same shape as OpenAILLM's request, so traces look alike -- but nothing is sent anywhere.
        request = {"model": "mock", "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}],
                   "max_completion_tokens": MAX_OUTPUT_TOKENS.get(role, DEFAULT_MAX_OUTPUT_TOKENS)}
        if schema:
            request["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": f"{role}_output", "schema": schema, "strict": True}}
        return request

    def complete(self, role: str, system: str, user: str, schema: dict | None = None) -> LLMResponse:
        import re

        self.calls.append({"role": role, "system": system, "user": user, "schema": schema})

        if role == "researcher":
            hits = re.findall(r'<source id="(\d+)" title="([^"]*)"', user)
            body = "\n".join(f"- {title} [{n}]" for n, title in hits) or "- (no sources)"
            text = f"[MOCK] Findings synthesised from the provided sources:\n{body}"
        elif role == "reviewer":
            verdict = self.review_verdicts.pop(0) if self.review_verdicts else True
            approved, score = verdict if isinstance(verdict, tuple) else (verdict, 8 if verdict else 4)
            text = json.dumps({
                "approved": approved,
                "feedback": "[MOCK] Research is adequately supported." if approved
                else "[MOCK] Missing concrete trade-offs; search for specific examples.",
                "unsupported_claims": [] if approved else ["[MOCK] claim without evidence"],
                "score": score,
            })
        elif role == "judge":  # used by the eval harness
            text = json.dumps({"faithfulness": 4, "completeness": 3, "citation_quality": 4,
                               "unsupported_claims": [], "reasoning": "[MOCK] canned judgement"})
        elif role == "finalizer":
            text = ("[MOCK] Final answer: learn the cloud fundamentals you need while building, "
                    "rather than all of AWS up front [1]. (Generated by the mock model.)")
        else:
            raise LLMError(f"MockLLM has no response for role '{role}'.")
        return LLMResponse(text=text, model="mock")

    def build_chat_request(self, role: str, messages: list[dict], tools: list[dict],
                           tool_choice: str = "auto") -> dict:
        return {"model": "mock", "messages": messages,
                "max_completion_tokens": MAX_OUTPUT_TOKENS.get(role, DEFAULT_MAX_OUTPUT_TOKENS),
                "tools": tools, "tool_choice": tool_choice}

    def chat(self, role: str, messages: list[dict], tools: list[dict], tool_choice: str = "auto") -> LLMResponse:
        """Scripted tool use: search once -> read the top 2 results -> write cited findings."""
        import re

        self.calls.append({"role": role, "system": messages[0]["content"], "user": transcript(messages[1:]),
                           "schema": None, "messages": list(messages), "tool_choice": tool_choice})
        called = [c["function"]["name"] for m in messages if m["role"] == "assistant"
                  for c in m.get("tool_calls") or []]
        tool_text = "\n".join(m["content"] for m in messages if m["role"] == "tool")
        n = len(called)

        if tool_choice != "none":
            if "web_search" not in called:
                user = messages[1]["content"]
                query = user.split("\n", 1)[0].removeprefix("Question: ").strip()
                if "Reviewer feedback" in user:
                    query += " concrete examples trade-offs"  # a revision searches differently
                return LLMResponse(text="", model="mock", finish_reason="tool_calls", tool_calls=[
                    {"id": f"call_{n + 1}", "name": "web_search", "arguments": json.dumps({"query": query})}])
            if "fetch_page" not in called:
                ids = re.findall(r'<source id="(\d+)"', tool_text)[:2]
                if ids:  # two calls in ONE turn = parallel tool calls
                    return LLMResponse(text="", model="mock", finish_reason="tool_calls", tool_calls=[
                        {"id": f"call_{n + i}", "name": "fetch_page", "arguments": json.dumps({"source_id": int(s)})}
                        for i, s in enumerate(ids, 1)])

        hits = dict(re.findall(r'<source id="(\d+)" title="([^"]*)"', tool_text))  # last title per id
        body = "\n".join(f"- {title} [{sid}]" for sid, title in hits.items()) or "- (no sources)"
        return LLMResponse(text=f"[MOCK] Findings from tool-gathered sources:\n{body}", model="mock")


class RecordingLLM:
    """Wraps any LLM and records every prompt/response, for the web UI's trace view.

    Agents don't know they are being recorded -- the wrapper has the same interface.
    """

    def __init__(self, inner: LLM):
        self.inner = inner
        self.name = inner.name
        self.pending: list[dict] = []
        # The full LLM interface, so another wrapper (telemetry.TracedLLM) can sit on top.
        self.build_request = inner.build_request
        self.build_chat_request = inner.build_chat_request

    def complete(self, role: str, system: str, user: str, schema: dict | None = None) -> LLMResponse:
        rec = {"role": role, "model": self.name, "structured": bool(schema),
               "system": system, "user": user, "response": None, "error": None,
               "request": self.inner.build_request(role, system, user, schema), "raw_response": None}
        start = time.perf_counter()
        try:
            resp = self.inner.complete(role, system, user, schema)
            rec.update(response=resp.text, input_tokens=resp.input_tokens,
                       output_tokens=resp.output_tokens, attempts=resp.attempts, finish_reason=resp.finish_reason,
                       raw_response=resp.raw)
            return resp
        except Exception as exc:
            rec["error"] = str(exc)
            raise
        finally:
            rec["duration_ms"] = round((time.perf_counter() - start) * 1000)
            self.pending.append(rec)

    def chat(self, role: str, messages: list[dict], tools: list[dict], tool_choice: str = "auto") -> LLMResponse:
        rec = {"role": role, "model": self.name, "structured": False, "kind": "chat",
               "system": messages[0]["content"], "user": transcript(messages[1:]),
               "response": None, "error": None, "tool_calls": [], "tool_choice": tool_choice,
               # copy: the caller keeps appending to `messages` after this turn
               "request": self.inner.build_chat_request(role, list(messages), tools, tool_choice),
               "raw_response": None}
        start = time.perf_counter()
        try:
            resp = self.inner.chat(role, messages, tools, tool_choice)
            rec.update(response=resp.text or describe_tool_calls(resp.tool_calls), tool_calls=resp.tool_calls,
                       input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                       attempts=resp.attempts, finish_reason=resp.finish_reason, raw_response=resp.raw)
            return resp
        except Exception as exc:
            rec["error"] = str(exc)
            raise
        finally:
            rec["duration_ms"] = round((time.perf_counter() - start) * 1000)
            self.pending.append(rec)

    def drain(self) -> list[dict]:
        """Return and clear the calls made since the last drain (i.e. by the last node)."""
        calls, self.pending = self.pending, []
        return calls


def build_llm(provider: str, model: str, timeout_s: float = 30, max_retries: int = 3) -> LLM:
    if provider == "openai":
        return OpenAILLM(model, timeout_s=timeout_s, max_retries=max_retries)
    if provider == "mock":
        return MockLLM()
    raise LLMError(f"Unknown LLM provider '{provider}'. Use 'openai' or 'mock'.")
