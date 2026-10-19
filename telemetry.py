"""Observability: OpenTelemetry traces and JSON logs, written locally as JSONL files.

One question = one TRACE. A trace is a tree of SPANS (timed operations with attributes):

    invoke_agent research-assistant                 the run              (research-app)
      node load_memory / node supervisor ...        one per graph step
      invoke_agent researcher
        chat gpt-4.1-mini                           one per model call:  tokens, cost, finish_reason
        execute_tool web_search                     what the agent asked for
          tools/call web_search      CLIENT         the MCP request      (only with TOOL_TRANSPORT=mcp)
            tools/call web_search    SERVER         ...in the MCP server process (mcp-server)
              web_search ddgs                       the provider call in tools.py
          GET en.wikipedia.org                      an outbound page fetch

Spans cross the process boundary because the MCP client puts the W3C `traceparent` in
`params._meta` of tools/call and the server continues the trace from it.

Every log line is JSON with the trace_id/span_id of the span that was active, so a trace
links to its logs and a log line to its trace.

Names follow the OpenTelemetry GenAI and MCP semantic conventions (gen_ai.*, mcp.*), so the
same spans work in Jaeger, Grafana Tempo, Langfuse or AWS X-Ray/CloudWatch when exported
over OTLP. Custom fields use the app.* prefix.

The same spans ALSO carry OpenInference attributes (openinference.span.kind, llm.*, tool.*,
input/output.value): Arize Phoenix builds its LLM views (span kinds, token counts, messages)
from those, not from gen_ai.*. Two vocabularies for one concept, until the tools converge.

Settings (environment):
  TELEMETRY=off                                       disable (default: on)
  TELEMETRY_DIR=...                                   where JSONL goes (default: traces/otel)
  OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
                                                      also record prompts, answers and tool inputs
                                                      (off by default: they can hold personal data)
  OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:6006   also export over OTLP/HTTP, e.g. to Phoenix
  PHOENIX_PROJECT_NAME=research-assistant             the Phoenix project the spans land in

Without setup() every span is a no-op, so tests and the labs pay nothing.
"""

import atexit
import contextlib
import functools
import json
import logging
import os
import re
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from opentelemetry import propagate, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import SpanKind, StatusCode

ROOT = Path(__file__).parent
DEFAULT_DIR = ROOT / "traces" / "otel"
CAPTURE_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
MAX_ATTR_CHARS = 8000
RUN_SPAN = "invoke_agent research-assistant"
OI_KIND = "openinference.span.kind"     # LLM | AGENT | TOOL | CHAIN | RETRIEVER ... (OpenInference)
AGENT_NODES = {"researcher", "reviewer", "finalizer"}   # nodes that call a model

# The global proxy tracer: a no-op until setup() installs a real provider.
tracer = trace.get_tracer("research-assistant")
_service: str | None = None
_lock = threading.Lock()


def enabled() -> bool:
    return os.getenv("TELEMETRY", "on").strip().lower() not in {"off", "0", "false", "no"}


def capture_content() -> bool:
    return os.getenv(CAPTURE_ENV, "false").strip().lower() in {"1", "true", "yes", "on"}


def telemetry_dir() -> Path:
    return Path(os.getenv("TELEMETRY_DIR") or DEFAULT_DIR)


def active() -> bool:
    return _service is not None


# ------------------------------------------------------------------ redaction
_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,})|((?i:bearer)\s+[A-Za-z0-9._\-~+/=]{8,})")


def redact(text: str) -> str:
    """Telemetry must never become a place where keys leak to."""
    return _SECRET_RE.sub(lambda m: "sk-...REDACTED" if m.group(1) else "Bearer ...REDACTED", text)


def _clip(text: str) -> str:
    text = redact(text)
    return text if len(text) <= MAX_ATTR_CHARS else text[:MAX_ATTR_CHARS] + f" …[{len(text) - MAX_ATTR_CHARS} more chars]"


def _attrs(attrs: dict | None) -> dict:
    """OTel attributes must be str/bool/int/float or lists of them; drop None, clip strings."""
    out = {}
    for k, v in (attrs or {}).items():
        if v is None:
            continue
        if isinstance(v, str):
            v = _clip(v)
        elif isinstance(v, (list, tuple)):
            v = [x if isinstance(x, (bool, int, float)) else _clip(str(x)) for x in v]
        elif not isinstance(v, (bool, int, float)):
            v = _clip(json.dumps(v, default=str, ensure_ascii=False))
        out[k] = v
    return out


# ------------------------------------------------------------------ exporters (files)
def _daily_file(directory: Path, kind: str) -> Path:
    # One file per process per day: several processes (app, MCP servers) never share a file.
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return directory / f"{kind}-{day}-{_service}-{os.getpid()}.jsonl"


def span_to_dict(s: ReadableSpan) -> dict:
    ctx = s.context
    return {
        "trace_id": f"{ctx.trace_id:032x}", "span_id": f"{ctx.span_id:016x}",
        "parent_span_id": f"{s.parent.span_id:016x}" if s.parent else None,
        "name": s.name, "kind": s.kind.name,
        "service": s.resource.attributes.get("service.name"),
        "start": s.start_time / 1e9, "end": s.end_time / 1e9,
        "duration_ms": round((s.end_time - s.start_time) / 1e6, 2),
        "status": s.status.status_code.name, "status_message": s.status.description,
        "attributes": dict(s.attributes or {}),
        "events": [{"name": e.name, "time": e.timestamp / 1e9, "attributes": dict(e.attributes or {})}
                   for e in s.events],
    }


class JsonlSpanExporter(SpanExporter):
    """Writes each finished span as one JSON line. Synchronous on purpose (see setup)."""

    def __init__(self, directory: Path):
        self.directory = directory

    def export(self, spans) -> SpanExportResult:
        lines = "".join(json.dumps(span_to_dict(s), default=str, ensure_ascii=False) + "\n" for s in spans)
        try:
            with _lock, _daily_file(self.directory, "spans").open("a", encoding="utf-8") as f:
                f.write(lines)
        except OSError:
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


class JsonLogHandler(logging.Handler):
    """Structured logs: one JSON object per line, tagged with the active trace and span."""

    def __init__(self, directory: Path):
        super().__init__(logging.INFO)
        self.directory = directory

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ctx = trace.get_current_span().get_span_context()
            entry = {
                "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
                "level": record.levelname, "logger": record.name, "message": redact(record.getMessage()),
                "service": _service, "pid": os.getpid(), "thread": record.threadName,
                "trace_id": f"{ctx.trace_id:032x}" if ctx.is_valid else None,
                "span_id": f"{ctx.span_id:016x}" if ctx.is_valid else None,
            }
            if record.exc_info:
                entry["exception"] = redact(logging.Formatter().formatException(record.exc_info))
            with _lock, _daily_file(self.directory, "logs").open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001 -- logging must never break the app
            self.handleError(record)


def setup(service: str) -> bool:
    """Install the tracer provider and the JSON log handler for this process (once).

    Call it AFTER logging.basicConfig(): the console keeps its level, files get INFO+.
    SimpleSpanProcessor writes each span the moment it ends. That costs a file write per span,
    but nothing is lost when a process is killed (MCP servers are often stopped abruptly).
    A collector export (OTLP) is batched instead: it goes over the network.
    """
    global _service
    if _service or not enabled():
        return _service is not None
    directory = telemetry_dir()
    directory.mkdir(parents=True, exist_ok=True)
    _service = service
    provider = TracerProvider(resource=Resource.create({
        "service.name": service,
        # Phoenix groups traces by project; the app and the MCP server must share one to show one trace.
        "openinference.project.name": os.getenv("PHOENIX_PROJECT_NAME", "research-assistant"),
    }))
    provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(directory)))
    if (endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")):
        _add_otlp(provider, endpoint)
    trace.set_tracer_provider(provider)
    atexit.register(provider.shutdown)

    root = logging.getLogger()
    for h in root.handlers:                     # console handlers keep showing what they showed before
        if h.level == logging.NOTSET:
            h.setLevel(root.level)
    root.setLevel(min(root.level, logging.INFO))
    root.addHandler(JsonLogHandler(directory))
    return True


def _reachable(endpoint: str, timeout: float = 0.5) -> bool:
    from urllib.parse import urlsplit
    import socket
    parts = urlsplit(endpoint)
    try:
        with socket.create_connection((parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)), timeout):
            return True
    except OSError:
        return False


def _add_otlp(provider: TracerProvider, endpoint: str) -> None:
    """Batch-export to an OTLP/HTTP backend (Phoenix, Jaeger, a collector) -- only if it is up.

    Without the check, a stopped backend makes the exporter retry for ~30 s at exit and fill
    the console with warnings. Observability must never slow down or break the app; the local
    JSONL files are written either way.
    """
    log = logging.getLogger("telemetry")
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        log.warning("OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry-exporter-otlp-proto-http is not installed")
        return
    if not _reachable(endpoint):
        log.warning("OTLP endpoint %s is not reachable (is Phoenix running?); spans go to %s only",
                    endpoint, telemetry_dir())
        return
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(timeout=5)))


def child_env() -> dict[str, str]:
    """Telemetry settings for a child process (the MCP server gets a minimal environment)."""
    if not active():
        return {"TELEMETRY": "off"}
    env = {"TELEMETRY": "on", "TELEMETRY_DIR": str(telemetry_dir())}
    if capture_content():
        env[CAPTURE_ENV] = "true"
    # Where to export, so the server's spans reach the same backend. Not OTEL_EXPORTER_OTLP_HEADERS:
    # those can hold a backend API key, and the server gets no secrets it doesn't need.
    for key in ("OTEL_EXPORTER_OTLP_ENDPOINT", "PHOENIX_PROJECT_NAME"):
        if os.getenv(key):
            env[key] = os.environ[key]
    return env


# ------------------------------------------------------------------ OpenInference helpers
def oi_messages(prefix: str, messages: list[dict]) -> dict:
    """Chat messages flattened the OpenInference way: llm.input_messages.0.message.role = ..."""
    out: dict = {}
    for i, m in enumerate(messages or []):
        p = f"{prefix}.{i}.message"
        out[f"{p}.role"] = m.get("role")
        content = m.get("content")
        if content is not None:
            out[f"{p}.content"] = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        if m.get("tool_call_id"):
            out[f"{p}.tool_call_id"] = m["tool_call_id"]
        for j, c in enumerate(m.get("tool_calls") or []):
            fn = c.get("function", c)                 # OpenAI message shape, or our {"id","name","arguments"}
            out[f"{p}.tool_calls.{j}.tool_call.id"] = c.get("id")
            out[f"{p}.tool_calls.{j}.tool_call.function.name"] = fn.get("name")
            out[f"{p}.tool_calls.{j}.tool_call.function.arguments"] = fn.get("arguments")
    return out


# ------------------------------------------------------------------ span helpers
@contextlib.contextmanager
def span(name: str, kind: SpanKind = SpanKind.INTERNAL, attributes: dict | None = None, context=None):
    """A span that records an exception (and sets ERROR status) if the block raises."""
    with tracer.start_as_current_span(name, kind=kind, attributes=_attrs(attributes), context=context) as s:
        try:
            yield s
        except BaseException as exc:
            s.set_attribute("error.type", type(exc).__name__)
            raise


def fail(s, message: str, error_type: str = "tool_error") -> None:
    """Mark a span as failed without an exception (tools return ok=False records)."""
    s.set_status(StatusCode.ERROR, redact(message)[:500])
    s.set_attribute("error.type", error_type)


def event(name: str, attributes: dict | None = None) -> None:
    """Add an event to whatever span is active (no-op when none is)."""
    trace.get_current_span().add_event(name, _attrs(attributes))


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return f"{ctx.trace_id:032x}" if ctx.is_valid else None


# ------------------------------------------------------------------ the run
@contextlib.contextmanager
def run_span(run_id: str, question: str, settings, entry: str):
    """The root span of one question. `entry` says who started it: cli, web, eval."""
    attrs = {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "research-assistant",
             "app.run_id": run_id, "app.entry": entry, "app.llm_provider": settings.llm_provider,
             "gen_ai.request.model": settings.openai_model if settings.llm_provider == "openai" else "mock",
             "app.search_provider": settings.search_provider, "app.researcher_mode": settings.researcher_mode,
             "app.tool_transport": settings.tool_transport, "app.max_revisions": settings.max_revisions,
             "app.question": question if capture_content() else None,
             OI_KIND: "AGENT", "input.value": question if capture_content() else None,
             "tag.tags": [entry, settings.researcher_mode, settings.tool_transport, settings.llm_provider]}
    with span(RUN_SPAN, attributes=attrs) as s:
        yield s


def finish_run(s, final: dict) -> None:
    """Summarise the final state on the root span: these become the per-run metrics."""
    usage = final.get("llm_usage") or []
    calls = final.get("tool_calls") or []
    failures = [c for c in calls if not c.get("ok")]
    s.set_attributes(_attrs({
        "app.outcome": final.get("outcome"), "app.review_status": final.get("review_status"),
        "app.revisions": final.get("revision_count", 0), "app.review_score": final.get("review_score"),
        "app.llm_calls": len(usage), "app.tool_calls": len(calls), "app.tool_failures": len(failures),
        "app.retries": sum(u.get("attempts", 1) - 1 for u in usage),
        "gen_ai.usage.input_tokens": sum(u.get("input_tokens", 0) for u in usage),
        "gen_ai.usage.output_tokens": sum(u.get("output_tokens", 0) for u in usage),
        "app.cost_usd": round(sum(u.get("cost_usd", 0.0) for u in usage), 6),
        "app.sources": len(final.get("sources") or []),
        "output.value": final.get("final_answer") if capture_content() else None,
    }))
    if final.get("outcome") == "failed":
        fail(s, final.get("error") or "run failed", "run_failed")


# ------------------------------------------------------------------ graph nodes
def traced_node(name: str, fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """Wrap a LangGraph node: one span per execution, with what it decided and changed."""
    is_agent = name in AGENT_NODES

    @functools.wraps(fn)
    def node(state: dict) -> dict:
        attrs = {"app.node": name, "app.revision": state.get("revision_count", 0),
                 OI_KIND: "AGENT" if is_agent else "CHAIN", "graph.node.id": name}
        if is_agent:
            attrs |= {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": name}
        with span(f"invoke_agent {name}" if is_agent else f"node {name}", attributes=attrs) as s:
            update = fn(state)
            s.set_attribute("app.update_keys", sorted(update or {}))
            if "next_step" in (update or {}):
                # The routing reason is built from state flags and counters, never from user text.
                reason = next((h["detail"] for h in update.get("history", []) if h["node"] == "supervisor"), None)
                s.set_attributes(_attrs({"app.route.next": update["next_step"], "app.route.reason": reason}))
            if (update or {}).get("error"):
                fail(s, update["error"], "node_error")
            if "review_status" in (update or {}) and name == "reviewer":
                s.set_attributes(_attrs({"app.review.status": update["review_status"],
                                         "app.review.score": update.get("review_score")}))
            return update
    return node


# ------------------------------------------------------------------ model calls
class TracedLLM:
    """Wraps any LLM: one CLIENT span per model call (GenAI semantic conventions).

    Same interface as the wrapped model, so agents don't know they are traced.
    """

    def __init__(self, inner, price_input_per_m: float = 0.0, price_output_per_m: float = 0.0):
        self.inner = inner
        self.name = inner.name
        self.provider, _, model = inner.name.partition(":")
        self.model = model or inner.name
        self.prices = (price_input_per_m, price_output_per_m)

    def __getattr__(self, item):       # build_request, calls, drain ... pass straight through
        return getattr(self.inner, item)

    def _span(self, role: str, request: dict):
        attrs = {"gen_ai.operation.name": "chat", "gen_ai.provider.name": self.provider,
                 "gen_ai.request.model": self.model, "app.role": role,
                 "gen_ai.request.max_tokens": request.get("max_completion_tokens"),
                 "gen_ai.request.temperature": request.get("temperature"),
                 "app.tool_choice": request.get("tool_choice"),
                 "app.tools_offered": [t["function"]["name"] for t in request.get("tools") or []] or None,
                 "app.structured_output": "response_format" in request,
                 "app.messages": len(request.get("messages") or []),
                 # OpenInference (Phoenix): the same facts under its names
                 OI_KIND: "LLM", "llm.model_name": self.model, "llm.provider": self.provider,
                 "llm.system": self.provider,
                 "llm.invocation_parameters": {k: request[k] for k in ("temperature", "max_completion_tokens", "tool_choice")
                                               if k in request}}
        for i, t in enumerate(request.get("tools") or []):        # schemas are code, not user data
            attrs[f"llm.tools.{i}.tool.json_schema"] = t
        if capture_content():
            messages = request.get("messages") or []
            attrs["gen_ai.input.messages"] = messages
            attrs |= oi_messages("llm.input_messages", messages)
            attrs["input.value"] = next((m["content"] for m in reversed(messages)
                                         if m.get("role") in ("user", "tool") and isinstance(m.get("content"), str)), None)
        return span(f"chat {self.model}", SpanKind.CLIENT, attrs)

    def _record(self, s, resp) -> None:
        cost = (resp.input_tokens * self.prices[0] + resp.output_tokens * self.prices[1]) / 1_000_000
        raw = resp.raw or {}
        s.set_attributes(_attrs({
            "gen_ai.response.model": raw.get("model") or resp.model, "gen_ai.response.id": raw.get("id"),
            "gen_ai.response.finish_reasons": [resp.finish_reason],
            "gen_ai.usage.input_tokens": resp.input_tokens, "gen_ai.usage.output_tokens": resp.output_tokens,
            "app.cost_usd": round(cost, 6), "app.attempts": resp.attempts,
            "app.tool_calls": [f"{c['name']}#{c['id']}" for c in resp.tool_calls] or None,
            "gen_ai.output.messages": (resp.text or json.dumps(resp.tool_calls)) if capture_content() else None,
            "llm.token_count.prompt": resp.input_tokens, "llm.token_count.completion": resp.output_tokens,
            "llm.token_count.total": resp.input_tokens + resp.output_tokens,
            "llm.cost.total": round(cost, 6),
        }))
        if capture_content():
            out = {"role": "assistant", "content": resp.text or None, "tool_calls": resp.tool_calls}
            s.set_attributes(_attrs({**oi_messages("llm.output_messages", [out]),
                                     "output.value": resp.text or json.dumps(resp.tool_calls)}))

    def _request(self, builder: str, *args) -> dict:
        """The request body, for span attributes only. Tracing must never break a model call."""
        try:
            return getattr(self.inner, builder)(*args)
        except Exception:  # noqa: BLE001
            return {}

    def complete(self, role: str, system: str, user: str, schema: dict | None = None):
        with self._span(role, self._request("build_request", role, system, user, schema)) as s:
            resp = self.inner.complete(role, system, user, schema)
            self._record(s, resp)
            return resp

    def chat(self, role: str, messages: list[dict], tools: list[dict], tool_choice: str = "auto"):
        with self._span(role, self._request("build_chat_request", role, list(messages), tools, tool_choice)) as s:
            resp = self.inner.chat(role, messages, tools, tool_choice)
            self._record(s, resp)
            return resp


# ------------------------------------------------------------------ tools
def traced_tool(tool: str, fn: Callable[[str], dict]) -> Callable[..., dict]:
    """Wrap a search/fetch callable: one span per tool execution the agent asked for."""

    def run(arg: str, call_id: str | None = None) -> dict:
        attrs = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool, "gen_ai.tool.type": "function",
                 "gen_ai.tool.call.id": call_id, "app.tool.input": arg if capture_content() else None,
                 OI_KIND: "TOOL", "tool.name": tool, "tool_call.id": call_id,
                 "input.value": arg if capture_content() else None}
        with span(f"execute_tool {tool}", attributes=attrs) as s:
            rec = fn(arg)
            if capture_content():
                s.set_attribute("output.value", _clip(json.dumps(
                    {k: rec.get(k) for k in ("ok", "error", "results", "content") if rec.get(k)}, ensure_ascii=False)))
            s.set_attributes(_attrs({"app.tool.ok": rec.get("ok"), "app.tool.provider": rec.get("provider"),
                                     "app.tool.transport": rec.get("transport", "local"),
                                     "app.tool.result_count": rec.get("result_count"),
                                     "app.tool.content_chars": len(rec.get("content") or "") or None}))
            if not rec.get("ok"):
                fail(s, rec.get("error") or "tool failed")
            return rec
    run.__wrapped__ = fn
    return run


# ------------------------------------------------------------------ context across processes (MCP)
def inject_meta() -> dict:
    """W3C trace context for the current span, to put in an MCP request's params._meta."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)          # adds "traceparent" (and "tracestate" if any)
    return carrier


def extract_meta(meta) -> object:
    """The parent context sent by an MCP client in params._meta (a pydantic model or a dict)."""
    if meta is None:
        return None
    get = meta.get if isinstance(meta, dict) else (lambda k: getattr(meta, k, None))
    carrier = {k: v for k in ("traceparent", "tracestate") if isinstance(v := get(k), str)}
    return propagate.extract(carrier) if carrier else None
