"""HTTP lab: call the OpenAI API with plain httpx -- no SDK -- and show the raw exchange.

A learning tool, separate from the research graph. It shows that a model call is just
an HTTPS POST with a JSON body, and lets you see everything that comes back: the status
line, response headers (request id, rate limits, timing) and the JSON body -- including
error bodies and streamed chunks, which the SDK normally hides behind objects/exceptions.

  python http_lab.py "Say hi in 3 words"      # one raw call, printed
  (or use the /http page of server.py)
"""

import json
import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

URL = "https://api.openai.com/v1/chat/completions"
MAX_OUTPUT_TOKENS_CAP = 500   # the lab is for looking at responses, not for long generations
MAX_STREAM_LINES = 400


def _redacted_headers() -> dict:
    return {"Authorization": "Bearer sk-...REDACTED", "Content-Type": "application/json"}


def send(body: dict, timeout: float = 60) -> dict:
    """POST `body` to the Chat Completions endpoint and return the raw exchange as a dict."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {"error": "OPENAI_API_KEY is not set in .env -- the lab needs a real key."}
    if not isinstance(body, dict):
        return {"error": "The request body must be a JSON object."}
    body = dict(body)
    for k in ("max_completion_tokens", "max_tokens"):
        if isinstance(body.get(k), int):
            body[k] = min(body[k], MAX_OUTPUT_TOKENS_CAP)
    if "max_completion_tokens" not in body and "max_tokens" not in body:
        body["max_completion_tokens"] = MAX_OUTPUT_TOKENS_CAP

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    exchange = {"request": {"method": "POST", "url": URL, "headers": _redacted_headers(), "body": body}}
    start = time.perf_counter()
    try:
        if body.get("stream"):
            # Streaming: the body arrives as Server-Sent Events -- "data: {json}" lines,
            # ending with "data: [DONE]". We keep the raw lines so you can see each chunk.
            with httpx.stream("POST", URL, headers=headers, json=body, timeout=timeout) as r:
                first_byte_ms = None
                lines = []
                for line in r.iter_lines():
                    if first_byte_ms is None:
                        first_byte_ms = round((time.perf_counter() - start) * 1000)
                    if line and len(lines) < MAX_STREAM_LINES:
                        lines.append(line)
                exchange["response"] = _response_meta(r, start)
                exchange["response"]["time_to_first_line_ms"] = first_byte_ms
                exchange["response"]["stream_lines"] = lines
                exchange["response"]["assembled_text"] = _assemble(lines)
        else:
            r = httpx.post(URL, headers=headers, json=body, timeout=timeout)
            exchange["response"] = _response_meta(r, start)
            try:
                exchange["response"]["body"] = r.json()
            except ValueError:
                exchange["response"]["body"] = r.text
    except httpx.HTTPError as exc:  # timeouts, DNS, connection refused: there IS no HTTP response
        exchange["transport_error"] = f"{type(exc).__name__}: {exc}"
    return exchange


def _response_meta(r: httpx.Response, start: float) -> dict:
    return {
        "status": r.status_code,
        "reason": r.reason_phrase,
        "http_version": r.http_version,
        "elapsed_ms": round((time.perf_counter() - start) * 1000),
        "headers": dict(r.headers),
    }


def _assemble(lines: list[str]) -> str:
    """Rebuild the full text from streamed deltas -- what the SDK does for you."""
    out = []
    for line in lines:
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            chunk = json.loads(line[6:])
        except ValueError:
            continue
        for choice in chunk.get("choices", []):
            out.append(choice.get("delta", {}).get("content") or "")
    return "".join(out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    prompt = " ".join(sys.argv[1:]) or "Say hi in 3 words."
    ex = send({"model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
               "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": 50})
    print(json.dumps(ex, indent=2, ensure_ascii=False))
