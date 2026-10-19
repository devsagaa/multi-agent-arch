"""MCP server: exposes our research tools over the Model Context Protocol.

The tool *code* is unchanged (tools.py). MCP only changes WHERE it runs and HOW it is
found and called: this file runs as a separate process, the client discovers the tools
with `tools/list` and invokes them with `tools/call` (JSON-RPC 2.0 over stdin/stdout).

Any MCP host can use this server -- our researcher, Claude Code, Claude Desktop, an IDE --
without knowing it is Python or how search is implemented.

Transports (same tools, same JSON-RPC messages):
  python mcp_server.py                       stdio: a client starts us as its subprocess
  python mcp_server.py --http                Streamable HTTP on http://127.0.0.1:8765/mcp
       MCP_TOKEN=...                         require "Authorization: Bearer <token>"
       --cert cert.pem --key key.pem         serve HTTPS (e.g. a local cert made with mkcert)
Inspect it:  python mcp_client.py            (stdio)   |   python mcp_client.py --url=http://127.0.0.1:8765/mcp

Security notes:
  - stdout IS the protocol channel. Never print() here; logging goes to stderr.
  - The server doesn't import config.py or read .env, so it never sees OPENAI_API_KEY.
    The client starts it with a minimal environment plus SEARCH_PROVIDER.
  - HTTP mode binds to 127.0.0.1 and keeps the SDK's DNS-rebinding protection (Host and
    Origin must be localhost), so a web page in your browser can't drive the server.
    Anything beyond localhost needs auth (MCP_TOKEN here; OAuth 2.1 in the MCP spec) and TLS.
  - fetch_url keeps the SSRF checks from tools.py: a tool server is still your code
    making network requests on someone else's say-so.
"""

import argparse
import hmac
import json
import logging
import os
import sys

import anyio
from mcp.server.fastmcp import Context, FastMCP

import telemetry
import tools
from telemetry import SpanKind
from tools import fetch_page, web_search

logging.basicConfig(level=logging.INFO, format="mcp-server %(levelname)s %(message)s")  # stderr

PROVIDER = os.getenv("SEARCH_PROVIDER", "ddgs").lower()   # "ddgs" or "mock"
FETCHER = "mock" if PROVIDER == "mock" else "http"

server = FastMCP("research-tools")

# ------------------------------------------------------------------ lab tracing (opt-in)
# The MCP lab (/mcp page) starts this server with MCP_LAB_TRACE=1 so it can show what happens
# *inside* the server: FastMCP calling tools.py, and every outbound HTTP request/response.
# Events go to stderr as "MCPLAB {json}" lines -- never stdout, which is the protocol channel.
LAB_TRACE = os.getenv("MCP_LAB_TRACE") == "1"


def _lab(frm: str, to: str, kind: str, title: str, raw: str = "", notes: list[str] | None = None) -> None:
    if LAB_TRACE:
        ev = {"from": frm, "to": to, "kind": kind, "title": title, "raw": raw, "notes": notes or []}
        sys.stderr.write("MCPLAB " + json.dumps(ev) + "\n")
        sys.stderr.flush()


if LAB_TRACE:
    # Record every request the fetcher sends and every response it gets (including redirects),
    # by wrapping the opener tools.py uses. Bodies are read later by tools.py; the extracted
    # text appears in the tool's result record.
    _real_open = tools._opener.open

    def _traced_open(req, data=None, timeout=None):
        parts = tools.urllib.parse.urlsplit(req.full_url)
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        hdrs = {"Host": parts.netloc, **dict(req.header_items())}
        _lab("server", "web", "web-request", f"GET {parts.netloc}",
             f"{req.get_method()} {path} HTTP/1.1\n" + "\n".join(f"{k}: {v}" for k, v in hdrs.items())
             + "\nAccept-Encoding: identity\nConnection: close",
             ["Sent by tools.fetch_page (urllib), after check_url_safe() approved the address.",
              "The last two headers are added by urllib/http.client at send time; the rest are set by tools.py.",
              "Redirects are followed by the opener and re-checked against the same SSRF rules."])
        try:
            resp = _real_open(req, data, timeout) if data is not None else _real_open(req, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 -- recorded, then re-raised for tools.py to handle
            status = getattr(exc, "code", None)
            raw = f"HTTP/1.1 {status} {getattr(exc, 'reason', '')}\n" + "\n".join(
                f"{k}: {v}" for k, v in (getattr(exc, "headers", None) or {}).items()) if status else f"{type(exc).__name__}: {exc}"
            _lab("web", "server", "web-response", f"{status or 'error'} from {parts.netloc}", raw,
                 ["tools.py turns this into ok=false with an error message; it never raises to FastMCP."])
            raise
        final = resp.geturl()
        _lab("web", "server", "web-response", f"{resp.status} {resp.reason} from {parts.netloc}",
             f"HTTP/1.1 {resp.status} {resp.reason}\n" + "\n".join(f"{k}: {v}" for k, v in resp.headers.items()),
             (["Redirected: final URL " + final] if final != req.full_url else [])
             + ["Body not shown here: tools.py reads up to 2 MB, strips HTML to text, and returns it in the record."])
        return resp

    tools._opener.open = _traced_open


def _run_traced(fn, *args):
    """Call a tools.py function, recording the call, the network step and the returned record."""
    shown = ", ".join(repr(a) for a in args)
    _lab("server", "server", "internal", f"FastMCP -> tools.{fn.__name__}()",
         f"tools.{fn.__name__}({shown})", [
             "FastMCP already validated the arguments against inputSchema and clamped limits.",
             "The MCP server is just a wrapper: the actual work is the same tools.py code the app uses locally."])
    if fn is web_search and PROVIDER == "ddgs":
        _lab("server", "web", "web-request", "search via the ddgs library",
             f"DDGS().text({args[0]!r}, max_results={args[2]})", [
                 "The ddgs library makes the HTTP requests to the search engines itself and doesn't expose them, "
                 "so the exact request bytes can't be shown -- only the call and what came back."])
    elif PROVIDER == "mock":
        _lab("server", "server", "internal", "mock provider: no network request", "", [
            "SEARCH_PROVIDER=mock: canned results, nothing leaves the machine."])
    record = fn(*args)
    if fn is web_search and PROVIDER == "ddgs":
        _lab("web", "server", "web-response", f"search results: {record['result_count']}",
             json.dumps(record.get("results", []), indent=2, ensure_ascii=False) if record["ok"] else str(record["error"]),
             ["Parsed results as returned by the ddgs library."])
    _lab("server", "server", "internal", f"tools.{fn.__name__}() returned ok={record['ok']}",
         json.dumps(record, indent=2, ensure_ascii=False), [
             "FastMCP serialises this dict to JSON text and sends it as result.content[0].text."])
    _lab("server", "server", "done", "tool finished", "")
    return record


def _server_span(tool: str, ctx: Context):
    """SERVER span for one tools/call, continuing the client's trace if it sent one in params._meta.

    A client that sends no trace context (Claude Desktop, an IDE) simply starts a new trace here.
    """
    req = ctx.request_context
    attrs = {"mcp.method.name": "tools/call", "gen_ai.tool.name": tool, "gen_ai.operation.name": "execute_tool",
             "jsonrpc.request.id": str(req.request_id), "app.search.provider": PROVIDER, telemetry.OI_KIND: "CHAIN"}
    return telemetry.span(f"tools/call {tool}", SpanKind.SERVER, attrs, context=telemetry.extract_meta(req.meta))


# The function signature becomes the tool's inputSchema and the docstring its description.
# The returned dict travels back as JSON text content (a plain `dict` has no fixed shape,
# so FastMCP publishes no outputSchema for it). A `Context` parameter is injected by FastMCP
# and is NOT part of the inputSchema: it gives the tool the request's id and _meta.
# Tools are async and push the blocking network work onto a thread, so several
# tools/call requests in flight at once (parallel page fetches) actually overlap.
# (anyio copies the context into that thread, so tools.py spans nest under the server span.)
@server.tool(name="web_search")
async def search(query: str, ctx: Context, max_results: int = 5) -> dict:
    """Search the web. Returns a record with ok, error and results (title, url, snippet)."""
    with _server_span("web_search", ctx):
        return await anyio.to_thread.run_sync(_run_traced, web_search, query, PROVIDER,
                                              max(1, min(max_results, 10)))


@server.tool()
async def fetch_url(url: str, ctx: Context, max_chars: int = 3000) -> dict:
    """Download a public http(s) page and return its readable text. Internal addresses are blocked."""
    with _server_span("fetch_url", ctx):
        return await anyio.to_thread.run_sync(_run_traced, fetch_page, url, FETCHER,
                                              max(200, min(max_chars, 20000)))


class BearerAuth:
    """Minimal ASGI middleware: reject HTTP requests without the shared token.

    A learning stand-in for real auth. The MCP spec's answer for remote servers is OAuth 2.1
    (the SDK has hooks: FastMCP(token_verifier=..., auth=...)); a static token is fine for
    one trusted client on your own machine.
    """

    def __init__(self, app, token: str):
        self.app, self.expected = app, f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            got = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(got, self.expected):      # constant-time compare
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": b'{"error": "missing or invalid bearer token"}'})
                return
        await self.app(scope, receive, send)               # also passes "lifespan" events through


def serve_http(host: str, port: int, cert: str | None, key: str | None) -> None:
    import uvicorn

    app = server.streamable_http_app()                     # Starlette app with POST/GET/DELETE on /mcp
    token = os.getenv("MCP_TOKEN")
    if token:
        app = BearerAuth(app, token)
    scheme = "https" if cert else "http"
    logging.info("MCP over Streamable HTTP at %s://%s:%d%s (auth: %s)", scheme, host, port,
                 server.settings.streamable_http_path, "bearer token" if token else "none -- localhost only")
    uvicorn.run(app, host=host, port=port, ssl_certfile=cert, ssl_keyfile=key, log_level="warning")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="research-tools MCP server")
    ap.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)    # not 8000: that's server.py
    ap.add_argument("--cert", help="TLS certificate (PEM) to serve HTTPS")
    ap.add_argument("--key", help="TLS private key (PEM)")
    a = ap.parse_args()
    # Spans and JSON logs go to files (never stdout: in stdio mode that's the protocol channel).
    telemetry.setup("mcp-server")
    if a.http:
        serve_http(a.host, a.port, a.cert, a.key)
    else:
        server.run(transport="stdio")
