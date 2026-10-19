"""MCP client: starts the tool server as a subprocess and calls its tools.

The MCP SDK is async; the LangGraph nodes here are plain synchronous functions. MCPToolbox
bridges the two: it runs one event loop in a background thread, keeps a single session
open for its lifetime, and exposes blocking list_tools() / call() methods.

Two transports, same protocol messages:
  - stdio (default): we start mcp_server.py as a subprocess and talk over its stdin/stdout.
  - Streamable HTTP: MCPToolbox(url="http://127.0.0.1:8765/mcp") connects to a server that is
    already running (python mcp_server.py --http). Each JSON-RPC message is an HTTP POST; the
    session is tracked with an Mcp-Session-Id header; auth is a normal Authorization header.

Lifecycle of a session (JSON-RPC 2.0, whichever transport carries it):
  1. initialize          client and server exchange protocol version and capabilities
  2. tools/list          server describes its tools: name, description, inputSchema
  3. tools/call (n×)     {"name": ..., "arguments": {...}} -> content / structuredContent / isError
  4. close               stdin is closed and the subprocess exits

Run `python mcp_client.py` to see discovery and one call of each tool.
"""

import asyncio
import atexit
import contextlib
import json
import os
import sys
import threading
from datetime import timedelta
from pathlib import Path

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.client.streamable_http import streamable_http_client

import telemetry
from state import ToolCall
from telemetry import SpanKind

SERVER_SCRIPT = Path(__file__).with_name("mcp_server.py")


class MCPError(RuntimeError):
    pass


def _root_cause(exc: BaseException) -> str:
    """anyio task groups wrap failures in ExceptionGroups; show the first real error instead."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


class MCPToolbox:
    """A blocking handle on one MCP server session (a subprocess, or an HTTP server by URL)."""

    def __init__(self, command: str = sys.executable, args: list[str] | None = None,
                 env: dict[str, str] | None = None, timeout_s: float = 30,
                 url: str | None = None, token: str | None = None, verify: str | bool = True):
        # HTTP mode: no subprocess. `token` is sent as "Authorization: Bearer ...";
        # `verify` is True (system CAs), False (don't!) or a CA/cert file for a local HTTPS cert.
        self.url, self.verify = url, verify
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        # get_default_environment() is a short allow-list (PATH, HOME, ...): the server
        # does not inherit our secrets. Pass anything else it needs explicitly.
        # telemetry.child_env(): where to write spans, so the server's spans join our traces.
        self.params = StdioServerParameters(
            command=command, args=args or [str(SERVER_SCRIPT)],
            env={**get_default_environment(), **telemetry.child_env(), **(env or {})}, cwd=str(SERVER_SCRIPT.parent))
        self._session_id = lambda: None             # HTTP only: the Mcp-Session-Id the server assigned
        self.timeout = timeout_s
        self.server_info: dict = {}
        self.tools: list = []
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._session: ClientSession | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-client")
        self._thread.start()
        self._main = asyncio.run_coroutine_threadsafe(self._run(), self._loop)
        if not self._ready.wait(timeout_s):
            self.close()
            raise MCPError(f"MCP server did not start within {timeout_s}s")
        if self._error:
            self.close()
            what = f"could not connect to {self.url}" if self.url else "MCP server failed to start"
            cause = _root_cause(self._error)
            if self.url and "Session terminated" in cause:
                # The SDK reports an HTTP 404 this way; the usual reason is a wrong path.
                cause += (f" -- the server answered 404 for {self.url!r}. Check the URL: it should end "
                          "with /mcp, and --url=... must be followed by a space before other options.")
            raise MCPError(f"{what}: {cause}")

    async def _run(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with contextlib.AsyncExitStack() as stack:
                if self.url:
                    # Auth and TLS are plain HTTP concerns: configure them on the httpx client.
                    # verify: True = system CAs; a file path = trust a local (mkcert / self-signed) cert.
                    http = await stack.enter_async_context(httpx.AsyncClient(
                        headers=self.headers, verify=self.verify, timeout=httpx.Timeout(self.timeout, read=300)))
                    read, write, self._session_id = await stack.enter_async_context(
                        streamable_http_client(self.url, http_client=http))
                else:
                    read, write = await stack.enter_async_context(stdio_client(self.params))
                session = await stack.enter_async_context(ClientSession(read, write))
                init = await session.initialize()                           # 1. handshake
                self.server_info = {"name": init.serverInfo.name, "version": init.serverInfo.version,
                                    "protocol": init.protocolVersion}
                self.tools = (await session.list_tools()).tools              # 2. discovery
                self._session = session
                self._ready.set()
                await self._stop.wait()                                      # 3. serve calls until close()
        except BaseException as exc:  # noqa: BLE001 -- reported to the constructor
            self._error = exc
        finally:
            self._session = None
            self._ready.set()

    def call(self, name: str, arguments: dict) -> dict:
        """tools/call. Returns structuredContent if the tool has an output schema, else parsed text."""
        if self._session is None:
            raise MCPError("MCP session is not open")
        # Span names and attributes follow the OTel MCP semantic conventions ("{method} {target}").
        attrs = {"mcp.method.name": "tools/call", "gen_ai.tool.name": name, "gen_ai.operation.name": "execute_tool",
                 "mcp.protocol.version": self.server_info.get("protocol"),
                 "network.transport": "tcp" if self.url else "pipe", "url.full": self.url,
                 "mcp.session.id": self._session_id() if self.url else None, telemetry.OI_KIND: "CHAIN"}
        with telemetry.span(f"tools/call {name}", SpanKind.CLIENT, attrs) as s:
            # The trace context travels inside the JSON-RPC message itself:
            #   {"method": "tools/call", "params": {"name": ..., "arguments": ..., "_meta": {"traceparent": "00-<trace>-<span>-01"}}}
            # so it works the same over stdio (no headers there) and over HTTP.
            meta = telemetry.inject_meta() or None
            coro = self._session.call_tool(name, arguments, read_timeout_seconds=timedelta(seconds=self.timeout),
                                           meta=meta)
            # This runs on the client's event-loop thread; the span stays open in ours until the reply.
            result = asyncio.run_coroutine_threadsafe(coro, self._loop).result(self.timeout + 5)
            text = "".join(getattr(c, "text", "") for c in result.content)
            s.set_attribute("app.mcp.is_error", bool(result.isError))
            if result.isError:
                # Protocol-level tool error (exception inside the tool, bad arguments...).
                telemetry.fail(s, text, "tool_error")
                raise MCPError(f"tool '{name}' failed: {text}")
        if result.structuredContent is not None:
            sc = result.structuredContent
            # FastMCP wraps non-object return types as {"result": ...}; ours are dicts already.
            return sc.get("result", sc) if set(sc) == {"result"} else sc
        return json.loads(text)

    def openai_tools(self) -> list[dict]:
        """Discovered MCP tools translated to OpenAI's `tools` format (for illustration).

        MCP's inputSchema is JSON Schema, which is also what OpenAI's `parameters` field takes,
        so the translation is mostly renaming. This is how generic MCP hosts expose a server's
        tools to a model.
        """
        return [{"type": "function", "function": {
            "name": t.name, "description": t.description or "", "parameters": t.inputSchema}}
            for t in self.tools]

    def close(self) -> None:
        if self._stop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
            try:
                self._main.result(10)
            except Exception:  # noqa: BLE001 -- best effort shutdown
                pass
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


_shared: dict[str, MCPToolbox] = {}
_shared_lock = threading.Lock()


def shared_toolbox(search_provider: str, url: str | None = None, token: str | None = None) -> MCPToolbox:
    """One long-lived server process per search provider, reused by every run.

    Starting a server costs ~1s (new Python process + handshake), so a web server or an
    eval run shouldn't pay it per question. Closed automatically at interpreter exit.
    """
    key = url or search_provider
    with _shared_lock:
        box = _shared.get(key)
        if box is None or box._session is None:
            # With a URL, the server was started separately and chose its own search provider.
            box = _shared[key] = (MCPToolbox(url=url, token=token) if url
                                  else MCPToolbox(env={"SEARCH_PROVIDER": search_provider}))
        return box


@atexit.register
def _close_shared() -> None:
    for box in _shared.values():
        box.close()


def mcp_tool_functions(toolbox: MCPToolbox, max_results: int, page_max_chars: int):
    """search(query) and fetch(url) callables with the same contract as tools.py:
    they always return a ToolCall record and never raise, so the researcher can't tell
    (and needn't care) whether a tool ran in-process or behind MCP."""

    def _wrap(tool: str, arg: str, fn) -> ToolCall:
        try:
            rec = fn()
        except Exception as exc:  # transport failure, timeout, server crash, tool error
            rec = {"tool": tool, "provider": "mcp", "input": arg, "ok": False, "result_count": 0,
                   "error": f"MCP call failed: {type(exc).__name__}: {exc}", "results": [], "content": ""}
        rec["transport"] = "mcp"
        return rec

    def search(query: str) -> ToolCall:
        return _wrap("web_search", query, lambda: toolbox.call("web_search", {"query": query, "max_results": max_results}))

    def fetch(url: str) -> ToolCall:
        return _wrap("fetch_page", url, lambda: toolbox.call("fetch_url", {"url": url, "max_chars": page_max_chars}))

    return search, fetch


# ------------------------------------------------------------------ demo
if __name__ == "__main__":
    provider = os.getenv("SEARCH_PROVIDER", "ddgs")
    if "--mock" in sys.argv:
        provider = "mock"
    url = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--url=")), None)
    if url:
        print(f"Connecting to {url} over Streamable HTTP ...\n")
        box_cm = MCPToolbox(url=url, token=os.getenv("MCP_TOKEN"), verify=os.getenv("MCP_CA_FILE") or True)
    else:
        print(f"Starting {SERVER_SCRIPT.name} as a subprocess (SEARCH_PROVIDER={provider}) ...\n")
        box_cm = MCPToolbox(env={"SEARCH_PROVIDER": provider})
    if "--call" in sys.argv:
        # One tools/call with your own input:  --call <tool> '<json arguments>'
        i = sys.argv.index("--call")
        tool = sys.argv[i + 1] if len(sys.argv) > i + 1 else ""
        try:
            arguments = json.loads(sys.argv[i + 2]) if len(sys.argv) > i + 2 else {}
        except json.JSONDecodeError as exc:
            sys.exit(f"arguments must be JSON, e.g. '{{\"query\": \"mcp\"}}' ({exc})")
        with box_cm as box:
            names = [t.name for t in box.tools]
            if tool not in names:
                sys.exit(f"unknown tool {tool!r}; the server offers {names}")
            print(f"tools/call {json.dumps({'name': tool, 'arguments': arguments})}\n")
            try:
                print(json.dumps(box.call(tool, arguments), indent=2, ensure_ascii=False))
            except MCPError as exc:
                print(f"error: {exc}")
        sys.exit(0)
    with box_cm as box:
        print(f"1. initialize -> server {box.server_info}\n")
        print("2. tools/list ->")
        for t in box.tools:
            print(f"   - {t.name}: {t.description}")
            print(f"     inputSchema:  {json.dumps(t.inputSchema)}")
            if t.outputSchema:
                print(f"     outputSchema: {json.dumps(t.outputSchema)[:160]}...")
        print("\n   ...as OpenAI `tools` (what a generic host would send to the model):")
        print("   " + json.dumps(box.openai_tools()[0])[:300] + " ...\n")

        print('3. tools/call web_search {"query": "model context protocol"} ->')
        rec = box.call("web_search", {"query": "model context protocol", "max_results": 3})
        print(f"   ok={rec['ok']} results={rec['result_count']} error={rec['error']}")
        for r in rec["results"]:
            print(f"   - {r['title'][:70]}  {r['url']}")

        if rec["ok"]:
            url = rec["results"][0]["url"]
            print(f"\n   tools/call fetch_url {{\"url\": \"{url}\"}} ->")
            page = box.call("fetch_url", {"url": url, "max_chars": 300})
            print(f"   ok={page['ok']} error={page['error']}\n   {page['content'][:300]!r}")

        print('\n   tools/call fetch_url {"url": "http://169.254.169.254/latest/meta-data/"} ->')
        if provider == "mock" or rec["results"][0]["url"].startswith("https://example.com/mock"):
            print("   (skipped: the mock fetcher never touches the network, so there is nothing to block)")
        else:
            blocked = box.call("fetch_url", {"url": "http://169.254.169.254/latest/meta-data/"})
            print(f"   ok={blocked['ok']} error={blocked['error']}")
    print("\n4. closed: the server subprocess has exited.")
