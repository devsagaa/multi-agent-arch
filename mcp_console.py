"""MCP console: you play the app. Send raw JSON-RPC to an MCP server and raw requests to the LLM.

The /mcp page runs the whole flow automatically. This console does nothing on its own: every
message is one you wrote (or a guided step filled in, which you can still edit). The server
keeps an MCP connection open between your clicks, so a session lasts across many requests:

  open   start mcp_server.py (stdio child process, or a temporary / your own HTTP server)
  send   write one raw message exactly as you typed it; return the raw reply, plus what
         happened inside the server (tools.py calls, outbound HTTP) when we can see it
  close  close stdin (stdio) or DELETE the session (HTTP), and stop the process

Sessions are kept in memory, limited in number and closed after IDLE_TIMEOUT_S without use.
"""

import atexit
import json
import queue
import subprocess
import sys
import threading
import time
import uuid

import httpx

import mcp_lab
from mcp_lab import ROOT, ServerLog, TemporaryHTTPServer, _http_request_text, _http_response_text, _lab_env

MAX_SESSIONS = 4
IDLE_TIMEOUT_S = 600
REPLY_TIMEOUT_S = 60
MAX_RAW_CHARS = 200_000


class StdioSession:
    transport = "stdio"

    def __init__(self, search_provider: str):
        self.proc = subprocess.Popen([sys.executable, str(ROOT / "mcp_server.py")], cwd=ROOT,
                                     env=_lab_env(search_provider), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, encoding="utf-8")
        self.log = ServerLog(self.proc.stderr)
        self.out: queue.Queue = queue.Queue()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        self.info = {"transport": "stdio", "pid": self.proc.pid, "search_provider": search_provider,
                     "started": f"{sys.executable.rsplit(chr(92), 1)[-1]} mcp_server.py   (child process; env: "
                                f"SEARCH_PROVIDER={search_provider}, MCP_LAB_TRACE=1, no API keys)"}

    def _read_stdout(self) -> None:
        for line in iter(self.proc.stdout.readline, ""):
            self.out.put(line.rstrip("\n"))
        self.out.put(None)                                   # EOF: the server exited

    def send(self, raw: str) -> dict:
        try:
            msg = json.loads(raw)
            line = json.dumps(msg)                           # stdio framing: exactly one line per message
        except json.JSONDecodeError:
            msg, line = None, " ".join(raw.split())          # send it anyway, so you can see the server's answer
        if self.proc.poll() is not None:
            return {"error": f"the server process has exited (code {self.proc.returncode}); reconnect"}
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        expects_reply = msg is None or (isinstance(msg, dict) and "id" in msg) or isinstance(msg, list)
        is_call = isinstance(msg, dict) and msg.get("method") == "tools/call"
        replies, deadline = [], time.monotonic() + (REPLY_TIMEOUT_S if expects_reply else 0.5)
        while time.monotonic() < deadline:
            try:
                reply = self.out.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                break
            if reply is None:
                replies.append("(server closed stdout: the process exited)")
                break
            replies.append(reply)
            try:
                data = json.loads(reply)
            except json.JSONDecodeError:
                continue
            # Done once we have the response to our id (or any error reply for unparsable input).
            if not isinstance(msg, dict) or data.get("id") == msg.get("id"):
                break
        internals = self.log.take(wait_for_done=is_call)
        return {
            "request_raw": line, "request_channel": f"stdin of process {self.proc.pid} (one line + \\n)",
            "response_raw": "\n".join(replies) if replies else "",
            "response_channel": f"stdout of process {self.proc.pid}",
            "no_reply_expected": not expects_reply, "server_internals": internals,
            "timed_out": expects_reply and not replies,
        }

    def close(self) -> dict:
        try:
            self.proc.stdin.close()
            self.proc.wait(5)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        return {"closed": f"closed stdin; process {self.proc.pid} exited with code {self.proc.returncode}"}


class HTTPSession:
    transport = "http"

    def __init__(self, search_provider: str, url: str | None, token: str | None):
        self.temp = None
        if not url:
            self.temp = TemporaryHTTPServer(search_provider)
            url, token = self.temp.url, self.temp.token
        self.url, self.token = url, token
        self.http = httpx.Client(timeout=REPLY_TIMEOUT_S)
        self.session_id: str | None = None
        self.protocol: str | None = None
        self.log = self.temp.log if self.temp else None
        self.info = {"transport": "http", "url": url, "search_provider": search_provider if self.temp else "server's choice",
                     "started": (f"mcp_server.py --http --port {self.temp.port}   (temporary; MCP_TOKEN=<random>, "
                                 f"SEARCH_PROVIDER={search_provider}, MCP_LAB_TRACE=1)") if self.temp
                     else "your own server (started separately); its internals can't be traced from here"}

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        if self.protocol:
            h["MCP-Protocol-Version"] = self.protocol
        return h

    def send(self, raw: str) -> dict:
        try:
            msg = json.loads(raw)
            body = json.dumps(msg)
        except json.JSONDecodeError:
            msg, body = None, raw
        is_init = isinstance(msg, dict) and msg.get("method") == "initialize"
        if is_init:
            # initialize starts a NEW session: it is sent without a session id or protocol header.
            self.session_id = self.protocol = None
        req = self.http.build_request("POST", self.url, content=body.encode(), headers=self._headers())
        is_call = isinstance(msg, dict) and msg.get("method") == "tools/call"
        try:
            r = self.http.send(req)
        except httpx.HTTPError as exc:
            return {"request_raw": _http_request_text(req), "error": f"{type(exc).__name__}: {exc}"}
        # Only a successful initialize creates a session; ids on error responses are ignored.
        if is_init and r.status_code == 200 and r.headers.get("mcp-session-id"):
            self.session_id = r.headers["mcp-session-id"]
        if is_init and r.status_code == 200:
            for line in r.text.splitlines():
                if line.startswith("data:") or line.startswith("{"):
                    try:
                        self.protocol = json.loads(line[5:] if line.startswith("data:") else line)["result"]["protocolVersion"]
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
        return {
            "request_raw": _http_request_text(req), "request_channel": f"HTTP POST {self.url}",
            "response_raw": _http_response_text(r), "response_channel": f"HTTP response ({r.status_code})",
            "no_reply_expected": isinstance(msg, dict) and "id" not in msg,
            "server_internals": self.log.take(wait_for_done=is_call and r.status_code == 200) if self.log else [],
            "session_id": self.session_id,
        }

    def close(self) -> dict:
        out = {}
        if self.session_id:
            req = self.http.build_request("DELETE", self.url, headers=self._headers())
            try:
                r = self.http.send(req)
                out = {"request_raw": _http_request_text(req), "response_raw": _http_response_text(r)}
            except httpx.HTTPError as exc:
                out = {"request_raw": _http_request_text(req), "error": str(exc)}
        self.http.close()
        if self.temp:
            self.temp.stop()
        out["closed"] = "session deleted" + ("; temporary server stopped" if self.temp else "")
        return out


_sessions: dict[str, tuple[float, object]] = {}
_lock = threading.Lock()


def _reap() -> None:
    now = time.monotonic()
    for sid, (used, sess) in list(_sessions.items()):
        if now - used > IDLE_TIMEOUT_S:
            _sessions.pop(sid, None)
            try:
                sess.close()
            except Exception:  # noqa: BLE001
                pass


def open_session(transport: str, search_provider: str, url: str | None = None, token: str | None = None) -> dict:
    with _lock:
        _reap()
        if len(_sessions) >= MAX_SESSIONS:
            return {"error": f"too many open sessions ({MAX_SESSIONS}); disconnect one first"}
    try:
        sess = HTTPSession(search_provider, url, token) if transport == "http" else StdioSession(search_provider)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not start: {type(exc).__name__}: {exc}"}
    sid = uuid.uuid4().hex[:12]
    with _lock:
        _sessions[sid] = (time.monotonic(), sess)
    return {"session": sid, **sess.info}


def send(sid: str, raw: str) -> dict:
    if len(raw) > MAX_RAW_CHARS:
        return {"error": "message too large"}
    with _lock:
        entry = _sessions.get(sid)
        if entry:
            _sessions[sid] = (time.monotonic(), entry[1])
    if not entry:
        return {"error": "no such session (it may have timed out); connect again"}
    t = time.perf_counter()
    out = entry[1].send(raw)
    out["elapsed_ms"] = round((time.perf_counter() - t) * 1000)
    return out


def close_session(sid: str) -> dict:
    with _lock:
        entry = _sessions.pop(sid, None)
    return entry[1].close() if entry else {"closed": "already closed"}


@atexit.register
def close_all() -> None:
    """Stop every server process this console started (also runs when server.py exits)."""
    for sid in list(_sessions):
        close_session(sid)


def llm(body: dict, mock: bool, turn: int) -> dict:
    """Send the model request. Real: http_lab.send (raw httpx, key redacted, output capped). Mock: local."""
    if not mock:
        import http_lab
        return http_lab.send(body)
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
        return {"error": "the mock model needs a body with a messages list"}
    try:
        status, headers, text = mcp_lab._mock_llm(body, turn)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        return {"error": f"the mock model couldn't read this body ({type(exc).__name__}: {exc}); "
                         "each message needs a role and content, each tool a function.name"}
    return {"request": {"method": "POST", "url": mcp_lab.OPENAI_URL,
                        "headers": {"Authorization": "Bearer sk-...REDACTED", "Content-Type": "application/json"},
                        "body": body},
            "response": {"status": status, "reason": "OK" if status == 200 else "Bad Request",
                         "http_version": "MOCK (nothing sent)", "elapsed_ms": 0,
                         "headers": headers, "body": json.loads(text)}}
