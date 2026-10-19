"""Local web UI for the research graph. Standard library only.

  python server.py            -> http://127.0.0.1:8000

Endpoints:
  GET    /                 the inspector (static/index.html)
  GET    /flow             the "how it works" explainer (static/flow.html)
  GET    /http             HTTP lab: raw OpenAI calls with httpx, no SDK (static/http.html)
  POST   /api/http-lab     {"body": {...}} send one raw request, return status/headers/body
  GET    /mcp              MCP lab: app <-> MCP client <-> MCP server <-> LLM, every raw message (static/mcp.html)
  GET    /mcp-console      MCP console: you play the app -- send raw JSON-RPC and raw LLM requests yourself
  POST   /api/mcp-console/{open,send,close,llm}   one live MCP session per browser tab
  POST   /api/mcp-lab      {"mode": "agent"|"tool", "question" | "tool"+"arguments", "transport": "stdio"|"http",
                             "url", "token", "mock_llm", "mock_search"} run one traced flow
  GET    /traces           observability: runs, latency/cost/failure summary, waterfall per trace
  GET    /api/traces       runs (root spans) + cross-run summary, from traces/otel/*.jsonl
  GET    /api/traces/<id>  every span (all processes) and every log line of one trace
  GET    /api/config       providers, max revisions, which keys each node reads
  GET    /api/run?...      runs the graph, streams every step as Server-Sent Events
  GET    /api/memory       list long-term memory
  POST   /api/memory       {"content": "..."} save a user preference
  DELETE /api/memory       clear long-term memory

How the trace is captured: LangGraph's stream_mode="debug" emits a "task" event
with the exact state a node receives, and a "task_result" event with the dict it
returns. Model prompts/responses come from RecordingLLM, drained after each node.
"""

import argparse
import dataclasses
import json
import logging
import os
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import http_lab
import mcp_console
import mcp_lab
import telemetry
import trace_store
from agents import NODE_READS, ResearchAgents
from config import load_settings
from langgraph.errors import GraphRecursionError

from graph import build_graph, run_config
from llm import RecordingLLM, build_llm
from memory import MemoryStore
from state import initial_state

ROOT = Path(__file__).parent
INDEX = ROOT / "static" / "index.html"


def _json(obj) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


class Handler(BaseHTTPRequestHandler):
    # ------------------------------------------------------------ helpers
    def _send(self, status: int, body: str, ctype: str = "application/json") -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _event(self, kind: str, payload: dict) -> None:
        self.wfile.write(f"event: {kind}\ndata: {_json(payload)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def log_message(self, fmt, *args):  # keep the console quiet except for runs
        pass

    # ------------------------------------------------------------ routes
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._send(200, INDEX.read_text(encoding="utf-8"), "text/html")
        elif url.path == "/flow":
            self._send(200, (ROOT / "static" / "flow.html").read_text(encoding="utf-8"), "text/html")
        elif url.path == "/http":
            self._send(200, (ROOT / "static" / "http.html").read_text(encoding="utf-8"), "text/html")
        elif url.path == "/mcp":
            self._send(200, (ROOT / "static" / "mcp.html").read_text(encoding="utf-8"), "text/html")
        elif url.path == "/mcp-console":
            self._send(200, (ROOT / "static" / "mcp-console.html").read_text(encoding="utf-8"), "text/html")
        elif url.path == "/traces":
            self._send(200, (ROOT / "static" / "traces.html").read_text(encoding="utf-8"), "text/html")
        elif url.path == "/api/traces":
            raw_days = (parse_qs(url.query).get("days") or ["7"])[0]
            days = int(raw_days) if raw_days.isdigit() else 7
            overview = trace_store.overview(telemetry.telemetry_dir(), max(1, min(days, 90)))
            self._send(200, _json({**overview, "capture_content": telemetry.capture_content()}))
        elif url.path.startswith("/api/traces/"):
            trace_id = url.path.rsplit("/", 1)[-1]
            if not trace_store.valid_trace_id(trace_id):
                return self._send(400, _json({"error": "a trace id is 32 hex characters"}))
            self._send(200, _json(trace_store.trace_detail(telemetry.telemetry_dir(), trace_id)))
        elif url.path == "/api/config":
            s = load_settings()
            self._send(200, _json({
                "llm_provider": s.llm_provider, "openai_model": s.openai_model,
                "search_provider": s.search_provider, "max_revisions": s.max_revisions,
                "researcher_mode": s.researcher_mode, "tool_transport": s.tool_transport,
                "node_reads": NODE_READS,
            }))
        elif url.path == "/api/memory":
            store = MemoryStore(load_settings().memory_db_path)
            try:
                self._send(200, _json(store.all()))
            finally:
                store.close()
        elif url.path == "/api/run":
            self._run(parse_qs(url.query))
        else:
            self._send(404, _json({"error": "not found"}))

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/http-lab":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}").get("body")
            except ValueError:
                return self._send(400, _json({"error": "request must be JSON"}))
            return self._send(200, _json(http_lab.send(body)))
        if path.startswith("/api/mcp-console/"):
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._send(400, _json({"error": "request must be JSON"}))
            action = path.rsplit("/", 1)[-1]
            if action == "open":
                url = str(req.get("url") or "").strip() or None
                if url and urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
                    return self._send(400, _json({"error": "the console only connects to MCP servers on localhost"}))
                provider = "mock" if req.get("mock_search") else load_settings().search_provider
                return self._send(200, _json(mcp_console.open_session(
                    "http" if req.get("transport") == "http" else "stdio", provider, url,
                    str(req.get("token") or "").strip() or None)))
            if action == "send":
                return self._send(200, _json(mcp_console.send(str(req.get("session")), str(req.get("raw") or ""))))
            if action == "close":
                return self._send(200, _json(mcp_console.close_session(str(req.get("session")))))
            if action == "llm":
                return self._send(200, _json(mcp_console.llm(req.get("body"), bool(req.get("mock")), int(req.get("turn") or 1))))
            return self._send(404, _json({"error": "not found"}))
        if path == "/api/mcp-lab":
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._send(400, _json({"error": "request must be JSON"}))
            transport = "http" if req.get("transport") == "http" else "stdio"
            url = str(req.get("url") or "").strip() or None
            token = str(req.get("token") or "").strip() or None
            # The lab only talks to MCP servers on this machine -- it must not become a proxy to other hosts.
            if url and urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
                return self._send(400, _json({"error": "the lab only connects to MCP servers on localhost"}))
            mock_search = bool(req.get("mock_search"))
            if req.get("mode") == "tool":
                tool, arguments = str(req.get("tool") or "").strip(), req.get("arguments")
                if not tool or not isinstance(arguments, dict):
                    return self._send(400, _json({"error": "tool (string) and arguments (JSON object) are required"}))
                return self._send(200, _json(mcp_lab.run_tool(tool, arguments, mock_search, transport, url, token)))
            question = str(req.get("question") or "").strip()[:500]
            if not question:
                return self._send(400, _json({"error": "question is required"}))
            return self._send(200, _json(mcp_lab.run(question, bool(req.get("mock_llm")), mock_search,
                                                     transport, url, token)))
        if path != "/api/memory":
            return self._send(404, _json({"error": "not found"}))
        length = int(self.headers.get("Content-Length", 0))
        content = (json.loads(self.rfile.read(length) or b"{}").get("content") or "").strip()
        if not content:
            return self._send(400, _json({"error": "content is required"}))
        store = MemoryStore(load_settings().memory_db_path)
        try:
            added = store.save(content, category="preference", source="user")
            self._send(200, _json({"added": added}))
        finally:
            store.close()

    def do_DELETE(self):
        if urlparse(self.path).path != "/api/memory":
            return self._send(404, _json({"error": "not found"}))
        store = MemoryStore(load_settings().memory_db_path)
        try:
            store.clear()
            self._send(200, _json({"cleared": True}))
        finally:
            store.close()

    # ------------------------------------------------------------ graph run
    def _run(self, q: dict) -> None:
        flag = lambda k: q.get(k, ["0"])[0] in {"1", "true", "on"}
        question = q.get("question", [""])[0].strip()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        if not question:
            return self._event("fail", {"message": "Please enter a question."})

        settings = load_settings()
        settings = dataclasses.replace(
            settings,
            llm_provider="mock" if flag("mock_llm") else settings.llm_provider,
            search_provider="mock" if flag("mock_search") else settings.search_provider,
            force_reject_first_review=flag("force_reject"),
            researcher_mode="pipeline" if flag("pipeline") else settings.researcher_mode,
            tool_transport="mcp" if flag("mcp") else settings.tool_transport,
        )
        store = MemoryStore(settings.memory_db_path)  # one connection per request thread
        try:
            llm = RecordingLLM(build_llm(settings.llm_provider, settings.openai_model,
                  settings.llm_timeout_s, settings.llm_max_retries))
            graph = build_graph(ResearchAgents(llm, settings, store))
            run_id = str(uuid.uuid4())
            config = run_config(run_id, settings.max_revisions)
            print(f"run: {question!r} llm={settings.llm_provider} search={settings.search_provider}")

            with telemetry.run_span(run_id, question, settings, entry="web") as root:
                self._event("start", {"question": question, "llm": llm.name,
                                      "search": settings.search_provider,
                                      "max_revisions": settings.max_revisions,
                                      "researcher_mode": settings.researcher_mode,
                                      "tool_transport": settings.tool_transport,
                                      "force_reject": settings.force_reject_first_review,
                                      "trace_id": telemetry.current_trace_id()})
                started: dict[str, float] = {}
                inputs: dict[str, dict] = {}
                for ev in graph.stream(initial_state(question), config, stream_mode="debug"):
                    p = ev["payload"]
                    if ev["type"] == "task":
                        started[p["id"]] = time.perf_counter()
                        inputs[p["id"]] = p["input"]
                        self._event("node_start", {"step": ev["step"], "node": p["name"]})
                    elif ev["type"] == "task_result":
                        ms = round((time.perf_counter() - started.pop(p["id"], time.perf_counter())) * 1000)
                        self._event("node_end", {
                            "step": ev["step"], "node": p["name"], "duration_ms": ms,
                            "input": inputs.pop(p["id"], {}),
                            "result": p["result"],
                            "error": str(p["error"]) if p.get("error") else None,
                            "llm_calls": llm.drain(),
                        })

                snapshot = graph.get_state(config)
                telemetry.finish_run(root, snapshot.values)
            self._event("done", {
                "state": snapshot.values,
                "checkpoints": sum(1 for _ in graph.get_state_history(config)),
                "memory": store.all(),
            })
        except GraphRecursionError:
            self._event("fail", {"message": f"Stopped: the graph exceeded its step limit "
                                            f"({config['recursion_limit']}), which means a routing bug created a cycle."})
        except Exception as exc:  # surface crashes in the UI instead of a silent stream
            traceback.print_exc()
            self._event("fail", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            store.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    load_settings()                       # loads .env, which may hold TELEMETRY settings
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")   # console
    telemetry.setup("research-app")       # + JSON logs (INFO and up) and spans in traces/otel
    # On Windows, SO_REUSEADDR lets a second server bind the same port silently
    # (and the old one keeps answering). Disable it so a stale server is an error.
    ThreadingHTTPServer.allow_reuse_address = os.name != "nt"
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)  # localhost only
    except OSError:
        raise SystemExit(
            f"Port {args.port} is already in use -- another server.py is probably still running.\n"
            f"Stop it, or start this one on another port:  python server.py --port {args.port + 1}"
        )
    print(f"Research graph UI: http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
