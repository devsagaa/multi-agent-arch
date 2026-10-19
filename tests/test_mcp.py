"""MCP transport: the same tools, served by mcp_server.py in a subprocess (mock search, offline)."""

import pytest

from llm import MockLLM
from mcp_client import MCPError, MCPToolbox, mcp_tool_functions, shared_toolbox
from test_workflow import make_settings, run, store  # noqa: F401 -- store is a fixture


@pytest.fixture(scope="module")
def box():
    with MCPToolbox(env={"SEARCH_PROVIDER": "mock"}) as b:
        yield b


def test_discovery_lists_tools_with_json_schemas(box):
    tools = {t.name: t for t in box.tools}
    assert set(tools) == {"web_search", "fetch_url"}
    assert tools["web_search"].inputSchema["required"] == ["query"]
    assert box.server_info["name"] == "research-tools"
    # MCP inputSchema is plain JSON Schema, so it maps straight onto OpenAI's `parameters`.
    fn = next(t["function"] for t in box.openai_tools() if t["function"]["name"] == "web_search")
    assert fn["parameters"]["properties"]["query"]["type"] == "string"


def test_tools_call_returns_the_same_record_as_a_local_call(box):
    rec = box.call("web_search", {"query": "agents", "max_results": 2})
    assert rec["ok"] and rec["result_count"] == 2 and rec["provider"] == "mock"
    page = box.call("fetch_url", {"url": rec["results"][0]["url"]})
    assert page["ok"] and "[MOCK]" in page["content"]


def test_bad_arguments_are_a_protocol_error(box):
    with pytest.raises(MCPError, match="web_search"):
        box.call("web_search", {})                     # missing required 'query'


def test_wrapped_functions_never_raise(box):
    search, fetch = mcp_tool_functions(box, max_results=3, page_max_chars=500)
    assert search("q")["transport"] == "mcp"
    dead = MCPToolbox(env={"SEARCH_PROVIDER": "mock"})
    dead.close()                                       # server gone: calls must fail as records
    rec = mcp_tool_functions(dead, 3, 500)[0]("q")
    assert rec["ok"] is False and "MCP call failed" in rec["error"] and rec["results"] == []


def test_server_does_not_inherit_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    with MCPToolbox(env={"SEARCH_PROVIDER": "mock"}) as b:
        assert "OPENAI_API_KEY" not in b.params.env and b.params.env["SEARCH_PROVIDER"] == "mock"


def test_server_that_fails_to_start_raises_clearly():
    with pytest.raises(MCPError, match="failed to start|did not start"):
        MCPToolbox(args=["-c", "import sys; sys.exit(3)"], timeout_s=10)


@pytest.mark.parametrize("mode", ["agentic", "pipeline"])
def test_full_graph_runs_with_tools_behind_mcp(store, mode):
    settings = make_settings(tool_transport="mcp", researcher_mode=mode)
    final = run(MockLLM(), store, settings)
    assert final["outcome"] == "approved"
    assert final["tool_calls"] and all(c.get("transport") == "mcp" for c in final["tool_calls"])
    assert shared_toolbox("mock") is shared_toolbox("mock")   # one server process reused


# ---------------------------------------------------------------- Streamable HTTP transport
@pytest.fixture(scope="module")
def http_server():
    """mcp_server.py --http on a free localhost port, with a bearer token. Stopped after the module."""
    import os, socket, subprocess, sys, time
    import httpx
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    env = {**os.environ, "SEARCH_PROVIDER": "mock", "MCP_TOKEN": "test-token"}
    proc = subprocess.Popen([sys.executable, "mcp_server.py", "--http", "--port", str(port)], env=env,
                            stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}/mcp"
    for _ in range(100):
        try:
            httpx.get(url, timeout=0.5); break
        except httpx.TransportError:
            time.sleep(0.1)
    yield url
    import mcp_client
    if (shared := mcp_client._shared.pop(url, None)):   # close our session before the server goes away
        shared.close()
    proc.terminate(); proc.wait(10)


def test_http_transport_lists_and_calls_tools(http_server):
    with MCPToolbox(url=http_server, token="test-token") as b:
        assert {t.name for t in b.tools} == {"web_search", "fetch_url"}
        assert b.call("web_search", {"query": "x", "max_results": 1})["result_count"] == 1


def test_http_transport_rejects_a_wrong_token(http_server):
    with pytest.raises(MCPError, match="401"):
        MCPToolbox(url=http_server, token="wrong", timeout_s=10)


def test_full_graph_uses_a_running_http_server(store, http_server):
    settings = make_settings(tool_transport="mcp", researcher_mode="agentic", mcp_url=http_server, mcp_token="test-token")
    final = run(MockLLM(), store, settings)
    assert final["outcome"] == "approved" and all(c.get("transport") == "mcp" for c in final["tool_calls"])


def test_mcp_lab_records_every_hop_offline():
    import mcp_lab
    out = mcp_lab.run("What is MCP?", mock_llm=True, mock_search=True)
    assert out["error"] is None and out["answer"].startswith("[MOCK]")
    kinds = [(e["from"], e["to"], e["kind"]) for e in out["events"]]
    assert ("client", "server", "jsonrpc-request") in kinds and ("app", "llm", "http-request") in kinds
    methods = [__import__("json").loads(e["raw"]).get("method") for e in out["events"] if e["kind"].startswith("jsonrpc") and e["from"] == "client"]
    assert methods[:3] == ["initialize", "notifications/initialized", "tools/list"] and "tools/call" in methods
    assert all("REDACTED" in e["raw"] for e in out["events"] if e["kind"] == "http-request")   # key never recorded


def test_mcp_lab_over_http_shows_session_header_and_redacts_token():
    import mcp_lab
    out = mcp_lab.run("What is MCP?", mock_llm=True, mock_search=True, transport="http")   # temporary server
    assert out["error"] is None
    posts = [e["raw"] for e in out["events"] if e["from"] == "client" and e["raw"].startswith("POST")]
    assert posts and all("Bearer ...REDACTED" in p for p in posts)
    assert "mcp-session-id" not in posts[0].lower() and all("mcp-session-id" in p.lower() for p in posts[1:])
    assert any(e["raw"].startswith("DELETE") for e in out["events"])                  # session ended explicitly


def test_mcp_lab_direct_tool_call_needs_no_model():
    import mcp_lab
    ok = mcp_lab.run_tool("web_search", {"query": "x", "max_results": 1}, mock_search=True)
    assert ok["error"] is None and '"ok": true' in ok["answer"]
    assert not any(e["to"] == "llm" for e in ok["events"])
    assert "Field required" in mcp_lab.run_tool("web_search", {}, mock_search=True)["answer"]
    assert "Unknown tool" in mcp_lab.run_tool("delete_files", {}, mock_search=True)["answer"]


# ---------------------------------------------------------------- MCP console (manual sessions)
def test_console_stdio_session_round_trip():
    import json, mcp_console
    o = mcp_console.open_session("stdio", "mock"); sid = o["session"]
    try:
        before = mcp_console.send(sid, '{"jsonrpc":"2.0","id":1,"method":"tools/list"}')
        assert '"error"' in before["response_raw"]                                 # not initialized yet
        mcp_console.send(sid, json.dumps({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}))
        assert mcp_console.send(sid, '{"jsonrpc":"2.0","method":"notifications/initialized"}')["no_reply_expected"]
        call = mcp_console.send(sid, '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"web_search","arguments":{"query":"x"}}}')
        assert '"isError":false' in call["response_raw"] and call["server_internals"]
        # Over stdio the SDK answers unparsable input with an error *log notification*, not a reply.
        bad = mcp_console.send(sid, "{not json")["response_raw"]
        assert '"notifications/message"' in bad and '"level":"error"' in bad
    finally:
        assert "exited" in mcp_console.close_session(sid)["closed"]


def test_console_mock_model_follows_tool_choice_and_api_rules():
    import json, mcp_console
    tools = [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}} for n in ("web_search", "fetch_url")]
    base = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    ask = lambda **b: mcp_console.llm({"model": "m", "messages": base, "tools": tools, **b}, mock=True, turn=1)["response"]
    choice = lambda r: r["body"]["choices"][0]
    assert choice(ask(tool_choice="none"))["finish_reason"] == "stop"
    assert choice(ask(tool_choice={"type": "function", "function": {"name": "fetch_url"}}))["message"]["tool_calls"][0]["function"]["name"] == "fetch_url"
    assert len(choice(ask(parallel_tool_calls=True))["message"]["tool_calls"]) == 2
    assert choice(ask(tool_choice="none", max_completion_tokens=3))["finish_reason"] == "length"
    assert ask(tool_choice={"type": "function", "function": {"name": "nope"}})["status"] == 400
    asked = {"role": "assistant", "content": None, "tool_calls": [
        {"id": c, "type": "function", "function": {"name": "web_search", "arguments": "{}"}} for c in ("A", "B")]}
    answer = lambda c: {"role": "tool", "tool_call_id": c, "content": json.dumps({"ok": True, "results": []})}
    missing = ask(messages=base + [asked, answer("A")])
    assert missing["status"] == 400 and "B" in missing["body"]["error"]["message"]          # every id needs a result
    assert ask(messages=base + [asked, answer("A"), answer("X")])["status"] == 400        # unknown tool_call_id
    assert ask(messages=base + [asked, answer("A"), answer("B")])["status"] == 200
    # "required" never lets the model stop, however many rounds have run.
    many = base + [asked, answer("A"), answer("B")] * 3
    assert choice(ask(tool_choice="required", messages=many))["finish_reason"] == "tool_calls"


def test_console_http_session_only_starts_on_successful_initialize():
    import json, mcp_console
    o = mcp_console.open_session("http", "mock"); sid = o["session"]
    try:
        early = mcp_console.send(sid, '{"jsonrpc":"2.0","id":1,"method":"tools/list"}')
        assert " 400 " in early["response_raw"].splitlines()[0] and early["session_id"] is None
        init = mcp_console.send(sid, json.dumps({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}))
        assert init["session_id"] and "mcp-session-id" not in init["request_raw"].lower()
        listed = mcp_console.send(sid, '{"jsonrpc":"2.0","id":3,"method":"tools/list"}')
        assert " 200 " in listed["response_raw"].splitlines()[0] and "mcp-session-id" in listed["request_raw"].lower()
    finally:
        mcp_console.close_session(sid)
