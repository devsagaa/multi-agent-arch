"""Observability: spans, cross-process trace context, structured logs and derived metrics (offline)."""

import json
import os
import uuid

import pytest

import telemetry
import trace_store
from agents import ResearchAgents
from graph import build_graph, run_config
from llm import MockLLM
from mcp_client import MCPToolbox, mcp_tool_functions
from state import initial_state
from test_workflow import make_settings, store  # noqa: F401 -- store is a fixture


@pytest.fixture(scope="module")
def otel_dir(tmp_path_factory):
    """Telemetry on, writing to a temp dir. The provider is global, so it stays on for the session."""
    if not telemetry.active():
        os.environ["TELEMETRY"] = "on"
        os.environ["TELEMETRY_DIR"] = str(tmp_path_factory.mktemp("otel"))
        os.environ.pop(telemetry.CAPTURE_ENV, None)
        assert telemetry.setup("research-app")
    return telemetry.telemetry_dir()


def run_graph(store, settings, llm=None, search=None, fetch=None, question="What is MCP?"):
    graph = build_graph(ResearchAgents(llm or MockLLM(), settings, store, search=search, fetch=fetch))
    run_id = str(uuid.uuid4())
    with telemetry.run_span(run_id, question, settings, entry="test") as root:
        final = graph.invoke(initial_state(question), run_config(run_id, settings.max_revisions))
        telemetry.finish_run(root, final)
        trace_id = telemetry.current_trace_id()
    return final, trace_id


def tree(detail: dict) -> tuple[dict, dict]:
    by_id = {s["span_id"]: s for s in detail["spans"]}
    parent = {s["span_id"]: by_id.get(s["parent_span_id"]) for s in detail["spans"]}
    return by_id, parent


def test_a_run_is_one_trace_of_nested_spans(otel_dir, store):
    final, trace_id = run_graph(store, make_settings(researcher_mode="agentic"))
    detail = trace_store.trace_detail(otel_dir, trace_id)
    _, parent = tree(detail)
    names = [s["name"] for s in detail["spans"]]
    root = next(s for s in detail["spans"] if s["name"] == telemetry.RUN_SPAN)
    assert root["parent_span_id"] is None and root["attributes"]["app.outcome"] == "approved"
    assert root["attributes"]["app.tool_calls"] == len(final["tool_calls"])
    assert {"node load_memory", "node supervisor", "invoke_agent researcher", "invoke_agent reviewer"} <= set(names)
    # Model calls follow the GenAI conventions and sit inside the agent that made them.
    chats = [s for s in detail["spans"] if s["attributes"].get("gen_ai.operation.name") == "chat"]
    assert chats and all(s["kind"] == "CLIENT" and parent[s["span_id"]]["name"].startswith("invoke_agent ") for s in chats)
    assert {s["attributes"]["app.role"] for s in chats} >= {"researcher", "reviewer", "finalizer"}
    # Tool executions carry the model's tool_call id, and the provider call nests under them.
    tool = next(s for s in detail["spans"] if s["name"] == "execute_tool web_search")
    assert tool["attributes"]["gen_ai.tool.call.id"].startswith("call_")
    assert any(parent[s["span_id"]] is tool for s in detail["spans"] if s["name"] == "web_search mock")
    # Every log line written during the run is tagged with the run's trace id.
    assert detail["logs"] and all(l["trace_id"] == trace_id for l in detail["logs"])


def test_parallel_fetches_keep_their_parent_span(otel_dir, store):
    """Thread-pool workers start with an empty context; _parallel_map carries the span over."""
    _, trace_id = run_graph(store, make_settings(researcher_mode="pipeline"))
    detail = trace_store.trace_detail(otel_dir, trace_id)
    _, parent = tree(detail)
    fetches = [s for s in detail["spans"] if s["name"] == "execute_tool fetch_page"]
    assert len(fetches) >= 2 and all(parent[s["span_id"]]["name"] == "invoke_agent researcher" for s in fetches)


def test_trace_continues_inside_the_mcp_server_process(otel_dir, store):
    box = MCPToolbox(env={"SEARCH_PROVIDER": "mock"})        # started after setup: inherits TELEMETRY_DIR
    try:
        search, fetch = mcp_tool_functions(box, 3, 500)
        _, trace_id = run_graph(store, make_settings(researcher_mode="agentic"), search=search, fetch=fetch)
    finally:
        box.close()                                          # the server exits; its spans are already on disk
    detail = trace_store.trace_detail(otel_dir, trace_id)
    _, parent = tree(detail)
    assert detail["services"] == ["mcp-server", "research-app"]
    server = [s for s in detail["spans"] if s["kind"] == "SERVER"]
    assert server and all(parent[s["span_id"]]["kind"] == "CLIENT"
                          and parent[s["span_id"]]["service"] == "research-app" for s in server)
    # The server's own work (tools.py) is a child of its SERVER span, and its logs share the trace id.
    assert any(parent[s["span_id"]] in server for s in detail["spans"] if s["name"] == "web_search mock")
    assert any(l["service"] == "mcp-server" for l in detail["logs"])


def test_content_is_not_recorded_by_default_and_secrets_never(otel_dir, store):
    question = "zebra-question-" + uuid.uuid4().hex
    run_graph(store, make_settings(), question=question)
    files = list(otel_dir.glob("*.jsonl"))
    assert files and not any(question in f.read_text(encoding="utf-8") for f in files if f.name.startswith("spans-"))
    assert telemetry.redact("key sk-abcdefghijklmnop and Authorization: Bearer abc.def.ghi123") == \
        "key sk-...REDACTED and Authorization: Bearer ...REDACTED"


def test_overview_metrics_and_failures_counted_once(otel_dir, store):
    def broken_search(query):
        return {"tool": "web_search", "provider": "test", "input": query, "ok": False, "result_count": 0,
                "error": "Search request failed: TimeoutError: 30 s", "results": []}
    final, trace_id = run_graph(store, make_settings(researcher_mode="pipeline"), search=broken_search)
    assert final["outcome"] == "failed"
    ov = trace_store.overview(otel_dir)
    run = next(r for r in ov["runs"] if r["trace_id"] == trace_id)
    assert run["outcome"] == "failed" and run["tool_failures"] == 1 and run["status"] == "ERROR"
    # One failed search marks execute_tool, the node and the run as failed -- but it's ONE failure.
    errs = [e for e in ov["errors"] if e["example_trace"] == trace_id]
    assert len(errs) == 1 and errs[0]["operation"] == "execute_tool web_search"
    assert errs[0]["message"] == "Search request failed: TimeoutError: # s"
    assert ov["summary"]["runs"] >= 2 and ov["summary"]["duration_p95_ms"] is not None
    assert any(o["operation"] == "model call: researcher" for o in ov["operations"])


def test_spans_also_carry_openinference_attributes_for_phoenix(otel_dir, store, monkeypatch):
    kinds = lambda d: {s["name"].split(" ")[0]: s["attributes"].get(telemetry.OI_KIND) for s in d["spans"]}
    _, trace_id = run_graph(store, make_settings(researcher_mode="agentic"))
    detail = trace_store.trace_detail(otel_dir, trace_id)
    k = kinds(detail)
    assert k["invoke_agent"] == "AGENT" and k["node"] == "CHAIN" and k["chat"] == "LLM" and k["execute_tool"] == "TOOL"
    chat = next(s for s in detail["spans"] if s["name"].startswith("chat "))["attributes"]
    assert chat["llm.model_name"] == "mock" and "llm.token_count.total" in chat and "llm.tools.0.tool.json_schema" in chat
    assert not any(key.startswith("llm.input_messages") for key in chat)            # content stays off by default
    # With content capture on, messages are flattened the OpenInference way.
    monkeypatch.setenv(telemetry.CAPTURE_ENV, "true")
    _, trace_id = run_graph(store, make_settings(researcher_mode="agentic"), question="capture me")
    spans = trace_store.trace_detail(otel_dir, trace_id)["spans"]
    chat = next(s for s in spans if s["name"].startswith("chat "))["attributes"]
    assert chat["llm.input_messages.0.message.role"] == "system" and "capture me" in chat["llm.input_messages.1.message.content"]
    assert chat["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"] == "web_search"
    root = next(s for s in spans if s["name"] == telemetry.RUN_SPAN)["attributes"]
    assert root["input.value"] == "capture me" and root["output.value"].startswith("[MOCK]")


def test_unreachable_otlp_endpoint_is_skipped_quickly():
    import time
    start = time.perf_counter()
    assert telemetry._reachable("http://127.0.0.1:9", timeout=0.3) is False
    assert time.perf_counter() - start < 2


def test_tracing_never_breaks_a_model_call():
    class Minimal:                                  # an LLM without the optional build_* helpers
        name = "mock"
        def complete(self, role, system, user, schema=None):
            return MockLLM().complete(role, system, user, schema)
    assert telemetry.TracedLLM(Minimal()).complete("finalizer", "s", "u").text.startswith("[MOCK]")


def test_percentiles_and_bad_lines():
    assert trace_store.pct([5, 1, 3, 2, 4], 50) == 3 and trace_store.pct(list(range(1, 21)), 95) == 19
    assert trace_store.pct([], 95) is None
    assert not trace_store.valid_trace_id("../../etc/passwd") and trace_store.valid_trace_id("a" * 32)


def test_trace_context_round_trips_through_meta(otel_dir):
    with telemetry.span("client") as s:
        meta = telemetry.inject_meta()
        sent = json.loads(json.dumps({"_meta": meta}))["_meta"]          # as it travels in JSON-RPC
    tid = f"{s.get_span_context().trace_id:032x}"
    assert sent["traceparent"].split("-")[1] == tid
    with telemetry.span("server", context=telemetry.extract_meta(sent)):
        assert telemetry.current_trace_id() == tid
    assert telemetry.extract_meta(None) is None and telemetry.extract_meta({}) is None
