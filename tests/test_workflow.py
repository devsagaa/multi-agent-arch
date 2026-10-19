"""Workflow tests. Fully offline: MockLLM + mock search + a temp SQLite file."""

import pytest

from agents import ResearchAgents, decide_next_step
from config import Settings
from graph import build_graph, recursion_limit, run_config
from llm import LLMError, MockLLM
from memory import MemoryStore
from state import initial_state
from tools import web_search

QUESTION = "Should a software engineer learn AWS before building production AI agents?"


def make_settings(**overrides) -> Settings:
    base = dict(llm_provider="mock", openai_model="unused", llm_timeout_s=5, llm_max_retries=0,
                price_input_per_m=0.4, price_output_per_m=1.6, search_provider="mock",
                search_max_results=3, fetch_pages=2, page_max_chars=500, max_revisions=1, force_reject_first_review=False,
                memory_db_path=":memory:", save_mock_answers=True,
                researcher_mode="pipeline")
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "mem.db"))
    yield s
    s.close()


def run(llm, store, settings=None, search=None, fetch=None):
    agents = ResearchAgents(llm, settings or make_settings(), store, search=search, fetch=fetch)
    graph = build_graph(agents)
    settings = settings or make_settings()
    return graph.invoke(initial_state(QUESTION), run_config("t", settings.max_revisions))


def nodes_run(state):
    return [h["node"] for h in state["history"]]


def calls_of(state, tool):
    return [c for c in state["tool_calls"] if c["tool"] == tool]


# ---------------------------------------------------------------- routing (pure)
@pytest.mark.parametrize("state,expected", [
    ({"review_status": "not_started"}, "researcher"),
    ({"review_status": "pending"}, "reviewer"),
    ({"review_status": "approved"}, "finalizer"),
    ({"review_status": "rejected", "revision_count": 0}, "researcher"),
    ({"review_status": "rejected", "revision_count": 1}, "finalizer"),
    ({"review_status": "pending", "error": "boom"}, "finalizer"),
    ({"review_status": "approved", "final_answer": "x"}, "save_memory"),
])
def test_decide_next_step(state, expected):
    assert decide_next_step(state, max_revisions=1)[0] == expected


# ---------------------------------------------------------------- shared state
def test_happy_path_updates_shared_state(store):
    final = run(MockLLM(), store)
    assert final["research_findings"].startswith("[MOCK]")
    assert len(final["sources"]) == 3
    assert len(calls_of(final, "web_search")) == 1 and final["tool_calls"][0]["ok"]
    assert len(calls_of(final, "fetch_page")) == 2   # fetch_pages=2
    assert final["review_status"] == "approved"
    assert final["outcome"] == "approved"
    assert final["final_answer"]
    assert nodes_run(final) == ["load_memory", "supervisor", "researcher", "supervisor",
                                "reviewer", "supervisor", "finalizer", "supervisor", "save_memory"]


def test_reviewer_reads_researcher_findings_from_state(store):
    llm = MockLLM()
    final = run(llm, store)
    reviewer_prompt = next(c["user"] for c in llm.calls if c["role"] == "reviewer")
    # The reviewer was never called by the researcher; it got the findings via state.
    assert final["research_findings"] in reviewer_prompt


# ---------------------------------------------------------------- retry path
def test_forced_rejection_follows_retry_path(store):
    final = run(MockLLM(), store, make_settings(force_reject_first_review=True))
    nodes = nodes_run(final)
    assert nodes.count("researcher") == 2
    assert nodes.count("reviewer") == 2
    assert final["revision_count"] == 1
    assert final["outcome"] == "approved"
    # The second search was steered by the reviewer's feedback.
    searches = calls_of(final, "web_search")
    assert searches[1]["input"] != searches[0]["input"]


def test_max_revisions_enforced_when_reviewer_always_rejects(store):
    llm = MockLLM(review_verdicts=[False, False, False, False])
    final = run(llm, store)
    nodes = nodes_run(final)
    assert nodes.count("researcher") == 2   # original + 1 revision, no more
    assert nodes.count("reviewer") == 2
    assert final["revision_count"] == 1
    assert final["outcome"] == "unapproved"
    assert "did NOT pass review" in final["final_answer"]
    assert store.all() == []  # unapproved answers are not saved to long-term memory


def test_max_revisions_is_configurable(store):
    # Scores keep improving (3 -> 5 -> 6), so every allowed revision is used.
    llm = MockLLM(review_verdicts=[(False, 3), (False, 5), (False, 6)])
    final = run(llm, store, make_settings(max_revisions=2))
    assert nodes_run(final).count("researcher") == 3


# ---------------------------------------------------------------- failures
def _failing_search(query):
    return {"tool": "web_search", "provider": "test", "input": query, "ok": False,
            "result_count": 0, "error": "network down", "results": []}


def test_search_failure_does_not_fabricate_results(store):
    llm = MockLLM()
    final = run(llm, store, search=_failing_search)
    assert final["outcome"] == "failed"
    assert final["research_findings"] == ""
    assert final["sources"] == []
    assert "network down" in final["final_answer"]
    assert "reviewer" not in nodes_run(final)
    assert llm.calls == []                  # no model was asked to "answer anyway"
    assert store.all() == []


class _BrokenLLM(MockLLM):
    def complete(self, role, system, user, schema=None):
        raise LLMError(f"simulated outage in {role}")


def test_model_failure_is_reported_not_hidden(store):
    final = run(_BrokenLLM(), store)
    assert final["outcome"] == "failed"
    assert "simulated outage" in final["final_answer"]
    assert final["research_findings"] == ""
    assert store.all() == []


def test_web_search_handles_exceptions_and_empty_results(monkeypatch):
    import tools

    def boom(q, n):
        raise ConnectionError("no internet")

    monkeypatch.setitem(tools.PROVIDERS, "ddgs", boom)
    rec = web_search("x", provider="ddgs")
    assert rec["ok"] is False and "no internet" in rec["error"] and rec["results"] == []

    monkeypatch.setitem(tools.PROVIDERS, "ddgs", lambda q, n: [])
    rec = web_search("x", provider="ddgs")
    assert rec["ok"] is False and rec["error"] == "Search returned no results."

    assert web_search("x", provider="nope")["ok"] is False


# ---------------------------------------------------------------- memory in the graph
def test_saved_memory_is_loaded_into_state_and_prompts(store):
    store.save("I prefer concise answers", category="preference")
    llm = MockLLM()
    final = run(llm, store)
    assert "User preference: I prefer concise answers" in final["memory_context"]
    finalizer_prompt = next(c["user"] for c in llm.calls if c["role"] == "finalizer")
    assert "I prefer concise answers" in finalizer_prompt
    assert any(m["source"] == "agent:mock" for m in store.all())  # approved answer saved


# ---------------------------------------------------------------- fetch_page
def test_full_page_text_reaches_the_researcher_prompt(store):
    llm = MockLLM()
    final = run(llm, store)
    prompt = next(c["user"] for c in llm.calls if c["role"] == "researcher")
    assert prompt.count("Full page text:") == 2          # top 2 pages read
    assert "[MOCK] Full page text for https://example.com/mock/1" in prompt
    assert "content" in final["sources"][0] and "content" not in final["sources"][2]


def test_failed_fetch_falls_back_to_snippet_without_failing_run(store):
    def flaky_fetch(url):
        ok = url.endswith("/1")
        return {"tool": "fetch_page", "provider": "test", "input": url, "ok": ok,
                "result_count": int(ok), "error": None if ok else "HTTP Error 403",
                "results": [], "content": "page one text" if ok else ""}

    llm = MockLLM()
    final = run(llm, store, fetch=flaky_fetch)
    assert final["outcome"] == "approved"
    failed = [c for c in calls_of(final, "fetch_page") if not c["ok"]]
    assert len(failed) == 1 and failed[0]["error"] == "HTTP Error 403"
    prompt = next(c["user"] for c in llm.calls if c["role"] == "researcher")
    assert prompt.count("Full page text:") == 1
    assert "1/2 pages read in full" in final["history"][2]["detail"]


def test_fetch_page_guards_and_html_extraction(monkeypatch):
    import tools
    from tools import fetch_page, html_to_text

    assert fetch_page("file:///etc/passwd")["ok"] is False
    assert fetch_page("https://x", provider="nope")["ok"] is False

    def boom(url, timeout):
        raise TimeoutError("timed out")
    monkeypatch.setitem(tools.FETCHERS, "http", boom)
    rec = fetch_page("https://example.com")
    assert rec["ok"] is False and "timed out" in rec["error"] and rec["content"] == ""

    monkeypatch.setitem(tools.FETCHERS, "http", lambda u, t: "x" * 50)
    rec = fetch_page("https://example.com", max_chars=10)
    assert rec["ok"] and rec["content"].startswith("x" * 10) and rec["content"].endswith("[truncated]")

    html = ("<html><head><script>var tracking = 1;</script><style>p{}</style></head><body>"
            "<nav>Home About Contact Pricing and many other links</nav>"
            "<p>Agents need observability &amp; retries in production systems.</p></body></html>")
    text = html_to_text(html)
    assert text == "Agents need observability & retries in production systems."


def test_asking_the_same_question_twice_keeps_one_memory_row(store):
    run(MockLLM(), store)
    run(MockLLM(), store)
    facts = [m for m in store.all() if m["category"] == "fact"]
    assert len(facts) == 1


def test_mock_runs_are_not_saved_to_real_memory_by_default(store):
    final = run(MockLLM(), store, make_settings(save_mock_answers=False))
    assert final["outcome"] == "approved"
    assert store.all() == []
    assert "mock" in final["history"][-1]["detail"]



# ---------------------------------------------------------------- stuck detection between nodes
def test_revisions_stop_when_the_score_does_not_improve(store):
    llm = MockLLM(review_verdicts=[(False, 5), (False, 5), (False, 9)])
    final = run(llm, store, make_settings(max_revisions=3))
    assert nodes_run(final).count("researcher") == 2          # 3 revisions allowed, only 1 used
    assert (final["previous_review_score"], final["review_score"]) == (5, 5)
    assert final["outcome"] == "unapproved"
    assert any("did not improve" in h["detail"] for h in final["history"])


@pytest.mark.parametrize("prev,cur,expected", [(None, 4, "researcher"), (4, 6, "researcher"), (6, 6, "finalizer"), (6, 3, "finalizer")])
def test_decide_next_step_revises_only_while_improving(prev, cur, expected):
    state = {"review_status": "rejected", "revision_count": 1, "previous_review_score": prev, "review_score": cur}
    assert decide_next_step(state, max_revisions=3)[0] == expected


def test_first_rejection_always_gets_a_revision():
    # Nothing to compare yet after round 1, whatever the score.
    state = {"review_status": "rejected", "revision_count": 0, "previous_review_score": None, "review_score": 0}
    assert decide_next_step(state, max_revisions=1)[0] == "researcher"


# ---------------------------------------------------------------- recursion limit
def test_recursion_limit_covers_the_longest_legitimate_path(store):
    for max_revisions in (0, 1, 2):
        llm = MockLLM(review_verdicts=[(False, i) for i in range(1, 10)])   # always improving, always rejected
        final = run(llm, store, make_settings(max_revisions=max_revisions))
        assert nodes_run(final).count("researcher") == 1 + max_revisions
        assert len(final["history"]) < recursion_limit(max_revisions)


def test_routing_bug_is_stopped_by_the_recursion_limit(store, monkeypatch):
    import agents
    from langgraph.errors import GraphRecursionError

    # Simulate a bug: the router always sends work back to the researcher.
    monkeypatch.setattr(agents, "decide_next_step", lambda state, m: ("researcher", "bug"))
    llm = MockLLM()
    with pytest.raises(GraphRecursionError):
        run(llm, store)
    assert len([c for c in llm.calls if c["role"] == "researcher"]) <= recursion_limit(1)
