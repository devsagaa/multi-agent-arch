"""Shared graph state: the "working memory" every node reads from and writes to.

Key idea: nodes never call each other. Each node receives the current state,
returns a *partial update* (a dict with only the keys it changed), and LangGraph
merges that update into the state before the next node runs.

How a key is merged is decided by its reducer:
  - plain keys       -> last write wins (the new value replaces the old one)
  - Annotated[list, operator.add] -> the new list is APPENDED to the old one
    (used for logs that must accumulate across nodes and retries)
"""

import operator
from typing import Annotated, Literal, NotRequired, TypedDict


class SearchHit(TypedDict):
    title: str
    url: str
    snippet: str
    content: NotRequired[str]   # full page text, filled in by fetch_page when it succeeds


class ToolCall(TypedDict):
    """One recorded tool invocation, kept for debugging and for the trace."""

    tool: str              # "web_search" or "fetch_page"
    provider: str          # "ddgs", "http" or "mock" -- never hidden from the user
    input: str
    ok: bool
    result_count: int
    error: str | None
    results: list[SearchHit]
    content: NotRequired[str]   # fetch_page only: extracted page text
    # Agentic researcher only: which model tool call asked for this, and with what arguments.
    call_id: NotRequired[str]
    arguments: NotRequired[str]
    transport: NotRequired[str]  # "mcp" when the tool ran in the MCP server process


class LLMUsage(TypedDict):
    """Token/cost accounting for one model call."""

    node: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    attempts: int
    latency_ms: int
    finish_reason: str   # always "stop" here: any other ending raises LLMError instead


class CitationCheck(TypedDict):
    """Deterministic (code, not model) check of the researcher's citations."""

    ok: bool
    cited: list[int]
    invalid: list[int]
    problems: list[str]


class HistoryEntry(TypedDict):
    node: str
    action: str
    detail: str


ReviewStatus = Literal["not_started", "pending", "approved", "rejected"]
NextStep = Literal["researcher", "reviewer", "finalizer", "save_memory"]


class ResearchState(TypedDict, total=False):
    # --- input ---
    question: str

    # --- long-term memory loaded for this run (read-only for agents) ---
    memory_context: list[str]

    # --- written by the researcher, read by the reviewer and finalizer ---
    research_findings: str
    sources: list[SearchHit]
    tool_calls: Annotated[list[ToolCall], operator.add]

    # --- written by the reviewer, read by the supervisor and researcher ---
    review_status: ReviewStatus
    review_feedback: str
    citation_check: CitationCheck
    # 0-10 quality score from the reviewer, and the one before it. The supervisor only
    # sends work back while the score is still improving (stuck detection between nodes).
    review_score: int | None
    previous_review_score: int | None

    # --- written by the supervisor ---
    next_step: NextStep
    revision_count: int

    # --- written by the finalizer ---
    final_answer: str
    outcome: Literal["approved", "unapproved", "failed"]

    # --- reliability and cost ---
    error: str | None
    llm_usage: Annotated[list[LLMUsage], operator.add]

    # --- trace of everything that happened (append-only) ---
    history: Annotated[list[HistoryEntry], operator.add]


def initial_state(question: str) -> ResearchState:
    """The state the graph starts from. Every other key is filled in by nodes."""
    return {
        "question": question,
        "memory_context": [],
        "research_findings": "",
        "sources": [],
        "tool_calls": [],
        "review_status": "not_started",
        "review_feedback": "",
        "review_score": None,
        "previous_review_score": None,
        "revision_count": 0,
        "final_answer": "",
        "error": None,
        "llm_usage": [],
        "history": [],
    }
