"""Wires the agents into a LangGraph StateGraph.

Topology (hub and spoke around the supervisor):

    START -> load_memory -> supervisor
    supervisor --(conditional)--> researcher | reviewer | finalizer | save_memory
    researcher -> supervisor
    reviewer   -> supervisor
    finalizer  -> supervisor
    save_memory -> END

Why every worker returns to the supervisor: routing decisions live in ONE
place (decide_next_step), so the retry limit cannot be bypassed by a worker.

Safety net: run_config() sets LangGraph's recursion_limit (max graph steps) just
above the longest legitimate path. If a routing bug ever creates a cycle, the run
fails fast with GraphRecursionError instead of spinning. Recent LangGraph versions
default this limit to ~10,000, so it must be set explicitly.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents import ResearchAgents
from state import ResearchState
from telemetry import traced_node


STEP_HEADROOM = 4


def recursion_limit(max_revisions: int) -> int:
    """Longest legitimate path through the graph, plus a little headroom.

    load_memory                                              1
    each research round: supervisor, researcher,
                         supervisor, reviewer                4 x (1 + max_revisions)
    supervisor, finalizer, supervisor, save_memory           4
    """
    return 1 + 4 * (1 + max_revisions) + 4 + STEP_HEADROOM


def run_config(thread_id: str, max_revisions: int) -> dict:
    """The config every run uses: its checkpoint thread and its step limit."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit(max_revisions)}


def route_from_supervisor(state: ResearchState) -> str:
    # The conditional edge only READS the decision the supervisor node wrote.
    return state["next_step"]


def build_graph(agents: ResearchAgents, checkpointer=None):
    g = StateGraph(ResearchState)

    # Each node is wrapped in a tracing span (telemetry.traced_node); the node itself is unchanged.
    for name in ("load_memory", "supervisor", "researcher", "reviewer", "finalizer", "save_memory"):
        g.add_node(name, traced_node(name, getattr(agents, name)))

    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"researcher": "researcher", "reviewer": "reviewer",
         "finalizer": "finalizer", "save_memory": "save_memory"},
    )
    for worker in ("researcher", "reviewer", "finalizer"):
        g.add_edge(worker, "supervisor")
    g.add_edge("save_memory", END)

    # Checkpointing = LangGraph snapshots of the graph state after each step
    # (per thread_id). It enables inspection/resume; it is NOT long-term memory.
    return g.compile(checkpointer=checkpointer or InMemorySaver())
