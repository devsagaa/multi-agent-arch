"""Terminal demo for the multi-agent research assistant.

Examples:
  python main.py "Should a software engineer learn AWS before building production AI agents?"
  python main.py --mock --force-reject "..."      # offline, deterministic, exercises retry
  python main.py --remember "I prefer concise answers with bullet points"
  python main.py --show-memory
  python main.py --show-prompts "..."             # print every prompt sent to the model + its output
  python main.py --trace "..."                    # save model calls to traces/<run id>.jsonl
"""

import argparse
import dataclasses
import json
import logging
import sys
import uuid
from pathlib import Path

import telemetry
from agents import ResearchAgents
from config import load_settings
from langgraph.errors import GraphRecursionError

from graph import build_graph, run_config
from llm import RecordingLLM, build_llm
from memory import MemoryStore
from state import initial_state

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def show_memory(store: MemoryStore) -> None:
    rows = store.all()
    print(f"\n=== Long-term memory ({store.db_path}): {len(rows)} entries ===")
    for r in rows:
        print(f"#{r['id']} [{r['category']}] ({r['source']}, {r['created_at']})\n    {r['content']}")


def print_update(node: str, update: dict) -> None:
    """Pretty-print one node's partial state update as it streams out of the graph."""
    for h in update.get("history", []):
        print(f"  [{h['node']:<11}] {h['action']}" + (f" -- {h['detail']}" if h["detail"] else ""))
    for call in update.get("tool_calls", []):
        if not call["ok"]:
            status = f"FAILED: {call['error']}"
        elif call["tool"] == "fetch_page":
            status = f"{len(call.get('content', ''))} chars of page text"
        else:
            status = f"{call['result_count']} results"
        via = " via MCP" if call.get("transport") == "mcp" else ""
        print(f"      tool {call['tool']}(provider={call['provider']}, input={call['input']!r}){via} -> {status}")
        for i, r in enumerate(call["results"], 1):
            print(f"        [{i}] {r['title'][:80]}  {r['url']}")
    if node == "load_memory":
        for m in update.get("memory_context", []):
            print(f"      memory: {m[:110]}")
    if update.get("research_findings"):
        print("      findings:\n" + "\n".join("        " + l for l in update["research_findings"].splitlines()))


def print_llm_calls(calls: list[dict], full: bool) -> None:
    """Show exactly what a node sent to the model and what came back."""
    limit = None if full else 1500
    clip = lambda s: s if limit is None or len(s) <= limit else s[:limit] + f"\n... [{len(s) - limit:,} more chars, use --full-prompts]"
    for c in calls:
        meta = f"{c['model']}" + (", strict JSON schema" if c["structured"] else "")
        if c.get("input_tokens") is not None:
            meta += (f", {c['input_tokens']:,} in / {c['output_tokens']:,} out tokens, {c['attempts']} attempt(s)"
                     f", finish_reason={c['finish_reason']}")
        print(f"\n      ----- LLM call: {c['role']} ({meta}, {c['duration_ms']} ms) -----")
        if c.get("kind") == "chat":
            meta += f", tool_choice={c['tool_choice']}"
        user_label = "MESSAGES (conversation after the system prompt)" if c.get("kind") == "chat" else "USER"
        out_label = "OUTPUT (tool calls requested)" if c.get("tool_calls") else "OUTPUT"
        for label, text in (("SYSTEM", c["system"]), (user_label, c["user"]),
                            (out_label, c["response"] if c["error"] is None else f"ERROR: {c['error']}")):
            print(f"      >>> {label}:")
            print("\n".join("        " + line for line in clip(text or "").splitlines()))
        if c.get("request"):
            # Message text is printed above; show the rest of the body exactly as sent
            # (tool_calls / tool_call_id fields included).
            body = {**c["request"], "messages": [{**m, "content": f"<{m['role']} text above>"} if m.get("content")
                                                 else m for m in c["request"]["messages"]]}
            if "tools" in body and not full:
                names = ", ".join(t["function"]["name"] for t in body["tools"])
                body["tools"] = f"<{len(body['tools'])} tool schemas: {names} -- shown in full with --full-prompts>"
            print("      >>> REQUEST BODY" + (" (mock: not sent):" if c["model"] == "mock"
                                             else " -> POST https://api.openai.com/v1/chat/completions:"))
            print("\n".join("        " + l for l in json.dumps(body, indent=2).splitlines()))
        if c.get("raw_response"):
            raw = {k: v for k, v in c["raw_response"].items() if k != "choices"}
            raw["finish_reason"] = c["raw_response"]["choices"][0].get("finish_reason")
            print("      >>> RAW RESPONSE (minus message text):")
            print("\n".join("        " + l for l in json.dumps(raw, indent=2, default=str).splitlines()))
    if calls:
        print()


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-agent research assistant (LangGraph)")
    p.add_argument("question", nargs="?", help="question to research")
    p.add_argument("--mock", action="store_true", help="mock model + mock search (offline, free)")
    p.add_argument("--force-reject", action="store_true", help="reviewer rejects the first draft")
    p.add_argument("--pipeline", action="store_true",
                   help="fixed researcher (code picks the tool calls) instead of model-driven tool calling")
    p.add_argument("--remember", metavar="TEXT", help="save a user preference to long-term memory")
    p.add_argument("--show-memory", action="store_true", help="print saved long-term memory")
    p.add_argument("--clear-memory", action="store_true", help="delete all saved memory")
    p.add_argument("--show-checkpoints", action="store_true", help="list LangGraph checkpoints after the run")
    p.add_argument("--show-prompts", action="store_true",
                   help="print every prompt sent to the model and its raw output")
    p.add_argument("--full-prompts", action="store_true", help="with --show-prompts: don't truncate long text")
    p.add_argument("--trace", action="store_true",
                   help="save every model call (prompts, output, tokens) to traces/<run id>.jsonl "
                        "(OpenTelemetry spans are always written to traces/otel unless TELEMETRY=off)")
    p.add_argument("-v", "--verbose", action="store_true", help="show Python logging output")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    settings = load_settings(force_mock=args.mock, force_reject_first=args.force_reject or None)
    telemetry.setup("research-app")       # after load_settings (reads .env) and basicConfig
    if args.pipeline:
        settings = dataclasses.replace(settings, researcher_mode="pipeline")
    store = MemoryStore(settings.memory_db_path)

    if args.clear_memory:
        store.clear()
        print("Memory cleared.")
    if args.remember:
        added = store.save(args.remember, category="preference", source="user")
        print("Saved preference." if added else "Preference already saved.")
    if args.show_memory and not args.question:
        show_memory(store)
    if (args.clear_memory or args.remember or args.show_memory) and not args.question:
        return

    question = args.question or input("Question: ").strip()
    if not question:
        sys.exit("No question given.")

    print(f"Model: {settings.llm_provider}"
          + (f" ({settings.openai_model})" if settings.llm_provider == "openai" else "")
          + f" | Search: {settings.search_provider} | Researcher: {settings.researcher_mode} | Tools: {settings.tool_transport} | Max revisions: {settings.max_revisions}"
          + f" | Forced first rejection: {settings.force_reject_first_review}")
    if settings.llm_provider == "mock":
        print("NOTE: using the MOCK model -- answers are canned, not real model output.")

    # RecordingLLM wraps the real model; agents don't know they're being recorded.
    llm = RecordingLLM(build_llm(settings.llm_provider, settings.openai_model,
                                 settings.llm_timeout_s, settings.llm_max_retries))
    graph = build_graph(ResearchAgents(llm, settings, store))
    run_id = str(uuid.uuid4())
    config = run_config(run_id, settings.max_revisions)
    trace_path = Path(__file__).parent / "traces" / f"{run_id}.jsonl"
    if args.trace:
        trace_path.parent.mkdir(exist_ok=True)

    print("\n=== Execution trace ===")
    with telemetry.run_span(run_id, question, settings, entry="cli") as root:
        trace_id = telemetry.current_trace_id()
        try:
            for chunk in graph.stream(initial_state(question), config, stream_mode="updates"):
                for node, update in chunk.items():
                    print_update(node, update or {})
                    calls = llm.drain()
                    if args.show_prompts:
                        print_llm_calls(calls, args.full_prompts)
                    if args.trace and calls:
                        with trace_path.open("a", encoding="utf-8") as f:
                            for c in calls:
                                f.write(json.dumps({"run_id": run_id, "node": node, **c}, ensure_ascii=False) + "\n")
        except GraphRecursionError:
            telemetry.fail(root, "graph step limit exceeded", "GraphRecursionError")
            sys.exit(f"\nStopped: the graph exceeded its step limit ({config['recursion_limit']}). "
                     "A routing bug created a cycle -- check decide_next_step().")
        final = graph.get_state(config).values
        telemetry.finish_run(root, final)

    print("\n=== Summary ===")
    print(f"Review status : {final.get('review_status')}  (feedback: {final.get('review_feedback') or '-'})")
    print(f"Retry occurred: {'yes' if final.get('revision_count', 0) else 'no'} "
          f"(revisions used: {final.get('revision_count', 0)}/{settings.max_revisions})")
    print(f"Outcome       : {final.get('outcome')}")
    usage = final.get("llm_usage", [])
    tokens = sum(u["input_tokens"] + u["output_tokens"] for u in usage)
    print(f"Model usage   : {len(usage)} calls, {tokens:,} tokens, ~${sum(u['cost_usd'] for u in usage):.4f}"
          + (f", {sum(u['attempts'] - 1 for u in usage)} retries" if any(u["attempts"] > 1 for u in usage) else ""))
    print("\n=== Final answer ===\n" + final.get("final_answer", ""))

    if args.show_checkpoints:
        print("\n=== LangGraph checkpoints (newest first) ===")
        for snap in graph.get_state_history(config):
            print(f"  step={snap.metadata.get('step'):>3}  next={snap.next}")
    if args.trace:
        print(f"\nModel calls saved to {trace_path}")
    if trace_id:
        print(f"\nTrace {trace_id}: python server.py, then http://127.0.0.1:8000/traces#{trace_id}")
    if args.show_memory:
        show_memory(store)
    store.close()


if __name__ == "__main__":
    main()
