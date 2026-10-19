"""Reads the spans and logs telemetry.py writes (traces/otel/*.jsonl) and derives metrics.

Metrics come from the spans themselves -- one source of truth:
  - per run:    the root span's attributes (duration, outcome, tokens, cost, tool failures...)
  - across runs: percentiles and counts over those runs
  - per operation: every span grouped by what it does (chat per role, tool, MCP call, fetch)
  - failures:   every span with ERROR status, grouped into categories

A real deployment would export the same spans to a backend (Jaeger, Tempo, X-Ray) and
compute these with its query language; this module is that query layer in miniature.
"""

import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from telemetry import RUN_SPAN

_TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
MAX_RUNS = 500


def valid_trace_id(trace_id: str) -> bool:
    return bool(_TRACE_ID.match(trace_id))


def _files(directory: Path, kind: str, days: float | None) -> list[Path]:
    if not directory.is_dir():
        return []
    cutoff = time.time() - days * 86400 if days else 0
    return sorted(p for p in directory.glob(f"{kind}-*.jsonl") if p.stat().st_mtime >= cutoff)


def _read(paths: Iterable[Path]) -> Iterator[dict]:
    for path in paths:
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except ValueError:      # a half-written last line from a killed process
                        continue
        except OSError:
            continue


def pct(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: p95 of 20 runs is the 19th fastest."""
    if not values:
        return None
    values = sorted(values)
    return values[max(0, min(len(values) - 1, math.ceil(p / 100 * len(values)) - 1))]


def operation(span: dict) -> str:
    """Group spans by what they do, not by their exact name (hosts and models vary)."""
    a = span.get("attributes", {})
    if a.get("gen_ai.operation.name") == "chat":
        return f"model call: {a.get('app.role', '?')}"
    if span["name"].startswith("GET "):
        return "page fetch (GET)"
    if span["name"].startswith("web_search "):
        return f"search provider: {a.get('app.search.provider', '?')}"
    if span["name"].startswith("tools/call "):
        return f"{span['name']} ({'MCP client' if span['kind'] == 'CLIENT' else 'MCP server'})"
    return span["name"]


def _run_row(root: dict, spans: list[dict]) -> dict:
    a = root["attributes"]
    return {
        "trace_id": root["trace_id"], "start": root["start"], "duration_ms": root["duration_ms"],
        "status": root["status"], "entry": a.get("app.entry"), "run_id": a.get("app.run_id"),
        "question": a.get("app.question"), "case": a.get("app.eval.case"),
        "outcome": a.get("app.outcome"), "revisions": a.get("app.revisions"), "score": a.get("app.review_score"),
        "llm": f"{a.get('app.llm_provider')}:{a.get('gen_ai.request.model')}",
        "search": a.get("app.search_provider"), "mode": a.get("app.researcher_mode"),
        "transport": a.get("app.tool_transport"),
        "cost_usd": a.get("app.cost_usd", 0.0), "input_tokens": a.get("gen_ai.usage.input_tokens", 0),
        "output_tokens": a.get("gen_ai.usage.output_tokens", 0), "llm_calls": a.get("app.llm_calls", 0),
        "tool_calls": a.get("app.tool_calls", 0), "tool_failures": a.get("app.tool_failures", 0),
        "retries": a.get("app.retries", 0),
        "spans": len(spans), "errors": sum(s["status"] == "ERROR" for s in spans),
        "services": sorted({s.get("service") or "?" for s in spans}),
    }


def _error_key(span: dict) -> tuple[str, str, str]:
    msg = re.sub(r"\d+", "#", (span.get("status_message") or "").split("\n")[0])[:90]
    return operation(span), span["attributes"].get("error.type", "error"), msg


def overview(directory: Path, days: float = 7) -> dict:
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for s in _read(_files(directory, "spans", days)):
        if isinstance(s, dict) and "trace_id" in s:
            by_trace[s["trace_id"]].append(s)

    runs, run_spans, orphans = [], [], 0
    for spans in by_trace.values():
        root = next((s for s in spans if s["name"] == RUN_SPAN and not s.get("parent_span_id")), None)
        if root is None:
            orphans += 1        # e.g. the MCP server called by another host, or a run still in progress
            continue
        runs.append(_run_row(root, spans))
        run_spans.extend(spans)
    runs.sort(key=lambda r: r["start"], reverse=True)

    durations = [r["duration_ms"] for r in runs]
    costs = [r["cost_usd"] for r in runs]
    tool_calls = sum(r["tool_calls"] for r in runs)

    ops: dict[str, list[dict]] = defaultdict(list)
    for s in run_spans:
        if s["name"] != RUN_SPAN:
            ops[operation(s)].append(s)
    operations = []
    for name, spans in ops.items():
        d = [s["duration_ms"] for s in spans]
        operations.append({
            "operation": name, "count": len(spans), "errors": sum(s["status"] == "ERROR" for s in spans),
            "p50_ms": pct(d, 50), "p95_ms": pct(d, 95), "max_ms": max(d),
            "cost_usd": round(sum(s["attributes"].get("app.cost_usd", 0.0) for s in spans), 6),
            "tokens": sum(s["attributes"].get("gen_ai.usage.input_tokens", 0)
                          + s["attributes"].get("gen_ai.usage.output_tokens", 0) for s in spans),
            "services": sorted({s.get("service") or "?" for s in spans}),
        })
    operations.sort(key=lambda o: -(o["p95_ms"] or 0))

    # Failure categories: only the span where an error STARTED. A failed tool also marks its
    # parents (execute_tool, the MCP spans) as failed; counting those too would triple-count it.
    errors: dict[tuple, dict] = {}
    has_failed_child = {s.get("parent_span_id") for s in run_spans if s["status"] == "ERROR"}
    for s in run_spans:
        if s["status"] != "ERROR" or s["name"] == RUN_SPAN or s["span_id"] in has_failed_child:
            continue                                # a child failed first: count that one
        key = _error_key(s)
        e = errors.setdefault(key, {"operation": key[0], "type": key[1], "message": key[2], "count": 0,
                                    "last_seen": 0, "example_trace": None})
        e["count"] += 1
        if s["start"] > e["last_seen"]:
            e["last_seen"], e["example_trace"] = s["start"], s["trace_id"]

    return {
        "directory": str(directory), "days": days, "orphan_traces": orphans,
        "runs": runs[:MAX_RUNS],
        "summary": {
            "runs": len(runs),
            "outcomes": dict(Counter(r["outcome"] or "unfinished" for r in runs)),
            "entries": dict(Counter(r["entry"] or "?" for r in runs)),
            "duration_p50_ms": pct(durations, 50), "duration_p95_ms": pct(durations, 95),
            "duration_max_ms": max(durations, default=None),
            "cost_total_usd": round(sum(costs), 6), "cost_mean_usd": round(sum(costs) / len(costs), 6) if costs else None,
            "cost_p95_usd": pct(costs, 95),
            "tokens_mean": round(sum(r["input_tokens"] + r["output_tokens"] for r in runs) / len(runs)) if runs else None,
            "tool_calls": tool_calls, "tool_failures": sum(r["tool_failures"] for r in runs),
            "tool_failure_rate": round(sum(r["tool_failures"] for r in runs) / tool_calls, 3) if tool_calls else None,
            "retries": sum(r["retries"] for r in runs),
            "runs_with_errors": sum(r["errors"] > 0 for r in runs),
        },
        "operations": operations,
        "errors": sorted(errors.values(), key=lambda e: -e["count"]),
    }


def trace_detail(directory: Path, trace_id: str) -> dict:
    """Every span of one trace (from every process that took part) and its log lines."""
    spans = sorted((s for s in _read(_files(directory, "spans", None)) if s.get("trace_id") == trace_id),
                   key=lambda s: s["start"])
    logs = sorted((l for l in _read(_files(directory, "logs", None)) if l.get("trace_id") == trace_id),
                  key=lambda l: l.get("time", ""))
    for s in spans:
        s["operation"] = operation(s)
    return {"trace_id": trace_id, "spans": spans, "logs": logs,
            "services": sorted({s.get("service") or "?" for s in spans})}
