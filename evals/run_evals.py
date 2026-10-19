"""Evaluation harness: run the whole graph on a fixed question set and score every run.

  python -m evals.run_evals                                   # agentic vs pipeline, 1 trial, no judge
  python -m evals.run_evals --judge --trials 2                # + LLM judge, 2 trials per case (flakiness)
  python -m evals.run_evals --cases lambda-timeout,rag-basics # a subset
  python -m evals.run_evals --judge --save-baseline           # record the current numbers as the baseline
  python -m evals.run_evals --judge --fail-on-regression      # exit 1 if worse than the baseline (CI gate)
  python -m evals.run_evals --mock                            # offline smoke test of the harness itself

Design choices:
  - Every run gets a fresh, empty memory database, so saved facts can't leak into
    answers (and evals never write to your real memory.db).
  - Each case can run several trials: models are nondeterministic, and a case that
    passes once and fails once is "flaky" -- a finding in itself.
  - A cost ceiling (--max-cost) stops the whole eval before it can run away.
  - Results go to evals/results/<timestamp>.json (everything) and .md (readable report).
"""

import argparse
import dataclasses
import json
import logging
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import telemetry
from agents import ResearchAgents
from config import load_settings
from graph import build_graph, run_config
from llm import build_llm
from memory import MemoryStore
from state import initial_state

from evals.scoring import judge, score_run

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset.json"
RESULTS = ROOT / "results"
BASELINE = ROOT / "baseline.json"

# How much worse than the baseline counts as a regression.
TOLERANCE = {"pass_rate": 0.10, "faithfulness": 0.5, "coverage": 0.10, "cost_mean": 0.30}


def load_cases(only: list[str] | None) -> list[dict]:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))["cases"]
    if only:
        unknown = set(only) - {c["id"] for c in cases}
        if unknown:
            sys.exit(f"Unknown case id(s): {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c["id"] in only]
    return cases


def run_once(case: dict, mode: str, base_settings, judge_llm, workdir: Path, trial: int) -> dict:
    """Run one case once, in isolation, and score it."""
    db = workdir / f"{case['id']}-{mode}-{trial}.db"
    settings = dataclasses.replace(base_settings, researcher_mode=mode, memory_db_path=str(db),
                                   force_reject_first_review=False, save_mock_answers=False)
    store = MemoryStore(str(db))
    record = {"case": case["id"], "category": case.get("category"), "mode": mode, "trial": trial}
    start = time.perf_counter()
    try:
        llm = build_llm(settings.llm_provider, settings.openai_model, settings.llm_timeout_s, settings.llm_max_retries)
        graph = build_graph(ResearchAgents(llm, settings, store))
        run_id = f"eval-{case['id']}-{mode}-{trial}"
        with telemetry.run_span(run_id, case["question"], settings, entry="eval") as root:
            root.set_attribute("app.eval.case", case["id"])
            record["trace_id"] = telemetry.current_trace_id()     # a failing case links to its trace
            final = graph.invoke(initial_state(case["question"]), run_config(run_id, settings.max_revisions))
            telemetry.finish_run(root, final)
        record["latency_s"] = round(time.perf_counter() - start, 1)
        judgement = judge(judge_llm, case["question"], final.get("final_answer", ""), final.get("sources", [])) \
            if judge_llm else None
        record.update(score_run(case, final, judgement))
        record["answer"] = final.get("final_answer", "")
    except Exception as exc:  # a crash is a failed run, not a crashed eval
        record.update(latency_s=round(time.perf_counter() - start, 1), passed=False, cost_usd=0.0,
                      fail_reasons=[f"crashed: {type(exc).__name__}: {exc}"], error=str(exc))
    finally:
        store.close()
    if judgement_tokens := (record.get("judge") or {}).get("tokens"):
        # Judge cost, using the same prices (approximation: all tokens at the input price).
        record["judge_cost_usd"] = round(judgement_tokens * base_settings.price_input_per_m / 1e6, 6)
    return record


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(p * (len(values) - 1)))]


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def summarize(records: list[dict]) -> dict:
    """Aggregate metrics for one mode."""
    by_case: dict[str, list[dict]] = {}
    for r in records:
        by_case.setdefault(r["case"], []).append(r)
    answerable = [r for r in records if "coverage" in r]
    unanswerable = [r for r in records if "abstained" in r]
    judged = [r["judge"] for r in records if r.get("judge") and "error" not in r["judge"]]
    costs = [r.get("cost_usd", 0) for r in records]
    lat = [r["latency_s"] for r in records if "latency_s" in r]
    return {
        "runs": len(records),
        "cases": len(by_case),
        "pass_rate": round(sum(r["passed"] for r in records) / len(records), 3) if records else None,
        "cases_always_pass": sum(all(r["passed"] for r in rs) for rs in by_case.values()),
        "flaky_cases": sorted(c for c, rs in by_case.items() if len({r["passed"] for r in rs}) > 1),
        "approved_first_try": mean([r["outcome"] == "approved" and r["revisions"] == 0 for r in answerable]),
        "coverage": mean([r["coverage"] for r in answerable]),
        "citations_ok": mean([r["citations_ok"] for r in answerable]),
        "grounding": mean([r.get("grounding") for r in answerable]),
        "uncited_claims": mean([r.get("uncited_claims") for r in answerable]),
        "abstain_rate": mean([r["abstained"] for r in unanswerable]),
        "faithfulness": mean([j["faithfulness"] for j in judged]),
        "completeness": mean([j["completeness"] for j in judged]),
        "citation_quality": mean([j["citation_quality"] for j in judged]),
        "model_calls": mean([r.get("model_calls") for r in records]),
        "searches": mean([r.get("searches") for r in records]),
        "pages_read": mean([r.get("pages_read") for r in records]),
        "stopped_early": sum(bool(r.get("stopped_early")) for r in records),
        "crashes": sum(bool(r.get("error")) for r in records),
        "cost_total": round(sum(costs), 4),
        "cost_mean": round(statistics.mean(costs), 5) if costs else None,
        "cost_p95": pct(costs, 0.95),
        "latency_mean_s": mean(lat),
        "latency_p95_s": pct(lat, 0.95),
    }


def compare(summary: dict, baseline: dict) -> list[str]:
    """Regressions vs the saved baseline, per mode. Empty list = no regression."""
    problems = []
    for mode, now in summary.items():
        before = baseline.get("summary", {}).get(mode)
        if not before:
            continue
        def drop(key):
            return before.get(key) is not None and now.get(key) is not None and before[key] - now[key]
        if (d := drop("pass_rate")) and d > TOLERANCE["pass_rate"]:
            problems.append(f"{mode}: pass rate {before['pass_rate']:.0%} -> {now['pass_rate']:.0%}")
        if (d := drop("faithfulness")) and d > TOLERANCE["faithfulness"]:
            problems.append(f"{mode}: judge faithfulness {before['faithfulness']} -> {now['faithfulness']}")
        if (d := drop("coverage")) and d > TOLERANCE["coverage"]:
            problems.append(f"{mode}: coverage {before['coverage']} -> {now['coverage']}")
        if before.get("cost_mean") and now.get("cost_mean") and \
                now["cost_mean"] > before["cost_mean"] * (1 + TOLERANCE["cost_mean"]):
            problems.append(f"{mode}: mean cost ${before['cost_mean']:.4f} -> ${now['cost_mean']:.4f}")
    return problems


def fmt(v, kind=""):
    if v is None:
        return "–"
    if kind == "%":
        return f"{v:.0%}"
    if kind == "$":
        return f"${v:.4f}"
    if kind == "s":
        return f"{v:.0f}s"
    return f"{v}"


ROWS = [  # (label, key, format, higher_is_better)
    ("Pass rate (runs)", "pass_rate", "%", True),
    ("Cases passing every trial", "cases_always_pass", "", True),
    ("Flaky cases", "flaky_cases", "", False),
    ("Approved on first try", "approved_first_try", "%", True),
    ("Key-point coverage", "coverage", "%", True),
    ("Valid citations", "citations_ok", "%", True),
    ("Grounded cited sentences (heuristic)", "grounding", "%", True),
    ("Uncited claims per answer", "uncited_claims", "", False),
    ("Abstained on unanswerable", "abstain_rate", "%", True),
    ("Judge: faithfulness (1-5)", "faithfulness", "", True),
    ("Judge: completeness (1-5)", "completeness", "", True),
    ("Judge: citation quality (1-5)", "citation_quality", "", True),
    ("Model calls per run", "model_calls", "", False),
    ("Searches per run", "searches", "", False),
    ("Pages read per run", "pages_read", "", None),
    ("Runs stopped early (stuck)", "stopped_early", "", None),
    ("Crashes", "crashes", "", False),
    ("Cost per run (mean)", "cost_mean", "$", False),
    ("Cost per run (p95)", "cost_p95", "$", False),
    ("Latency (mean)", "latency_mean_s", "s", False),
    ("Latency (p95)", "latency_p95_s", "s", False),
    ("Total cost", "cost_total", "$", None),
]


def render_markdown(meta: dict, summary: dict, records: list[dict], regressions: list[str] | None) -> str:
    modes = list(summary)
    lines = [f"# Eval report · {meta['started']}", "",
             f"Model: `{meta['model']}` · search: `{meta['search']}` · trials per case: {meta['trials']} · "
             f"judge: {meta['judge'] or 'off'} · cases: {meta['cases']}", ""]
    lines += ["## Summary", "", "| Metric | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for label, key, kind, _ in ROWS:
        vals = []
        for m in modes:
            v = summary[m][key]
            vals.append(", ".join(v) if isinstance(v, list) and v else ("none" if isinstance(v, list) else fmt(v, kind)))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    if regressions is not None:
        lines += ["", "## Against baseline", ""]
        lines += [f"- ❌ {p}" for p in regressions] or ["- ✅ no regressions beyond tolerance"]
    lines += ["", "## Per case", "", "| Case | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for cid in dict.fromkeys(r["case"] for r in records):
        cells = []
        for m in modes:
            rs = [r for r in records if r["case"] == cid and r["mode"] == m]
            marks = "".join("✅" if r["passed"] else "❌" for r in rs)
            cov = mean([r.get("coverage") for r in rs])
            cost = mean([r.get("cost_usd") for r in rs])
            cells.append(f"{marks} cov {fmt(cov, '%')} · {fmt(cost, '$')}")
        lines.append(f"| `{cid}` | " + " | ".join(cells) + " |")
    failures = [r for r in records if not r["passed"]]
    if failures:
        lines += ["", "## Why runs failed", ""]
        for r in failures:
            lines.append(f"- `{r['case']}` · {r['mode']} · trial {r['trial']}: " + "; ".join(r["fail_reasons"]))
            if r.get("judge", {}).get("unsupported_claims"):
                lines.append("  - judge flagged: " + " | ".join(r["judge"]["unsupported_claims"][:3]))
    lines += ["", "## How to read this", "",
              "- **Coverage** and **grounding** are cheap string checks: useful for spotting regressions, "
              "but paraphrases score lower than they deserve.",
              "- **Judge** scores are a model's opinion against a rubric. With the same model as the system, "
              "expect some self-preference; use a stronger judge model (`--judge-model`) for decisions.",
              "- **Flaky** cases passed in some trials and failed in others: nondeterminism worth investigating.",
              "- Live web search changes over time, so compare runs made close together."]
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Evaluate the research graph on a fixed question set.")
    ap.add_argument("--modes", default="agentic,pipeline", help="comma-separated researcher modes")
    ap.add_argument("--cases", help="comma-separated case ids (default: all)")
    ap.add_argument("--trials", type=int, default=1, help="runs per case per mode (>1 reveals flakiness)")
    ap.add_argument("--judge", action="store_true", help="score answers with an LLM judge")
    ap.add_argument("--judge-model", help="model for the judge (default: the system's own model)")
    ap.add_argument("--mock", action="store_true", help="mock model + mock search: tests the harness, not quality")
    ap.add_argument("--max-cost", type=float, default=1.00, help="stop the eval when spend reaches this many USD")
    ap.add_argument("--save-baseline", action="store_true", help=f"save this summary as {BASELINE.name}")
    ap.add_argument("--fail-on-regression", action="store_true", help="exit 1 if worse than the baseline")
    args = ap.parse_args()

    base = load_settings(force_mock=args.mock)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    telemetry.setup("research-app")       # every eval run becomes a trace (entry=eval), linked from results
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    cases = load_cases(args.cases.split(",") if args.cases else None)
    judge_llm = None
    if args.judge:
        judge_llm = build_llm(base.llm_provider, args.judge_model or base.openai_model,
                              base.llm_timeout_s, base.llm_max_retries)
    meta = {"started": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": base.llm_provider + (
                f":{base.openai_model}" if base.llm_provider == "openai" else ""),
            "search": base.search_provider, "trials": args.trials, "cases": len(cases),
            "judge": (args.judge_model or base.openai_model) if args.judge and base.llm_provider == "openai"
            else ("mock" if args.judge else None), "modes": modes}

    total = len(cases) * len(modes) * args.trials
    print(f"Running {len(cases)} cases x {len(modes)} modes x {args.trials} trial(s) = {total} runs "
          f"({meta['model']}, search={base.search_provider}, judge={meta['judge'] or 'off'})")
    if args.mock:
        print("NOTE: mock mode checks that the harness works; the quality scores are meaningless.")

    records, spent = [], 0.0
    plan = [(t, c, m) for t in range(1, args.trials + 1) for c in cases for m in modes]
    with tempfile.TemporaryDirectory(prefix="evals-") as tmp:
        for trial, case, mode in plan:
            if spent >= args.max_cost:
                print(f"Cost ceiling ${args.max_cost:.2f} reached after {len(records)} runs -- stopping early.")
                break
            r = run_once(case, mode, base, judge_llm, Path(tmp), trial)
            spent += r.get("cost_usd", 0) + r.get("judge_cost_usd", 0)
            records.append(r)
            status = "PASS" if r["passed"] else "FAIL"
            extra = "" if r["passed"] else "  <- " + "; ".join(r["fail_reasons"])[:110]
            print(f"  [{len(records):>3}/{total}] {status} {case['id']:<24} {mode:<9} "
                  f"{r.get('latency_s', 0):>5.0f}s ${r.get('cost_usd', 0):.4f}{extra}")

    summary = {m: summarize([r for r in records if r["mode"] == m]) for m in modes if any(r["mode"] == m for r in records)}
    regressions = None
    if BASELINE.exists() and not args.save_baseline:
        regressions = compare(summary, json.loads(BASELINE.read_text(encoding="utf-8")))

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"{stamp}.json").write_text(json.dumps({"meta": meta, "summary": summary, "records": records},
                                                      indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_markdown(meta, summary, records, regressions)
    (RESULTS / f"{stamp}.md").write_text(report, encoding="utf-8")

    print("\n" + report.split("## Per case")[0].strip())
    print(f"\nFull report: {RESULTS / (stamp + '.md')}   (eval spend ≈ ${spent:.4f})")
    if args.save_baseline:
        BASELINE.write_text(json.dumps({"meta": meta, "summary": summary}, indent=2), encoding="utf-8")
        print(f"Baseline saved to {BASELINE}")
    if args.fail_on_regression and regressions:
        sys.exit("Regression against baseline: " + "; ".join(regressions))


if __name__ == "__main__":
    main()
