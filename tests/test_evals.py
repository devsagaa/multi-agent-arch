"""The eval harness must itself be trustworthy: test its scoring on hand-made runs."""

from pathlib import Path

from evals.run_evals import compare, load_cases, run_once, summarize
from evals.scoring import abstained, coverage, grounding, score_run
from test_workflow import make_settings

SOURCES = [
    {"title": "Lambda quotas", "url": "https://a.example", "snippet": "Functions can run for up to 900 seconds.",
     "content": "The maximum timeout for a Lambda function is 15 minutes (900 seconds). Memory ranges up to 10,240 MB."},
    {"title": "Step Functions", "url": "https://b.example", "snippet": "Durable orchestration with retries."},
]
CASE = {"id": "lambda-timeout", "question": "Max Lambda timeout?", "key_points": [["15 minutes", "900 seconds"]],
        "min_coverage": 1.0}


def final_state(answer, outcome="approved", sources=SOURCES, **extra):
    return {"final_answer": answer, "outcome": outcome, "sources": sources, "revision_count": 0,
            "tool_calls": [], "llm_usage": [], "history": [], **extra}


# ---------------------------------------------------------------- deterministic checks
def test_coverage_accepts_any_phrasing_in_a_group():
    groups = [["15 minutes", "900 seconds"], ["memory"], ["vpc"]]
    result = coverage("Lambda allows 900 seconds and 10 GB of MEMORY.", groups)
    assert result["score"] == 2 / 3 and result["missing"] == ["vpc"]


def test_grounding_separates_supported_from_unsupported_sentences():
    answer = ("The maximum timeout for a Lambda function is 15 minutes [1]. "
              "Lambda natively supports quantum accelerators for training giant models [1].")
    g = grounding(answer, SOURCES)
    assert g["cited_sentences"] == 2 and g["supported"] == 1 and g["rate"] == 0.5
    assert "quantum" in g["weakest"][0][1]


def test_grounding_counts_long_uncited_claims():
    g = grounding("Lambda is the best compute service for every workload ever built by anyone. Short.", SOURCES)
    assert g["uncited_claims"] == 1 and g["rate"] is None


def test_abstention_detection():
    assert abstained("The sources do not cover re:Invent 2031.", "approved")
    assert abstained("anything", "failed")
    assert not abstained("AWS announced Quantum Lambda and S4.", "approved")


# ---------------------------------------------------------------- pass / fail decisions
def test_good_answer_passes():
    r = score_run(CASE, final_state("Lambda functions can run for at most 15 minutes (900 seconds) [1]."))
    assert r["passed"], r["fail_reasons"]
    assert r["coverage"] == 1.0 and r["citations_ok"] and r["grounding"] == 1.0


def test_each_problem_is_reported_with_its_reason():
    r = score_run(CASE, final_state("Lambda has a generous timeout [7].", outcome="unapproved"))
    assert not r["passed"]
    joined = " ".join(r["fail_reasons"])
    assert "outcome=unapproved" in joined
    assert "do not exist" in joined                 # [7] with 2 sources
    assert "coverage 0.00" in joined and "15 minutes" in joined


def test_judge_faithfulness_below_threshold_fails_the_case():
    judgement = {"faithfulness": 2, "completeness": 4, "citation_quality": 3, "unsupported_claims": ["x"], "reasoning": ""}
    r = score_run(CASE, final_state("Lambda runs for up to 15 minutes [1]."), judgement)
    assert not r["passed"] and r["fail_reasons"] == ["judge faithfulness 2 < 4"]


def test_unanswerable_case_passes_only_when_the_system_declines():
    case = {"id": "future", "kind": "unanswerable", "question": "re:Invent 2031?", "key_points": []}
    assert score_run(case, final_state("I could not produce a reliable answer.", outcome="failed"))["passed"]
    assert score_run(case, final_state("The sources do not cover 2031 [1]."))["passed"]
    invented = score_run(case, final_state("AWS launched Bedrock Ultra and Lambda Infinity [1]."))
    assert not invented["passed"] and "should have declined" in invented["fail_reasons"][0]


# ---------------------------------------------------------------- aggregation and regressions
def test_summary_flags_flaky_cases():
    recs = [{"case": "a", "mode": "m", "passed": True, "cost_usd": 0.01, "latency_s": 10},
            {"case": "a", "mode": "m", "passed": False, "cost_usd": 0.02, "latency_s": 20},
            {"case": "b", "mode": "m", "passed": True, "cost_usd": 0.01, "latency_s": 10}]
    s = summarize(recs)
    assert s["pass_rate"] == round(2 / 3, 3) and s["flaky_cases"] == ["a"] and s["cases_always_pass"] == 1


def test_regressions_beyond_tolerance_are_reported():
    baseline = {"summary": {"agentic": {"pass_rate": 0.9, "faithfulness": 4.5, "coverage": 0.9, "cost_mean": 0.006}}}
    same = {"agentic": {"pass_rate": 0.85, "faithfulness": 4.3, "coverage": 0.88, "cost_mean": 0.007}}
    worse = {"agentic": {"pass_rate": 0.6, "faithfulness": 3.5, "coverage": 0.7, "cost_mean": 0.01}}
    assert compare(same, baseline) == []
    problems = compare(worse, baseline)
    assert len(problems) == 4 and any("pass rate" in p for p in problems)


# ---------------------------------------------------------------- harness plumbing
def test_dataset_is_well_formed():
    cases = load_cases(None)
    assert len({c["id"] for c in cases}) == len(cases) >= 10
    for c in cases:
        if c.get("kind") == "unanswerable":
            continue
        assert c["key_points"] and all(isinstance(g, list) and g for g in c["key_points"])


def test_each_run_uses_its_own_empty_memory(tmp_path: Path):
    settings = make_settings(save_mock_answers=True)          # even if saving were on...
    case = load_cases(["lambda-timeout"])[0]
    r = run_once(case, "agentic", settings, judge_llm=None, workdir=tmp_path, trial=1)
    assert "fail_reasons" in r and r["mode"] == "agentic"
    assert (tmp_path / "lambda-timeout-agentic-1.db").exists()  # ...it lands in the eval's temp DB
    assert r["model_calls"] == 5                               # 3 researcher turns + reviewer + finalizer
