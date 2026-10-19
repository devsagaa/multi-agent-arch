"""Scoring for one finished run. Deterministic checks first, an optional LLM judge second.

Why both:
  - Deterministic checks are free, repeatable and can't be talked into a good score,
    but they only see surface features (keywords, citation ids, word overlap).
  - An LLM judge can assess meaning (is this claim really supported?), but it is
    noisy, costs tokens and can be biased toward text written by the same model.
A case passes only when every check that applies to it passes.
"""

import json
import re

from agents import UNTRUSTED_NOTICE, _format_sources, check_citations
from llm import LLM, LLMError

# Phrases that show the system admitted it couldn't answer (for unanswerable cases).
ABSTAIN_PHRASES = [
    "could not produce a reliable answer", "do not cover", "don't cover", "does not cover", "not covered",
    "no information", "not available", "cannot find", "could not find", "unable to", "no reliable",
    "has not happened", "hasn't happened", "not yet", "future event", "in the future", "no sources",
    "not mentioned", "do not mention", "don't mention", "do not contain", "don't contain", "no details",
]

_STOP = set("""a an the and or but of to in on for with by from as at is are was were be been being this that these those
it its into than then there their they them can could should would will may might also such not no yes more most
other some any each which what when where who how why about over under between within without your you our we""".split())
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-\.]{2,}", text.lower()) if w not in _STOP}


def coverage(answer: str, key_points: list[list[str]]) -> dict:
    """Which key points the answer mentions. Each key point is a list of acceptable phrasings."""
    text = answer.lower()
    hits = [any(alt.lower() in text for alt in group) for group in key_points]
    missing = [group[0] for group, hit in zip(key_points, hits) if not hit]
    return {"score": sum(hits) / len(hits) if hits else 1.0, "missing": missing}


def grounding(answer: str, sources: list[dict], threshold: float = 0.5) -> dict:
    """Heuristic check that cited sentences resemble the sources they cite.

    For each sentence with [n] citations: the share of its content words that also
    appear in the cited sources' title/snippet/page text. A sentence counts as
    supported at >= threshold. Paraphrasing lowers the score, so treat this as a
    cheap signal, not proof -- the judge checks meaning.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if s.strip()]
    cited, supported, uncited_claims, weakest = 0, 0, 0, []
    for sentence in sentences:
        ids = {int(n) for group in _CITE.findall(sentence) for n in group.split(",")}
        body = _CITE.sub("", sentence)
        words = _words(body)
        if not ids:
            # Long declarative sentences without a citation are likely unsupported claims.
            if len(body) > 60 and not body.rstrip().endswith((":", "?")):
                uncited_claims += 1
            continue
        if not words:
            continue
        cited += 1
        src_text = " ".join(
            f"{s.get('title', '')} {s.get('snippet', '')} {s.get('content', '')}"
            for i, s in enumerate(sources, 1) if i in ids)
        overlap = len(words & _words(src_text)) / len(words)
        if overlap >= threshold:
            supported += 1
        else:
            weakest.append((round(overlap, 2), body[:120]))
    weakest.sort()
    return {"cited_sentences": cited, "supported": supported,
            "rate": supported / cited if cited else None,
            "uncited_claims": uncited_claims, "weakest": weakest[:3]}


def abstained(answer: str, outcome: str | None) -> bool:
    text = answer.lower()
    return outcome == "failed" or any(p in text for p in ABSTAIN_PHRASES)


# ------------------------------------------------------------------ LLM judge
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "integer", "description": "1-5: every claim is supported by the cited source text."},
        "completeness": {"type": "integer", "description": "1-5: covers what a knowledgeable answer to this question should."},
        "citation_quality": {"type": "integer", "description": "1-5: citations point at the sources that actually support each claim."},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["faithfulness", "completeness", "citation_quality", "unsupported_claims", "reasoning"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = (
    "You are an impartial evaluator of research answers. Score the ANSWER against the SOURCES only, "
    "not against your own knowledge, except for completeness.\n"
    "Rubric (integers 1-5):\n"
    "- faithfulness: 5 = every factual claim is supported by the cited sources; 3 = some claims unsupported "
    "or overstated; 1 = mostly unsupported or contradicted.\n"
    "- completeness: 5 = covers the important aspects a knowledgeable expert would expect; 3 = misses notable "
    "aspects; 1 = barely answers.\n"
    "- citation_quality: 5 = each citation points at a source that supports that specific claim; "
    "3 = some citations are misplaced or vague; 1 = citations are mostly wrong or missing.\n"
    "List every unsupported claim verbatim or closely paraphrased. Be strict; do not reward fluency.\n\n"
    + UNTRUSTED_NOTICE)


def judge(llm: LLM, question: str, answer: str, sources: list[dict]) -> dict:
    user = (f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nSOURCES:\n{_format_sources(sources) or '(none)'}")
    try:
        resp = llm.complete("judge", JUDGE_SYSTEM, user, schema=JUDGE_SCHEMA)
        verdict = json.loads(resp.text)
    except (LLMError, json.JSONDecodeError) as exc:
        return {"error": str(exc)}
    for k in ("faithfulness", "completeness", "citation_quality"):
        verdict[k] = max(1, min(5, int(verdict.get(k, 1))))
    verdict["tokens"] = resp.input_tokens + resp.output_tokens
    return verdict


# ------------------------------------------------------------------ one case
def score_run(case: dict, final: dict, judgement: dict | None = None,
              min_faithfulness: int = 4) -> dict:
    """All checks for one run, plus pass/fail with the reasons it failed."""
    answer = final.get("final_answer", "") or ""
    sources = final.get("sources", []) or []
    outcome = final.get("outcome")
    usage = final.get("llm_usage", []) or []
    history = final.get("history", []) or []
    reasons: list[str] = []

    result = {
        "outcome": outcome,
        "revisions": final.get("revision_count", 0),
        "review_score": final.get("review_score"),
        "sources": len(sources),
        "pages_read": sum(bool(s.get("content")) for s in sources),
        "searches": sum(c["tool"] == "web_search" for c in final.get("tool_calls", [])),
        "model_calls": len(usage),
        "tokens": sum(u["input_tokens"] + u["output_tokens"] for u in usage),
        "cost_usd": round(sum(u["cost_usd"] for u in usage), 6),
        "stopped_early": any("stopped early" in (h.get("detail") or "") for h in history),
        "answer_chars": len(answer),
    }

    if case.get("kind") == "unanswerable":
        result["abstained"] = abstained(answer, outcome)
        if not result["abstained"]:
            reasons.append("answered a question it should have declined")
    else:
        cov = coverage(answer, case["key_points"])
        cites = check_citations(answer, len(sources))
        ground = grounding(answer, sources)
        result.update(coverage=round(cov["score"], 2), missing=cov["missing"],
                      citations_ok=cites["ok"], invalid_citations=cites["invalid"],
                      grounding=None if ground["rate"] is None else round(ground["rate"], 2),
                      uncited_claims=ground["uncited_claims"], weakest_sentences=ground["weakest"])
        if outcome != "approved":
            reasons.append(f"outcome={outcome}")
        if not cites["ok"]:
            reasons.append("citation problems: " + " ".join(cites["problems"]))
        if cov["score"] < case.get("min_coverage", 0.67):
            reasons.append(f"coverage {cov['score']:.2f} < {case.get('min_coverage', 0.67)} (missing: {', '.join(cov['missing'])})")

    if judgement:
        result["judge"] = judgement
        if "error" in judgement:
            reasons.append(f"judge failed: {judgement['error']}")
        elif case.get("kind") != "unanswerable" and judgement["faithfulness"] < min_faithfulness:
            reasons.append(f"judge faithfulness {judgement['faithfulness']} < {min_faithfulness}")

    result["passed"] = not reasons
    result["fail_reasons"] = reasons
    return result
