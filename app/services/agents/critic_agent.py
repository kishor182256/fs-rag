import re

from app.schemas.agentic import CriticReport
from app.services.retrieval_types import RetrievalCandidate

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "give",
    "in",
    "is",
    "it",
    "list",
    "of",
    "on",
    "that",
    "the",
    "to",
    "what",
    "with",
}


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", text or "")
        if len(token) > 2 and token.lower() not in STOPWORDS
    }


def evaluate_answer(
    *,
    answer: str,
    candidates: list[RetrievalCandidate],
    require_citations: bool,
    query: str = "",
) -> CriticReport:
    issues: list[str] = []
    if not candidates:
        issues.append("no_evidence_candidates")

    citation_count = len(re.findall(r"\((?:chunk|image)_[A-Za-z0-9_-]+\s+p\d+(?:-\d+)?\)", answer or ""))
    if require_citations and citation_count == 0:
        issues.append("missing_citations")

    query_terms = _tokens(query)
    answer_terms = _tokens(answer)
    if query_terms:
        overlap_ratio = len(query_terms & answer_terms) / max(1, len(query_terms))
        if overlap_ratio < 0.20:
            issues.append("low_query_alignment")

    lowered_query = (query or "").lower()
    if "list" in lowered_query and "countr" in lowered_query:
        bullet_lines = [line for line in (answer or "").splitlines() if line.strip().startswith("-")]
        if len(bullet_lines) < 2:
            issues.append("low_list_completeness")

    faithfulness_score = 0.9 if candidates else 0.0
    consistency_score = 0.9
    if "Insufficient evidence" in answer:
        consistency_score = 0.8
    if issues:
        faithfulness_score = max(0.0, faithfulness_score - 0.5)
        consistency_score = max(0.0, consistency_score - 0.4)

    if not issues:
        return CriticReport(
            passed=True,
            faithfulness_score=round(faithfulness_score, 3),
            consistency_score=round(consistency_score, 3),
            issues=[],
            recommendation="accept",
        )

    recommendation = "retry"
    if "no_evidence_candidates" in issues:
        recommendation = "abstain"

    return CriticReport(
        passed=False,
        faithfulness_score=round(faithfulness_score, 3),
        consistency_score=round(consistency_score, 3),
        issues=issues,
        recommendation=recommendation,
    )
