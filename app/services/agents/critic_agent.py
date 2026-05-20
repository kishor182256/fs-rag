import re

from app.schemas.agentic import CriticReport
from app.services.retrieval_types import RetrievalCandidate


def evaluate_answer(
    *,
    answer: str,
    candidates: list[RetrievalCandidate],
    require_citations: bool,
) -> CriticReport:
    issues: list[str] = []
    if not candidates:
        issues.append("no_evidence_candidates")

    citation_count = len(re.findall(r"\((?:chunk|image)_[A-Za-z0-9_-]+\s+p\d+(?:-\d+)?\)", answer or ""))
    if require_citations and citation_count == 0:
        issues.append("missing_citations")

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
