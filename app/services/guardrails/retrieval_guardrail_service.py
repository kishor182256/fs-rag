import re
from collections import defaultdict

from app.schemas.agentic import RetrievalGuardrailReport
from app.services.retrieval_types import RetrievalCandidate


BLOCKED_SOURCE_PATTERNS = [
    r"test",
    r"sandbox",
    r"draft",
]


def _extract_years(text: str) -> set[int]:
    years = set()
    for match in re.findall(r"\b(20\d{2})\b", text):
        year = int(match)
        if 2000 <= year <= 2100:
            years.add(year)
    return years


def _query_requires_single_document_scope(query: str) -> bool:
    lowered = (query or "").lower()
    markers = [
        "uploaded document",
        "uploaded pdf",
        "as mentioned in document",
        "as mentioned in the document",
        "from the uploaded",
        "from uploaded document",
        "in the document",
    ]
    return any(marker in lowered for marker in markers)


def apply_retrieval_guardrails(
    candidates: list[RetrievalCandidate],
    query: str = "",
) -> tuple[list[RetrievalCandidate], RetrievalGuardrailReport]:
    blocked_sources: list[str] = []
    stale_warnings: list[str] = []

    kept: list[RetrievalCandidate] = []
    for candidate in candidates:
        source_lower = candidate.source_file.lower()
        if any(re.search(pattern, source_lower) for pattern in BLOCKED_SOURCE_PATTERNS):
            blocked_sources.append(candidate.source_file)
            continue
        kept.append(candidate)

    if kept and _query_requires_single_document_scope(query):
        score_by_source: dict[str, float] = defaultdict(float)
        count_by_source: dict[str, int] = defaultdict(int)
        for item in kept:
            score_by_source[item.source_file] += float(item.rerank_score)
            count_by_source[item.source_file] += 1

        dominant_source = max(
            score_by_source.keys(),
            key=lambda src: (score_by_source[src], count_by_source[src]),
        )
        kept = [item for item in kept if item.source_file == dominant_source]

    conflicts: list[str] = []
    if len(kept) >= 2:
        first_years = _extract_years(kept[0].text)
        second_years = _extract_years(kept[1].text)
        if first_years and second_years and first_years.isdisjoint(second_years):
            conflicts.append(
                f"Potential temporal conflict between {kept[0].chunk_id} and {kept[1].chunk_id}."
            )

    for candidate in kept:
        years = sorted(_extract_years(candidate.text))
        if years and years[-1] < 2024:
            stale_warnings.append(f"{candidate.chunk_id} contains older year {years[-1]}.")

    report = RetrievalGuardrailReport(
        allowed=len(kept) > 0,
        blocked_sources=sorted(set(blocked_sources)),
        stale_warnings=stale_warnings[:5],
        conflicts=conflicts[:5],
    )
    return kept, report
