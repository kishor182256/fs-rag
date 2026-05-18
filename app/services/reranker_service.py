import re

from app.schemas.ingestion import QueryRequest
from app.services.retrieval_types import RetrievalCandidate

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")


def _tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if len(token) > 1]


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if abs(v_max - v_min) < 1e-9:
        return [1.0 for _ in values]
    return [(value - v_min) / (v_max - v_min) for value in values]


def _compute_proximity(query_terms: list[str], text: str) -> float:
    tokens = _tokenize(text)
    if not tokens or not query_terms:
        return 0.0

    positions: dict[str, list[int]] = {term: [] for term in query_terms}
    for idx, token in enumerate(tokens):
        if token in positions:
            positions[token].append(idx)

    present = [term for term in query_terms if positions[term]]
    if len(present) <= 1:
        return 0.5 if present else 0.0

    ordered = []
    for term in present:
        ordered.append(positions[term][0])
    ordered.sort()

    avg_gap = sum(ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)) / max(1, len(ordered) - 1)
    return max(0.0, 1.0 - min(avg_gap / 25.0, 1.0))


def _build_snippet(text: str, query_terms: list[str], max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return ""

    if not query_terms:
        return clean[: max_chars - 3] + "..." if len(clean) > max_chars else clean

    lower = clean.lower()
    first_pos = min((lower.find(term) for term in query_terms if lower.find(term) != -1), default=-1)
    if first_pos < 0:
        return clean[: max_chars - 3] + "..." if len(clean) > max_chars else clean

    start = max(0, first_pos - max_chars // 4)
    end = min(len(clean), start + max_chars)
    snippet = clean[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(clean):
        snippet += "..."
    return snippet


def rerank_candidates(
    *,
    query: str,
    candidates: list[RetrievalCandidate],
    request: QueryRequest,
) -> list[RetrievalCandidate]:
    if not candidates:
        return []

    query_terms = _tokenize(query)
    bm25_norm = _normalize_scores([c.bm25_score for c in candidates])
    vector_norm = _normalize_scores([c.vector_score for c in candidates])

    for idx, candidate in enumerate(candidates):
        matched = set(candidate.matched_terms)
        coverage = (len(matched) / len(set(query_terms))) if query_terms else 0.0
        phrase_bonus = 0.12 if query.lower() in candidate.text.lower() else 0.0
        proximity = _compute_proximity(query_terms=query_terms, text=candidate.text)

        has_vector = candidate.vector_score > 0
        w_bm25 = 0.48
        w_vector = 0.30 if has_vector else 0.0
        w_coverage = 0.14
        w_proximity = 0.06
        w_phrase = 0.02

        total_weight = w_bm25 + w_vector + w_coverage + w_proximity + w_phrase
        if total_weight <= 0:
            total_weight = 1.0

        score = (
            w_bm25 * bm25_norm[idx]
            + w_vector * vector_norm[idx]
            + w_coverage * coverage
            + w_proximity * proximity
            + w_phrase * phrase_bonus
        ) / total_weight

        candidate.rerank_score = round(score, 6)

        max_chars = request.max_snippet_chars
        if request.response_mode == "compact":
            max_chars = min(max_chars, 500)
        elif request.response_mode == "full":
            max_chars = max(max_chars, 2000)
        candidate.text = re.sub(r"\s+", " ", candidate.text).strip()
        candidate_snippet = _build_snippet(candidate.text, query_terms, max_chars)
        if candidate_snippet:
            candidate.snippet = candidate_snippet

    candidates.sort(key=lambda item: item.rerank_score, reverse=True)
    return candidates
