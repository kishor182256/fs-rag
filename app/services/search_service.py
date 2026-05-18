import json
import re

from app.core.config import settings
from app.schemas.ingestion import ChunkMetadata, QueryHit, QueryRequest, QueryResponse


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
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")


def _tokenize(text: str) -> set[str]:
    return {token for token in TOKEN_PATTERN.findall(text.lower()) if len(token) > 1}


def _tokenize_keywords(text: str) -> set[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def _jaccard_score(query_tokens: set[str], text_tokens: set[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    intersection = len(query_tokens & text_tokens)
    if intersection == 0:
        return 0.0
    union = len(query_tokens | text_tokens)
    return round(intersection / union, 4)


def _passes_filters(chunk: dict, request: QueryRequest) -> bool:
    metadata = chunk.get("metadata", {})
    chunk_months = {m.lower() for m in metadata.get("months", [])}
    chunk_topics = {t.lower() for t in metadata.get("topics", [])}

    if request.months:
        requested_months = {m.lower() for m in request.months}
        if not (requested_months & chunk_months):
            return False

    if request.topics:
        requested_topics = {t.lower() for t in request.topics}
        if not (requested_topics & chunk_topics):
            return False

    return True


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_segments(text: str) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []

    segments = re.split(r"(?<=[.!?])\s+|\s+\u2022\s+|\s+\|\s+", cleaned)
    segments = [segment.strip() for segment in segments if segment.strip()]

    if len(segments) <= 1 and len(cleaned) > 240:
        pieces = re.split(r"\s(?=\d{1,2}\s+[A-Z])", cleaned)
        segments = [piece.strip() for piece in pieces if piece.strip()]

    return segments


def _is_list_query(query: str) -> bool:
    q = query.lower()
    list_markers = ("list", "winners", "winner", "top", "rank", "recipients", "complete")
    return any(marker in q for marker in list_markers)


def _build_snippet(query_tokens: set[str], text: str, max_chars: int, list_query: bool) -> tuple[str, list[str]]:
    segments = _split_segments(text)
    if not segments:
        return "", []

    scored: list[tuple[int, int, str, set[str]]] = []
    for idx, segment in enumerate(segments):
        segment_tokens = _tokenize_keywords(segment)
        matched = query_tokens & segment_tokens
        if not matched:
            continue
        scored.append((len(matched), idx, segment, matched))

    if not scored:
        snippet = segments[0]
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 3].rstrip() + "..."
        return snippet, []

    scored.sort(key=lambda item: item[0], reverse=True)

    if list_query:
        best_idx = scored[0][1]
        window_start = max(0, best_idx - 2)
        window_end = min(len(segments), best_idx + 4)
        top = []
        for idx in range(window_start, window_end):
            segment = segments[idx]
            matched = query_tokens & _tokenize_keywords(segment)
            top.append((len(matched), idx, segment, matched))
    else:
        top = scored[:3]
        top.sort(key=lambda item: item[1])

    picked_segments: list[str] = []
    matched_terms: set[str] = set()
    for _, _, segment, matched in top:
        if segment not in picked_segments:
            picked_segments.append(segment)
        matched_terms.update(matched)

    snippet = " ".join(picked_segments).strip()
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3].rstrip() + "..."

    return snippet, sorted(matched_terms)


def _score_chunk(query_tokens: set[str], text: str, matched_terms: set[str]) -> float:
    text_tokens = _tokenize(text)
    base = _jaccard_score(query_tokens, text_tokens)
    if not query_tokens:
        return base

    coverage = len(matched_terms) / len(query_tokens)

    final_score = base + (0.35 * coverage)
    return round(final_score, 4)


def _prune_hits(hits: list[QueryHit]) -> list[QueryHit]:
    if not hits:
        return hits

    best_score = hits[0].score
    dynamic_threshold = max(settings.search_min_score, best_score * settings.search_relative_score_ratio)
    kept = [hit for hit in hits if hit.score >= dynamic_threshold]
    if kept:
        return kept
    return [hits[0]]


def search_chunks(request: QueryRequest) -> QueryResponse:
    manifests = sorted(settings.processed_dir.glob("*.json"))
    if not manifests:
        return QueryResponse(query=request.query, hits=[])

    scoring_tokens = _tokenize(request.query)
    match_tokens = _tokenize_keywords(request.query)
    if not match_tokens:
        match_tokens = scoring_tokens

    list_query = _is_list_query(request.query)
    snippet_limit = request.max_snippet_chars
    if list_query and request.response_mode != "compact":
        snippet_limit = max(snippet_limit, 2200)

    scored_hits: list[QueryHit] = []

    for manifest_path in manifests:
        try:
            doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        doc_id = doc.get("doc_id", "")
        source_file = doc.get("source_file", "")

        for chunk in doc.get("chunks", []):
            if not _passes_filters(chunk, request):
                continue

            chunk_text = chunk.get("text", "")
            snippet, matched_terms = _build_snippet(
                query_tokens=match_tokens,
                text=chunk_text,
                max_chars=snippet_limit,
                list_query=list_query,
            )
            matched_set = set(matched_terms)
            coverage = (len(matched_set) / len(match_tokens)) if match_tokens else 0.0
            if match_tokens and coverage < settings.search_min_keyword_coverage:
                continue

            score = _score_chunk(match_tokens, chunk_text, matched_set)
            if score <= 0:
                continue

            metadata = chunk.get("metadata", {})
            include_text = request.include_full_text or request.response_mode == "full"

            scored_hits.append(
                QueryHit(
                    doc_id=doc_id,
                    source_file=source_file,
                    chunk_id=chunk.get("chunk_id", ""),
                    page_start=chunk.get("page_start", 0),
                    page_end=chunk.get("page_end", 0),
                    score=score,
                    snippet=snippet,
                    matched_terms=matched_terms,
                    metadata=ChunkMetadata(
                        months=metadata.get("months", []),
                        topics=metadata.get("topics", []),
                        entities=metadata.get("entities", []),
                    ),
                    text=chunk_text if include_text else None,
                )
            )

    scored_hits.sort(key=lambda hit: hit.score, reverse=True)
    pruned_hits = _prune_hits(scored_hits)
    return QueryResponse(query=request.query, hits=pruned_hits[: request.top_k])
