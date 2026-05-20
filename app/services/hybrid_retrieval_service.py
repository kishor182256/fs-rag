import re

from app.core.config import settings
from app.schemas.ingestion import QueryRequest
from app.services.bm25_service import search_bm25
from app.services.reranker_service import rerank_candidates
from app.services.retrieval_types import RetrievalCandidate
from app.services.vector_store_service import (
    load_text_chunks_for_bm25,
    search_chunks_semantic,
    search_images_semantic,
)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")
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
IMAGE_INTENT_MARKERS = {
    "image",
    "images",
    "diagram",
    "diagrams",
    "figure",
    "figures",
    "chart",
    "charts",
    "graph",
    "graphs",
    "visual",
    "visuals",
    "show",
}


def _query_keywords(query: str) -> list[str]:
    tokens = [token for token in TOKEN_PATTERN.findall(query.lower()) if len(token) > 1]
    keywords = [token for token in tokens if token not in STOPWORDS]
    return keywords or tokens


def _compute_matched_terms(query: str, text: str) -> list[str]:
    query_terms = {token.strip().lower() for token in query.split() if token.strip()}
    text_terms = {token.strip().lower() for token in text.split() if token.strip()}
    if not query_terms or not text_terms:
        return []
    return sorted(query_terms.intersection(text_terms))


def _needs_image_routing(query: str, include_images: bool) -> bool:
    if include_images:
        return True
    lowered = query.lower()
    return any(marker in lowered for marker in IMAGE_INTENT_MARKERS)


def _merge_candidates(
    bm25_candidates: list[RetrievalCandidate],
    vector_candidates,
    request: QueryRequest,
) -> list[RetrievalCandidate]:
    def _passes_filters(metadata: dict) -> bool:
        chunk_months = {m.lower() for m in metadata.get("months", [])}
        chunk_topics = {t.lower() for t in metadata.get("topics", [])}
        if request.months and not ({m.lower() for m in request.months} & chunk_months):
            return False
        if request.topics and not ({t.lower() for t in request.topics} & chunk_topics):
            return False
        return True

    def _key(doc_id: str, chunk_id: str, modality: str) -> str:
        return f"{doc_id}::{modality}::{chunk_id}"

    merged: dict[str, RetrievalCandidate] = {
        _key(candidate.doc_id, candidate.chunk_id, candidate.modality): candidate for candidate in bm25_candidates
    }

    for hit in vector_candidates:
        if not _passes_filters(hit.metadata):
            continue
        hit_key = _key(hit.doc_id, hit.chunk_id, getattr(hit, "modality", "text"))
        existing = merged.get(hit_key)
        if existing:
            existing.vector_score = max(existing.vector_score, float(hit.score))
            if not existing.text:
                existing.text = hit.text
            continue

        merged[hit_key] = RetrievalCandidate(
            doc_id=hit.doc_id,
            source_file=hit.source_file,
            chunk_id=hit.chunk_id,
            page_start=hit.page_start,
            page_end=hit.page_end,
            text=hit.text,
            metadata=hit.metadata,
            matched_terms=_compute_matched_terms(request.query, hit.text),
            modality=getattr(hit, "modality", "text"),
            image_path=getattr(hit, "image_path", ""),
            image_name=getattr(hit, "image_name", ""),
            bm25_score=0.0,
            vector_score=float(hit.score),
        )

    return list(merged.values())


def _post_filter_candidates(candidates: list[RetrievalCandidate], request: QueryRequest) -> list[RetrievalCandidate]:
    if not candidates:
        return []

    query_terms = set(_query_keywords(request.query))
    top_score = max(candidates[0].rerank_score, 1e-6)
    coverage_threshold = settings.search_min_keyword_coverage
    if len(query_terms) >= 8:
        coverage_threshold = min(coverage_threshold, 0.25)
    if len(query_terms) >= 12:
        coverage_threshold = min(coverage_threshold, 0.20)

    kept: list[RetrievalCandidate] = []
    seen_ids: set[str] = set()
    seen_signatures: set[str] = set()

    for candidate in candidates:
        candidate_key = f"{candidate.doc_id}::{candidate.chunk_id}::{candidate.modality}"
        if candidate_key in seen_ids:
            continue
        seen_ids.add(candidate_key)

        signature = re.sub(r"\s+", " ", candidate.text.lower()).strip()[:280]
        if signature and signature in seen_signatures:
            continue
        if signature:
            seen_signatures.add(signature)

        matched = {term.lower() for term in candidate.matched_terms}
        lexical_coverage = (len(matched & query_terms) / len(query_terms)) if query_terms else 1.0
        relative_score = candidate.rerank_score / top_score

        strong_anchor = (
            candidate.rerank_score >= max(0.8, top_score * 0.90)
            or bool(set(re.findall(r"\b[A-Z]{2,}\b", request.query)) & set(re.findall(r"\b[A-Z]{2,}\b", candidate.text)))
        )
        passes_quality_gate = (
            candidate.rerank_score >= settings.search_min_score
            and relative_score >= settings.search_relative_score_ratio
            and (lexical_coverage >= coverage_threshold or strong_anchor)
        )

        if passes_quality_gate or not kept:
            kept.append(candidate)

    return kept


def _ensure_image_presence(
    *,
    candidates: list[RetrievalCandidate],
    ranked_candidates: list[RetrievalCandidate],
    top_k: int,
    needs_images: bool,
) -> list[RetrievalCandidate]:
    if not needs_images:
        return candidates[:top_k]

    top = candidates[:top_k]
    if any(candidate.modality == "image" for candidate in top):
        return top

    fallback_image = next((candidate for candidate in ranked_candidates if candidate.modality == "image"), None)
    if fallback_image is None:
        return top

    deduped = []
    seen: set[str] = set()
    for candidate in top + [fallback_image]:
        key = f"{candidate.doc_id}:{candidate.chunk_id}:{candidate.modality}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    if len(deduped) > top_k:
        # keep best scored candidates after ensuring one image is present
        deduped.sort(key=lambda item: item.rerank_score, reverse=True)
        image_kept = next((item for item in deduped if item.modality == "image"), None)
        top_only = deduped[:top_k]
        if image_kept and all(item.modality != "image" for item in top_only):
            top_only[-1] = image_kept
        return top_only

    return deduped[:top_k]


async def run_hybrid_retrieval(request: QueryRequest) -> tuple[list[RetrievalCandidate], str]:
    needs_images = _needs_image_routing(query=request.query, include_images=request.include_images)
    effective_request = request.model_copy(update={"include_images": needs_images})

    chunks = load_text_chunks_for_bm25(limit=5000)
    bm25_candidates: list[RetrievalCandidate] = []
    if chunks:
        bm25_candidates = search_bm25(
            chunks=chunks,
            request=effective_request,
            bm25_k1=settings.bm25_k1,
            bm25_b=settings.bm25_b,
            limit=max(request.top_k * 4, 20),
        )

    vector_status = "disabled"
    vector_candidates = []
    if request.use_vector:
        vector_candidates, vector_status = await search_chunks_semantic(
            query=request.query,
            limit=max(request.vector_top_k, settings.vector_query_limit),
        )
        if needs_images and settings.enable_multimodal_ingest:
            image_hits, image_vector_status = await search_images_semantic(
                query=request.query,
                limit=max(request.vector_top_k, settings.vector_query_limit),
            )
            vector_candidates.extend(image_hits)
            if vector_status == "used" or image_vector_status == "used":
                vector_status = "used"
            elif vector_status == "error" or image_vector_status == "error":
                vector_status = "error"
            elif vector_status == "unavailable" or image_vector_status == "unavailable":
                vector_status = "unavailable"

    merged = _merge_candidates(
        bm25_candidates=bm25_candidates,
        vector_candidates=vector_candidates,
        request=effective_request,
    )
    if not merged:
        return [], vector_status
    reranked = rerank_candidates(query=request.query, candidates=merged, request=request)
    filtered = _post_filter_candidates(reranked, request)
    final_candidates = _ensure_image_presence(
        candidates=filtered,
        ranked_candidates=reranked,
        top_k=request.top_k,
        needs_images=needs_images,
    )
    return final_candidates, vector_status
