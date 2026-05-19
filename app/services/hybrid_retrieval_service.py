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


def _compute_matched_terms(query: str, text: str) -> list[str]:
    query_terms = {token.strip().lower() for token in query.split() if token.strip()}
    text_terms = {token.strip().lower() for token in text.split() if token.strip()}
    if not query_terms or not text_terms:
        return []
    return sorted(query_terms.intersection(text_terms))


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


async def run_hybrid_retrieval(request: QueryRequest) -> tuple[list[RetrievalCandidate], str]:
    chunks = load_text_chunks_for_bm25(limit=5000)
    bm25_candidates: list[RetrievalCandidate] = []
    if chunks:
        bm25_candidates = search_bm25(
            chunks=chunks,
            request=request,
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
        if request.include_images and settings.enable_multimodal_ingest:
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
        request=request,
    )
    if not merged:
        return [], vector_status
    reranked = rerank_candidates(query=request.query, candidates=merged, request=request)
    return reranked[: request.top_k], vector_status
