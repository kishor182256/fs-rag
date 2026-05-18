import json

from app.core.config import settings
from app.schemas.ingestion import QueryRequest
from app.services.bm25_service import search_bm25
from app.services.reranker_service import rerank_candidates
from app.services.retrieval_types import RetrievalCandidate
from app.services.vector_store_service import search_chunks_semantic


def _load_chunk_records() -> list[dict]:
    manifests = sorted(settings.processed_dir.glob("*.json"))
    records: list[dict] = []

    for manifest_path in manifests:
        try:
            doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        doc_id = str(doc.get("doc_id", ""))
        source_file = str(doc.get("source_file", ""))
        for chunk in doc.get("chunks", []):
            records.append(
                {
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "chunk_id": str(chunk.get("chunk_id", "")),
                    "page_start": int(chunk.get("page_start", 0) or 0),
                    "page_end": int(chunk.get("page_end", 0) or 0),
                    "text": str(chunk.get("text", "")),
                    "metadata": chunk.get("metadata", {}) if isinstance(chunk.get("metadata", {}), dict) else {},
                }
            )

    return records


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

    merged: dict[str, RetrievalCandidate] = {candidate.chunk_id: candidate for candidate in bm25_candidates}

    for hit in vector_candidates:
        if not _passes_filters(hit.metadata):
            continue
        existing = merged.get(hit.chunk_id)
        if existing:
            existing.vector_score = max(existing.vector_score, float(hit.score))
            if not existing.text:
                existing.text = hit.text
            continue

        merged[hit.chunk_id] = RetrievalCandidate(
            doc_id=hit.doc_id,
            source_file=hit.source_file,
            chunk_id=hit.chunk_id,
            page_start=hit.page_start,
            page_end=hit.page_end,
            text=hit.text,
            metadata=hit.metadata,
            matched_terms=[],
            bm25_score=0.0,
            vector_score=float(hit.score),
        )

    return list(merged.values())


async def run_hybrid_retrieval(request: QueryRequest) -> tuple[list[RetrievalCandidate], str]:
    chunks = _load_chunk_records()
    if not chunks:
        return [], "disabled"

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

    merged = _merge_candidates(
        bm25_candidates=bm25_candidates,
        vector_candidates=vector_candidates,
        request=request,
    )
    reranked = rerank_candidates(query=request.query, candidates=merged, request=request)
    return reranked[: request.top_k], vector_status
