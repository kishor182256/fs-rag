from app.schemas.ingestion import ChunkMetadata
from app.schemas.ingestion import QueryHit, QueryRequest, QueryResponse
from app.services.hybrid_retrieval_service import run_hybrid_retrieval
from app.services.llm_answer_service import synthesize_answer


async def run_query_pipeline(request: QueryRequest) -> QueryResponse:
    candidates, vector_status = await run_hybrid_retrieval(request)

    hits: list[QueryHit] = []
    for candidate in candidates:
        include_text = request.include_full_text or request.response_mode == "full"
        hits.append(
            QueryHit(
                doc_id=candidate.doc_id,
                source_file=candidate.source_file,
                chunk_id=candidate.chunk_id,
                page_start=candidate.page_start,
                page_end=candidate.page_end,
                score=round(candidate.rerank_score, 4),
                snippet=candidate.snippet or candidate.text[: min(request.max_snippet_chars, 1200)],
                matched_terms=candidate.matched_terms,
                metadata=ChunkMetadata(
                    months=candidate.metadata.get("months", []),
                    topics=candidate.metadata.get("topics", []),
                    entities=candidate.metadata.get("entities", []),
                ),
                text=candidate.text if include_text else None,
            )
        )

    result = QueryResponse(query=request.query, hits=hits, vector_status=vector_status)

    if not request.use_llm:
        result.answer_status = "disabled"
        return result

    answer, status, model = await synthesize_answer(request.query, result.hits)
    result.answer = answer
    result.answer_status = status
    result.answer_model = model
    return result
