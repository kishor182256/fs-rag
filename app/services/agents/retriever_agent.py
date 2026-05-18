from app.schemas.agentic import AgenticQueryRequest
from app.services.hybrid_retrieval_service import run_hybrid_retrieval
from app.services.retrieval_types import RetrievalCandidate


async def gather_evidence(request: AgenticQueryRequest) -> tuple[list[RetrievalCandidate], str]:
    # Reuse current hybrid retrieval by mapping shared request fields.
    from app.schemas.ingestion import QueryRequest

    retrieval_request = QueryRequest(
        query=request.query,
        top_k=request.top_k,
        months=request.months,
        topics=request.topics,
        use_vector=request.use_vector,
        include_images=request.include_images,
        vector_top_k=request.vector_top_k,
        use_llm=False,
        include_full_text=False,
        response_mode=request.response_mode,
        max_snippet_chars=request.max_snippet_chars,
    )
    return await run_hybrid_retrieval(retrieval_request)
