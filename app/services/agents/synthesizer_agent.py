from app.core.config import settings
from app.schemas.ingestion import ChunkMetadata, QueryHit
from app.services.llm_answer_service import synthesize_answer
from app.services.retrieval_types import RetrievalCandidate


def _to_query_hits(candidates: list[RetrievalCandidate]) -> list[QueryHit]:
    hits: list[QueryHit] = []
    for candidate in candidates:
        hits.append(
            QueryHit(
                doc_id=candidate.doc_id,
                source_file=candidate.source_file,
                chunk_id=candidate.chunk_id,
                page_start=candidate.page_start,
                page_end=candidate.page_end,
                score=round(candidate.rerank_score, 4),
                snippet=candidate.snippet or candidate.text[: min(len(candidate.text), 500)],
                matched_terms=candidate.matched_terms,
                metadata=ChunkMetadata(
                    months=candidate.metadata.get("months", []),
                    topics=candidate.metadata.get("topics", []),
                    entities=candidate.metadata.get("entities", []),
                ),
                modality="image" if candidate.modality == "image" else "text",
                image_path=candidate.image_path or None,
                image_name=candidate.image_name or None,
                text=candidate.text,
            )
        )
    return hits


def _deterministic_answer(candidates: list[RetrievalCandidate], query: str) -> str:
    if not candidates:
        return "Insufficient evidence found for the query."
    lines: list[str] = [f"Query: {query}"]
    for candidate in candidates[: settings.llm_max_context_hits]:
        snippet = (candidate.snippet or candidate.text).strip()
        lines.append(f"- {snippet} ({candidate.chunk_id} p{candidate.page_start}-{candidate.page_end})")
    return "\n".join(lines)


async def generate_answer(
    *,
    query: str,
    candidates: list[RetrievalCandidate],
    use_llm: bool,
) -> tuple[str, str, str | None]:
    if not candidates:
        return "Insufficient evidence found for the query.", "no_hits", None

    if not use_llm:
        return _deterministic_answer(candidates, query), "disabled", None

    hits = _to_query_hits(candidates)
    answer, status, model = await synthesize_answer(query, hits)
    if not answer:
        return _deterministic_answer(candidates, query), "llm_error", model
    return answer, status, model
