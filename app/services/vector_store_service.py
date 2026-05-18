from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import uuid

from app.core.config import settings
from app.services.embedding_service import EmbeddingUnavailableError, embed_text, embed_texts


class VectorStoreUnavailableError(Exception):
    pass


@dataclass
class VectorSearchHit:
    doc_id: str
    source_file: str
    chunk_id: str
    page_start: int
    page_end: int
    score: float
    text: str
    metadata: dict


def _client() -> Any:
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:
        raise VectorStoreUnavailableError("qdrant-client not installed.") from exc

    try:
        return QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            timeout=10,
        )
    except Exception as exc:
        raise VectorStoreUnavailableError("Cannot initialize Qdrant client.") from exc


def _distance():
    from qdrant_client.http import models as qmodels

    value = settings.vector_distance.lower()
    if value == "dot":
        return qmodels.Distance.DOT
    if value == "euclid":
        return qmodels.Distance.EUCLID
    return qmodels.Distance.COSINE


def _point_id(doc_id: str, chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{chunk_id}"))


def _ensure_collection(client: Any, size: int) -> None:
    from qdrant_client.http import models as qmodels

    try:
        exists = client.collection_exists(settings.qdrant_collection)
    except Exception as exc:
        raise VectorStoreUnavailableError("Unable to check Qdrant collection.") from exc

    if exists:
        return

    try:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=qmodels.VectorParams(size=size, distance=_distance()),
        )
    except Exception as exc:
        raise VectorStoreUnavailableError("Unable to create Qdrant collection.") from exc


async def index_chunks(
    doc_id: str,
    source_file: str,
    chunks: list[dict],
) -> dict:
    if not settings.enable_vector_indexing:
        return {"status": "disabled", "points_indexed": 0}

    if not chunks:
        return {"status": "empty", "points_indexed": 0}

    texts = [chunk.get("text", "") for chunk in chunks]
    try:
        vectors = await embed_texts(texts)
    except EmbeddingUnavailableError:
        return {"status": "embedding_unavailable", "points_indexed": 0}

    try:
        client = _client()
        _ensure_collection(client, len(vectors[0]))
    except VectorStoreUnavailableError:
        return {"status": "qdrant_unavailable", "points_indexed": 0}

    try:
        from qdrant_client.http import models as qmodels
    except Exception:
        return {"status": "qdrant_unavailable", "points_indexed": 0}

    points: list[qmodels.PointStruct] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        points.append(
            qmodels.PointStruct(
                id=_point_id(doc_id=doc_id, chunk_id=chunk["chunk_id"]),
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "chunk_id": chunk["chunk_id"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                },
            )
        )

    try:
        client.upsert(collection_name=settings.qdrant_collection, points=points, wait=False)
    except Exception:
        return {"status": "qdrant_error", "points_indexed": 0}

    return {"status": "indexed", "points_indexed": len(points)}


async def search_chunks_semantic(query: str, limit: int) -> tuple[list[VectorSearchHit], str]:
    if not settings.enable_vector_indexing:
        return [], "disabled"

    try:
        query_vector = await embed_text(query)
    except EmbeddingUnavailableError:
        return [], "unavailable"

    try:
        client = _client()
        _ensure_collection(client, len(query_vector))
    except VectorStoreUnavailableError:
        return [], "unavailable"

    try:
        search_result = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=limit,
        )
        points = search_result.points
    except Exception:
        try:
            points = client.search(
                collection_name=settings.qdrant_collection,
                query_vector=query_vector,
                limit=limit,
            )
        except Exception:
            return [], "error"

    hits: list[VectorSearchHit] = []
    for point in points:
        payload = point.payload or {}
        hits.append(
            VectorSearchHit(
                doc_id=str(payload.get("doc_id", "")),
                source_file=str(payload.get("source_file", "")),
                chunk_id=str(payload.get("chunk_id", "")),
                page_start=int(payload.get("page_start", 0) or 0),
                page_end=int(payload.get("page_end", 0) or 0),
                score=float(getattr(point, "score", 0.0) or 0.0),
                text=str(payload.get("text", "")),
                metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {},
            )
        )

    return hits, "used"
