from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import uuid

from app.core.config import settings
from app.services.embedding_service import EmbeddingUnavailableError, embed_text, embed_texts
from app.services.multimodal_service import MultimodalUnavailableError, embed_images_clip, embed_text_clip


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
    modality: str = "text"
    image_path: str = ""
    image_name: str = ""


def qdrant_health() -> tuple[str, str]:
    if not settings.enable_vector_indexing:
        return "disabled", "vector indexing disabled by config"
    try:
        client = _client()
    except VectorStoreUnavailableError as exc:
        return "down", str(exc)

    try:
        exists = client.collection_exists(settings.qdrant_collection)
        return (
            "up",
            f"host={settings.qdrant_host} port={settings.qdrant_port} collection={settings.qdrant_collection} exists={exists}",
        )
    except Exception as exc:
        return "down", f"Unable to query collection state: {exc}"


def _client() -> Any:
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:
        raise VectorStoreUnavailableError("qdrant-client not installed.") from exc

    try:
        return QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            https=bool(settings.qdrant_https),
            api_key=(settings.qdrant_api_key or None),
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
    _ensure_collection_named(client=client, size=size, collection_name=settings.qdrant_collection)


def _ensure_collection_named(client: Any, size: int, collection_name: str) -> None:
    from qdrant_client.http import models as qmodels

    try:
        exists = client.collection_exists(collection_name)
    except Exception as exc:
        raise VectorStoreUnavailableError("Unable to check Qdrant collection.") from exc

    if exists:
        return

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=size, distance=_distance()),
        )
    except Exception as exc:
        raise VectorStoreUnavailableError("Unable to create Qdrant collection.") from exc


def _collection_exists(client: Any, collection_name: str) -> bool:
    try:
        return bool(client.collection_exists(collection_name))
    except Exception:
        return False


def find_duplicate_by_file_hash(file_sha256: str) -> dict | None:
    normalized = str(file_sha256 or "").strip().lower()
    if not normalized:
        return None
    if not settings.enable_vector_indexing:
        return None

    try:
        client = _client()
        from qdrant_client.http import models as qmodels
    except Exception:
        return None

    collection_name = settings.qdrant_collection
    if not _collection_exists(client, collection_name):
        return None

    hash_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="file_sha256",
                match=qmodels.MatchValue(value=normalized),
            )
        ]
    )

    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=hash_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        return None

    if not points:
        return None

    payload = points[0].payload or {}
    return {
        "doc_id": str(payload.get("doc_id", "")),
        "source_file": str(payload.get("source_file", "")),
        "file_sha256": normalized,
    }


async def index_chunks(
    doc_id: str,
    source_file: str,
    chunks: list[dict],
    file_sha256: str = "",
) -> dict:
    if not settings.enable_vector_indexing:
        return {"status": "disabled", "points_indexed": 0}

    if not chunks:
        return {"status": "empty", "points_indexed": 0}

    texts = [chunk.get("text", "") for chunk in chunks]
    try:
        vectors = await embed_texts(texts)
    except EmbeddingUnavailableError as exc:
        return {"status": "embedding_unavailable", "points_indexed": 0, "error": str(exc)}

    try:
        client = _client()
        _ensure_collection(client, len(vectors[0]))
    except VectorStoreUnavailableError as exc:
        return {"status": "qdrant_unavailable", "points_indexed": 0, "error": str(exc)}

    try:
        from qdrant_client.http import models as qmodels
    except Exception as exc:
        return {"status": "qdrant_unavailable", "points_indexed": 0, "error": str(exc)}

    points: list[qmodels.PointStruct] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        points.append(
            qmodels.PointStruct(
                id=_point_id(doc_id=doc_id, chunk_id=chunk["chunk_id"]),
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "file_sha256": str(file_sha256 or "").strip().lower(),
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
    except Exception as exc:
        return {"status": "qdrant_error", "points_indexed": 0, "error": str(exc)}

    return {"status": "indexed", "points_indexed": len(points)}


async def index_multimodal_image_records(
    doc_id: str,
    source_file: str,
    image_records: list[dict],
    file_sha256: str = "",
) -> dict:
    if not settings.enable_multimodal_ingest:
        return {"status": "disabled", "points_indexed": 0}
    if not settings.enable_vector_indexing:
        return {"status": "vector_indexing_disabled", "points_indexed": 0}
    if not image_records:
        return {"status": "empty", "points_indexed": 0}

    texts = [str(record.get("embedding_text", "") or record.get("caption", "") or "").strip() for record in image_records]
    if any(not text for text in texts):
        texts = [
            text if text else f"Image content from {source_file}, page {int(image_records[idx].get('page', 0) or 0)}."
            for idx, text in enumerate(texts)
        ]

    try:
        vectors = await embed_texts(texts)
    except EmbeddingUnavailableError as exc:
        return {"status": "embedding_unavailable", "points_indexed": 0, "error": str(exc)}

    try:
        client = _client()
        _ensure_collection_named(
            client=client,
            size=len(vectors[0]),
            collection_name=settings.multimodal_qdrant_collection,
        )
    except VectorStoreUnavailableError as exc:
        return {"status": "qdrant_unavailable", "points_indexed": 0, "error": str(exc)}

    try:
        from qdrant_client.http import models as qmodels
    except Exception as exc:
        return {"status": "qdrant_unavailable", "points_indexed": 0, "error": str(exc)}

    caption_points: list[qmodels.PointStruct] = []
    for record, vector, text in zip(image_records, vectors, texts, strict=True):
        image_id = str(record.get("image_id", "") or record.get("chunk_id", ""))
        if not image_id:
            continue
        payload = {
            "doc_id": doc_id,
            "source_file": source_file,
            "file_sha256": str(file_sha256 or "").strip().lower(),
            "modality": "image",
            "image_id": image_id,
            "page": int(record.get("page", 0) or 0),
            "image_path": str(record.get("image_path", "")),
            "image_name": str(record.get("image_name", "")),
            "caption": str(record.get("caption", "")),
            "embedding_text": text,
            "metadata": record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {},
        }
        caption_points.append(
            qmodels.PointStruct(
                id=_point_id(doc_id=doc_id, chunk_id=f"image:{image_id}"),
                vector=vector,
                payload=payload,
            )
        )

    if not caption_points:
        return {"status": "empty", "points_indexed": 0}

    try:
        client.upsert(collection_name=settings.multimodal_qdrant_collection, points=caption_points, wait=False)
    except Exception as exc:
        return {"status": "qdrant_error", "points_indexed": 0, "error": str(exc)}

    clip_points_indexed = 0
    clip_error = ""
    if settings.enable_clip_image_vectors:
        try:
            clip_vectors = embed_images_clip([str(record.get("image_path", "")) for record in image_records])
            if clip_vectors:
                _ensure_collection_named(
                    client=client,
                    size=len(clip_vectors[0]),
                    collection_name=settings.multimodal_clip_qdrant_collection,
                )
                clip_points: list[qmodels.PointStruct] = []
                for record, vector in zip(image_records, clip_vectors, strict=True):
                    image_id = str(record.get("image_id", "") or record.get("chunk_id", ""))
                    if not image_id:
                        continue
                    payload = {
                        "doc_id": doc_id,
                        "source_file": source_file,
                        "file_sha256": str(file_sha256 or "").strip().lower(),
                        "modality": "image",
                        "image_id": image_id,
                        "page": int(record.get("page", 0) or 0),
                        "image_path": str(record.get("image_path", "")),
                        "image_name": str(record.get("image_name", "")),
                        "caption": str(record.get("caption", "")),
                        "metadata": record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {},
                    }
                    clip_points.append(
                        qmodels.PointStruct(
                            id=_point_id(doc_id=doc_id, chunk_id=f"image-clip:{image_id}"),
                            vector=vector,
                            payload=payload,
                        )
                    )
                if clip_points:
                    client.upsert(
                        collection_name=settings.multimodal_clip_qdrant_collection,
                        points=clip_points,
                        wait=False,
                    )
                    clip_points_indexed = len(clip_points)
        except (MultimodalUnavailableError, VectorStoreUnavailableError, Exception) as exc:  # noqa: PERF203
            clip_error = str(exc)

    result = {
        "status": "indexed",
        "points_indexed": len(caption_points),
        "collection": settings.multimodal_qdrant_collection,
        "clip_collection": settings.multimodal_clip_qdrant_collection if settings.enable_clip_image_vectors else "",
        "clip_points_indexed": clip_points_indexed,
    }
    if clip_error:
        result["clip_error"] = clip_error
    return result


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
                modality="text",
            )
        )

    return hits, "used"


async def search_images_semantic(query: str, limit: int) -> tuple[list[VectorSearchHit], str]:
    if not settings.enable_multimodal_ingest:
        return [], "disabled"
    if not settings.enable_vector_indexing:
        return [], "disabled"

    client = None
    try:
        client = _client()
    except VectorStoreUnavailableError:
        return [], "unavailable"

    merged: dict[tuple[str, str], VectorSearchHit] = {}

    # Caption/text semantic search for images
    try:
        caption_query_vector = await embed_text(query)
        _ensure_collection_named(
            client=client,
            size=len(caption_query_vector),
            collection_name=settings.multimodal_qdrant_collection,
        )
        try:
            search_result = client.query_points(
                collection_name=settings.multimodal_qdrant_collection,
                query=caption_query_vector,
                limit=limit,
            )
            caption_points = search_result.points
        except Exception:
            caption_points = client.search(
                collection_name=settings.multimodal_qdrant_collection,
                query_vector=caption_query_vector,
                limit=limit,
            )
        for point in caption_points:
            payload = point.payload or {}
            page = int(payload.get("page", 0) or 0)
            image_id = str(payload.get("image_id", "") or str(getattr(point, "id", "")))
            key = (str(payload.get("doc_id", "")), image_id)
            merged[key] = VectorSearchHit(
                doc_id=str(payload.get("doc_id", "")),
                source_file=str(payload.get("source_file", "")),
                chunk_id=image_id,
                page_start=page,
                page_end=page,
                score=float(getattr(point, "score", 0.0) or 0.0),
                text=str(payload.get("caption", "") or payload.get("embedding_text", "") or ""),
                metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {},
                modality="image",
                image_path=str(payload.get("image_path", "")),
                image_name=str(payload.get("image_name", "")),
            )
    except Exception:
        pass

    # CLIP text->image search
    if settings.enable_clip_image_vectors:
        try:
            clip_query_vector = embed_text_clip(query)
            _ensure_collection_named(
                client=client,
                size=len(clip_query_vector),
                collection_name=settings.multimodal_clip_qdrant_collection,
            )
            try:
                clip_result = client.query_points(
                    collection_name=settings.multimodal_clip_qdrant_collection,
                    query=clip_query_vector,
                    limit=limit,
                )
                clip_points = clip_result.points
            except Exception:
                clip_points = client.search(
                    collection_name=settings.multimodal_clip_qdrant_collection,
                    query_vector=clip_query_vector,
                    limit=limit,
                )

            for point in clip_points:
                payload = point.payload or {}
                page = int(payload.get("page", 0) or 0)
                image_id = str(payload.get("image_id", "") or str(getattr(point, "id", "")))
                key = (str(payload.get("doc_id", "")), image_id)
                clip_score = float(getattr(point, "score", 0.0) or 0.0)
                existing = merged.get(key)
                if existing is None or clip_score > existing.score:
                    merged[key] = VectorSearchHit(
                        doc_id=str(payload.get("doc_id", "")),
                        source_file=str(payload.get("source_file", "")),
                        chunk_id=image_id,
                        page_start=page,
                        page_end=page,
                        score=clip_score,
                        text=str(payload.get("caption", "") or ""),
                        metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {},
                        modality="image",
                        image_path=str(payload.get("image_path", "")),
                        image_name=str(payload.get("image_name", "")),
                    )
                else:
                    existing.score = max(existing.score, clip_score)
        except Exception:
            pass

    if not merged:
        return [], "unavailable"

    hits = sorted(merged.values(), key=lambda item: item.score, reverse=True)[:limit]
    return hits, "used"


def load_text_chunks_for_bm25(limit: int = 5000) -> list[dict]:
    if not settings.enable_vector_indexing:
        return []

    try:
        client = _client()
    except VectorStoreUnavailableError:
        return []

    if not _collection_exists(client, settings.qdrant_collection):
        return []

    rows: list[dict] = []
    offset = None
    page_size = 256

    while len(rows) < limit:
        try:
            points, next_offset = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=min(page_size, limit - len(rows)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            break

        if not points:
            break

        for point in points:
            payload = point.payload or {}
            text = str(payload.get("text", "")).strip()
            if not text:
                continue
            rows.append(
                {
                    "doc_id": str(payload.get("doc_id", "")),
                    "source_file": str(payload.get("source_file", "")),
                    "chunk_id": str(payload.get("chunk_id", "")),
                    "page_start": int(payload.get("page_start", 0) or 0),
                    "page_end": int(payload.get("page_end", 0) or 0),
                    "text": text,
                    "metadata": payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {},
                }
            )
            if len(rows) >= limit:
                break

        if next_offset is None:
            break
        offset = next_offset

    return rows
