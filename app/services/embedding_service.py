import json

import httpx

from app.core.config import settings


class EmbeddingUnavailableError(Exception):
    pass


def _extract_vectors(payload: dict) -> list[list[float]]:
    data = payload.get("data", [])
    vectors: list[list[float]] = []
    for item in data:
        embedding = item.get("embedding")
        if isinstance(embedding, list) and embedding:
            vectors.append([float(value) for value in embedding])
    return vectors


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    endpoint = settings.openai_base_url.rstrip("/") + "/embeddings"
    headers = {"Content-Type": "application/json"}
    if settings.openai_api_key:
        headers["Authorization"] = f"Bearer {settings.openai_api_key}"

    payload = {
        "model": settings.embedding_model,
        "input": texts,
    }

    try:
        async with httpx.AsyncClient(timeout=settings.embedding_timeout_seconds) as client:
            response = await client.post(endpoint, headers=headers, content=json.dumps(payload))
            response.raise_for_status()
            response_payload = response.json()
    except httpx.HTTPError as exc:
        raise EmbeddingUnavailableError("Embedding endpoint unavailable.") from exc

    vectors = _extract_vectors(response_payload)
    if len(vectors) != len(texts):
        raise EmbeddingUnavailableError("Embedding response shape mismatch.")
    return vectors


async def embed_text(text: str) -> list[float]:
    vectors = await embed_texts([text])
    if not vectors:
        raise EmbeddingUnavailableError("No embedding returned.")
    return vectors[0]
