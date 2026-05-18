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


def _is_max_tokens_per_request_error(exc: httpx.HTTPStatusError) -> bool:
    try:
        payload = exc.response.json()
        error_obj = payload.get("error", {})
        code = str(error_obj.get("code", "") or "").strip().lower()
        message = str(error_obj.get("message", "") or "").strip().lower()
        return code == "max_tokens_per_request" or "max" in message and "tokens per request" in message
    except Exception:
        return False


async def _embed_batch(texts: list[str]) -> list[list[float]]:
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
    except httpx.HTTPStatusError as exc:
        if _is_max_tokens_per_request_error(exc):
            if len(texts) <= 1:
                raise EmbeddingUnavailableError(
                    "Single input exceeds embedding token limit. Split chunking further or reduce chunk size."
                ) from exc
            midpoint = len(texts) // 2
            left = await _embed_batch(texts[:midpoint])
            right = await _embed_batch(texts[midpoint:])
            return left + right
        body = ""
        try:
            body = exc.response.text[:400]
        except Exception:
            body = ""
        raise EmbeddingUnavailableError(
            f"Embedding HTTP {exc.response.status_code} from {endpoint}. Response: {body}"
        ) from exc
    except httpx.HTTPError as exc:
        raise EmbeddingUnavailableError(f"Embedding endpoint unavailable at {endpoint}: {exc}") from exc

    vectors = _extract_vectors(response_payload)
    if len(vectors) != len(texts):
        raise EmbeddingUnavailableError(
            f"Embedding response shape mismatch. expected={len(texts)} got={len(vectors)}"
        )
    return vectors


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return await _embed_batch(texts)


async def embed_text(text: str) -> list[float]:
    vectors = await embed_texts([text])
    if not vectors:
        raise EmbeddingUnavailableError("No embedding returned.")
    return vectors[0]
