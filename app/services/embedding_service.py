import asyncio
import json

import httpx

from app.core.config import settings


class EmbeddingUnavailableError(Exception):
    pass


def _embedding_provider() -> str:
    return str(getattr(settings, "embedding_provider", "openai") or "openai").strip().lower()


def _bedrock_region() -> str:
    region = (getattr(settings, "bedrock_region", "") or "").strip()
    if region:
        return region
    fallback = (getattr(settings, "aws_region", "") or "").strip()
    if fallback:
        return fallback
    raise EmbeddingUnavailableError("Bedrock region missing. Set BEDROCK_REGION or AWS_REGION.")


def _extract_vectors(payload: dict) -> list[list[float]]:
    data = payload.get("data", [])
    vectors: list[list[float]] = []
    for item in data:
        embedding = item.get("embedding")
        if isinstance(embedding, list) and embedding:
            vectors.append([float(value) for value in embedding])
    return vectors


def _extract_bedrock_embedding(payload: dict) -> list[float]:
    embedding = payload.get("embedding")
    if isinstance(embedding, list) and embedding:
        return [float(value) for value in embedding]

    by_type = payload.get("embeddingsByType", {})
    if isinstance(by_type, dict):
        candidate = by_type.get("float")
        if isinstance(candidate, list) and candidate:
            return [float(value) for value in candidate]

    raise EmbeddingUnavailableError("Bedrock embedding response missing embedding vector.")


def _is_max_tokens_per_request_error(exc: httpx.HTTPStatusError) -> bool:
    try:
        payload = exc.response.json()
        error_obj = payload.get("error", {})
        code = str(error_obj.get("code", "") or "").strip().lower()
        message = str(error_obj.get("message", "") or "").strip().lower()
        return code == "max_tokens_per_request" or "max" in message and "tokens per request" in message
    except Exception:
        return False


async def _embed_batch_openai(texts: list[str]) -> list[list[float]]:
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
            left = await _embed_batch_openai(texts[:midpoint])
            right = await _embed_batch_openai(texts[midpoint:])
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


def _embed_single_bedrock_sync(text: str) -> list[float]:
    try:
        import boto3
    except Exception as exc:
        raise EmbeddingUnavailableError(f"boto3 unavailable for Bedrock embeddings: {exc}") from exc

    model_id = (getattr(settings, "bedrock_embedding_model_id", "") or "").strip()
    if not model_id:
        raise EmbeddingUnavailableError("BEDROCK_EMBEDDING_MODEL_ID is required for Bedrock embeddings.")

    client = boto3.client("bedrock-runtime", region_name=_bedrock_region())
    body = json.dumps({"inputText": text})

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        raw_body = response["body"].read()
        payload = json.loads(raw_body)
    except Exception as exc:
        raise EmbeddingUnavailableError(f"Bedrock embedding invocation failed: {exc}") from exc

    return _extract_bedrock_embedding(payload)


async def _embed_batch_bedrock(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vectors.append(await asyncio.to_thread(_embed_single_bedrock_sync, text))
    return vectors


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    provider = _embedding_provider()
    if provider == "bedrock":
        return await _embed_batch_bedrock(texts)
    if provider == "openai":
        return await _embed_batch_openai(texts)

    raise EmbeddingUnavailableError(f"Unsupported embedding provider: {provider}")


async def embed_text(text: str) -> list[float]:
    vectors = await embed_texts([text])
    if not vectors:
        raise EmbeddingUnavailableError("No embedding returned.")
    return vectors[0]
