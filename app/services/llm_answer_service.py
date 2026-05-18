import json

import httpx

from app.core.config import settings
from app.schemas.ingestion import QueryHit


def _extract_output_text(payload: dict) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text", "")
                if text:
                    parts.append(text)

    return "\n".join(parts).strip()


def _build_context(hits: list[QueryHit]) -> str:
    context_blocks: list[str] = []
    for idx, hit in enumerate(hits[: settings.llm_max_context_hits], start=1):
        content = hit.text or hit.snippet
        context_blocks.append(
            "\n".join(
                [
                    f"[HIT {idx}]",
                    f"chunk_id: {hit.chunk_id}",
                    f"source_file: {hit.source_file}",
                    f"pages: {hit.page_start}-{hit.page_end}",
                    f"score: {hit.score}",
                    f"content: {content}",
                ]
            )
        )

    return "\n\n".join(context_blocks)


async def synthesize_answer(query: str, hits: list[QueryHit]) -> tuple[str | None, str, str | None]:
    if not hits:
        return "No relevant information found in retrieved context.", "no_hits", None

    instructions = (
        "You are a strict RAG answer engine. "
        "Answer only from provided context. "
        "Return only what the user asked; do not include unrelated sections. "
        "If the user asks for a list, return only that list. "
        "Do not add external facts. "
        "Keep response concise and high-signal. "
        "Cite each line using (chunk_id pX-Y)."
    )

    user_prompt = (
        f"User query: {query}\n\n"
        "Retrieved context follows.\n"
        "Use it to produce the final answer.\n\n"
        f"{_build_context(hits)}"
    )

    payload = {
        "model": settings.openai_model,
        "instructions": instructions,
        "input": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    }

    endpoint = settings.openai_base_url.rstrip("/") + "/responses"
    headers = {"Content-Type": "application/json"}
    if settings.openai_api_key:
        headers["Authorization"] = f"Bearer {settings.openai_api_key}"

    try:
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(endpoint, headers=headers, content=json.dumps(payload))
            response.raise_for_status()
            response_payload = response.json()
    except httpx.HTTPError:
        return None, "llm_error", settings.openai_model

    answer = _extract_output_text(response_payload)
    if not answer:
        return None, "llm_error", settings.openai_model

    return answer, "generated", settings.openai_model
