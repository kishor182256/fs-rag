import json
import re

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


def _length_hint(query: str) -> str:
    range_match = re.search(r"\b(\d{1,2})\s*[\-\u2013]\s*(\d{1,2})\s*lines?\b", query.lower())
    if range_match:
        lo = int(range_match.group(1))
        hi = int(range_match.group(2))
        if 1 <= lo <= hi <= 60:
            return f"Length constraint: keep the answer between {lo} and {hi} lines."

    fixed_match = re.search(r"\b(\d{1,2})\s*lines?\b", query.lower())
    if fixed_match:
        value = int(fixed_match.group(1))
        if 1 <= value <= 60:
            return f"Length constraint: keep the answer around {value} lines."

    return ""


def _response_style_hint(query: str) -> str:
    lowered = query.lower()
    list_only_markers = ["give list", "list of", "table of", "winners list"]
    explanatory_markers = [
        "explain",
        "describe",
        "overview",
        "how",
        "why",
        "analysis",
        "impact",
        "key highlights",
        "highlights of",
    ]
    is_list_only = any(marker in lowered for marker in list_only_markers) and not any(
        marker in lowered for marker in ["explain", "describe", "overview", "analysis"]
    )
    is_explanatory = any(marker in lowered for marker in explanatory_markers)

    if is_list_only:
        return (
            "Output format: return only the requested list, with no intro or conclusion. "
            "Keep each bullet factual and concise."
        )

    if is_explanatory:
        return (
            "Output format: provide three parts in plain text labels: Basic Info, Detailed Insights, Conclusion. "
            "Basic Info should give 2-4 concise bullets. "
            "Detailed Insights should give 3-6 richer points from evidence. "
            "Conclusion should summarize significance in 2-3 sentences."
        )

    return "Output format: start with a short direct answer, then supporting points, then a brief conclusion."


async def synthesize_answer(query: str, hits: list[QueryHit]) -> tuple[str | None, str, str | None]:
    if not hits:
        return "No relevant information found in retrieved context.", "no_hits", None

    length_hint = _length_hint(query)
    style_hint = _response_style_hint(query)
    instructions = (
        "You are a strict production RAG answer engine. "
        "Answer only from provided context. "
        "Never add external facts, assumptions, or guessed details. "
        "If evidence is weak or conflicting, state 'Insufficient local evidence.' "
        "Return only what the user asked and avoid unrelated details. "
        "Use plain text output and avoid markdown headings. "
        "For list/highlights queries, return a clean bullet list. "
        "Cite every bullet or sentence with (chunk_id pX-Y). "
        "Do not include citations that are not used in the answer. "
        f"{style_hint}"
    )

    length_line = f"{length_hint}\n\n" if length_hint else ""
    user_prompt = (
        f"User query: {query}\n\n"
        f"{length_line}"
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
