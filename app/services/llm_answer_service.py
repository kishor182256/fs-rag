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
    return _build_context_with_filter(hits=hits, relevant_image_chunk_ids=None)


def _build_context_with_filter(hits: list[QueryHit], relevant_image_chunk_ids: set[str] | None) -> str:
    context_blocks: list[str] = []
    max_scan = max(settings.llm_max_context_hits * 4, settings.llm_max_context_hits)
    serial = 1

    for hit in hits[:max_scan]:
        if hit.modality == "image" and relevant_image_chunk_ids is not None and hit.chunk_id not in relevant_image_chunk_ids:
            continue

        content = hit.text or hit.snippet
        context_blocks.append(
            "\n".join(
                [
                    f"[HIT {serial}]",
                    f"modality: {hit.modality}",
                    f"chunk_id: {hit.chunk_id}",
                    f"source_file: {hit.source_file}",
                    f"pages: {hit.page_start}-{hit.page_end}",
                    f"score: {hit.score}",
                    f"content: {content}",
                ]
            )
        )

        serial += 1
        if len(context_blocks) >= settings.llm_max_context_hits:
            break

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


def _query_keywords(query: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9][a-z0-9\-']*", query.lower()) if len(token) > 2]


def _diagram_requested(query: str) -> bool:
    lowered = query.lower()
    return any(token in lowered for token in ["diagram", "figure", "chart", "graph", "image", "visual"])


def _text_terms(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9][a-z0-9\-']*", value.lower()) if len(token) > 2}


def _image_relevance_score(query_terms: set[str], image_text: str) -> float:
    if not query_terms:
        return 0.0
    image_terms = _text_terms(image_text)
    if not image_terms:
        return 0.0
    overlap = query_terms.intersection(image_terms)
    return len(overlap) / max(1, len(query_terms))


def _relevant_image_hits(query: str, hits: list[QueryHit]) -> list[QueryHit]:
    query_terms = set(_query_keywords(query))
    if not query_terms:
        return []

    relevant: list[QueryHit] = []
    for hit in hits:
        if hit.modality != "image":
            continue

        image_text = " ".join(
            part
            for part in [
                hit.snippet or "",
                hit.text or "",
                " ".join(hit.metadata.topics or []),
                " ".join(hit.metadata.entities or []),
            ]
            if part
        )

        score = _image_relevance_score(query_terms, image_text)
        min_threshold = 0.12 if len(query_terms) >= 6 else 0.08
        if score >= min_threshold:
            relevant.append(hit)

    return relevant


def _diagram_instruction(query: str, relevant_images: list[QueryHit]) -> str:
    if not _diagram_requested(query):
        return ""
    if relevant_images:
        return (
            "Add a final section titled 'Diagram'. "
            "Give up to 2 relevant diagram/image references with one short relevance note each. "
            "Use exact citation format (image_XXXXX pX-Y)."
        )
    return "If no image evidence exists in context, explicitly state: 'Insufficient local evidence for a diagram.'"


def _suppress_false_diagram_fallback(answer: str, has_relevant_images: bool) -> str:
    if not has_relevant_images:
        return answer
    lines = [line for line in answer.splitlines() if "insufficient local evidence for a diagram" not in line.lower()]
    return "\n".join(lines).strip()


def _dedupe_inline_citations(answer: str) -> str:
    citation_pattern = re.compile(r"\((?:chunk|image)_[A-Za-z0-9_-]+\s+p\d+(?:-\d+)?\)")

    seen: set[str] = set()

    def _replace(match: re.Match[str]) -> str:
        marker = match.group(0)
        if marker in seen:
            return ""
        seen.add(marker)
        return marker

    text = citation_pattern.sub(_replace, answer)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\(\s+\)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def synthesize_answer(query: str, hits: list[QueryHit]) -> tuple[str | None, str, str | None]:
    if not hits:
        return "No relevant information found in retrieved context.", "no_hits", None

    diagram_requested = _diagram_requested(query)
    relevant_images = _relevant_image_hits(query=query, hits=hits) if diagram_requested else []
    relevant_image_ids = {hit.chunk_id for hit in relevant_images}

    length_hint = _length_hint(query)
    style_hint = _response_style_hint(query)
    diagram_hint = _diagram_instruction(query, relevant_images)
    instructions = (
        "You are a strict production RAG answer engine. "
        "Answer only from provided context. "
        "Never add external facts, assumptions, or guessed details. "
        "If evidence is weak or conflicting, state 'Insufficient local evidence.' "
        "Return only what the user asked and avoid unrelated details. "
        "Use plain text output and avoid markdown headings. "
        "For list/highlights queries, return a clean bullet list. "
        "Cite every bullet or sentence with exact IDs like (chunk_00004 p5-8) or (image_00018 p5-5). "
        "Do not output placeholders like chunk_id. "
        "Do not include citations that are not used in the answer. "
        f"{style_hint} "
        f"{diagram_hint}"
    )

    length_line = f"{length_hint}\n\n" if length_hint else ""
    user_prompt = (
        f"User query: {query}\n\n"
        f"{length_line}"
        "Retrieved context follows.\n"
        "Use it to produce the final answer.\n\n"
        f"{_build_context_with_filter(hits=hits, relevant_image_chunk_ids=(relevant_image_ids if diagram_requested else None))}"
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

    answer = _suppress_false_diagram_fallback(answer=answer, has_relevant_images=bool(relevant_images))
    answer = _dedupe_inline_citations(answer)

    return answer, "generated", settings.openai_model
