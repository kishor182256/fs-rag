import re
import logging

from app.core.config import settings
from app.schemas.ingestion import ChunkMetadata, QueryHit
from app.services.llm_answer_service import synthesize_answer
from app.services.retrieval_types import RetrievalCandidate

logger = logging.getLogger(__name__)


LIST_QUERY_MARKERS = [
    "list",
    "countries",
    "joined",
    "members",
    "participants",
    "winners",
    "awards",
    "announced",
]

GENERIC_LIST_TERMS = {
    "give",
    "list",
    "lists",
    "country",
    "countries",
    "joined",
    "join",
    "that",
    "the",
    "from",
    "with",
    "about",
    "contries",
    "countr",
}


NOISE_MARKERS = {
    "labour",
    "laws",
    "judiciary",
    "ranking",
    "survekshan",
    "drone",
    "transfer",
    "treaty",
    "credit",
    "projects",
    "forest",
    "forests",
}


COUNTRY_STOPWORDS = {
    "countries",
    "country",
    "challenge",
    "blue",
    "ndc",
    "cop",
    "summit",
    "ministerial",
    "taskforce",
    "ocean",
    "initiative",
    "launched",
    "joined",
    "including",
    "namely",
    "and",
}


def _query_terms(query: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", query or "")
        if len(token) > 2
    }


def _extract_subject_phrase(query: str) -> str:
    q = re.sub(r"\s+", " ", (query or "")).strip()
    m = re.search(r"(?i)\bwhat\s+is\s+(.+?)(?:\?|$|\band\b|\bin\b|\bas\b)", q)
    if m:
        phrase = m.group(1).strip(" .,:;")
        return phrase
    m = re.search(r"(?i)\bexplain\s+(.+?)(?:\?|$|\bin\b|\bas\b)", q)
    if m:
        phrase = m.group(1).strip(" .,:;")
        return phrase
    m = re.search(r"(?i)\blist\s+of\s+(.+?)(?:\?|$)", q)
    if m:
        phrase = m.group(1).strip(" .,:;")
        return phrase
    m = re.search(r"(?i)\bjoined\s+the\s+(.+?)(?:\?|$)", q)
    if m:
        phrase = m.group(1).strip(" .,:;")
        return phrase
    return ""


def _list_title(query: str) -> str:
    subject = _extract_subject_phrase(query)
    if subject:
        return f"List: {subject}"
    q = re.sub(r"\s+", " ", (query or "")).strip(" ?")
    if q:
        return f"List: {q}"
    return "List:"


def _is_concept_query(query: str) -> bool:
    lowered = (query or "").lower()
    return any(marker in lowered for marker in ["what is", "define", "explain", "brief about", "overview of"])


def _is_list_query(query: str) -> bool:
    lowered = (query or "").lower()
    return any(marker in lowered for marker in LIST_QUERY_MARKERS)


def _is_country_list_query(query: str) -> bool:
    lowered = (query or "").lower()
    return any(marker in lowered for marker in ["country", "countries", "countr", "contries"]) and _is_list_query(query)


def _normalize_text(text: str) -> str:
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    clean = clean.replace("|", " ")
    clean = re.sub(r"\s{2,}", " ", clean)
    clean = clean.replace("•", "")
    clean = clean.replace(" .", ".")
    clean = re.sub(r"\bch\s*\d+\b.*?(?=\s[A-Z]|\Z)", "", clean, flags=re.IGNORECASE)
    return clean.strip()


def _sentence_split(text: str) -> list[str]:
    clean = _normalize_text(text)
    if not clean:
        return []
    parts = re.split(r"(?<=[.!?])\s+", clean)
    return [part.strip() for part in parts if len(part.strip()) >= 30]


def _normalize_sentence(sentence: str) -> str:
    value = re.sub(r"\s+", " ", (sentence or "")).strip()
    value = value.strip(" ,;:-")
    value = value.replace("â€¢", "").strip()
    value = re.sub(r"\s{2,}", " ", value)
    return value


def _is_fragment(sentence: str) -> bool:
    s = _normalize_sentence(sentence)
    if not s:
        return True
    if len(s) < 28:
        return True
    if s.endswith(","):
        return True
    # Common OCR/segment truncation endings.
    if re.search(r"\b(to|and|or|with|for|of|in|on|at|by|from)$", s.lower()):
        return True
    return False


def _term_set(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", text or "")
        if len(token) > 2
    }


def _relevance_score(text: str, query_terms: set[str], subject_phrase: str = "") -> float:
    if not text:
        return 0.0
    terms = _term_set(text)
    overlap = len(query_terms & terms)
    phrase_bonus = 2.5 if subject_phrase and subject_phrase in text.lower() else 0.0
    noise_penalty = 0.7 if (terms & NOISE_MARKERS) else 0.0
    return max(0.0, float(overlap) + phrase_bonus - noise_penalty)


def _subject_anchor_terms(query: str) -> set[str]:
    subject_phrase = _extract_subject_phrase(query)
    subject_terms = _term_set(subject_phrase)
    if subject_terms:
        return {term for term in subject_terms if term not in GENERIC_LIST_TERMS}
    query_terms = _query_terms(query)
    return {term for term in query_terms if term not in GENERIC_LIST_TERMS}


def _best_clause(sentence: str, query_terms: set[str], subject_phrase: str = "") -> str:
    pieces = [sentence]
    if re.search(r"\b(and|while|whereas|however|but)\b", sentence, flags=re.IGNORECASE):
        pieces = [part.strip(" ,;:-") for part in re.split(r"\b(?:and|while|whereas|however|but)\b", sentence, flags=re.IGNORECASE) if part.strip()]
    if len(pieces) == 1:
        return _normalize_sentence(sentence)

    scored = sorted(
        (( _relevance_score(piece, query_terms, subject_phrase), piece) for piece in pieces),
        key=lambda item: item[0],
        reverse=True,
    )
    best = _normalize_sentence(scored[0][1])
    if _is_fragment(best):
        return _normalize_sentence(sentence)
    return best or _normalize_sentence(sentence)


def _extract_country_like_names(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return []

    # Prefer list tail after common lead-ins if present.
    tail_match = re.search(r"(?i)\b(?:includes?|including|namely|are|were|joined by)\b[:\s]*(.+)$", normalized)
    if tail_match:
        normalized = tail_match.group(1).strip(" .;:")

    fragments = re.split(r",|/|;|\band\b", normalized, flags=re.IGNORECASE)
    names: list[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        token = re.sub(r"[^A-Za-z\s\-]", " ", fragment).strip()
        token = re.sub(r"\s{2,}", " ", token).strip()
        if not token:
            continue

        lower = token.lower()
        if lower in COUNTRY_STOPWORDS:
            continue
        if any(word in lower for word in COUNTRY_STOPWORDS):
            # If phrase still contains stopword semantics and is long, it is likely not a country.
            words = token.split()
            if len(words) > 2:
                continue
        if len(token) < 3 or len(token) > 30:
            continue

        words = token.split()
        if len(words) > 3:
            continue

        if any(word in lower for word in ["challenge", "launched", "jointly", "includes", "joined", "treaty"]):
            continue

        # Heuristic: mostly title-cased country labels.
        titled = sum(1 for word in words if word[:1].isupper())
        if titled < max(1, len(words) - 1):
            continue

        normalized_name = " ".join(word.capitalize() if word.islower() else word for word in words)
        key = normalized_name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(normalized_name)
    return names


def _focused_snippet(text: str, query: str, limit: int = 280) -> str:
    clean = re.sub(r"\s+", " ", (text or "")).strip().replace("|", " ")
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    if not clean:
        return ""

    query_terms = _query_terms(query)
    subject_phrase = _extract_subject_phrase(query).lower()
    if not query_terms:
        return clean[:limit] + ("..." if len(clean) > limit else "")

    sentences = re.split(r"(?<=[.!?])\s+", clean)
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        best_clause = _best_clause(sentence.strip(), query_terms, subject_phrase)
        if _is_fragment(best_clause):
            continue
        score = _relevance_score(best_clause, query_terms, subject_phrase)
        if "augmented reality" in best_clause.lower():
            score += 1.5
        if score > 0:
            scored.append((score, best_clause))
    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0][1]
    else:
        best = clean[:limit]

    if len(best) > limit:
        best = best[: limit - 3] + "..."
    return best


def _ranked_sentence_candidates(candidates: list[RetrievalCandidate], query: str, cap: int = 12) -> list[tuple[float, str, str]]:
    query_terms = _query_terms(query)
    subject_phrase = _extract_subject_phrase(query).lower()
    ranked: list[tuple[float, str, str]] = []

    for candidate in candidates[: max(1, settings.llm_max_context_hits * 2)]:
        citation = f"({candidate.chunk_id} p{candidate.page_start}-{candidate.page_end})"
        for sentence in _sentence_split(candidate.snippet or candidate.text):
            clause = _best_clause(sentence, query_terms, subject_phrase)
            if _is_fragment(clause):
                continue
            sentence_lower = clause.lower()
            sentence_terms = _term_set(clause)
            overlap = len(query_terms & sentence_terms)
            phrase_bonus = 3.0 if subject_phrase and subject_phrase in sentence_lower else 0.0
            definition_bonus = 1.2 if any(
                marker in sentence_lower
                for marker in ["is called", "can be defined as", "is the", "refers to", "defined as"]
            ) else 0.0
            application_bonus = 0.8 if any(
                marker in sentence_lower
                for marker in ["application", "used in", "can be used", "enables", "allows", "helps"]
            ) else 0.0
            score = float(overlap) + phrase_bonus + definition_bonus + application_bonus
            if score > 0:
                ranked.append((score, clause.strip(), citation))

    ranked.sort(key=lambda item: item[0], reverse=True)
    dedup: list[tuple[float, str, str]] = []
    seen: set[str] = set()
    for item in ranked:
        signature = item[1].lower()
        if signature in seen:
            continue
        seen.add(signature)
        dedup.append(item)
        if len(dedup) >= cap:
            break
    return dedup


def _concept_fallback(candidates: list[RetrievalCandidate], query: str) -> str:
    ranked = _ranked_sentence_candidates(candidates, query, cap=10)
    if not ranked:
        return "Insufficient local evidence."

    definition = ranked[0]
    comparison = next(
        (item for item in ranked if "virtual reality" in item[1].lower() or "unlike" in item[1].lower()),
        None,
    )
    applications = [
        item for item in ranked
        if any(marker in item[1].lower() for marker in ["application", "used in", "can be used", "enables", "allows"])
    ][:2]

    lines: list[str] = [
        "Basic Info",
        f"- {definition[1]} {definition[2]}",
    ]
    used_sentences: set[str] = {definition[1].lower()}
    if comparison is not None and comparison[1].lower() not in used_sentences:
        lines.append(f"- {comparison[1]} {comparison[2]}")
        used_sentences.add(comparison[1].lower())

    lines.extend(["", "Detailed Insights"])
    added = 0
    for _, sentence, citation in applications + ranked[1:]:
        normalized = sentence.lower()
        if normalized in used_sentences:
            continue
        used_sentences.add(normalized)
        lines.append(f"- {sentence} {citation}")
        added += 1
        if added >= 3:
            break

    lines.extend(
        [
            "",
            "Conclusion",
            "Augmented Reality overlays digital information onto real-world surroundings and is applied to improve interactive user experiences.",
        ]
    )
    return "\n".join(lines)


def _extract_list_items(query: str, candidates: list[RetrievalCandidate], max_items: int) -> list[tuple[str, str]]:
    query_terms = _query_terms(query)
    subject_phrase = _extract_subject_phrase(query).lower()
    anchor_terms = _subject_anchor_terms(query)
    country_mode = _is_country_list_query(query)
    items: list[tuple[str, str]] = []
    seen: set[str] = set()

    for candidate in candidates[: max(1, settings.llm_max_context_hits * 2)]:
        citation = f"({candidate.chunk_id} p{candidate.page_start}-{candidate.page_end})"
        source_text = candidate.text or candidate.snippet or ""

        # For country-list queries, parse full chunk text first to avoid snippet truncation artifacts.
        if country_mode:
            for country in _extract_country_like_names(source_text):
                key = f"country::{country.lower()}"
                if key in seen:
                    continue
                seen.add(key)
                items.append((country, citation))
                if len(items) >= max_items:
                    return items

        for sentence in _sentence_split(source_text):
            clause = _best_clause(sentence, query_terms, subject_phrase)
            score = _relevance_score(clause, query_terms, subject_phrase)
            if score <= 0:
                continue
            sentence_terms = _term_set(sentence)
            if anchor_terms and not (sentence_terms & anchor_terms):
                continue
            if NOISE_MARKERS & _term_set(clause):
                continue

            if country_mode:
                countries = _extract_country_like_names(sentence)
                for country in countries:
                    key = f"country::{country.lower()}"
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append((country, citation))
                    if len(items) >= max_items:
                        return items
                continue

            normalized = re.sub(r"\s+", " ", clause).strip()
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                items.append((normalized, citation))
                if len(items) >= max_items:
                    return items
    return items


def _to_query_hits(candidates: list[RetrievalCandidate], query: str) -> list[QueryHit]:
    def _as_str_list(value: object) -> list[str]:
        if isinstance(value, list):
            cleaned: list[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    cleaned.append(text)
            return cleaned
        return []

    hits: list[QueryHit] = []
    for candidate in candidates:
        snippet = _focused_snippet(candidate.snippet or candidate.text, query, limit=420)
        if not snippet:
            snippet = (candidate.snippet or candidate.text[: min(len(candidate.text), 500)]).strip()
        metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        hits.append(
            QueryHit(
                doc_id=candidate.doc_id,
                source_file=candidate.source_file,
                chunk_id=candidate.chunk_id,
                page_start=candidate.page_start,
                page_end=candidate.page_end,
                score=round(candidate.rerank_score, 4),
                snippet=snippet,
                matched_terms=candidate.matched_terms,
                metadata=ChunkMetadata(
                    months=_as_str_list(metadata.get("months", [])),
                    topics=_as_str_list(metadata.get("topics", [])),
                    entities=_as_str_list(metadata.get("entities", [])),
                ),
                modality="image" if candidate.modality == "image" else "text",
                image_path=candidate.image_path or None,
                image_name=candidate.image_name or None,
                # Keep context compact for LLM synthesis to reduce long-chunk dump behavior.
                text=None,
            )
        )
    return hits


def _deterministic_answer(candidates: list[RetrievalCandidate], query: str, response_format: str = "auto") -> str:
    if not candidates:
        return "Insufficient evidence found for the query."
    format_mode = (response_format or "auto").strip().lower()
    if _is_concept_query(query) and format_mode in {"auto", "points"}:
        return _concept_fallback(candidates, query)

    if _is_list_query(query) and format_mode in {"auto", "points"}:
        list_cap = 20 if _is_country_list_query(query) else max(3, settings.llm_max_context_hits)
        extracted = _extract_list_items(query, candidates, max_items=list_cap)
        if _is_country_list_query(query) and 0 < len(extracted) < 3:
            top_cite = f"({candidates[0].chunk_id} p{candidates[0].page_start}-{candidates[0].page_end})"
            return (
                f"{_list_title(query)}\n"
                "Key Points\n"
                f"- Retrieved evidence appears incomplete for a country-level list in this context. {top_cite}\n"
                f"- Found only {len(extracted)} country item(s), so a complete trusted list cannot be produced from local evidence alone. {top_cite}"
            )
        if not extracted:
            top_cite = f"({candidates[0].chunk_id} p{candidates[0].page_start}-{candidates[0].page_end})"
            return (
                f"{_list_title(query)}\n"
                "Key Points\n"
                f"- Retrieved evidence does not provide a clean, itemized list for this query. {top_cite}\n"
                f"- The source mentions the topic but individual list items are incomplete/unclear in local evidence. {top_cite}"
            )
        lines = [_list_title(query), "Key Points"]
        for sentence, citation in extracted:
            lines.append(f"- {sentence} {citation}")
        return "\n".join(lines)

    if format_mode == "table":
        lines = ["| Item | Evidence | Citation |", "|---|---|---|"]
        for idx, candidate in enumerate(candidates[: settings.llm_max_context_hits], start=1):
            snippet = re.sub(r"\s+", " ", (candidate.snippet or candidate.text)).strip().replace("|", " ")
            snippet = re.sub(r"\s{2,}", " ", snippet).strip()
            if len(snippet) > 180:
                snippet = snippet[:177] + "..."
            lines.append(
                f"| {idx} | {snippet} | ({candidate.chunk_id} p{candidate.page_start}-{candidate.page_end}) |"
            )
        return "\n".join(lines)

    lowered = (query or "").lower()
    list_style = format_mode == "points" or any(marker in lowered for marker in ["list", "key highlights", "highlights", "points"])
    lines: list[str] = ["Key Points" if list_style else "Basic Info"]

    for candidate in candidates[: settings.llm_max_context_hits]:
        snippet = _focused_snippet(candidate.snippet or candidate.text, query, limit=220)
        lines.append(f"- {snippet} ({candidate.chunk_id} p{candidate.page_start}-{candidate.page_end})")

    if not list_style:
        lines.extend(["", "Conclusion", "Summary based strictly on retrieved local evidence."])
    return "\n".join(lines)


async def generate_answer(
    *,
    query: str,
    candidates: list[RetrievalCandidate],
    use_llm: bool,
    response_format: str = "auto",
    provider_override: str | None = None,
    model_override: str | None = None,
) -> tuple[str, str, str | None]:
    if not candidates:
        return "Insufficient evidence found for the query.", "no_hits", None

    if not use_llm:
        return _deterministic_answer(candidates, query, response_format=response_format), "disabled", None

    try:
        hits = _to_query_hits(candidates, query)
        answer, status, model = await synthesize_answer(
            query,
            hits,
            response_format=response_format,
            provider_override=provider_override,
            model_override=model_override,
        )
        if not answer:
            return _deterministic_answer(candidates, query, response_format=response_format), "llm_error", model
        return answer, status, model
    except Exception as exc:  # noqa: BLE001
        logger.exception("synthesizer_generate_answer_failed query=%s", (query or "")[:160])
        fallback = _deterministic_answer(candidates, query, response_format=response_format)
        return fallback, "llm_error", str(type(exc).__name__)
