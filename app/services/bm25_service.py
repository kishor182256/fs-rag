from collections import Counter
from dataclasses import dataclass
from math import log
import re

from app.schemas.ingestion import QueryRequest
from app.services.retrieval_types import RetrievalCandidate

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


@dataclass
class _ChunkRecord:
    doc_id: str
    source_file: str
    chunk_id: str
    page_start: int
    page_end: int
    text: str
    metadata: dict


def _tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if len(token) > 1]


def _query_terms(text: str) -> list[str]:
    tokens = _tokenize(text)
    keywords = [token for token in tokens if token not in STOPWORDS]
    return keywords or tokens


def _passes_filters(metadata: dict, request: QueryRequest) -> bool:
    chunk_months = {m.lower() for m in metadata.get("months", [])}
    chunk_topics = {t.lower() for t in metadata.get("topics", [])}

    if request.months:
        requested_months = {m.lower() for m in request.months}
        if not (requested_months & chunk_months):
            return False

    if request.topics:
        requested_topics = {t.lower() for t in request.topics}
        if not (requested_topics & chunk_topics):
            return False

    return True


def _idf(n_docs: int, df: int) -> float:
    return log(1 + (n_docs - df + 0.5) / (df + 0.5))


def search_bm25(
    *,
    chunks: list[dict],
    request: QueryRequest,
    bm25_k1: float,
    bm25_b: float,
    limit: int,
) -> list[RetrievalCandidate]:
    terms = _query_terms(request.query)
    if not terms:
        return []

    records: list[_ChunkRecord] = []
    tf_maps: list[Counter] = []
    doc_lengths: list[int] = []
    df_counter: Counter = Counter()

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if not _passes_filters(metadata, request):
            continue

        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        tokens = _tokenize(text)
        if not tokens:
            continue

        tf = Counter(tokens)
        present_terms = set()
        for term in terms:
            if tf[term] > 0:
                present_terms.add(term)

        for term in present_terms:
            df_counter[term] += 1

        records.append(
            _ChunkRecord(
                doc_id=str(chunk.get("doc_id", "")),
                source_file=str(chunk.get("source_file", "")),
                chunk_id=str(chunk.get("chunk_id", "")),
                page_start=int(chunk.get("page_start", 0) or 0),
                page_end=int(chunk.get("page_end", 0) or 0),
                text=text,
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )
        tf_maps.append(tf)
        doc_lengths.append(len(tokens))

    n_docs = len(records)
    if n_docs == 0:
        return []

    avgdl = sum(doc_lengths) / max(1, n_docs)
    candidates: list[RetrievalCandidate] = []

    for record, tf_map, doc_len in zip(records, tf_maps, doc_lengths, strict=True):
        matched_terms: list[str] = []
        score = 0.0

        for term in terms:
            freq = tf_map.get(term, 0)
            if freq <= 0:
                continue

            matched_terms.append(term)
            idf = _idf(n_docs=n_docs, df=df_counter.get(term, 0))
            denom = freq + bm25_k1 * (1 - bm25_b + bm25_b * (doc_len / max(1e-6, avgdl)))
            score += idf * ((freq * (bm25_k1 + 1)) / denom)

        if score <= 0:
            continue

        candidates.append(
            RetrievalCandidate(
                doc_id=record.doc_id,
                source_file=record.source_file,
                chunk_id=record.chunk_id,
                page_start=record.page_start,
                page_end=record.page_end,
                text=record.text,
                metadata=record.metadata,
                matched_terms=sorted(set(matched_terms)),
                bm25_score=round(score, 6),
            )
        )

    candidates.sort(key=lambda item: item.bm25_score, reverse=True)
    return candidates[:limit]
