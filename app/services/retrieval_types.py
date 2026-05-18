from dataclasses import dataclass


@dataclass
class RetrievalCandidate:
    doc_id: str
    source_file: str
    chunk_id: str
    page_start: int
    page_end: int
    text: str
    metadata: dict
    matched_terms: list[str]
    modality: str = "text"
    image_path: str = ""
    image_name: str = ""
    snippet: str = ""
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float = 0.0
