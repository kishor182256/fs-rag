from pydantic import BaseModel, Field
from typing import Literal


class ChunkMetadata(BaseModel):
    months: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)


class IngestedChunk(BaseModel):
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    token_estimate: int = Field(..., description="Simple word-count estimate.")
    metadata: ChunkMetadata


class IngestionResponse(BaseModel):
    doc_id: str
    source_file: str
    file_size_bytes: int
    pages_processed: int
    chunks_created: int
    ingestion_pipeline: Literal["legacy", "hf_enriched", "hf_fallback_legacy"] = "legacy"
    manifest_path: str
    message: str
    months_detected: list[str] = Field(default_factory=list)
    topics_detected: list[str] = Field(default_factory=list)
    vector_index_summary: dict = Field(default_factory=dict)


class IngestionJobAcceptedResponse(BaseModel):
    status: Literal["accepted"] = "accepted"
    message: str
    job_id: str
    doc_id: str
    source_file: str
    ingestion_pipeline: Literal["legacy", "hf_enriched", "hf_fallback_legacy"] = "legacy"
    queue: str
    s3_bucket: str
    s3_key: str


class IngestionJobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed", "unknown"]
    doc_id: str | None = None
    source_file: str | None = None
    pipeline: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    message_id: str | None = None
    error: str | None = None
    result: dict | None = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=2, ge=1, le=20)
    months: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    use_vector: bool = True
    vector_top_k: int = Field(default=8, ge=1, le=40)
    use_llm: bool = True
    include_images: bool = False
    include_full_text: bool = False
    response_mode: Literal["compact", "balanced", "full"] = "balanced"
    max_snippet_chars: int = Field(default=1200, ge=120, le=4000)


class QueryHit(BaseModel):
    doc_id: str
    source_file: str
    chunk_id: str
    page_start: int
    page_end: int
    score: float
    snippet: str
    matched_terms: list[str] = Field(default_factory=list)
    metadata: ChunkMetadata
    modality: Literal["text", "image"] = "text"
    image_path: str | None = None
    image_name: str | None = None
    text: str | None = None


class QueryResponse(BaseModel):
    query: str
    hits: list[QueryHit]
    answer: str | None = None
    answer_status: Literal["generated", "llm_unavailable", "llm_error", "no_hits", "disabled"] = "disabled"
    answer_model: str | None = None
    vector_status: Literal["used", "disabled", "unavailable", "error"] = "disabled"
