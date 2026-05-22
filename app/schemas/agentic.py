from pydantic import BaseModel, Field
from typing import Literal

from app.schemas.ingestion import ChunkMetadata


class AgenticQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    months: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    use_vector: bool = True
    include_images: bool = True
    vector_top_k: int = Field(default=8, ge=1, le=40)
    use_llm: bool = True
    response_mode: Literal["compact", "balanced", "full"] = "balanced"
    response_format: Literal["auto", "points", "table"] = "auto"
    max_snippet_chars: int = Field(default=1200, ge=120, le=4000)
    require_citations: bool = True
    max_corrections: int = Field(default=2, ge=0, le=3)
    compare_models: bool = False
    primary_model: str | None = None
    secondary_model: str | None = None
    model_provider: Literal["auto", "bedrock", "openai"] = "auto"
    primary_provider: Literal["auto", "bedrock", "openai"] = "auto"
    secondary_provider: Literal["auto", "bedrock", "openai"] = "auto"


class GuardrailResult(BaseModel):
    allowed: bool = True
    sanitized_query: str
    risk_flags: list[str] = Field(default_factory=list)
    action: Literal["allow", "block", "sanitize"] = "allow"


class AgentPlan(BaseModel):
    intent: Literal["fact_lookup", "comparison", "multi_hop", "action_request", "unknown"] = "unknown"
    sub_queries: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    score: float
    snippet: str
    metadata: ChunkMetadata
    modality: Literal["text", "image"] = "text"
    image_path: str | None = None
    image_name: str | None = None
    citation: str


class RetrievalGuardrailReport(BaseModel):
    allowed: bool = True
    blocked_sources: list[str] = Field(default_factory=list)
    stale_warnings: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)


class CriticReport(BaseModel):
    passed: bool
    faithfulness_score: float = 0.0
    consistency_score: float = 0.0
    issues: list[str] = Field(default_factory=list)
    recommendation: Literal["accept", "retry", "abstain"] = "accept"


class AgentStep(BaseModel):
    step: str
    status: Literal["ok", "warn", "blocked", "error"]
    detail: str


class ResponseSource(BaseModel):
    document: str
    pages: str


class ResponseMetadata(BaseModel):
    model: str | None = None
    retrieval_method: str = "unknown"
    grounded: bool = False


class ModelOutput(BaseModel):
    provider: str
    model: str | None = None
    status: str
    answer: str


class ImageReference(BaseModel):
    document: str
    page: int
    image_name: str | None = None
    image_path: str | None = None
    caption: str | None = None
    citation: str | None = None


class AgenticQueryResponse(BaseModel):
    query: str
    status: Literal["completed", "blocked", "abstained", "failed"]
    final_answer: str | None = None
    sources: list[ResponseSource] = Field(default_factory=list)
    image_references: list[ImageReference] = Field(default_factory=list)
    metadata: ResponseMetadata | None = None
    citations: list[str] = Field(default_factory=list)
    planner: AgentPlan | None = None
    input_guardrails: GuardrailResult | None = None
    retrieval_guardrails: RetrievalGuardrailReport | None = None
    critic: CriticReport | None = None
    evidence: list[EvidenceItem] | None = None
    steps: list[AgentStep] | None = None
    answer_model: str | None = None
    model_outputs: list[ModelOutput] | None = None
    vector_status: Literal["used", "disabled", "unavailable", "error"] = "disabled"
