from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "PDF Ingestion API"
    app_env: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8000

    max_upload_size_mb: int = 25
    chunk_size: int = 700
    chunk_overlap: int = 100
    min_chunk_chars: int = 120
    min_page_text_chars: int = 80
    enable_ocr_fallback: bool = False

    upload_dir: Path = Path("data/uploads")
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_timeout_seconds: int = 30
    embedding_timeout_seconds: int = 30
    enable_multimodal_ingest: bool = False
    multimodal_qdrant_collection: str = "rag_chunks_mm"
    multimodal_clip_qdrant_collection: str = "rag_chunks_mm_clip"
    multimodal_max_images_per_doc: int = 80
    multimodal_min_image_bytes: int = 1024
    enable_clip_image_vectors: bool = False
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    enable_vlm_captions: bool = False
    vlm_caption_model: str = "gpt-4o-mini"
    vlm_caption_max_tokens: int = 80

    enable_vector_indexing: bool = True
    qdrant_single_source: bool = True
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_https: bool = False
    qdrant_api_key: str | None = None
    qdrant_collection: str = "rag_chunks"
    vector_distance: str = "cosine"
    vector_query_limit: int = 8
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    search_min_score: float = 0.01
    search_relative_score_ratio: float = 0.6
    search_min_keyword_coverage: float = 0.35
    llm_max_context_hits: int = 3

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


settings = Settings()


def _resolve_storage_path(path_value: Path) -> Path:
    return path_value if path_value.is_absolute() else (PROJECT_ROOT / path_value).resolve()


settings.upload_dir = _resolve_storage_path(settings.upload_dir)
settings.upload_dir.mkdir(parents=True, exist_ok=True)
