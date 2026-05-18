from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    processed_dir: Path = Path("data/processed")
    openai_api_key: str | None = None
    openai_base_url: str = "http://localhost:11434/v1"
    openai_model: str = "qwen2.5:7b"
    embedding_model: str = "nomic-embed-text"
    llm_timeout_seconds: int = 30
    embedding_timeout_seconds: int = 30

    enable_vector_indexing: bool = True
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
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
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.processed_dir.mkdir(parents=True, exist_ok=True)
