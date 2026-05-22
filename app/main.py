from fastapi import FastAPI
import logging
import threading

from app.api.router import api_router
from app.core.config import settings
from app.services.vector_store_service import qdrant_health
from app.workers.async_ingestion_worker import AsyncIngestionWorker

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="PDF ingestion pipeline starter for exam-style documents.",
)
app.include_router(api_router)

logger = logging.getLogger("uvicorn.error")
_embedded_worker_thread: threading.Thread | None = None


def _run_embedded_worker() -> None:
    try:
        worker = AsyncIngestionWorker(
            poll_wait_seconds=int(settings.async_worker_poll_wait_seconds),
            visibility_timeout_seconds=int(settings.async_worker_visibility_timeout_seconds),
            max_messages=int(settings.async_worker_max_messages),
            max_attempts=int(settings.async_worker_max_attempts),
        )
        worker.run_forever()
    except Exception as exc:
        logger.error("Embedded async ingestion worker failed to start: %s", exc)


@app.on_event("startup")
async def startup_event() -> None:
    llm_provider = str(getattr(settings, "llm_provider", "openai") or "openai").strip().lower()
    embedding_provider = str(getattr(settings, "embedding_provider", "openai") or "openai").strip().lower()
    openai_required = llm_provider == "openai" or embedding_provider == "openai"
    key_present = bool((settings.openai_api_key or "").strip())
    logger.info(
        "Provider config: llm_provider=%s embedding_provider=%s openai_base_url=%s openai_model=%s embedding_model=%s openai_api_key_present=%s",
        llm_provider,
        embedding_provider,
        settings.openai_base_url,
        settings.openai_model,
        settings.embedding_model,
        key_present,
    )
    if openai_required and not key_present:
        logger.warning("OPENAI_API_KEY is missing but provider requires OpenAI.")

    qdrant_state, qdrant_detail = qdrant_health()
    if qdrant_state == "up":
        logger.info("Qdrant Vector DB status: UP (%s)", qdrant_detail)
    elif qdrant_state == "disabled":
        logger.warning("Qdrant Vector DB status: DISABLED (%s)", qdrant_detail)
    else:
        logger.error("Qdrant Vector DB status: DOWN (%s)", qdrant_detail)

    global _embedded_worker_thread
    if settings.enable_async_ingestion and settings.auto_start_async_worker and _embedded_worker_thread is None:
        _embedded_worker_thread = threading.Thread(
            target=_run_embedded_worker,
            name="async-ingestion-worker",
            daemon=True,
        )
        _embedded_worker_thread.start()
        logger.info("Embedded async ingestion worker started in API process.")
