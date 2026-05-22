import json
import logging
import threading
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import settings
from app.services.vector_store_service import qdrant_health
from app.workers.async_ingestion_worker import AsyncIngestionWorker

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="PDF ingestion pipeline starter for exam-style documents.",
)
app.include_router(api_router)

logger = logging.getLogger("uvicorn.error")
_embedded_worker_thread: threading.Thread | None = None


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    try:
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": latency_ms,
                    "client_ip": request.client.host if request.client else "",
                },
                ensure_ascii=True,
            )
        )
        response.headers["X-Request-Id"] = request_id
        return response
    except Exception:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception(
            json.dumps(
                {
                    "event": "http_unhandled_exception",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": latency_ms,
                    "client_ip": request.client.host if request.client else "",
                },
                ensure_ascii=True,
            )
        )
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(
        json.dumps(
            {
                "event": "global_exception_handler",
                "method": request.method,
                "path": request.url.path,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            ensure_ascii=True,
        )
    )
    return JSONResponse(status_code=500, content={"message": "Internal server error"})


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

    qdrant_host = str(getattr(settings, "qdrant_host", "") or "").strip().lower()
    if qdrant_host in {"localhost", "127.0.0.1"}:
        logger.warning(
            "QDRANT_HOST is set to '%s'. In ECS/Fargate this usually breaks semantic retrieval. "
            "Set QDRANT_HOST to your Qdrant Cloud host and QDRANT_HTTPS=true.",
            qdrant_host,
        )

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
