from fastapi import FastAPI
import logging

from app.api.router import api_router
from app.core.config import settings
from app.services.vector_store_service import qdrant_health

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="PDF ingestion pipeline starter for exam-style documents.",
)
app.include_router(api_router)

logger = logging.getLogger("uvicorn.error")


@app.on_event("startup")
async def startup_event() -> None:
    key_present = bool((settings.openai_api_key or "").strip())
    logger.info(
        "OpenAI config: base_url=%s model=%s embedding_model=%s api_key_present=%s",
        settings.openai_base_url,
        settings.openai_model,
        settings.embedding_model,
        key_present,
    )
    if not key_present:
        logger.warning("OPENAI_API_KEY is missing. LLM/embedding calls will fail.")

    qdrant_state, qdrant_detail = qdrant_health()
    if qdrant_state == "up":
        logger.info("Qdrant Vector DB status: UP (%s)", qdrant_detail)
    elif qdrant_state == "disabled":
        logger.warning("Qdrant Vector DB status: DISABLED (%s)", qdrant_detail)
    else:
        logger.error("Qdrant Vector DB status: DOWN (%s)", qdrant_detail)
