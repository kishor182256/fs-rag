from fastapi import APIRouter
from app.core.config import settings
from app.services.vector_store_service import qdrant_health

router = APIRouter()


@router.get("/health")
def healthcheck() -> dict:
    vector_state, vector_detail = qdrant_health()
    return {
        "status": "ok",
        "service": "ingesto",
        "llm_provider": str(getattr(settings, "llm_provider", "openai")),
        "embedding_provider": str(getattr(settings, "embedding_provider", "openai")),
        "qdrant": {
            "state": vector_state,
            "detail": vector_detail,
            "host": settings.qdrant_host,
            "port": settings.qdrant_port,
            "https": bool(settings.qdrant_https),
        },
    }
