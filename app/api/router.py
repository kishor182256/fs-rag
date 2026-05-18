from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.ingestion import router as ingestion_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(ingestion_router, prefix="/v1", tags=["ingestion"])
