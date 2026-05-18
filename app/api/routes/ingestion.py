from fastapi import APIRouter, File, UploadFile

from app.schemas.ingestion import IngestionResponse, QueryRequest, QueryResponse
from app.services.ingestion_service import ingest_pdf
from app.services.query_service import run_query_pipeline

router = APIRouter()


@router.post("/ingest/pdf", response_model=IngestionResponse)
async def ingest_pdf_endpoint(file: UploadFile = File(...)) -> IngestionResponse:
    return await ingest_pdf(file)


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
async def query_chunks(payload: QueryRequest) -> QueryResponse:
    return await run_query_pipeline(payload)
