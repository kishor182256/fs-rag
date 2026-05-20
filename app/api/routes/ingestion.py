from typing import Literal

from fastapi import APIRouter, File, Query, UploadFile

from app.schemas.agentic import AgenticQueryRequest, AgenticQueryResponse
from app.schemas.ingestion import IngestionResponse, QueryRequest, QueryResponse
from app.services.agentic_orchestrator_service import run_agentic_query
from app.services.ingestion_service import ingest_pdf
from app.services.query_service import run_query_pipeline

router = APIRouter()


@router.post("/ingest/pdf", response_model=IngestionResponse)
async def ingest_pdf_endpoint(
    file: UploadFile = File(...),
    pipeline: Literal["auto", "legacy", "hf"] = Query(default="auto"),
) -> IngestionResponse:
    return await ingest_pdf(file, pipeline=pipeline)


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
async def query_chunks(payload: QueryRequest) -> QueryResponse:
    return await run_query_pipeline(payload)


@router.post("/agentic/query", response_model=AgenticQueryResponse, response_model_exclude_none=True)
async def agentic_query(payload: AgenticQueryRequest) -> AgenticQueryResponse:
    return await run_agentic_query(payload)
