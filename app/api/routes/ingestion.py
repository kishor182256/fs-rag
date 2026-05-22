from typing import Literal

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.schemas.agentic import AgenticQueryRequest, AgenticQueryResponse
from app.schemas.ingestion import (
    IngestionJobAcceptedResponse,
    IngestionJobStatusResponse,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.agentic_orchestrator_service import run_agentic_query
from app.services.ingestion_service import get_ingestion_job_status, ingest_pdf, ingest_pdf_stream
from app.services.query_service import run_query_pipeline

router = APIRouter()


@router.post("/ingest/pdf", response_model=IngestionResponse | IngestionJobAcceptedResponse)
async def ingest_pdf_endpoint(
    file: UploadFile = File(...),
    pipeline: Literal["auto", "legacy", "hf"] = Query(default="auto"),
    async_mode: bool | None = Query(default=None),
    wait_for_completion: bool | None = Query(default=None),
    stream: bool = Query(default=False, description="When true, returns text/event-stream progress events."),
) -> IngestionResponse | IngestionJobAcceptedResponse | StreamingResponse:
    if stream:
        return await ingest_pdf_stream(
            file,
            pipeline=pipeline,
            async_mode=async_mode,
        )
    return await ingest_pdf(
        file,
        pipeline=pipeline,
        async_mode=async_mode,
        wait_for_completion=wait_for_completion,
    )


@router.get("/ingest/jobs/{job_id}", response_model=IngestionJobStatusResponse)
async def ingestion_job_status(job_id: str) -> IngestionJobStatusResponse:
    return get_ingestion_job_status(job_id)


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
async def query_chunks(payload: QueryRequest) -> QueryResponse:
    return await run_query_pipeline(payload)


@router.post("/agentic/query", response_model=AgenticQueryResponse, response_model_exclude_none=True)
async def agentic_query(payload: AgenticQueryRequest) -> AgenticQueryResponse:
    return await run_agentic_query(payload)
