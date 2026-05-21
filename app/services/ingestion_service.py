import hashlib
import asyncio
from pathlib import Path
import re
from typing import Any, Literal
import uuid

from fastapi import HTTPException, UploadFile
from pypdf import PdfReader

from app.core.config import settings
from app.schemas.ingestion import IngestionJobAcceptedResponse, IngestionJobStatusResponse, IngestionResponse
from app.services.aws_async_ingestion_service import AwsAsyncIngestionService
from app.services.chunker import build_chunks
from app.services.metadata_enricher import extract_metadata
from app.services.multimodal_service import generate_vlm_caption
from app.services.pdf_extractor import extract_pdf_pages
from app.services.s3_storage_service import S3UploadService
from app.services.vector_store_service import (
    find_duplicate_by_file_hash,
    index_chunks,
    index_multimodal_image_records,
)


HF_TOPIC_LABELS = [
    "science and technology",
    "cloud computing",
    "big data and analytics",
    "artificial intelligence",
    "robotics",
    "banking and finance",
    "economy and policy",
    "government schemes",
    "international relations",
    "awards and honors",
    "sports events",
    "education and examinations",
]

HF_CHUNK_TYPE_LABELS = {
    "concept_explanation": "concept explanation paragraph",
    "definition": "definition statement",
    "list": "bullet list or enumeration",
    "table_or_facts": "table or numeric facts",
    "timeline_or_news": "timeline or event update",
    "qa_or_exam": "question and answer or exam content",
    "procedure": "procedure or step by step instructions",
}


class HFChunkEnricher:
    def __init__(self) -> None:
        self.enabled = bool(settings.enable_hf_chunk_enrichment)
        self._initialized = False
        self._available = False
        self._init_error = ""
        self._zero_shot: Any | None = None
        self._ner: Any | None = None

    def _initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        if not self.enabled:
            self._init_error = "disabled_by_config"
            return

        try:
            from transformers import pipeline

            self._zero_shot = pipeline(
                task="zero-shot-classification",
                model=settings.hf_topic_model_name,
            )
            self._ner = pipeline(
                task="token-classification",
                model=settings.hf_ner_model_name,
                aggregation_strategy="simple",
            )
            self._available = True
        except Exception as exc:
            self._available = False
            self._init_error = str(exc)

    @staticmethod
    def _normalize_topic(label: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
        return normalized or "unknown_topic"

    def enrich(self, text: str) -> tuple[dict, bool, str]:
        self._initialize()
        if not self._available or not self._zero_shot or not self._ner:
            return {}, False, self._init_error or "hf_unavailable"

        sample = " ".join((text or "").split())[: max(256, int(settings.hf_enrichment_max_chars))]
        if not sample:
            return {}, False, "empty_text"

        try:
            topic_result = self._zero_shot(sample, candidate_labels=HF_TOPIC_LABELS, multi_label=False)
            topic_label = str((topic_result.get("labels") or [""])[0])
            topic_score = float((topic_result.get("scores") or [0.0])[0])

            type_result = self._zero_shot(sample, candidate_labels=list(HF_CHUNK_TYPE_LABELS.values()), multi_label=False)
            chunk_type_label = str((type_result.get("labels") or [""])[0])
            chunk_type_score = float((type_result.get("scores") or [0.0])[0])

            chunk_type = "unknown"
            for key, label in HF_CHUNK_TYPE_LABELS.items():
                if label == chunk_type_label:
                    chunk_type = key
                    break

            entities_raw = self._ner(sample)
            entities: list[str] = []
            seen: set[str] = set()
            entity_limit = max(1, int(settings.hf_enrichment_entity_limit))
            for entity in entities_raw:
                word = str(entity.get("word", "")).strip()
                if not word:
                    continue
                cleaned = re.sub(r"\s+", " ", word).replace(" ##", "").strip()
                lowered = cleaned.lower()
                if len(cleaned) < 2 or lowered in seen:
                    continue
                seen.add(lowered)
                entities.append(cleaned)
                if len(entities) >= entity_limit:
                    break

            topic = ""
            if topic_label and topic_score >= float(settings.hf_chunk_type_min_score):
                topic = self._normalize_topic(topic_label)

            return (
                {
                    "topic": topic,
                    "topic_label": topic_label,
                    "topic_score": topic_score,
                    "chunk_type": chunk_type,
                    "chunk_type_label": chunk_type_label,
                    "chunk_type_score": chunk_type_score,
                    "entities": entities,
                },
                True,
                "",
            )
        except Exception as exc:
            return {}, False, str(exc)

    def is_available(self) -> bool:
        self._initialize()
        return bool(self._available)


_hf_enricher = HFChunkEnricher()
_s3_uploader = S3UploadService()
_aws_async_service = AwsAsyncIngestionService()


def _hf_collection_name() -> str:
    return str(settings.hf_qdrant_collection or "").strip()


def _requested_text_collection(pipeline: Literal["auto", "legacy", "hf"]) -> str:
    if pipeline == "legacy":
        return settings.qdrant_collection
    if pipeline == "hf":
        target = _hf_collection_name()
        if target:
            return target
        raise HTTPException(status_code=400, detail="HF pipeline requested but HF_QDRANT_COLLECTION is not configured.")

    if settings.enable_hf_chunk_enrichment:
        target = _hf_collection_name()
        if target:
            return target
    return settings.qdrant_collection


def _resolve_pipeline_target(
    pipeline: Literal["auto", "legacy", "hf"],
) -> tuple[str, str, str, bool]:
    requested_collection = _requested_text_collection(pipeline=pipeline)
    hf_requested = pipeline == "hf" or (pipeline == "auto" and requested_collection == _hf_collection_name())
    hf_available = bool(_hf_enricher.is_available()) if hf_requested else False
    if hf_requested and hf_available:
        target_collection = requested_collection
        pipeline_used = "hf_enriched"
    elif hf_requested:
        target_collection = settings.qdrant_collection
        pipeline_used = "hf_fallback_legacy"
    else:
        target_collection = settings.qdrant_collection
        pipeline_used = "legacy"

    duplicate_scope_collection = requested_collection if hf_requested else settings.qdrant_collection
    return target_collection, pipeline_used, duplicate_scope_collection, hf_requested


def _safe_filename(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in {"-", "_", "."}).strip(".") or "upload.pdf"


def _resolve_async_mode(
    *,
    pipeline: Literal["auto", "legacy", "hf"],
    async_mode: bool | None,
) -> bool:
    if async_mode is not None:
        return bool(async_mode)

    # Default behavior: HF pipeline prefers async S3->SQS path when enabled.
    if pipeline == "hf" and settings.enable_async_ingestion:
        return True
    return False


def _resolve_wait_for_completion(
    *,
    effective_async_mode: bool,
    wait_for_completion: bool | None,
) -> bool:
    if not effective_async_mode:
        return False
    if wait_for_completion is not None:
        return bool(wait_for_completion)
    return bool(settings.async_wait_for_completion_default)


async def _wait_for_job_terminal_state(job_id: str) -> dict | None:
    timeout_seconds = max(1, int(settings.async_wait_timeout_seconds))
    poll_seconds = max(0.2, float(settings.async_wait_poll_seconds))
    deadline = asyncio.get_running_loop().time() + timeout_seconds

    while True:
        item, ok, error = _aws_async_service.get_job(job_id=job_id)
        if not ok:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "job_status_unavailable",
                    "message": "Could not fetch ingestion job status while waiting for completion.",
                    "reason": error or "unknown_error",
                },
            )
        if item is not None:
            status = str(item.get("status", "")).strip().lower()
            if status in {"completed", "failed"}:
                return item

        if asyncio.get_running_loop().time() >= deadline:
            return None
        await asyncio.sleep(poll_seconds)


async def _save_upload(file: UploadFile, destination: Path) -> tuple[int, str]:
    total_size = 0
    hasher = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as out_file:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > settings.max_upload_bytes:
                out_file.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max allowed is {settings.max_upload_size_mb} MB.",
                )
            hasher.update(chunk)
            out_file.write(chunk)
    return total_size, hasher.hexdigest()


def _queue_ingestion_job(
    *,
    doc_id: str,
    source_file: str,
    pipeline: str,
    file_sha256: str,
    s3_bucket: str,
    s3_key: str,
) -> tuple[dict, bool, str]:
    job_item, created, create_error = _aws_async_service.create_job(
        doc_id=doc_id,
        source_file=source_file,
        pipeline=pipeline,
        file_sha256=file_sha256,
        s3_bucket=s3_bucket,
        s3_key=s3_key,
    )
    if not created:
        return {}, False, create_error

    message_payload = {
        "job_id": job_item["job_id"],
        "doc_id": doc_id,
        "source_file": source_file,
        "pipeline": pipeline,
        "file_sha256": file_sha256,
        "s3_bucket": s3_bucket,
        "s3_key": s3_key,
        "submitted_at": job_item.get("created_at", ""),
    }

    message_id, sent, send_error = _aws_async_service.enqueue_job(job_payload=message_payload)
    if not sent:
        _aws_async_service.mark_job_failed(job_id=job_item["job_id"], error=send_error)
        return {}, False, send_error

    _aws_async_service.update_job_message_id(job_id=job_item["job_id"], message_id=message_id)
    job_item["message_id"] = message_id
    return job_item, True, ""


async def _extract_pdf_image_records(file_path: Path, doc_id: str, source_file: str) -> tuple[list[dict], dict]:
    if not settings.enable_multimodal_ingest:
        return [], {"status": "disabled", "images_extracted": 0}

    images_root = settings.upload_dir / "images" / doc_id
    images_root.mkdir(parents=True, exist_ok=True)

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:
        return [], {"status": "reader_error", "images_extracted": 0, "error": str(exc)}

    image_records: list[dict] = []
    max_images = max(0, int(settings.multimodal_max_images_per_doc))
    min_bytes = max(0, int(settings.multimodal_min_image_bytes))

    for page_index, page in enumerate(reader.pages):
        if max_images and len(image_records) >= max_images:
            break

        try:
            images = list(getattr(page, "images", []) or [])
        except Exception:
            images = []

        for image_index, image in enumerate(images, start=1):
            if max_images and len(image_records) >= max_images:
                break
            try:
                image_bytes = bytes(getattr(image, "data", b"") or b"")
                if len(image_bytes) < min_bytes:
                    continue
                image_name = str(getattr(image, "name", "") or f"page_{page_index+1}_img_{image_index}.bin")
                image_id = f"image_{len(image_records) + 1:05d}"
                safe_name = "".join(ch for ch in image_name if ch.isalnum() or ch in {"-", "_", "."}).strip(".")
                if not safe_name:
                    safe_name = f"{image_id}.bin"
                output_path = images_root / f"{image_id}_{safe_name}"
                output_path.write_bytes(image_bytes)

                caption = f"Extracted figure/image from page {page_index + 1} of {source_file}."
                vlm_caption = await generate_vlm_caption(
                    str(output_path),
                    source_file=source_file,
                    page=page_index + 1,
                )
                if vlm_caption:
                    caption = vlm_caption
                image_records.append(
                    {
                        "image_id": image_id,
                        "page": page_index + 1,
                        "image_name": image_name,
                        "image_path": str(output_path),
                        "caption": caption,
                        "embedding_text": caption,
                        "metadata": {
                            "source": "pdf_image",
                            "bytes": len(image_bytes),
                            "vlm_caption_used": bool(vlm_caption),
                        },
                    }
                )
            except Exception:
                continue

    summary = {
        "status": "enabled",
        "images_extracted": len(image_records),
        "max_images_per_doc": max_images,
        "vlm_captions_enabled": bool(settings.enable_vlm_captions),
        "clip_vectors_enabled": bool(settings.enable_clip_image_vectors),
    }
    return image_records, summary


async def process_saved_pdf(
    *,
    file_path: Path,
    doc_id: str,
    source_file: str,
    file_size: int,
    file_sha256: str,
    pipeline: Literal["auto", "legacy", "hf"] = "auto",
    skip_duplicate_check: bool = False,
) -> IngestionResponse:
    target_collection, pipeline_used, duplicate_scope_collection, _ = _resolve_pipeline_target(pipeline=pipeline)

    if not skip_duplicate_check:
        duplicate = find_duplicate_by_file_hash(file_sha256=file_sha256, collection_name=duplicate_scope_collection)
        if duplicate is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "duplicate_file",
                    "message": "This file has already been ingested.",
                    "file_sha256": file_sha256,
                    "existing_doc_id": duplicate.get("doc_id", ""),
                    "existing_source_file": duplicate.get("source_file", ""),
                    "collection_name": duplicate.get("collection_name", duplicate_scope_collection),
                    "duplicate_scope_collection": duplicate_scope_collection,
                },
            )

    pages = extract_pdf_pages(
        file_path,
        min_page_text_chars=settings.min_page_text_chars,
        enable_ocr_fallback=settings.enable_ocr_fallback,
    )
    if not pages:
        raise HTTPException(
            status_code=422,
            detail="No readable text found. Enable OCR fallback for scanned PDFs.",
        )

    chunks = build_chunks(
        pages=pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_chars=settings.min_chunk_chars,
    )

    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks produced from extracted text.")

    chunk_records = []
    doc_months: set[str] = set()
    doc_topics: set[str] = set()
    extraction_methods: dict[str, int] = {}
    likely_scanned_pages = 0
    hf_chunks_attempted = 0
    hf_chunks_enriched = 0
    hf_fallback_chunks = 0
    hf_errors: list[str] = []

    for page in pages:
        extraction_methods[page.extraction_method] = extraction_methods.get(page.extraction_method, 0) + 1
        if page.likely_scanned:
            likely_scanned_pages += 1

    for chunk in chunks:
        metadata = extract_metadata(chunk.text)
        merged_topics = list(metadata.topics)
        merged_entities = list(metadata.entities)

        hf_chunks_attempted += 1
        hf_enrichment, hf_used, hf_error = _hf_enricher.enrich(chunk.text)
        if hf_used:
            hf_chunks_enriched += 1
            hf_topic = str(hf_enrichment.get("topic", "")).strip()
            if hf_topic and hf_topic not in merged_topics:
                merged_topics.append(hf_topic)
            for entity in hf_enrichment.get("entities", []):
                normalized = str(entity).strip()
                if normalized and normalized not in merged_entities:
                    merged_entities.append(normalized)
        else:
            hf_fallback_chunks += 1
            if hf_error and hf_error not in hf_errors:
                hf_errors.append(hf_error)

        doc_months.update(metadata.months)
        doc_topics.update(merged_topics)

        chunk_records.append(
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "token_estimate": chunk.token_estimate,
                "metadata": {
                    "months": metadata.months,
                    "topics": merged_topics,
                    "entities": merged_entities,
                    "hf_enrichment": {
                        **hf_enrichment,
                        "enabled": bool(settings.enable_hf_chunk_enrichment),
                        "applied": bool(hf_used),
                        "fallback_used": not bool(hf_used),
                    },
                },
            }
        )

    vector_index_summary = await index_chunks(
        doc_id=doc_id,
        source_file=source_file,
        chunks=chunk_records,
        file_sha256=file_sha256,
        collection_name=target_collection,
    )

    image_records, image_extraction_summary = await _extract_pdf_image_records(
        file_path=file_path,
        doc_id=doc_id,
        source_file=source_file,
    )
    multimodal_index_summary = await index_multimodal_image_records(
        doc_id=doc_id,
        source_file=source_file,
        image_records=image_records,
    )
    return IngestionResponse(
        doc_id=doc_id,
        source_file=source_file,
        file_size_bytes=file_size,
        pages_processed=len(pages),
        chunks_created=len(chunks),
        ingestion_pipeline=pipeline_used,
        manifest_path=f"qdrant://{target_collection}/{doc_id}",
        message="Ingestion completed into Qdrant only.",
        months_detected=sorted(doc_months),
        topics_detected=sorted(doc_topics),
        vector_index_summary={
            **vector_index_summary,
            "target_collection": target_collection,
            "duplicate_scope_collection": duplicate_scope_collection,
            "pipeline_used": pipeline_used,
            "hf_enrichment_summary": {
                "enabled": bool(settings.enable_hf_chunk_enrichment),
                "chunks_attempted": hf_chunks_attempted,
                "chunks_enriched": hf_chunks_enriched,
                "fallback_chunks": hf_fallback_chunks,
                "errors": hf_errors[:3],
            },
            "extraction_summary": {
                "methods": extraction_methods,
                "likely_scanned_pages": likely_scanned_pages,
            },
            "multimodal_extraction_summary": image_extraction_summary,
            "multimodal_vector_index_summary": multimodal_index_summary,
        },
    )


async def ingest_pdf(
    file: UploadFile,
    pipeline: Literal["auto", "legacy", "hf"] = "auto",
    async_mode: bool | None = None,
    wait_for_completion: bool | None = None,
) -> IngestionResponse | IngestionJobAcceptedResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    original_name = _safe_filename(file.filename)
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    stored_name = f"{doc_id}_{original_name}"
    destination = settings.upload_dir / stored_name

    file_size, file_sha256 = await _save_upload(file, destination)
    effective_async_mode = _resolve_async_mode(pipeline=pipeline, async_mode=async_mode)
    effective_wait_for_completion = _resolve_wait_for_completion(
        effective_async_mode=effective_async_mode,
        wait_for_completion=wait_for_completion,
    )

    _, pipeline_used, duplicate_scope_collection, _ = _resolve_pipeline_target(pipeline=pipeline)
    if effective_async_mode:
        duplicate_job, duplicate_check_ok, duplicate_check_error = _aws_async_service.find_duplicate_by_file_hash(
            file_sha256=file_sha256,
            pipeline=pipeline,
        )
        if not duplicate_check_ok:
            destination.unlink(missing_ok=True)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "duplicate_check_failed",
                    "message": "Could not validate duplicate file in AWS async pipeline.",
                    "reason": duplicate_check_error or "unknown_error",
                },
            )
        if duplicate_job is not None:
            destination.unlink(missing_ok=True)
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "duplicate_file",
                    "message": "This file has already been ingested in AWS async pipeline.",
                    "file_sha256": file_sha256,
                    "existing_job_id": str(duplicate_job.get("job_id", "")),
                    "existing_doc_id": str(duplicate_job.get("doc_id", "")),
                    "existing_source_file": str(duplicate_job.get("source_file", "")),
                    "pipeline": str(duplicate_job.get("pipeline", pipeline)),
                    "status": str(duplicate_job.get("status", "")),
                    "duplicate_scope_collection": "aws_async_pipeline",
                },
            )
    else:
        duplicate = find_duplicate_by_file_hash(file_sha256=file_sha256, collection_name=duplicate_scope_collection)
        if duplicate is not None:
            destination.unlink(missing_ok=True)
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "duplicate_file",
                    "message": "This file has already been ingested.",
                    "file_sha256": file_sha256,
                    "existing_doc_id": duplicate.get("doc_id", ""),
                    "existing_source_file": duplicate.get("source_file", ""),
                    "collection_name": duplicate.get("collection_name", duplicate_scope_collection),
                    "duplicate_scope_collection": duplicate_scope_collection,
                },
            )

    if effective_async_mode:
        if not settings.enable_async_ingestion:
            destination.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "async_ingestion_disabled",
                    "message": "Async ingestion is disabled. Set ENABLE_ASYNC_INGESTION=true.",
                },
            )

        s3_upload_summary, s3_uploaded, s3_error = _s3_uploader.upload_pdf(
            file_path=destination,
            doc_id=doc_id,
            source_file=original_name,
        )
        if not s3_uploaded:
            destination.unlink(missing_ok=True)
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "s3_upload_failed",
                    "message": "Could not upload file to S3 for async ingestion.",
                    "reason": s3_error or "unknown_error",
                    "s3_upload_summary": s3_upload_summary,
                },
            )

        job_item, queued, queue_error = _queue_ingestion_job(
            doc_id=doc_id,
            source_file=original_name,
            pipeline=pipeline,
            file_sha256=file_sha256,
            s3_bucket=str(s3_upload_summary.get("bucket", "")),
            s3_key=str(s3_upload_summary.get("key", "")),
        )
        destination.unlink(missing_ok=True)
        if not queued:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "job_queue_failed",
                    "message": "Could not enqueue ingestion job.",
                    "reason": queue_error or "unknown_error",
                },
            )

        accepted_response = IngestionJobAcceptedResponse(
            status="accepted",
            message="File accepted and queued for async ingestion.",
            job_id=str(job_item.get("job_id", "")),
            doc_id=doc_id,
            source_file=original_name,
            ingestion_pipeline=pipeline_used,
            queue=str(settings.sqs_queue_url or ""),
            s3_bucket=str(s3_upload_summary.get("bucket", "")),
            s3_key=str(s3_upload_summary.get("key", "")),
        )
        if not effective_wait_for_completion:
            return accepted_response

        terminal_item = await _wait_for_job_terminal_state(job_id=accepted_response.job_id)
        if terminal_item is None:
            return IngestionJobAcceptedResponse(
                status="accepted",
                message="File queued and processing; completion wait timed out. Check job status endpoint.",
                job_id=accepted_response.job_id,
                doc_id=accepted_response.doc_id,
                source_file=accepted_response.source_file,
                ingestion_pipeline=accepted_response.ingestion_pipeline,
                queue=accepted_response.queue,
                s3_bucket=accepted_response.s3_bucket,
                s3_key=accepted_response.s3_key,
            )

        terminal_status = str(terminal_item.get("status", "")).strip().lower()
        if terminal_status == "failed":
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "ingestion_job_failed",
                    "message": "Queued ingestion job failed.",
                    "job_id": str(terminal_item.get("job_id", accepted_response.job_id)),
                    "reason": str(terminal_item.get("error", "") or "unknown_error"),
                },
            )

        result = terminal_item.get("result", {})
        if isinstance(result, dict):
            payload = result.get("ingestion_response")
            if isinstance(payload, dict):
                try:
                    return IngestionResponse.model_validate(payload)
                except Exception:
                    pass
        return IngestionJobAcceptedResponse(
            status="accepted",
            message="Ingestion completed, but full response payload was unavailable. Check job status result.",
            job_id=accepted_response.job_id,
            doc_id=accepted_response.doc_id,
            source_file=accepted_response.source_file,
            ingestion_pipeline=accepted_response.ingestion_pipeline,
            queue=accepted_response.queue,
            s3_bucket=accepted_response.s3_bucket,
            s3_key=accepted_response.s3_key,
        )

    return await process_saved_pdf(
        file_path=destination,
        doc_id=doc_id,
        source_file=original_name,
        file_size=file_size,
        file_sha256=file_sha256,
        pipeline=pipeline,
        skip_duplicate_check=True,
    )


def get_ingestion_job_status(job_id: str) -> IngestionJobStatusResponse:
    item, ok, error = _aws_async_service.get_job(job_id=job_id)
    if not ok:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "job_status_unavailable",
                "message": "Could not fetch ingestion job status.",
                "reason": error or "unknown_error",
            },
        )
    if item is None:
        return IngestionJobStatusResponse(job_id=job_id, status="unknown")

    status = str(item.get("status", "unknown")).strip().lower() or "unknown"
    if status not in {"queued", "processing", "completed", "failed", "unknown"}:
        status = "unknown"

    return IngestionJobStatusResponse(
        job_id=str(item.get("job_id", job_id)),
        status=status,
        doc_id=str(item.get("doc_id", "")) or None,
        source_file=str(item.get("source_file", "")) or None,
        pipeline=str(item.get("pipeline", "")) or None,
        created_at=str(item.get("created_at", "")) or None,
        updated_at=str(item.get("updated_at", "")) or None,
        message_id=str(item.get("message_id", "")) or None,
        error=str(item.get("error", "")) or None,
        result=item.get("result", {}) if isinstance(item.get("result", {}), dict) else None,
    )
