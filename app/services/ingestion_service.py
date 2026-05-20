import hashlib
from pathlib import Path
import re
from typing import Any, Literal
import uuid

from fastapi import HTTPException, UploadFile
from pypdf import PdfReader

from app.core.config import settings
from app.schemas.ingestion import IngestionResponse
from app.services.chunker import build_chunks
from app.services.metadata_enricher import extract_metadata
from app.services.multimodal_service import generate_vlm_caption
from app.services.pdf_extractor import extract_pdf_pages
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


def _safe_filename(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in {"-", "_", "."}).strip(".") or "upload.pdf"


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


async def ingest_pdf(file: UploadFile, pipeline: Literal["auto", "legacy", "hf"] = "auto") -> IngestionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    original_name = _safe_filename(file.filename)
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    stored_name = f"{doc_id}_{original_name}"
    destination = settings.upload_dir / stored_name

    file_size, file_sha256 = await _save_upload(file, destination)

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

    pages = extract_pdf_pages(
        destination,
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
        source_file=original_name,
        chunks=chunk_records,
        file_sha256=file_sha256,
        collection_name=target_collection,
    )

    image_records, image_extraction_summary = await _extract_pdf_image_records(
        file_path=destination,
        doc_id=doc_id,
        source_file=original_name,
    )
    multimodal_index_summary = await index_multimodal_image_records(
        doc_id=doc_id,
        source_file=original_name,
        image_records=image_records,
    )
    return IngestionResponse(
        doc_id=doc_id,
        source_file=original_name,
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
