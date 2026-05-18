from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import uuid

from fastapi import HTTPException, UploadFile
from pypdf import PdfReader

from app.core.config import settings
from app.schemas.ingestion import IngestionResponse
from app.services.chunker import build_chunks
from app.services.metadata_enricher import extract_metadata
from app.services.multimodal_service import generate_vlm_caption
from app.services.pdf_extractor import extract_pdf_pages
from app.services.vector_store_service import index_chunks, index_multimodal_image_records


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


def _sha256_of_file(file_path: Path) -> str | None:
    if not file_path.exists() or not file_path.is_file():
        return None
    hasher = hashlib.sha256()
    with file_path.open("rb") as in_file:
        while True:
            chunk = in_file.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _find_duplicate_manifest(file_sha256: str) -> dict | None:
    normalized_hash = str(file_sha256 or "").strip().lower()
    if not normalized_hash:
        return None

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    for manifest_path in settings.processed_dir.glob("*.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        manifest_hash = str(manifest.get("file_sha256", "")).strip().lower()
        if not manifest_hash:
            stored_file = str(manifest.get("stored_file", "")).strip()
            if stored_file:
                computed_hash = _sha256_of_file(Path(stored_file))
                if computed_hash:
                    manifest_hash = computed_hash.lower()
                    manifest["file_sha256"] = manifest_hash
                    try:
                        manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
                    except Exception:
                        pass

        if manifest_hash and manifest_hash == normalized_hash:
            return {
                "doc_id": str(manifest.get("doc_id", "")),
                "source_file": str(manifest.get("source_file", "")),
                "manifest_path": str(manifest_path),
                "stored_file": str(manifest.get("stored_file", "")),
            }

    return None


async def _extract_pdf_image_records(file_path: Path, doc_id: str, source_file: str) -> tuple[list[dict], dict]:
    if not settings.enable_multimodal_ingest:
        return [], {"status": "disabled", "images_extracted": 0}

    images_root = settings.processed_dir / "images" / doc_id
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


async def ingest_pdf(file: UploadFile) -> IngestionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    original_name = _safe_filename(file.filename)
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    stored_name = f"{doc_id}_{original_name}"
    destination = settings.upload_dir / stored_name

    file_size, file_sha256 = await _save_upload(file, destination)

    duplicate = _find_duplicate_manifest(file_sha256)
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
                "existing_manifest_path": duplicate.get("manifest_path", ""),
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

    for page in pages:
        extraction_methods[page.extraction_method] = extraction_methods.get(page.extraction_method, 0) + 1
        if page.likely_scanned:
            likely_scanned_pages += 1

    for chunk in chunks:
        metadata = extract_metadata(chunk.text)
        doc_months.update(metadata.months)
        doc_topics.update(metadata.topics)

        chunk_records.append(
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "token_estimate": chunk.token_estimate,
                "metadata": {
                    "months": metadata.months,
                    "topics": metadata.topics,
                    "entities": metadata.entities,
                },
            }
        )

    manifest = {
        "doc_id": doc_id,
        "source_file": original_name,
        "stored_file": str(destination),
        "file_sha256": file_sha256,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "file_size_bytes": file_size,
        "pages_processed": len(pages),
        "chunks_created": len(chunks),
        "extraction_summary": {
            "methods": extraction_methods,
            "likely_scanned_pages": likely_scanned_pages,
        },
        "doc_metadata": {
            "months": sorted(doc_months),
            "topics": sorted(doc_topics),
        },
        "chunks": chunk_records,
    }

    vector_index_summary = await index_chunks(
        doc_id=doc_id,
        source_file=original_name,
        chunks=chunk_records,
    )
    manifest["vector_index_summary"] = vector_index_summary

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
    manifest["multimodal_extraction_summary"] = image_extraction_summary
    manifest["multimodal_vector_index_summary"] = multimodal_index_summary
    manifest["images"] = image_records

    manifest_path = settings.processed_dir / f"{doc_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    return IngestionResponse(
        doc_id=doc_id,
        source_file=original_name,
        file_size_bytes=file_size,
        pages_processed=len(pages),
        chunks_created=len(chunks),
        manifest_path=str(manifest_path),
        message="Ingestion completed.",
        months_detected=sorted(doc_months),
        topics_detected=sorted(doc_topics),
        vector_index_summary=vector_index_summary,
    )
