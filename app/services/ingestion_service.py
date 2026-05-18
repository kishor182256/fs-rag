from datetime import UTC, datetime
import json
from pathlib import Path
import uuid

from fastapi import HTTPException, UploadFile

from app.core.config import settings
from app.schemas.ingestion import IngestionResponse
from app.services.chunker import build_chunks
from app.services.metadata_enricher import extract_metadata
from app.services.pdf_extractor import extract_pdf_pages
from app.services.vector_store_service import index_chunks


def _safe_filename(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in {"-", "_", "."}).strip(".") or "upload.pdf"


async def _save_upload(file: UploadFile, destination: Path) -> int:
    total_size = 0
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
            out_file.write(chunk)
    return total_size


async def ingest_pdf(file: UploadFile) -> IngestionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    original_name = _safe_filename(file.filename)
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    stored_name = f"{doc_id}_{original_name}"
    destination = settings.upload_dir / stored_name

    file_size = await _save_upload(file, destination)

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

    manifest_path = settings.processed_dir / f"{doc_id}.json"
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
    )
