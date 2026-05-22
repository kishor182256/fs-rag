from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
import socket
from typing import Any, Literal, cast

from app.core.config import settings
from app.services.aws_async_ingestion_service import AwsAsyncIngestionService
from app.services.ingestion_service import process_saved_pdf

LOGGER = logging.getLogger("async_ingestion_worker")


def _build_worker_id() -> str:
    host = socket.gethostname().strip() or "worker"
    return f"{host}:{Path.cwd().name}"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _safe_name(name: str) -> str:
    return "".join(ch for ch in (name or "") if ch.isalnum() or ch in {"-", "_", "."}).strip(".") or "document.pdf"


def _normalize_pipeline(value: str) -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"legacy", "hf", "auto"}:
        return normalized
    return "auto"


class AsyncIngestionWorker:
    def __init__(
        self,
        *,
        poll_wait_seconds: int = 20,
        visibility_timeout_seconds: int = 900,
        max_messages: int = 1,
        max_attempts: int = 3,
    ) -> None:
        self.poll_wait_seconds = max(1, min(20, poll_wait_seconds))
        self.visibility_timeout_seconds = max(30, visibility_timeout_seconds)
        self.max_messages = max(1, min(10, max_messages))
        self.max_attempts = max(1, max_attempts)
        self.worker_id = _build_worker_id()
        self.async_service = AwsAsyncIngestionService()

        region = str(settings.aws_region or "").strip()
        queue_url = str(settings.sqs_queue_url or "").strip()
        if not region:
            raise RuntimeError("AWS_REGION is required for async worker.")
        if not queue_url:
            raise RuntimeError("SQS_QUEUE_URL is required for async worker.")

        self.queue_url = queue_url
        self._sqs: Any | None = None
        self._s3: Any | None = None

        import boto3

        self._sqs = boto3.client("sqs", region_name=region)
        self._s3 = boto3.client("s3", region_name=region, endpoint_url=(settings.s3_endpoint_url or None))

    def run_forever(self) -> None:
        LOGGER.info(
            "worker_started queue=%s wait=%ss visibility=%ss max_messages=%s max_attempts=%s",
            self.queue_url,
            self.poll_wait_seconds,
            self.visibility_timeout_seconds,
            self.max_messages,
            self.max_attempts,
        )
        while True:
            messages = self._receive_messages()
            if not messages:
                continue
            for message in messages:
                self._handle_message(message)

    def _receive_messages(self) -> list[dict]:
        assert self._sqs is not None
        response = self._sqs.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=self.max_messages,
            WaitTimeSeconds=self.poll_wait_seconds,
            VisibilityTimeout=self.visibility_timeout_seconds,
            MessageAttributeNames=["All"],
            AttributeNames=["All"],
        )
        return list(response.get("Messages", []) or [])

    def _delete_message(self, receipt_handle: str) -> None:
        assert self._sqs is not None
        self._sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)

    def _download_pdf(self, *, bucket: str, key: str, job_id: str, source_file: str) -> Path:
        assert self._s3 is not None
        local_dir = settings.upload_dir / "async_jobs" / job_id
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / _safe_name(source_file)
        self._s3.download_file(bucket, key, str(local_path))
        return local_path

    def _mark_failed(self, *, job_id: str, error: str) -> None:
        ok, mark_error = self.async_service.mark_job_failed(job_id=job_id, error=error)
        if not ok:
            LOGGER.error("job_status_update_failed job_id=%s reason=%s", job_id, mark_error)

    def _mark_progress(self, *, job_id: str, phase: str, progress: int, message: str) -> None:
        ok, progress_error = self.async_service.update_job_progress(
            job_id=job_id,
            phase=phase,
            progress_percent=progress,
            status_message=message,
        )
        if not ok:
            LOGGER.warning("job_progress_update_failed job_id=%s phase=%s reason=%s", job_id, phase, progress_error)

    def _handle_message(self, message: dict) -> None:
        receipt_handle = str(message.get("ReceiptHandle", "")).strip()
        attributes = message.get("Attributes", {}) if isinstance(message.get("Attributes", {}), dict) else {}
        receive_count_raw = attributes.get("ApproximateReceiveCount", "1")
        receive_count = int(str(receive_count_raw or "1"))

        body_raw = str(message.get("Body", "")).strip()
        if not body_raw:
            LOGGER.error("empty_message_body: deleting message")
            if receipt_handle:
                self._delete_message(receipt_handle)
            return

        payload: dict[str, Any]
        try:
            payload = json.loads(body_raw)
        except Exception as exc:
            LOGGER.error("invalid_message_json error=%s", exc)
            if receipt_handle:
                self._delete_message(receipt_handle)
            return

        job_id = str(payload.get("job_id", "")).strip()
        doc_id = str(payload.get("doc_id", "")).strip()
        source_file = str(payload.get("source_file", "")).strip()
        s3_bucket = str(payload.get("s3_bucket", "")).strip()
        s3_key = str(payload.get("s3_key", "")).strip()
        pipeline = _normalize_pipeline(str(payload.get("pipeline", "auto")))
        pipeline_value = cast(Literal["auto", "legacy", "hf"], pipeline)
        file_sha256 = str(payload.get("file_sha256", "")).strip().lower()

        if not job_id or not doc_id or not source_file or not s3_bucket or not s3_key:
            LOGGER.error("invalid_job_payload job_id=%s payload=%s", job_id, payload)
            if job_id:
                self._mark_failed(job_id=job_id, error="invalid_job_payload")
            if receipt_handle:
                self._delete_message(receipt_handle)
            return

        if receive_count > self.max_attempts:
            LOGGER.error("max_attempts_exceeded job_id=%s attempts=%s", job_id, receive_count)
            self._mark_failed(job_id=job_id, error=f"max_attempts_exceeded:{receive_count}")
            if receipt_handle:
                self._delete_message(receipt_handle)
            return

        ok, processing_error = self.async_service.mark_job_processing(job_id=job_id, worker_id=self.worker_id)
        if not ok:
            LOGGER.warning("mark_processing_failed job_id=%s reason=%s", job_id, processing_error)
        self._mark_progress(job_id=job_id, phase="processing", progress=15, message="Worker picked up job from queue.")

        local_pdf_path: Path | None = None
        try:
            self._mark_progress(job_id=job_id, phase="downloading", progress=25, message="Downloading PDF from S3.")
            local_pdf_path = self._download_pdf(
                bucket=s3_bucket,
                key=s3_key,
                job_id=job_id,
                source_file=source_file,
            )
            self._mark_progress(
                job_id=job_id,
                phase="downloaded",
                progress=35,
                message="PDF downloaded. Starting extraction and indexing.",
            )
            resolved_sha = file_sha256 or _sha256_file(local_pdf_path)
            file_size = int(local_pdf_path.stat().st_size)

            self._mark_progress(
                job_id=job_id,
                phase="indexing",
                progress=70,
                message="Running chunking, enrichment, embeddings, and vector indexing.",
            )
            response = asyncio.run(
                process_saved_pdf(
                    file_path=local_pdf_path,
                    doc_id=doc_id,
                    source_file=source_file,
                    file_size=file_size,
                    file_sha256=resolved_sha,
                    pipeline=pipeline_value,
                    skip_duplicate_check=True,
                )
            )

            self._mark_progress(
                job_id=job_id,
                phase="finalizing",
                progress=90,
                message="Persisting ingestion result to job store.",
            )
            result = {
                "doc_id": response.doc_id,
                "source_file": response.source_file,
                "ingestion_pipeline": response.ingestion_pipeline,
                "pages_processed": response.pages_processed,
                "chunks_created": response.chunks_created,
                "manifest_path": response.manifest_path,
                "vector_index_summary": response.vector_index_summary,
                "ingestion_response": response.model_dump(),
            }
            complete_ok, complete_error = self.async_service.mark_job_completed(job_id=job_id, result=result)
            if not complete_ok:
                LOGGER.error("mark_completed_failed job_id=%s reason=%s", job_id, complete_error)

            LOGGER.info(
                "job_completed job_id=%s doc_id=%s pages=%s chunks=%s",
                job_id,
                doc_id,
                response.pages_processed,
                response.chunks_created,
            )
            if receipt_handle:
                self._delete_message(receipt_handle)
        except Exception as exc:  # noqa: BLE001
            error_text = f"worker_processing_failed:{exc}"
            LOGGER.exception("job_processing_failed job_id=%s error=%s", job_id, exc)
            self._mark_failed(job_id=job_id, error=error_text)
            if receive_count >= self.max_attempts and receipt_handle:
                self._delete_message(receipt_handle)
        finally:
            if local_pdf_path is not None:
                try:
                    local_pdf_path.unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    local_dir = local_pdf_path.parent
                    if local_dir.exists() and not any(local_dir.iterdir()):
                        local_dir.rmdir()
                except Exception:
                    pass


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Async ingestion worker (SQS -> S3 -> Qdrant)")
    parser.add_argument("--poll-wait-seconds", type=int, default=20, help="SQS long-poll wait (1-20).")
    parser.add_argument("--visibility-timeout-seconds", type=int, default=900, help="SQS visibility timeout.")
    parser.add_argument("--max-messages", type=int, default=1, help="Max messages per poll (1-10).")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max receive attempts before drop.")
    parser.add_argument("--log-level", default="INFO", help="Python log level (INFO, DEBUG, WARNING).")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    log_level = str(args.log_level or "INFO").strip().upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    worker = AsyncIngestionWorker(
        poll_wait_seconds=int(args.poll_wait_seconds),
        visibility_timeout_seconds=int(args.visibility_timeout_seconds),
        max_messages=int(args.max_messages),
        max_attempts=int(args.max_attempts),
    )

    try:
        worker.run_forever()
    except KeyboardInterrupt:
        LOGGER.info("worker_stopped_by_signal")


if __name__ == "__main__":
    main()
