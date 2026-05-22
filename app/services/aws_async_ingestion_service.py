from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from typing import Any
import uuid

from app.core.config import settings

LOGGER = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class AwsAsyncIngestionService:
    def __init__(self) -> None:
        self.enabled = bool(settings.enable_async_ingestion)
        self.region = str(settings.aws_region or "").strip()
        self.queue_url = str(settings.sqs_queue_url or "").strip()
        self.jobs_table = str(settings.dynamodb_jobs_table or "").strip()
        self.job_pk = str(settings.dynamodb_job_pk or "job_id").strip() or "job_id"
        self.resolved_job_pk = self.job_pk
        self._resolved_key_schema: list[dict[str, str]] = []
        self.dynamodb_endpoint_url = str(settings.dynamodb_endpoint_url or "").strip() or None
        self._initialized = False
        self._init_error = ""
        self._sqs_client: Any | None = None
        self._dynamodb_resource: Any | None = None

    def _initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        if not self.enabled:
            self._init_error = "disabled_by_config"
            return
        if not self.region:
            self._init_error = "missing_aws_region"
            return
        if not self.queue_url:
            self._init_error = "missing_sqs_queue_url"
            return
        if not self.jobs_table:
            self._init_error = "missing_dynamodb_jobs_table"
            return

        try:
            import boto3

            self._sqs_client = boto3.client("sqs", region_name=self.region)
            self._dynamodb_resource = boto3.resource(
                "dynamodb",
                region_name=self.region,
                endpoint_url=self.dynamodb_endpoint_url,
            )
            self._resolve_table_partition_key()
        except Exception as exc:
            self._init_error = f"aws_client_init_failed:{exc}"

    def is_available(self) -> bool:
        self._initialize()
        return self._sqs_client is not None and self._dynamodb_resource is not None

    def _table(self) -> Any:
        assert self._dynamodb_resource is not None
        return self._dynamodb_resource.Table(self.jobs_table)

    def _resolve_table_partition_key(self) -> None:
        try:
            table = self._table()
            table.load()
            key_schema = list(getattr(table, "key_schema", []) or [])
            self._resolved_key_schema = [
                {
                    "AttributeName": str(entry.get("AttributeName", "")),
                    "KeyType": str(entry.get("KeyType", "")).strip().upper(),
                }
                for entry in key_schema
                if str(entry.get("AttributeName", ""))
            ]
            partition_key = next(
                (
                    str(entry.get("AttributeName", ""))
                    for entry in self._resolved_key_schema
                    if str(entry.get("KeyType", "")).strip().upper() == "HASH"
                ),
                "",
            )
            if partition_key:
                self.resolved_job_pk = partition_key
                if partition_key != self.job_pk:
                    LOGGER.warning(
                        "dynamodb_job_pk_mismatch configured=%r table=%r using=%r",
                        self.job_pk,
                        partition_key,
                        partition_key,
                    )
        except Exception as exc:
            # Keep configured fallback so service remains available.
            LOGGER.warning("dynamodb_pk_autodetect_failed using_configured_pk=%s reason=%s", self.job_pk, exc)

    def _job_key(self, job_id: str) -> dict[str, str]:
        return {self.resolved_job_pk: job_id}

    def _ensure_required_keys(self, *, item: dict, job_id: str) -> None:
        if not self._resolved_key_schema:
            return

        for entry in self._resolved_key_schema:
            attr_name = str(entry.get("AttributeName", "")).strip()
            key_type = str(entry.get("KeyType", "")).strip().upper()
            if not attr_name:
                continue
            if attr_name in item and str(item.get(attr_name, "")).strip():
                continue
            if key_type == "HASH":
                item[attr_name] = job_id
            elif key_type == "RANGE":
                # deterministic sort-key fallback when table has composite PK
                item[attr_name] = _utc_now_iso()

    def create_job(
        self,
        *,
        doc_id: str,
        source_file: str,
        pipeline: str,
        file_sha256: str,
        s3_bucket: str,
        s3_key: str,
    ) -> tuple[dict, bool, str]:
        self._initialize()
        if not self.is_available():
            return {}, False, self._init_error or "aws_async_unavailable"

        job_id = str(uuid.uuid4())
        now = _utc_now_iso()
        item = {
            "job_id": job_id,
            "status": "queued",
            "phase": "queued",
            "progress_percent": 5,
            "status_message": "Ingestion job queued.",
            "created_at": now,
            "updated_at": now,
            "doc_id": doc_id,
            "source_file": source_file,
            "pipeline": pipeline,
            "file_sha256": file_sha256,
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
            "message_id": "",
            "error": "",
            "result": {},
        }
        item[self.resolved_job_pk] = job_id
        if self.job_pk not in item:
            item[self.job_pk] = job_id
        self._ensure_required_keys(item=item, job_id=job_id)

        try:
            self._table().put_item(Item=item)
        except Exception as exc:
            item_keys = sorted(list(item.keys()))
            return (
                {},
                False,
                (
                    f"dynamodb_put_failed:{exc};item_keys={item_keys};"
                    f"resolved_pk={self.resolved_job_pk!r};job_pk={self.job_pk!r};"
                    f"resolved_key_schema={self._resolved_key_schema!r}"
                ),
            )

        return item, True, ""

    def enqueue_job(self, *, job_payload: dict) -> tuple[str, bool, str]:
        self._initialize()
        if not self.is_available():
            return "", False, self._init_error or "aws_async_unavailable"
        try:
            response = self._sqs_client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(job_payload, ensure_ascii=True),
            )
            return str(response.get("MessageId", "")), True, ""
        except Exception as exc:
            return "", False, f"sqs_send_failed:{exc}"

    def update_job_message_id(self, *, job_id: str, message_id: str) -> tuple[bool, str]:
        self._initialize()
        if not self.is_available():
            return False, self._init_error or "aws_async_unavailable"
        try:
            self._table().update_item(
                Key=self._job_key(job_id),
                UpdateExpression="SET message_id=:m, updated_at=:u",
                ExpressionAttributeValues={":m": message_id, ":u": _utc_now_iso()},
            )
            return True, ""
        except Exception as exc:
            return False, f"dynamodb_update_failed:{exc}"

    def mark_job_processing(self, *, job_id: str, worker_id: str = "") -> tuple[bool, str]:
        self._initialize()
        if not self.is_available():
            return False, self._init_error or "aws_async_unavailable"
        try:
            self._table().update_item(
                Key=self._job_key(job_id),
                UpdateExpression="SET #s=:s, phase=:p, progress_percent=:pp, status_message=:m, worker_id=:w, #err=:e, updated_at=:u",
                ExpressionAttributeNames={"#s": "status", "#err": "error"},
                ExpressionAttributeValues={
                    ":s": "processing",
                    ":p": "processing",
                    ":pp": 15,
                    ":m": "Worker started processing.",
                    ":w": (worker_id or "").strip(),
                    ":e": "",
                    ":u": _utc_now_iso(),
                },
            )
            return True, ""
        except Exception as exc:
            return False, f"dynamodb_update_failed:{exc}"

    def mark_job_failed(self, *, job_id: str, error: str) -> tuple[bool, str]:
        self._initialize()
        if not self.is_available():
            return False, self._init_error or "aws_async_unavailable"
        try:
            self._table().update_item(
                Key=self._job_key(job_id),
                UpdateExpression="SET #s=:s, phase=:p, progress_percent=:pp, status_message=:m, #err=:e, updated_at=:u",
                ExpressionAttributeNames={"#s": "status", "#err": "error"},
                ExpressionAttributeValues={
                    ":s": "failed",
                    ":p": "failed",
                    ":pp": 100,
                    ":m": "Ingestion failed.",
                    ":e": (error or "unknown_error")[:1200],
                    ":u": _utc_now_iso(),
                },
            )
            return True, ""
        except Exception as exc:
            return False, f"dynamodb_update_failed:{exc}"

    def mark_job_completed(self, *, job_id: str, result: dict) -> tuple[bool, str]:
        self._initialize()
        if not self.is_available():
            return False, self._init_error or "aws_async_unavailable"
        safe_result = result if isinstance(result, dict) else {}
        try:
            self._table().update_item(
                Key=self._job_key(job_id),
                UpdateExpression="SET #s=:s, phase=:p, progress_percent=:pp, status_message=:m, #res=:r, #err=:e, updated_at=:u",
                ExpressionAttributeNames={"#s": "status", "#res": "result", "#err": "error"},
                ExpressionAttributeValues={
                    ":s": "completed",
                    ":p": "completed",
                    ":pp": 100,
                    ":m": "Ingestion completed successfully.",
                    ":r": safe_result,
                    ":e": "",
                    ":u": _utc_now_iso(),
                },
            )
            return True, ""
        except Exception as exc:
            return False, f"dynamodb_update_failed:{exc}"

    def update_job_progress(
        self,
        *,
        job_id: str,
        phase: str,
        progress_percent: int,
        status_message: str = "",
    ) -> tuple[bool, str]:
        self._initialize()
        if not self.is_available():
            return False, self._init_error or "aws_async_unavailable"

        safe_phase = (phase or "processing").strip().lower()[:64]
        safe_progress = max(0, min(100, int(progress_percent)))
        safe_message = (status_message or "").strip()[:300]
        try:
            self._table().update_item(
                Key=self._job_key(job_id),
                UpdateExpression="SET phase=:p, progress_percent=:pp, status_message=:m, updated_at=:u",
                ExpressionAttributeValues={
                    ":p": safe_phase,
                    ":pp": safe_progress,
                    ":m": safe_message,
                    ":u": _utc_now_iso(),
                },
            )
            return True, ""
        except Exception as exc:
            return False, f"dynamodb_update_failed:{exc}"

    def get_job(self, *, job_id: str) -> tuple[dict | None, bool, str]:
        self._initialize()
        if not self.is_available():
            return None, False, self._init_error or "aws_async_unavailable"
        try:
            response = self._table().get_item(Key=self._job_key(job_id))
            item = response.get("Item")
            if not item:
                return None, True, ""
            normalized = dict(item)
            if not normalized.get("job_id"):
                normalized["job_id"] = str(normalized.get(self.job_pk, job_id))
            return normalized, True, ""
        except Exception as exc:
            return None, False, f"dynamodb_get_failed:{exc}"

    def find_duplicate_by_file_hash(self, *, file_sha256: str, pipeline: str) -> tuple[dict | None, bool, str]:
        self._initialize()
        if not self.is_available():
            return None, False, self._init_error or "aws_async_unavailable"

        normalized_hash = str(file_sha256 or "").strip().lower()
        normalized_pipeline = str(pipeline or "").strip().lower()
        if not normalized_hash:
            return None, True, ""

        # Dynamo table is keyed by job id, so we scan with a targeted filter to find
        # existing queued/processing/completed jobs for the same file hash and pipeline.
        try:
            from boto3.dynamodb.conditions import Attr

            status_filter = Attr("status").is_in(["queued", "processing", "completed"])
            hash_filter = Attr("file_sha256").eq(normalized_hash)
            pipeline_filter = Attr("pipeline").eq(normalized_pipeline)
            scan_filter = hash_filter & pipeline_filter & status_filter

            response = self._table().scan(
                FilterExpression=scan_filter,
                Limit=1,
            )
            items = response.get("Items", []) or []
            if not items:
                return None, True, ""

            item = dict(items[0])
            if not item.get("job_id"):
                item["job_id"] = str(item.get(self.job_pk, ""))
            return item, True, ""
        except Exception as exc:
            return None, False, f"dynamodb_scan_failed:{exc}"
