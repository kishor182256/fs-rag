from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import settings


class S3UploadService:
    def __init__(self) -> None:
        self.enabled = bool(settings.enable_s3_upload)
        self.required = bool(settings.s3_upload_required)
        self.bucket_name = str(settings.s3_bucket_name or "").strip()
        self.region = str(settings.aws_region or "").strip()
        self.key_prefix = str(settings.s3_key_prefix or "").strip().strip("/")
        self.endpoint_url = str(settings.s3_endpoint_url or "").strip() or None
        self._client: Any | None = None
        self._init_error = ""
        self._initialized = False

    def _initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        if not self.enabled:
            self._init_error = "disabled_by_config"
            return
        if not self.bucket_name:
            self._init_error = "missing_bucket_name"
            return
        if not self.region:
            self._init_error = "missing_aws_region"
            return

        try:
            import boto3

            self._client = boto3.client(
                "s3",
                region_name=self.region,
                endpoint_url=self.endpoint_url,
            )
        except Exception as exc:
            self._init_error = f"s3_client_init_failed:{exc}"

    def _build_key(self, *, doc_id: str, source_file: str) -> str:
        if self.key_prefix:
            return f"{self.key_prefix}/{doc_id}/{source_file}"
        return f"{doc_id}/{source_file}"

    def upload_pdf(self, *, file_path: Path, doc_id: str, source_file: str) -> tuple[dict, bool, str]:
        self._initialize()
        if self._client is None:
            return (
                {
                    "status": "disabled" if not self.enabled else "unavailable",
                    "bucket": self.bucket_name,
                    "key": "",
                },
                False,
                self._init_error or "s3_unavailable",
            )

        key = self._build_key(doc_id=doc_id, source_file=source_file)
        try:
            self._client.upload_file(
                Filename=str(file_path),
                Bucket=self.bucket_name,
                Key=key,
            )
        except Exception as exc:
            return (
                {
                    "status": "error",
                    "bucket": self.bucket_name,
                    "key": key,
                },
                False,
                f"s3_upload_failed:{exc}",
            )

        return (
            {
                "status": "uploaded",
                "bucket": self.bucket_name,
                "key": key,
                "s3_uri": f"s3://{self.bucket_name}/{key}",
            },
            True,
            "",
        )
