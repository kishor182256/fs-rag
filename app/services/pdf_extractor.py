from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import re

from pypdf import PdfReader


BlockType = Literal["heading", "list_item", "table_row", "paragraph"]


@dataclass
class ExtractedBlock:
    page_number: int
    text: str
    block_type: BlockType


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    blocks: list[ExtractedBlock]
    extraction_method: Literal["layout", "plain", "ocr"]
    likely_scanned: bool


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_page_text(page) -> tuple[str, str]:
    try:
        layout_text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False) or ""
    except Exception:
        layout_text = ""

    plain_text = page.extract_text() or ""

    if len(layout_text.strip()) >= len(plain_text.strip()):
        return layout_text, "layout"
    return plain_text, "plain"


def _line_kind(line: str) -> BlockType:
    stripped = line.strip()
    if not stripped:
        return "paragraph"

    if re.match(r"^(\d+[\.\)]\s+|[A-Za-z][\.\)]\s+|[-*•]\s+)", stripped):
        return "list_item"

    if " | " in stripped or re.search(r"\s{3,}", stripped):
        return "table_row"

    upper_ratio = 0.0
    alpha_chars = [ch for ch in stripped if ch.isalpha()]
    if alpha_chars:
        upper_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)

    if (upper_ratio > 0.8 and len(stripped.split()) <= 16) or stripped.endswith(":"):
        return "heading"

    return "paragraph"


def _normalize_table_separators(line: str) -> str:
    if " | " in line:
        return _normalize_whitespace(line)
    if re.search(r"\s{3,}", line):
        parts = [part.strip() for part in re.split(r"\s{3,}", line) if part.strip()]
        if len(parts) > 1:
            return " | ".join(parts)
    return _normalize_whitespace(line)


def _build_blocks(page_number: int, raw_text: str) -> list[ExtractedBlock]:
    lines = raw_text.replace("\r", "\n").split("\n")
    blocks: list[ExtractedBlock] = []

    paragraph_buffer: list[str] = []
    table_buffer: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_buffer:
            blocks.append(
                ExtractedBlock(
                    page_number=page_number,
                    text=_normalize_whitespace(" ".join(paragraph_buffer)),
                    block_type="paragraph",
                )
            )
            paragraph_buffer.clear()

    def flush_table() -> None:
        if table_buffer:
            for row in table_buffer:
                blocks.append(
                    ExtractedBlock(
                        page_number=page_number,
                        text=row,
                        block_type="table_row",
                    )
                )
            table_buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_table()
            continue

        kind = _line_kind(stripped)
        if kind == "table_row":
            flush_paragraph()
            table_buffer.append(_normalize_table_separators(stripped))
            continue

        flush_table()

        if kind in {"heading", "list_item"}:
            flush_paragraph()
            blocks.append(
                ExtractedBlock(
                    page_number=page_number,
                    text=_normalize_whitespace(stripped),
                    block_type=kind,
                )
            )
            continue

        paragraph_buffer.append(stripped)

    flush_paragraph()
    flush_table()
    return [block for block in blocks if block.text]


def _prune_repeating_page_noise(pages: list[ExtractedPage]) -> list[ExtractedPage]:
    if len(pages) < 3:
        return pages

    candidate_counts: dict[str, int] = {}
    for page in pages:
        first_last_lines = []
        page_lines = [line.strip() for line in page.text.split("\n") if line.strip()]
        first_last_lines.extend(page_lines[:2])
        first_last_lines.extend(page_lines[-2:])
        for line in set(first_last_lines):
            normalized = _normalize_whitespace(line)
            if 6 <= len(normalized) <= 120:
                candidate_counts[normalized] = candidate_counts.get(normalized, 0) + 1

    threshold = max(3, int(len(pages) * 0.6))
    repeating = {line for line, count in candidate_counts.items() if count >= threshold}
    if not repeating:
        return pages

    cleaned_pages: list[ExtractedPage] = []
    for page in pages:
        filtered_blocks = [block for block in page.blocks if _normalize_whitespace(block.text) not in repeating]
        page_text = "\n".join(block.text for block in filtered_blocks)
        cleaned_pages.append(
            ExtractedPage(
                page_number=page.page_number,
                text=page_text,
                blocks=filtered_blocks,
                extraction_method=page.extraction_method,
                likely_scanned=page.likely_scanned,
            )
        )
    return cleaned_pages


def extract_pdf_pages(
    file_path: Path,
    *,
    min_page_text_chars: int = 80,
    enable_ocr_fallback: bool = False,
) -> list[ExtractedPage]:
    reader = PdfReader(str(file_path))
    pages: list[ExtractedPage] = []

    for idx, page in enumerate(reader.pages, start=1):
        raw_text, method = _extract_page_text(page)
        blocks = _build_blocks(page_number=idx, raw_text=raw_text)
        page_text = "\n".join(block.text for block in blocks).strip()
        likely_scanned = len(_normalize_whitespace(page_text)) < min_page_text_chars

        if likely_scanned and enable_ocr_fallback:
            from app.services.ocr_service import ocr_page_with_fallback

            ocr_text = ocr_page_with_fallback(file_path=file_path, page_number=idx)
            if ocr_text:
                blocks = _build_blocks(page_number=idx, raw_text=ocr_text)
                page_text = "\n".join(block.text for block in blocks).strip()
                method = "ocr"
                likely_scanned = False

        if page_text:
            pages.append(
                ExtractedPage(
                    page_number=idx,
                    text=page_text,
                    blocks=blocks,
                    extraction_method=method,
                    likely_scanned=likely_scanned,
                )
            )

    return _prune_repeating_page_noise(pages)
