from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def ocr_page_with_fallback(file_path: Path, page_number: int) -> str:
    """Best-effort OCR fallback for scanned pages.

    Returns empty string when OCR dependencies are unavailable or OCR fails.
    """
    try:
        import fitz  # type: ignore
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        return ""

    try:
        document = fitz.open(str(file_path))
        page = document[page_number - 1]
        pixmap = page.get_pixmap(dpi=220)
        image = Image.open(BytesIO(pixmap.tobytes("png")))
        text = pytesseract.image_to_string(image)
        document.close()
    except Exception:
        return ""

    return _normalize_whitespace(text)
