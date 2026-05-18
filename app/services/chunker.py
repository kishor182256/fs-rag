from dataclasses import dataclass
import re

from app.services.pdf_extractor import ExtractedPage


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    token_estimate: int


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _take_overlap_blocks(
    block_window: list[tuple[int, str, int]],
    overlap_tokens: int,
) -> list[tuple[int, str, int]]:
    if overlap_tokens <= 0 or not block_window:
        return []

    kept: list[tuple[int, str, int]] = []
    token_count = 0
    for block in reversed(block_window):
        kept.insert(0, block)
        token_count += block[2]
        if token_count >= overlap_tokens:
            break
    return kept


def build_chunks(
    pages: list[ExtractedPage],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
) -> list[Chunk]:
    blocks: list[tuple[int, str, int]] = []
    for page in pages:
        page_blocks = page.blocks if getattr(page, "blocks", None) else []
        if not page_blocks:
            cleaned_page = _normalize_whitespace(page.text)
            if cleaned_page:
                word_count = len(cleaned_page.split())
                blocks.append((page.page_number, cleaned_page, word_count))
            continue

        for block in page_blocks:
            cleaned_text = _normalize_whitespace(block.text)
            if not cleaned_text:
                continue
            word_count = len(cleaned_text.split())
            if word_count == 0:
                continue
            blocks.append((page.page_number, cleaned_text, word_count))

    if not blocks:
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    chunks: list[Chunk] = []
    block_window: list[tuple[int, str, int]] = []
    token_window = 0

    chunk_index = 1
    for page_number, block_text, block_tokens in blocks:
        if block_tokens > chunk_size and not block_window:
            # Avoid giant single-block chunks by cutting oversized blocks.
            words = block_text.split()
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                segment = " ".join(words[start:end]).strip()
                if len(segment) >= min_chunk_chars:
                    chunks.append(
                        Chunk(
                            chunk_id=f"chunk_{chunk_index:05d}",
                            text=segment,
                            page_start=page_number,
                            page_end=page_number,
                            token_estimate=end - start,
                        )
                    )
                    chunk_index += 1
                if end >= len(words):
                    break
                start = max(end - chunk_overlap, start + 1)
            continue

        if block_window and token_window + block_tokens > chunk_size:
            chunk_text = _normalize_whitespace(" ".join(text for _, text, _ in block_window))
            if len(chunk_text) >= min_chunk_chars:
                chunks.append(
                    Chunk(
                        chunk_id=f"chunk_{chunk_index:05d}",
                        text=chunk_text,
                        page_start=block_window[0][0],
                        page_end=block_window[-1][0],
                        token_estimate=token_window,
                    )
                )
                chunk_index += 1

            block_window = _take_overlap_blocks(block_window, chunk_overlap)
            token_window = sum(tokens for _, _, tokens in block_window)

        block_window.append((page_number, block_text, block_tokens))
        token_window += block_tokens

    if block_window:
        chunk_text = _normalize_whitespace(" ".join(text for _, text, _ in block_window))
        if len(chunk_text) >= min_chunk_chars:
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{chunk_index:05d}",
                    text=chunk_text,
                    page_start=block_window[0][0],
                    page_end=block_window[-1][0],
                    token_estimate=token_window,
                )
            )

    return chunks
