from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sys
from typing import Any

import httpx

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import settings

WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return WS_RE.sub(" ", text).strip()


def _extract_output_text(payload: dict[str, Any]) -> str:
    out = payload.get("output_text")
    if isinstance(out, str) and out.strip():
        return out.strip()

    parts: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text", "")
                if text:
                    parts.append(text)
    return "\n".join(parts).strip()


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _chunk_quality_score(chunk_text: str) -> int:
    score = 0
    if len(chunk_text) > 300:
        score += 2
    if any(token in chunk_text for token in ["2025", "2026", "%", "launched", "announced", "awarded", "capital", "currency"]):
        score += 3
    if chunk_text.count(".") >= 2:
        score += 2
    if sum(ch.isdigit() for ch in chunk_text) >= 5:
        score += 1
    return score


def _prepare_chunks(input_dir: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for manifest in sorted(input_dir.glob("*.json")):
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        doc_id = str(doc.get("doc_id", ""))
        source_file = str(doc.get("source_file", ""))
        for chunk in doc.get("chunks", []):
            text = _clean(str(chunk.get("text", "")))
            if len(text) < 160:
                continue
            chunks.append(
                {
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "chunk_id": str(chunk.get("chunk_id", "")),
                    "page_start": int(chunk.get("page_start", 0) or 0),
                    "page_end": int(chunk.get("page_end", 0) or 0),
                    "text": text,
                    "score": _chunk_quality_score(text),
                }
            )
    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks


async def _generate_for_chunk(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = (
        "Generate up to 2 high-quality exam-style QA pairs from the context. "
        "Rules: return JSON array only. No markdown. "
        "Each item must have keys: question, answer, question_type, difficulty, evidence_text. "
        "question_type must be one of: date_recall, fact_recall, numeric_recall. "
        "difficulty must be one of: easy, medium, hard. "
        "Questions must be clear, standalone, and factual. "
        "Answer must be concise. "
        "evidence_text must be an exact short excerpt from context. "
        "If no good QA can be formed, return [] only.\n\n"
        f"Context:\n{chunk['text'][:1800]}"
    )

    payload = {
        "model": settings.openai_model,
        "instructions": "You create clean gold QA datasets for exam prep.",
        "input": [{"role": "user", "content": prompt}],
    }

    endpoint = settings.openai_base_url.rstrip("/") + "/responses"
    headers = {"Content-Type": "application/json"}
    if settings.openai_api_key:
        headers["Authorization"] = f"Bearer {settings.openai_api_key}"

    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        response = await client.post(endpoint, headers=headers, content=json.dumps(payload))
        response.raise_for_status()
        out_text = _extract_output_text(response.json())

    return _extract_json_array(out_text)


def _load_existing_rows(output_file: Path) -> list[dict[str, Any]]:
    if not output_file.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in output_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _write_rows(output_file: Path, rows: list[dict[str, Any]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_state(state_file: Path) -> dict[str, Any]:
    if not state_file.exists():
        return {"next_chunk_index": 0, "processed_chunk_ids": []}
    try:
        parsed = json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"next_chunk_index": 0, "processed_chunk_ids": []}
    if not isinstance(parsed, dict):
        return {"next_chunk_index": 0, "processed_chunk_ids": []}
    parsed.setdefault("next_chunk_index", 0)
    parsed.setdefault("processed_chunk_ids", [])
    return parsed


def _write_state(state_file: Path, state: dict[str, Any]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")


def _is_valid_item(item: dict[str, Any], chunk_text: str) -> bool:
    q = _clean(str(item.get("question", "")))
    a = _clean(str(item.get("answer", "")))
    q_type = _clean(str(item.get("question_type", "")))
    diff = _clean(str(item.get("difficulty", "")))
    ev = _clean(str(item.get("evidence_text", "")))

    if q_type not in {"date_recall", "fact_recall", "numeric_recall"}:
        return False
    if diff not in {"easy", "medium", "hard"}:
        return False
    if len(q.split()) < 6 or len(q.split()) > 30:
        return False
    if len(a.split()) < 1 or len(a.split()) > 18:
        return False
    if len(ev) < 25:
        return False
    if ev.lower() not in chunk_text.lower():
        return False
    return True


async def build_dataset(
    input_dir: Path,
    output_file: Path,
    target: int,
    *,
    batch_size: int,
    resume: bool,
    fresh: bool,
    state_file: Path,
) -> dict[str, Any]:
    chunks = _prepare_chunks(input_dir)
    if fresh:
        output_file.unlink(missing_ok=True)
        state_file.unlink(missing_ok=True)

    selected: list[dict[str, Any]] = _load_existing_rows(output_file) if resume and not fresh else []
    seen: set[tuple[str, str]] = {
        (_clean(str(item.get("question", ""))).lower(), _clean(str(item.get("answer", ""))).lower())
        for item in selected
    }

    state = _load_state(state_file) if resume and not fresh else {"next_chunk_index": 0, "processed_chunk_ids": []}
    next_chunk_index = int(state.get("next_chunk_index", 0) or 0)
    processed_chunk_ids = set(str(x) for x in state.get("processed_chunk_ids", []))

    processed_this_run = 0
    generated_this_run = 0

    for idx in range(next_chunk_index, len(chunks)):
        chunk = chunks[idx]
        if len(selected) >= target:
            break
        if processed_this_run >= batch_size:
            break
        if chunk["chunk_id"] in processed_chunk_ids:
            continue

        processed_this_run += 1
        try:
            generated = await _generate_for_chunk(chunk)
        except Exception:
            processed_chunk_ids.add(chunk["chunk_id"])
            state["next_chunk_index"] = idx + 1
            state["processed_chunk_ids"] = sorted(processed_chunk_ids)
            _write_state(state_file, state)
            continue

        for item in generated:
            if len(selected) >= target:
                break
            if not _is_valid_item(item, chunk["text"]):
                continue

            question = _clean(str(item["question"]))
            answer = _clean(str(item["answer"]))
            key = (question.lower(), answer.lower())
            if key in seen:
                continue
            seen.add(key)

            selected.append(
                {
                    "id": f"qa_{len(selected)+1:04d}",
                    "question": question,
                    "answer": answer,
                    "question_type": _clean(str(item["question_type"])),
                    "difficulty": _clean(str(item["difficulty"])),
                    "evidence_text": _clean(str(item["evidence_text"])),
                    "source": {
                        "doc_id": chunk["doc_id"],
                        "source_file": chunk["source_file"],
                        "chunk_id": chunk["chunk_id"],
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                    },
                }
            )
            generated_this_run += 1

        processed_chunk_ids.add(chunk["chunk_id"])
        state["next_chunk_index"] = idx + 1
        state["processed_chunk_ids"] = sorted(processed_chunk_ids)
        _write_rows(output_file, selected)
        _write_state(state_file, state)

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_chunks": len(chunks),
        "selected_count": len(selected),
        "generated_this_run": generated_this_run,
        "processed_chunks_this_run": processed_this_run,
        "resume_mode": resume,
        "fresh_mode": fresh,
        "batch_size": batch_size,
        "next_chunk_index": state.get("next_chunk_index", 0),
        "target": target,
        "output_file": str(output_file),
        "state_file": str(state_file),
    }


async def _amain(args: argparse.Namespace) -> None:
    state_file = args.state_file or args.output_file.with_suffix(".state.json")
    summary = await build_dataset(
        args.input_dir,
        args.output_file,
        args.target,
        batch_size=args.batch_size,
        resume=args.resume,
        fresh=args.fresh,
        state_file=state_file,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gold QA set using LLM from ingested manifests.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-file", type=Path, default=Path("datasets/gold_qa_set_llm_120.jsonl"))
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output/state files.")
    parser.add_argument("--fresh", action="store_true", help="Start from scratch and overwrite output/state.")
    parser.add_argument("--state-file", type=Path, default=None, help="Checkpoint state file path.")
    args = parser.parse_args()

    import asyncio

    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
