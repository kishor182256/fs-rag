from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import httpx


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Gold set not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _pick_context(hit: dict[str, Any]) -> str:
    text = hit.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    snippet = hit.get("snippet", "")
    return snippet.strip() if isinstance(snippet, str) else ""


async def _fetch_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    base_url: str,
    row: dict[str, Any],
    top_k: int,
    use_llm: bool,
    use_vector: bool,
    response_mode: str,
) -> dict[str, Any]:
    payload = {
        "query": row["question"],
        "top_k": top_k,
        "use_llm": use_llm,
        "use_vector": use_vector,
        "response_mode": response_mode,
        "include_full_text": False,
    }

    async with semaphore:
        response = await client.post(base_url, json=payload)
        response.raise_for_status()
        result = response.json()

    hits = result.get("hits", [])
    contexts = [_pick_context(hit) for hit in hits if _pick_context(hit)]
    answer = (result.get("answer") or "").strip()
    if not answer and contexts:
        answer = contexts[0]

    return {
        "id": row.get("id", ""),
        "question": row["question"],
        "ground_truth": row["answer"],
        "ground_truth_source": row.get("source", {}),
        "generated_answer": answer,
        "answer_status": result.get("answer_status", "disabled"),
        "vector_status": result.get("vector_status", "disabled"),
        "retrieved_contexts": contexts,
        "retrieved_chunk_ids": [hit.get("chunk_id", "") for hit in hits],
        "retrieved_sources": [
            {
                "source_file": hit.get("source_file", ""),
                "page_start": hit.get("page_start", 0),
                "page_end": hit.get("page_end", 0),
            }
            for hit in hits
        ],
    }


async def build_eval_records(
    *,
    gold_file: Path,
    output_file: Path,
    query_url: str,
    top_k: int,
    use_llm: bool,
    use_vector: bool,
    response_mode: str,
    timeout_seconds: int,
    concurrency: int,
    limit: int | None,
) -> dict[str, Any]:
    rows = _load_jsonl(gold_file)
    if limit:
        rows = rows[:limit]

    semaphore = asyncio.Semaphore(max(1, concurrency))
    timeout = httpx.Timeout(timeout_seconds)

    records: list[dict[str, Any]] = []
    failures = 0
    error_samples: list[str] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            _fetch_one(
                client=client,
                semaphore=semaphore,
                base_url=query_url,
                row=row,
                top_k=top_k,
                use_llm=use_llm,
                use_vector=use_vector,
                response_mode=response_mode,
            )
            for row in rows
        ]
        for future in asyncio.as_completed(tasks):
            try:
                records.append(await future)
            except Exception as exc:
                failures += 1
                if len(error_samples) < 5:
                    error_samples.append(f"{type(exc).__name__}: {str(exc)}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_gold_rows": len(rows),
        "records_written": len(records),
        "failures": failures,
        "query_url": query_url,
        "top_k": top_k,
        "use_llm": use_llm,
        "use_vector": use_vector,
        "output_file": str(output_file),
        "error_samples": error_samples,
    }
    summary_file = output_file.with_suffix(".summary.json")
    summary_file.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG eval records from a gold QA set.")
    parser.add_argument("--gold-file", type=Path, default=Path("datasets/gold_qa_set_150.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("datasets/eval_records.jsonl"))
    parser.add_argument("--query-url", type=str, default="http://localhost:8000/v1/query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-llm", action="store_true", default=False)
    parser.add_argument("--use-vector", action="store_true", default=True)
    parser.add_argument("--response-mode", type=str, default="balanced")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summary = asyncio.run(
        build_eval_records(
            gold_file=args.gold_file,
            output_file=args.output_file,
            query_url=args.query_url,
            top_k=args.top_k,
            use_llm=args.use_llm,
            use_vector=args.use_vector,
            response_mode=args.response_mode,
            timeout_seconds=args.timeout_seconds,
            concurrency=args.concurrency,
            limit=args.limit,
        )
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
