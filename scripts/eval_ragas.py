from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any
import warnings
import math


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Eval records not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _build_dataset_rows(records: list[dict[str, Any]]) -> dict[str, list[Any]]:
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for row in records:
        q = str(row.get("question", "")).strip()
        gt = str(row.get("ground_truth", "")).strip()
        ans = str(row.get("generated_answer", "")).strip()
        ctx = row.get("retrieved_contexts", [])
        if not q or not gt or not ans or not isinstance(ctx, list) or not ctx:
            continue
        questions.append(q)
        answers.append(ans)
        contexts.append([str(x) for x in ctx if str(x).strip()])
        ground_truths.append(gt)

    # Keep both legacy and newer RAGAS column names for compatibility.
    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    }


def _build_metrics(metric_names: list[str]) -> list[Any]:
    # Prefer explicit class instantiation, which is accepted by current RAGAS APIs.
    try:
        from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy
        registry = {
            "faithfulness": Faithfulness,
            "answer_relevancy": ResponseRelevancy,
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall,
        }
        return [registry[name]() for name in metric_names if name in registry]
    except Exception:
        pass

    # Fallback for versions exposing classes/instances from ragas.metrics.
    try:
        from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy
        registry = {
            "faithfulness": Faithfulness,
            "answer_relevancy": ResponseRelevancy,
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall,
        }
        return [registry[name]() for name in metric_names if name in registry]
    except Exception:
        pass

    # Final fallback for versions exposing metric instances directly.
    try:
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        registry = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        return [registry[name] for name in metric_names if name in registry]
    except Exception:
        pass

    raise RuntimeError("Could not import expected RAGAS metrics.")


def run_ragas(
    *,
    records_file: Path,
    output_file: Path,
    llm_model: str,
    embedding_model: str,
    base_url: str,
    api_key: str | None,
    limit: int | None,
    max_workers: int,
    timeout_seconds: int,
    max_retries: int,
    batch_size: int | None,
    metric_names: list[str],
    max_contexts: int,
    max_context_chars: int,
) -> dict[str, Any]:
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.run_config import RunConfig
    except Exception as exc:
        raise RuntimeError(
            "Missing eval dependencies. Install with: pip install -r requirements-eval.txt"
        ) from exc

    metrics = _build_metrics(metric_names)
    if not metrics:
        raise RuntimeError("No valid metrics selected.")

    records = _load_records(records_file)
    if limit:
        records = records[:limit]
    data = _build_dataset_rows(records)
    if not data["question"]:
        raise RuntimeError("No valid records for RAGAS evaluation.")

    # Trim contexts to reduce timeout pressure on local models.
    if max_contexts > 0:
        data["contexts"] = [ctxs[:max_contexts] for ctxs in data["contexts"]]
        data["retrieved_contexts"] = [ctxs[:max_contexts] for ctxs in data["retrieved_contexts"]]
    if max_context_chars > 0:
        data["contexts"] = [[c[:max_context_chars] for c in ctxs] for ctxs in data["contexts"]]
        data["retrieved_contexts"] = [[c[:max_context_chars] for c in ctxs] for ctxs in data["retrieved_contexts"]]

    dataset = Dataset.from_dict(data)
    api_key_final = api_key or os.getenv("OPENAI_API_KEY") or "dummy"
    llm = ChatOpenAI(
        model=llm_model,
        base_url=base_url,
        api_key=api_key_final,
        temperature=0,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )
    need_embeddings = any(name == "answer_relevancy" for name in metric_names)
    embeddings = (
        OpenAIEmbeddings(
            model=embedding_model,
            base_url=base_url,
            api_key=api_key_final,
            request_timeout=timeout_seconds,
            max_retries=max_retries,
        )
        if need_embeddings
        else None
    )

    run_config = RunConfig(
        timeout=timeout_seconds,
        max_retries=max_retries,
        max_workers=max_workers,
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        batch_size=batch_size,
        raise_exceptions=False,
    )

    metric_scores: dict[str, float] = {}
    metric_non_null_counts: dict[str, int] = {}

    # Preferred path for ragas 0.4.x: per-sample frame with metric columns.
    try:
        frame = result.to_pandas()
        if frame is not None and not frame.empty:
            for col in frame.columns:
                if col in {
                    "question",
                    "answer",
                    "contexts",
                    "ground_truth",
                    "user_input",
                    "response",
                    "retrieved_contexts",
                    "reference",
                }:
                    continue
                series = frame[col]
                numeric_values: list[float] = []
                for value in series.tolist():
                    if isinstance(value, (float, int)):
                        fv = float(value)
                        if not math.isnan(fv):
                            numeric_values.append(fv)
                if numeric_values:
                    metric_scores[col] = sum(numeric_values) / len(numeric_values)
                    metric_non_null_counts[col] = len(numeric_values)
    except Exception:
        pass

    # fallback: result object direct numeric fields
    if not metric_scores and hasattr(result, "__dict__"):
        for key, value in result.__dict__.items():
            if isinstance(value, (float, int)):
                metric_scores[key] = float(value)

    # fallback: mapping-like result
    if not metric_scores:
        try:
            raw = dict(result)
            for key, value in raw.items():
                if isinstance(value, (float, int)):
                    metric_scores[key] = float(value)
        except Exception:
            pass

    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "records_file": str(records_file),
        "num_records_used": len(data["question"]),
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "limit": limit,
        "max_workers": max_workers,
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
        "batch_size": batch_size,
        "scores": metric_scores,
        "metric_non_null_counts": metric_non_null_counts,
        "metrics_used": metric_names,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = argparse.ArgumentParser(description="Run RAGAS metrics on eval records.")
    parser.add_argument("--records-file", type=Path, default=Path("datasets/eval_records.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("datasets/ragas_report.json"))
    parser.add_argument("--llm-model", type=str, default="qwen2.5:7b")
    parser.add_argument("--embedding-model", type=str, default="nomic-embed-text")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--metrics",
        type=str,
        default="faithfulness,context_precision,context_recall,answer_relevancy",
        help="Comma-separated: faithfulness,context_precision,context_recall,answer_relevancy",
    )
    parser.add_argument("--max-contexts", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=700)
    args = parser.parse_args()
    metric_names = [x.strip() for x in args.metrics.split(",") if x.strip()]

    summary = run_ragas(
        records_file=args.records_file,
        output_file=args.output_file,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        base_url=args.base_url,
        api_key=args.api_key,
        limit=args.limit,
        max_workers=args.max_workers,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        batch_size=args.batch_size,
        metric_names=metric_names,
        max_contexts=args.max_contexts,
        max_context_chars=args.max_context_chars,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
