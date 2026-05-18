from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean
from typing import Any


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


def run_deepeval(
    *,
    records_file: Path,
    output_file: Path,
    min_faithfulness: float,
    min_answer_relevancy: float,
    min_context_precision: float,
    min_context_recall: float,
) -> dict[str, Any]:
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except Exception as exc:
        raise RuntimeError(
            "Missing eval dependencies. Install with: pip install -r requirements-eval.txt"
        ) from exc

    records = _load_records(records_file)
    if not records:
        raise RuntimeError("No eval records found.")

    metrics = {
        "faithfulness": FaithfulnessMetric(threshold=min_faithfulness, include_reason=True),
        "answer_relevancy": AnswerRelevancyMetric(threshold=min_answer_relevancy, include_reason=True),
        "contextual_precision": ContextualPrecisionMetric(threshold=min_context_precision, include_reason=True),
        "contextual_recall": ContextualRecallMetric(threshold=min_context_recall, include_reason=True),
    }

    per_case: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = {name: [] for name in metrics}

    for row in records:
        question = str(row.get("question", "")).strip()
        ground_truth = str(row.get("ground_truth", "")).strip()
        answer = str(row.get("generated_answer", "")).strip()
        contexts = [str(x) for x in row.get("retrieved_contexts", []) if str(x).strip()]
        if not question or not answer or not contexts:
            continue

        case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth,
            retrieval_context=contexts,
        )

        metric_scores: dict[str, float] = {}
        metric_reasons: dict[str, str] = {}
        for name, metric in metrics.items():
            try:
                metric.measure(case)
                score = float(metric.score or 0.0)
                reason = str(metric.reason or "")
            except Exception:
                score = 0.0
                reason = "metric_error"
            metric_values[name].append(score)
            metric_scores[name] = score
            metric_reasons[name] = reason

        per_case.append(
            {
                "id": row.get("id", ""),
                "question": question,
                "scores": metric_scores,
                "reasons": metric_reasons,
            }
        )

    averages = {
        name: (mean(values) if values else 0.0)
        for name, values in metric_values.items()
    }

    gates = {
        "faithfulness": averages["faithfulness"] >= min_faithfulness,
        "answer_relevancy": averages["answer_relevancy"] >= min_answer_relevancy,
        "contextual_precision": averages["contextual_precision"] >= min_context_precision,
        "contextual_recall": averages["contextual_recall"] >= min_context_recall,
    }
    ci_pass = all(gates.values())

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "records_file": str(records_file),
        "num_cases_evaluated": len(per_case),
        "averages": averages,
        "thresholds": {
            "faithfulness": min_faithfulness,
            "answer_relevancy": min_answer_relevancy,
            "contextual_precision": min_context_precision,
            "contextual_recall": min_context_recall,
        },
        "gates": gates,
        "ci_pass": ci_pass,
        "per_case": per_case,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepEval metrics with CI gates.")
    parser.add_argument("--records-file", type=Path, default=Path("datasets/eval_records.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("datasets/deepeval_report.json"))
    parser.add_argument("--min-faithfulness", type=float, default=0.80)
    parser.add_argument("--min-answer-relevancy", type=float, default=0.75)
    parser.add_argument("--min-context-precision", type=float, default=0.70)
    parser.add_argument("--min-context-recall", type=float, default=0.70)
    parser.add_argument("--fail-on-gate", action="store_true", default=False)
    args = parser.parse_args()

    report = run_deepeval(
        records_file=args.records_file,
        output_file=args.output_file,
        min_faithfulness=args.min_faithfulness,
        min_answer_relevancy=args.min_answer_relevancy,
        min_context_precision=args.min_context_precision,
        min_context_recall=args.min_context_recall,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "per_case"}, ensure_ascii=True, indent=2))
    if args.fail_on_gate and not report["ci_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
