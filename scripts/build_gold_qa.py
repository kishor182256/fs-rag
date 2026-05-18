from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import random
import re
from typing import Iterable


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\s+[\u2022|]\s+|\n+")
WS_RE = re.compile(r"\s+")

KEY_VALUE_RE = re.compile(r"\b([A-Za-z][A-Za-z &/()\-]{2,50})\s*:\s*([^.;]{2,90})")
DATE_RE = re.compile(r"\b((?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}|20\d{2})\b", re.IGNORECASE)
PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?%)\b")
EVENT_DATE_RE = re.compile(
    r"\b(?P<event>[A-Z][A-Za-z0-9,'()&\-\s]{20,150}?)\s(?:on|in|from)\s(?P<date>(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}|20\d{2})\b",
    re.IGNORECASE,
)
GOOD_EVENT_TERMS = {
    "launched",
    "announced",
    "released",
    "visited",
    "signed",
    "won",
    "awarded",
    "conducted",
    "commissioned",
    "observed",
    "celebrated",
    "established",
    "declared",
    "joined",
    "inaugurated",
    "approved",
    "held",
    "adopted",
    "issued",
}
ALLOWED_FACT_KEYS = {
    "capital",
    "currency",
    "president",
    "prime minister",
    "ceo",
    "headquarters",
    "founded",
    "launched",
    "launch date",
    "theme",
    "objective",
    "outlay",
    "coverage",
}


@dataclass
class QAItem:
    question: str
    answer: str
    question_type: str
    difficulty: str
    evidence_text: str
    source: dict


def _clean(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("|", " ")
    return WS_RE.sub(" ", text).strip()


def _sentence_candidates(chunk_text: str) -> list[str]:
    raw_parts = SENTENCE_SPLIT_RE.split(chunk_text)
    out: list[str] = []
    for part in raw_parts:
        s = _clean(part)
        if len(s) < 35 or len(s) > 260:
            continue
        if s.lower().startswith("ga capsule"):
            continue
        if sum(ch.isdigit() for ch in s) > 12:
            continue
        if s.count("-") > 8:
            continue
        if not s[0].isalpha():
            continue
        out.append(s)
    return out


def _build_from_key_value(sentence: str, source: dict) -> list[QAItem]:
    items: list[QAItem] = []
    for lhs, rhs in KEY_VALUE_RE.findall(sentence):
        lhs_c = _clean(lhs)
        rhs_c = _clean(rhs).strip("- ")
        if len(rhs_c.split()) > 12 or len(lhs_c.split()) > 10:
            continue
        lhs_n = lhs_c.lower().strip()
        if lhs_n not in ALLOWED_FACT_KEYS:
            continue
        if any(token in rhs_c.lower() for token in ["rank", "zone", "category", "table"]):
            continue

        q = f"What is the {lhs_n}?"
        if lhs_c.lower().startswith(("ceo", "president", "capital", "currency", "headquarters", "founded", "launched")):
            q = f"What is the {lhs_c}?"
        items.append(
            QAItem(
                question=q,
                answer=rhs_c,
                question_type="fact_recall",
                difficulty="easy",
                evidence_text=sentence,
                source=source,
            )
        )
    return items


def _subject_prefix(sentence: str, answer_text: str) -> str:
    idx = sentence.lower().find(answer_text.lower())
    prefix = sentence[:idx].strip(" ,:-") if idx > 0 else sentence
    prefix = re.sub(r"\b(in|on|from|at|to|and)$", "", prefix, flags=re.IGNORECASE).strip(" ,:-")
    return prefix


def _build_from_date(sentence: str, source: dict) -> list[QAItem]:
    items: list[QAItem] = []
    for match in EVENT_DATE_RE.finditer(sentence):
        event = _clean(match.group("event"))
        ans = _clean(match.group("date"))
        event_low = event.lower()
        if not any(term in event_low for term in GOOD_EVENT_TERMS):
            continue
        if len(event.split()) < 5 or len(event.split()) > 24:
            continue
        if event.endswith(("to", "from", "and")):
            continue
        q = f"When did this occur: {event}?"
        items.append(
            QAItem(
                question=q,
                answer=ans,
                question_type="date_recall",
                difficulty="medium",
                evidence_text=sentence,
                source=source,
            )
        )
    return items


def _build_from_percent(sentence: str, source: dict) -> list[QAItem]:
    items: list[QAItem] = []
    for match in PERCENT_RE.finditer(sentence):
        ans = match.group(1)
        subj = _subject_prefix(sentence, ans)
        if len(subj.split()) < 4:
            continue
        if "theme" in subj.lower():
            continue
        if any(bad in subj.lower() for bad in ["rank", "table", "zone", "category"]):
            continue
        q = f"What percentage is mentioned for: {subj}?"
        items.append(
            QAItem(
                question=q,
                answer=ans,
                question_type="numeric_recall",
                difficulty="medium",
                evidence_text=sentence,
                source=source,
            )
        )
    return items


def extract_qas_from_chunk(chunk: dict, doc_meta: dict) -> list[QAItem]:
    chunk_text = str(chunk.get("text", ""))
    source = {
        "doc_id": doc_meta["doc_id"],
        "source_file": doc_meta["source_file"],
        "chunk_id": chunk.get("chunk_id", ""),
        "page_start": chunk.get("page_start", 0),
        "page_end": chunk.get("page_end", 0),
    }
    out: list[QAItem] = []
    for sentence in _sentence_candidates(chunk_text):
        out.extend(_build_from_key_value(sentence, source))
        out.extend(_build_from_date(sentence, source))
        out.extend(_build_from_percent(sentence, source))
    return out


def _dedupe(items: Iterable[QAItem]) -> list[QAItem]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[QAItem] = []
    for item in items:
        q = _clean(item.question)
        a = _clean(item.answer)
        if len(a) < 2 or len(a) > 120:
            continue
        if len(q.split()) < 6 or len(q.split()) > 30:
            continue
        if "the following occur:" in q.lower() and len(q.split()) < 10:
            continue
        if q.lower().endswith(("to?", "from?", "and?")):
            continue
        if any(bad in q.lower() for bad in ["theme:", "rank |", "zone |", "category |"]):
            continue
        key = (q.lower(), a.lower())
        if key in seen:
            continue
        seen.add(key)
        item.question = q
        item.answer = a
        deduped.append(item)
    return deduped


def _quality_sort(items: list[QAItem]) -> list[QAItem]:
    def score(i: QAItem) -> tuple[int, int, int]:
        s = 0
        if i.question_type in {"date_recall", "numeric_recall", "fact_recall"}:
            s += 2
        if 3 <= len(i.answer.split()) <= 8:
            s += 1
        if any(c.isdigit() for c in i.answer):
            s += 1
        if any(w in i.question.lower() for w in ["what percentage", "when did", "what is"]):
            s += 1
        return (s, len(i.question), -len(i.answer))

    return sorted(items, key=score, reverse=True)


def build_dataset(input_dir: Path, output_file: Path, target: int, seed: int) -> dict:
    manifests = sorted(input_dir.glob("*.json"))
    all_items: list[QAItem] = []

    for manifest in manifests:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        doc_meta = {"doc_id": data.get("doc_id", ""), "source_file": data.get("source_file", "")}
        for chunk in data.get("chunks", []):
            all_items.extend(extract_qas_from_chunk(chunk, doc_meta))

    deduped = _dedupe(all_items)
    ranked = _quality_sort(deduped)

    if len(ranked) < target:
        target = len(ranked)

    # Keep a balanced mix across question types for exam-style usage.
    quotas = {
        "date_recall": int(target * 0.5),
        "numeric_recall": int(target * 0.3),
        "fact_recall": int(target * 0.2),
    }

    selected: list[QAItem] = []
    counts: dict[str, int] = {}
    for item in ranked:
        q_type = item.question_type
        limit = quotas.get(q_type, target)
        if counts.get(q_type, 0) >= limit:
            continue
        selected.append(item)
        counts[q_type] = counts.get(q_type, 0) + 1
        if len(selected) >= target:
            break

    if len(selected) < target:
        used_ids = {id(x) for x in selected}
        for item in ranked:
            if id(item) in used_ids:
                continue
            selected.append(item)
            if len(selected) >= target:
                break

    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, item in enumerate(selected, start=1):
        rows.append(
            {
                "id": f"qa_{idx:04d}",
                "question": item.question,
                "answer": item.answer,
                "question_type": item.question_type,
                "difficulty": item.difficulty,
                "evidence_text": item.evidence_text,
                "source": item.source,
            }
        )

    with output_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_manifests": len(manifests),
        "candidate_count": len(all_items),
        "deduped_count": len(deduped),
        "selected_count": len(rows),
        "output_file": str(output_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exam-style gold QA set from ingested manifests.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-file", type=Path, default=Path("datasets/gold_qa_set.jsonl"))
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = build_dataset(args.input_dir, args.output_file, args.target, args.seed)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
