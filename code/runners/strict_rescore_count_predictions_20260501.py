#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gold_counts(benchmark_dir: Path) -> dict[str, int]:
    answers_path = benchmark_dir / "answers.json"
    if answers_path.exists():
        answers = load_json(answers_path)
        return {str(row["id"]): int(float(row["answer"])) for row in answers}

    manifest_path = benchmark_dir / "manifest.csv"
    if manifest_path.exists():
        out: dict[str, int] = {}
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                sample_id = str(row.get("sample_id") or row.get("id") or "")
                gold = row.get("gold_count")
                if sample_id and gold not in {None, ""}:
                    out[sample_id] = int(float(gold))
        return out

    raise FileNotFoundError(f"Neither answers.json nor manifest.csv exists in {benchmark_dir}")


def strict_parse_count(text: str, *, min_value: int = 0, max_value: int = 6) -> int | None:
    cleaned = (text or "").strip()
    cleaned = cleaned.strip("`'\" \t\r\n")
    match = re.fullmatch(r"[+-]?\d+", cleaned)
    if not match:
        return None
    value = int(cleaned)
    if value < min_value or value > max_value:
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-kind", default="unknown")
    parser.add_argument("--min-value", type=int, default=0)
    parser.add_argument("--max-value", type=int, default=6)
    args = parser.parse_args()

    questions = load_json(args.benchmark_dir / "vqa_count_llava.json")
    gold_by_id = load_gold_counts(args.benchmark_dir)
    pred_by_id: dict[str, dict] = {}
    for row in read_jsonl(args.predictions):
        sample_id = str(row.get("sample_id") or row.get("id") or row.get("question_id"))
        pred_by_id[sample_id] = row

    scored = []
    for q in questions:
        sample_id = str(q["id"])
        pred = pred_by_id.get(sample_id, {})
        text = str(pred.get("text") or pred.get("answer") or "").strip()
        parsed = strict_parse_count(text, min_value=args.min_value, max_value=args.max_value)
        gold = gold_by_id.get(sample_id)
        scored.append(
            {
                "sample_id": sample_id,
                "gold_count": gold,
                "pred_text": text,
                "parsed_count": parsed,
                "correct": parsed is not None and gold is not None and parsed == gold,
            }
        )

    total = len(scored)
    parsed_rows = [row for row in scored if row["parsed_count"] is not None]
    correct = sum(1 for row in scored if row["correct"])
    parsed_correct = sum(1 for row in parsed_rows if row["correct"])
    abs_err_sum = sum(abs(int(row["parsed_count"]) - int(row["gold_count"])) for row in parsed_rows)
    within_one = sum(abs(int(row["parsed_count"]) - int(row["gold_count"])) <= 1 for row in parsed_rows)
    summary = {
        "model_kind": args.model_kind,
        "benchmark_dir": str(args.benchmark_dir),
        "task": "count",
        "predictions_path": str(args.predictions),
        "metrics": {
            "overall": {
                "total": total,
                "parsed": len(parsed_rows),
                "unparsed": total - len(parsed_rows),
                "parsed_rate": (len(parsed_rows) / total) if total else 0.0,
                "accuracy": (correct / total) if total else 0.0,
                "parsed_accuracy": (parsed_correct / len(parsed_rows)) if parsed_rows else 0.0,
                "mean_abs_error": (abs_err_sum / len(parsed_rows)) if parsed_rows else 0.0,
                "within_one_accuracy": (within_one / len(parsed_rows)) if parsed_rows else 0.0,
            }
        },
        "scoring_source": "strict full-output integer parser",
        "strict_count_parser": {
            "accepted_regex": r"^[+-]?\d+$",
            "min_value": args.min_value,
            "max_value": args.max_value,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "count_summary_strict.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "count_sample_scores_strict.jsonl").open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
