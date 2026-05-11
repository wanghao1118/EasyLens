from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluate_normal_rrg import clean_prompt, normalize_space, score_texts


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_refs(benchmark_dir: Path) -> dict[str, dict[str, str]]:
    data = read_json(benchmark_dir / "vqa_report_llava.json")
    refs: dict[str, dict[str, str]] = {}
    for item in data:
        conv = item.get("conversations", [])
        prompt = clean_prompt(conv[0].get("value", "") if conv else "")
        reference = normalize_space(conv[1].get("value", "") if len(conv) > 1 else "")
        refs[str(item["id"])] = {"prompt": prompt, "reference": reference}
    return refs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-kind", required=True)
    parser.add_argument("--model-label", default=None)
    args = parser.parse_args()

    refs = load_refs(args.benchmark_dir)
    pred_rows = read_jsonl(args.predictions)
    paired: list[dict[str, Any]] = []
    for row in pred_rows:
        sample_id = str(row.get("sample_id") or row.get("id") or row.get("question_id"))
        if sample_id not in refs:
            continue
        pred = normalize_space(row.get("report_text") or row.get("prediction") or "")
        paired.append(
            {
                "sample_id": sample_id,
                "prompt": refs[sample_id]["prompt"],
                "reference": refs[sample_id]["reference"],
                "prediction": pred,
                "error": str(row.get("error") or ""),
            }
        )

    valid = [row for row in paired if not row["error"]]
    metrics = score_texts([row["prediction"] for row in valid], [row["reference"] for row in valid])
    summary = {
        "model_kind": args.model_kind,
        "model_label": args.model_label or args.model_kind,
        "benchmark_dir": str(args.benchmark_dir),
        "predictions_path": str(args.predictions),
        "total": len(paired),
        "valid": len(valid),
        "metrics": metrics,
        "metric_note": "RRG metrics computed with evaluate_normal_rrg.score_texts over generated report_text and benchmark references.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rrg_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "rrg_sample_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "METEOR", "ROUGE_L", "BL_1", "BL_4"])
        writer.writeheader()
        for row in valid:
            writer.writerow({"sample_id": row["sample_id"], **score_texts([row["prediction"]], [row["reference"]])})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
