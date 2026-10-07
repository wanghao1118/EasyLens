from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def coord_to_cell(x: int, y: int) -> str | None:
    if x < 1 or x > 3 or y < 1 or y > 3:
        return None
    return f"{chr(ord('A') + y - 1)}{x}"


def option_map_from_prompt(prompt: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for opt, x_text, y_text in re.findall(r"\b([A-D])\s*\.\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)", (prompt or "").upper()):
        cell = coord_to_cell(int(x_text), int(y_text))
        if cell:
            out[opt] = cell
    return out


def parse_cell_set(text: str, option_map: dict[str, str]) -> tuple[str, ...]:
    text_u = (text or "").upper()
    cells: list[str] = []
    for token in re.findall(r"\b([A-C][1-3])\b", text_u):
        cells.append(token)
    for x_text, y_text in re.findall(r"\(\s*([1-3])\s*,\s*([1-3])\s*\)", text_u):
        cell = coord_to_cell(int(x_text), int(y_text))
        if cell:
            cells.append(cell)
    if not cells:
        for opt in re.findall(r"\b([A-D])\b", text_u):
            cell = option_map.get(opt)
            if cell:
                cells.append(cell)
    return tuple(sorted(dict.fromkeys(cells)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-kind", default="unknown")
    parser.add_argument("--limit", type=int, default=0, help="Score the first N benchmark questions; 0 scores all.")
    args = parser.parse_args()

    questions = load_json(args.benchmark_dir / "vqa_multiselect_llava.json")
    if args.limit > 0:
        questions = questions[: args.limit]
    answers = load_json(args.benchmark_dir / "answers.json")
    answer_by_id = {str(row["id"]): str(row["answer"]).strip().upper() for row in answers}
    pred_by_id = {str(row.get("sample_id") or row.get("id") or row.get("question_id")): row for row in read_jsonl(args.predictions_path)}

    scored = []
    for q in questions:
        sid = str(q["id"])
        prompt = q.get("conversations", [{}])[0].get("value", "")
        opt_map = option_map_from_prompt(prompt)
        gold_option = answer_by_id.get(sid, "")
        gold_cells = tuple([opt_map[gold_option]]) if gold_option in opt_map else parse_cell_set(gold_option, opt_map)
        pred = pred_by_id.get(sid, {})
        answer_text = str(pred.get("text") or pred.get("answer") or "").strip()
        raw_text = str(pred.get("raw_text") or "").strip()
        pred_cells = parse_cell_set(answer_text, opt_map)
        if not pred_cells:
            pred_cells = parse_cell_set(raw_text, opt_map)
        exact = pred_cells == tuple(sorted(gold_cells))
        scored.append(
            {
                "sample_id": sid,
                "gold_answer": gold_option,
                "gold_cells": ",".join(gold_cells),
                "pred_text": answer_text,
                "raw_text": raw_text,
                "pred_cells": ",".join(pred_cells),
                "exact": exact,
                "empty_pred": not bool(pred_cells),
            }
        )

    total = len(scored)
    correct = sum(1 for row in scored if row["exact"])
    empty = sum(1 for row in scored if row["empty_pred"])
    summary = {
        "model_kind": args.model_kind,
        "benchmark_dir": str(args.benchmark_dir),
        "task": "multiselect",
        "limit": args.limit,
        "predictions_path": str(args.predictions_path),
        "metrics": {
            "overall": {
                "total": total,
                "exact_match_rate": correct / total if total else 0.0,
                "hit_any_rate": correct / total if total else 0.0,
                "empty_pred_rate": empty / total if total else 0.0,
                "mean_precision": correct / total if total else 0.0,
                "mean_recall": correct / total if total else 0.0,
                "mean_f1": correct / total if total else 0.0,
                "mean_iou": correct / total if total else 0.0,
            }
        },
        "scoring_source": "set-exact grid-cell parser over canonicalized prediction text",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "multiselect_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "multiselect_sample_scores.jsonl").open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary["metrics"]["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
