from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from evaluate_rex_report_baseline_token_probs import build_runner


@dataclass(frozen=True)
class ReportRecord:
    sample_id: str
    image_path: Path
    image_text: str
    prompt: str
    reference: str


TOKEN_RE = re.compile(r"[a-z0-9]+")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_space(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split()).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def clean_prompt(prompt: str) -> str:
    prompt = prompt.replace("\r\n", "\n")
    if prompt.startswith("<image>\n"):
        return prompt.split("\n", 1)[1].strip()
    return prompt.replace("<image>", "").strip()


def load_records(benchmark_dir: Path, limit: int | None = None) -> list[ReportRecord]:
    data = json.loads((benchmark_dir / "vqa_report_llava.json").read_text(encoding="utf-8"))
    rows: list[ReportRecord] = []
    for item in data:
        conv = item.get("conversations", [])
        prompt = clean_prompt(conv[0].get("value", "") if conv else "")
        reference = normalize_space(conv[1].get("value", "") if len(conv) > 1 else "")
        image_text = str(item["image"])
        rows.append(
            ReportRecord(
                sample_id=str(item["id"]),
                image_path=Path(image_text),
                image_text=image_text,
                prompt=prompt,
                reference=reference,
            )
        )
    return rows[:limit] if limit else rows


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def corpus_bleu(preds: list[str], refs: list[str], max_n: int) -> float:
    pred_tokens = [tokenize(x) for x in preds]
    ref_tokens = [tokenize(x) for x in refs]
    pred_len = sum(len(x) for x in pred_tokens)
    ref_len = sum(len(x) for x in ref_tokens)
    if pred_len == 0 or ref_len == 0:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        clipped = 0
        total = 0
        for hyp, ref in zip(pred_tokens, ref_tokens):
            hyp_counts = ngrams(hyp, n)
            ref_counts = ngrams(ref, n)
            total += sum(hyp_counts.values())
            clipped += sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append((clipped + 1e-9) / (total + 1e-9))
    if any(p <= 0.0 for p in precisions):
        return 0.0
    bp = 1.0 if pred_len > ref_len else math.exp(1.0 - (ref_len / max(pred_len, 1)))
    return float(bp * math.exp(sum(math.log(p) for p in precisions) / max_n))


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        cur = [0]
        for j, tok_b in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if tok_a == tok_b else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_score(pred: str, ref: str) -> float:
    hyp = tokenize(pred)
    gold = tokenize(ref)
    if not hyp or not gold:
        return 0.0
    lcs = lcs_len(hyp, gold)
    prec = lcs / len(hyp)
    rec = lcs / len(gold)
    return float((2 * prec * rec) / (prec + rec)) if prec + rec else 0.0


def meteor_score(pred: str, ref: str) -> float:
    hyp = tokenize(pred)
    gold = tokenize(ref)
    if not hyp or not gold:
        return 0.0
    hyp_counts = Counter(hyp)
    gold_counts = Counter(gold)
    matches = sum(min(hyp_counts[tok], gold_counts[tok]) for tok in hyp_counts)
    if matches == 0:
        return 0.0
    precision = matches / len(hyp)
    recall = matches / len(gold)
    f_mean = (10 * precision * recall) / (recall + 9 * precision) if recall + 9 * precision else 0.0

    matched_positions = []
    used = set()
    for tok in hyp:
        for idx, gold_tok in enumerate(gold):
            if idx not in used and tok == gold_tok:
                used.add(idx)
                matched_positions.append(idx)
                break
    chunks = 1
    for prev, cur in zip(matched_positions, matched_positions[1:]):
        if cur != prev + 1:
            chunks += 1
    penalty = 0.5 * (chunks / matches) ** 3
    return float(f_mean * (1 - penalty))


def score_texts(preds: list[str], refs: list[str]) -> dict[str, float]:
    n = len(preds)
    rouge_l = sum(rouge_l_score(p, r) for p, r in zip(preds, refs)) / n if n else 0.0
    meteor = sum(meteor_score(p, r) for p, r in zip(preds, refs)) / n if n else 0.0
    return {
        "METEOR": float(meteor),
        "ROUGE_L": float(rouge_l),
        "BL_1": corpus_bleu(preds, refs, 1),
        "BL_4": corpus_bleu(preds, refs, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate official-report RRG benchmarks.")
    parser.add_argument("--model-kind", choices=["llava-med", "hulu-med-7b", "lingshu", "medgemma", "medgemma1.5", "radfm"], required=True)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--model-path", type=Path, required=False)
    parser.add_argument("--radfm-quick-demo-root", type=Path, default=Path("/path/to/RadFM/Quick_demo"))
    parser.add_argument("--radfm-checkpoint", type=Path, default=Path("/path/to/RadFM/pytorch_model.bin"))
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.model_kind != "radfm" and args.model_path is None:
        parser.error("--model-path is required unless --model-kind radfm")
    if args.model_label is None:
        args.model_label = args.model_kind
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ensure_dir(args.output_dir)
    records = load_records(args.benchmark_dir, limit=(args.limit if args.limit > 0 else None))
    runner = build_runner(args)
    pred_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(records, start=1):
        image = Image.open(rec.image_path).convert("RGB")
        start_time = time.perf_counter()
        text, _outputs, _generated = runner.generate_with_scores(image, rec.image_path, rec.prompt, int(args.max_new_tokens))
        elapsed_sec = time.perf_counter() - start_time
        pred = normalize_space(text)
        pred_rows.append(
            {
                "sample_id": rec.sample_id,
                "image": rec.image_text,
                "prompt": rec.prompt,
                "reference": rec.reference,
                "prediction": pred,
            }
        )
        sample_score = score_texts([pred], [rec.reference])
        sample_rows.append({"sample_id": rec.sample_id, **sample_score})
        runtime_rows.append(
            {
                "sample_id": rec.sample_id,
                "elapsed_sec": float(elapsed_sec),
            }
        )
        if idx % 10 == 0 or idx == len(records):
            print(f"[{args.model_label}] {args.benchmark_dir.name} RRG {idx}/{len(records)}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    preds = [row["prediction"] for row in pred_rows]
    refs = [row["reference"] for row in pred_rows]
    metrics = score_texts(preds, refs)
    summary = {
        "model_kind": args.model_kind,
        "model_label": args.model_label,
        "benchmark_dir": str(args.benchmark_dir),
        "total": len(pred_rows),
        "metrics": metrics,
        "metric_note": "METEOR is an exact-token METEOR implementation with fragmentation penalty; ROUGE-L is mean sample LCS-F1; BLEU is corpus BLEU.",
    }
    write_jsonl(args.output_dir / "report_answers.jsonl", pred_rows)
    write_jsonl(args.output_dir / "runtime_per_sample.jsonl", runtime_rows)
    valid_runtime = [row["elapsed_sec"] for row in runtime_rows if row.get("elapsed_sec") is not None]
    write_json(
        args.output_dir / "runtime_summary.json",
        {
            "model_kind": args.model_kind,
            "with_easylens": False,
            "total": len(runtime_rows),
            "valid": len(valid_runtime),
            "mean_sec_per_sample": float(sum(valid_runtime) / len(valid_runtime)) if valid_runtime else 0.0,
            "by_task": {
                "report": {
                    "n": len(valid_runtime),
                    "mean_sec_per_sample": float(sum(valid_runtime) / len(valid_runtime)) if valid_runtime else 0.0,
                }
            },
        },
    )
    with (args.output_dir / "sample_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "METEOR", "ROUGE_L", "BL_1", "BL_4"])
        writer.writeheader()
        writer.writerows(sample_rows)
    write_json(args.output_dir / "rrg_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
