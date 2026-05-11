from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable


TEXT_FIELDS = [
    "report_text",
    "pred_report_text",
    "prediction",
    "pred_text",
    "parsed_text",
    "answer_text",
    "generated_text",
    "answer",
    "text",
    "output",
    "response",
    "raw_text",
]

SIDE_TERMS = {
    "left": re.compile(r"\bleft\b"),
    "right": re.compile(r"\bright\b"),
}

LOBE_PATTERNS = {
    "upper": re.compile(r"\bupper lobe(?:s)?\b"),
    "middle": re.compile(r"\bmiddle lobe\b"),
    "lower": re.compile(r"\blower lobe(?:s)?\b"),
}

SMALLNESS_PATTERNS = [
    re.compile(r"\btiny\b"),
    re.compile(r"\bsmall\b"),
    re.compile(r"\bmicro\b"),
    re.compile(r"\bsub-?centi(?:meter|metre)\b"),
    re.compile(r"\bsubcenti(?:meter|metre)\b"),
    re.compile(r"\bsub-?cm\b"),
]

CONFOUND_PATTERNS = [
    re.compile(r"\badditional abnormal"),
    re.compile(r"\bother abnormal"),
    re.compile(r"\bconcurrent abnormal"),
    re.compile(r"\blimited by"),
    re.compile(r"\bcomplicat"),
    re.compile(r"\bin addition to"),
    re.compile(r"\bbeyond the target"),
]

MULTIPLE_PATTERNS = [
    re.compile(r"\bmultiple\b"),
    re.compile(r"\bseveral\b"),
    re.compile(r"\bfew\b"),
    re.compile(r"\bbilateral\b"),
    re.compile(r"\bboth lungs?\b"),
    re.compile(r"\bnodules\b"),
    re.compile(r"\blesions\b"),
    re.compile(r"\bopacities\b"),
    re.compile(r"\bdensities\b"),
]

SINGLE_PATTERNS = [
    re.compile(r"\bsingle\b"),
    re.compile(r"\bone\b"),
    re.compile(r"\ba pulmonary nodule\b"),
    re.compile(r"\ba nonspecific pulmonary nodule\b"),
    re.compile(r"\ba subpleural nodule\b"),
    re.compile(r"\ba visible micro pulmonary lesion\b"),
    re.compile(r"\bvisible micro pulmonary lesion\b"),
]

LESION_TERMS = [
    re.compile(r"\bpulmonary nodule(?:s)?\b"),
    re.compile(r"\bnodule(?:s)?\b"),
    re.compile(r"\bnodular\b"),
    re.compile(r"\bpulmonary lesion(?:s)?\b"),
    re.compile(r"\blesion(?:s)?\b"),
    re.compile(r"\bnodular opacit(?:y|ies)\b"),
    re.compile(r"\bnodular densit(?:y|ies)\b"),
]

DENIAL_TERMS = [
    re.compile(r"\bno\b"),
    re.compile(r"\bwithout\b"),
    re.compile(r"\babsent\b"),
    re.compile(r"\bnegative for\b"),
    re.compile(r"\bnot seen\b"),
    re.compile(r"\bnot identified\b"),
    re.compile(r"\bnone identified\b"),
    re.compile(r"\bnone seen\b"),
    re.compile(r"\bnone detected\b"),
    re.compile(r"\bno visible\b"),
    re.compile(r"\bno definite\b"),
]

WEIGHTS = {
    "lesion_mention_hit": 0.35,
    "smallness_hit": 0.20,
    "laterality_match": 0.15,
    "lobe_match": 0.10,
    "multiplicity_match": 0.10,
    "context_consistency_hit": 0.10,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def clean_text(text: str | None) -> str:
    return " ".join(str(text or "").strip().split())


def lower_text(text: str | None) -> str:
    return clean_text(text).lower()


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_text(value)
    return text.lower() in {"true", "1", "yes"}


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"[\n.;]+", lower_text(text))
    return [piece.strip() for piece in pieces if piece.strip()]


def has_lesion_term(text: str) -> bool:
    return any(pattern.search(text) for pattern in LESION_TERMS)


def has_denial_term(text: str) -> bool:
    return any(pattern.search(text) for pattern in DENIAL_TERMS)


def detect_lesion_mention(text: str) -> tuple[bool, bool]:
    positive = False
    explicit_denial = False
    for sentence in split_sentences(text):
        if not has_lesion_term(sentence):
            continue
        if has_denial_term(sentence):
            explicit_denial = True
            continue
        positive = True
    return positive, explicit_denial and not positive


def extract_mm_sizes(text: str) -> list[float]:
    sizes: list[float] = []
    for raw in re.findall(r"\b([0-9]+(?:\.[0-9]+)?)\s*mm\b", lower_text(text)):
        try:
            sizes.append(float(raw))
        except ValueError:
            continue
    return sizes


def detect_smallness(text: str) -> bool:
    lowered = lower_text(text)
    if any(pattern.search(lowered) for pattern in SMALLNESS_PATTERNS):
        return True
    sizes = extract_mm_sizes(lowered)
    return any(size <= 10.0 for size in sizes)


def detect_sides(text: str) -> set[str]:
    lowered = lower_text(text)
    if "bilateral" in lowered or "both lungs" in lowered or "both lung" in lowered:
        return {"left", "right"}
    return {name for name, pattern in SIDE_TERMS.items() if pattern.search(lowered)}


def detect_lobes(text: str) -> set[str]:
    lowered = lower_text(text)
    return {name for name, pattern in LOBE_PATTERNS.items() if pattern.search(lowered)}


def detect_multiplicity(text: str) -> str | None:
    lowered = lower_text(text)
    if any(pattern.search(lowered) for pattern in MULTIPLE_PATTERNS):
        return "multiple"
    if any(pattern.search(lowered) for pattern in SINGLE_PATTERNS):
        return "single"
    if re.search(r"\bnodule\b", lowered) or re.search(r"\blesion\b", lowered):
        return "single"
    return None


def detect_context_signal(text: str) -> bool:
    lowered = lower_text(text)
    return any(pattern.search(lowered) for pattern in CONFOUND_PATTERNS)


def choose_text_field(row: dict, preferred: str | None) -> tuple[str, str]:
    if preferred:
        return preferred, clean_text(row.get(preferred, ""))
    for field in TEXT_FIELDS:
        value = clean_text(row.get(field, ""))
        if value:
            return field, value
    return "missing", ""


def load_predictions(path: Path, preferred_field: str | None) -> tuple[dict[str, str], str]:
    rows: dict[str, str] = {}
    selected_field = preferred_field or ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = clean_text(row.get("sample_id") or row.get("id"))
            if not sample_id:
                continue
            field_name, text = choose_text_field(row, preferred_field)
            if not selected_field and field_name != "missing":
                selected_field = field_name
            rows[sample_id] = text
    return rows, selected_field or "missing"


def compute_micro_score(row: dict) -> float:
    weighted = 0.0
    total = 0.0
    for key, weight in WEIGHTS.items():
        value = row.get(key)
        if value is None:
            continue
        total += weight
        weighted += weight if bool(value) else 0.0
    if total <= 0.0:
        return 0.0
    return weighted / total


def evaluate_pair(gold_row: dict[str, str], pred_text: str) -> dict:
    gold_text = clean_text(gold_row.get("report_text", ""))
    pred_text = clean_text(pred_text)

    gold_has_lesion, _ = detect_lesion_mention(gold_text)
    pred_has_lesion, pred_denial = detect_lesion_mention(pred_text)

    gold_small = detect_smallness(gold_text)
    pred_small = detect_smallness(pred_text)

    gold_sides = sorted(detect_sides(gold_text))
    pred_sides = sorted(detect_sides(pred_text))

    gold_lobes = sorted(detect_lobes(gold_text))
    pred_lobes = sorted(detect_lobes(pred_text))

    gold_multi = detect_multiplicity(gold_text)
    pred_multi = detect_multiplicity(pred_text)

    gold_context_flag = parse_bool(gold_row.get("context_flag"))
    pred_context_signal = detect_context_signal(pred_text)

    lesion_mention_hit = pred_has_lesion and not pred_denial
    smallness_hit = lesion_mention_hit and (pred_small if gold_small else True)

    laterality_applicable = bool(gold_sides)
    laterality_match = lesion_mention_hit and pred_sides == gold_sides if laterality_applicable else None

    lobe_applicable = bool(gold_lobes)
    lobe_match = lesion_mention_hit and pred_lobes == gold_lobes if lobe_applicable else None

    multiplicity_applicable = gold_multi is not None
    multiplicity_match = lesion_mention_hit and pred_multi == gold_multi if multiplicity_applicable else None

    location_parts = [value for value in [laterality_match, lobe_match] if value is not None]
    full_location_match = all(location_parts) if location_parts else None

    context_consistency_hit = lesion_mention_hit and (pred_context_signal == gold_context_flag)

    row = {
        "sample_id": gold_row["sample_id"],
        "source_sample_id": clean_text(gold_row.get("source_sample_id", "")),
        "clean": parse_bool(gold_row.get("clean")),
        "context_flag": gold_context_flag,
        "gold_report_text": gold_text,
        "pred_report_text": pred_text,
        "gold_has_lesion": gold_has_lesion,
        "pred_has_lesion": pred_has_lesion,
        "pred_explicit_lesion_denial": pred_denial,
        "gold_smallness": gold_small,
        "pred_smallness": pred_small,
        "gold_sizes_mm": extract_mm_sizes(gold_text),
        "pred_sizes_mm": extract_mm_sizes(pred_text),
        "gold_sides": gold_sides,
        "pred_sides": pred_sides,
        "gold_lobes": gold_lobes,
        "pred_lobes": pred_lobes,
        "gold_multiplicity": gold_multi,
        "pred_multiplicity": pred_multi,
        "pred_context_signal": pred_context_signal,
        "lesion_mention_hit": lesion_mention_hit,
        "smallness_hit": smallness_hit,
        "core_micro_hit": lesion_mention_hit and smallness_hit,
        "laterality_match": laterality_match,
        "lobe_match": lobe_match,
        "full_location_match": full_location_match,
        "multiplicity_match": multiplicity_match,
        "context_consistency_hit": context_consistency_hit,
        "context_signal_hallucination": (not gold_context_flag) and pred_context_signal,
        "context_recall_hit": pred_context_signal if gold_context_flag else None,
        "strict_perception_hit": None,
    }
    strict_values = [
        row["lesion_mention_hit"],
        row["smallness_hit"],
        row["context_consistency_hit"],
    ]
    strict_values.extend(value for value in [laterality_match, lobe_match, multiplicity_match] if value is not None)
    row["strict_perception_hit"] = all(strict_values)
    row["micro_perception_score"] = round(compute_micro_score(row), 6)
    return row


def summarize(rows: list[dict]) -> dict:
    def _rate(key: str) -> tuple[int, int, float | None]:
        applicable = [row[key] for row in rows if row.get(key) is not None]
        if not applicable:
            return 0, 0, None
        hits = sum(1 for value in applicable if bool(value))
        return hits, len(applicable), hits / len(applicable)

    def _mean(key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            return None
        return sum(values) / len(values)

    lesion_hits, lesion_total, lesion_rate = _rate("lesion_mention_hit")
    small_hits, small_total, small_rate = _rate("smallness_hit")
    core_hits, core_total, core_rate = _rate("core_micro_hit")
    lat_hits, lat_total, lat_rate = _rate("laterality_match")
    lobe_hits, lobe_total, lobe_rate = _rate("lobe_match")
    loc_hits, loc_total, loc_rate = _rate("full_location_match")
    mult_hits, mult_total, mult_rate = _rate("multiplicity_match")
    ctx_hits, ctx_total, ctx_rate = _rate("context_consistency_hit")
    strict_hits, strict_total, strict_rate = _rate("strict_perception_hit")
    denial_hits, denial_total, denial_rate = _rate("pred_explicit_lesion_denial")
    ctx_recall_hits, ctx_recall_total, ctx_recall_rate = _rate("context_recall_hit")
    halluc_hits, halluc_total, halluc_rate = _rate("context_signal_hallucination")

    return {
        "total": len(rows),
        "lesion_mention": {"hits": lesion_hits, "applicable": lesion_total, "rate": lesion_rate},
        "smallness": {"hits": small_hits, "applicable": small_total, "rate": small_rate},
        "core_micro": {"hits": core_hits, "applicable": core_total, "rate": core_rate},
        "laterality_match": {"hits": lat_hits, "applicable": lat_total, "rate": lat_rate},
        "lobe_match": {"hits": lobe_hits, "applicable": lobe_total, "rate": lobe_rate},
        "full_location_match": {"hits": loc_hits, "applicable": loc_total, "rate": loc_rate},
        "multiplicity_match": {"hits": mult_hits, "applicable": mult_total, "rate": mult_rate},
        "context_consistency": {"hits": ctx_hits, "applicable": ctx_total, "rate": ctx_rate},
        "strict_perception": {"hits": strict_hits, "applicable": strict_total, "rate": strict_rate},
        "context_recall": {"hits": ctx_recall_hits, "applicable": ctx_recall_total, "rate": ctx_recall_rate},
        "context_signal_hallucination": {"hits": halluc_hits, "applicable": halluc_total, "rate": halluc_rate},
        "explicit_lesion_denial": {"hits": denial_hits, "applicable": denial_total, "rate": denial_rate},
        "micro_perception_score_mean": _mean("micro_perception_score"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate diagnostic reports for micro-lesion perception on the ReX manual-authored report set."
    )
    parser.add_argument("--benchmark-dir", required=True, help="Directory containing manifest.csv with gold report_text.")
    parser.add_argument("--predictions-jsonl", required=True, help="JSONL with sample_id and generated report text.")
    parser.add_argument("--output-dir", required=True, help="Where to write per-sample and summary outputs.")
    parser.add_argument(
        "--report-field",
        default=None,
        help="Optional field name to read from the predictions JSONL. Defaults to the first populated known text field.",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    predictions_jsonl = Path(args.predictions_jsonl)
    output_dir = Path(args.output_dir)

    gold_rows = load_manifest(benchmark_dir / "manifest.csv")
    pred_map, selected_field = load_predictions(predictions_jsonl, args.report_field)

    per_sample: list[dict] = []
    missing_predictions = 0
    for gold_row in gold_rows:
        sample_id = gold_row["sample_id"]
        pred_text = pred_map.get(sample_id, "")
        if not clean_text(pred_text):
            missing_predictions += 1
        per_sample.append(evaluate_pair(gold_row, pred_text))

    clean_rows = [row for row in per_sample if bool(row["clean"])]
    context_rows = [row for row in per_sample if bool(row["context_flag"])]

    summary = {
        "benchmark_dir": str(benchmark_dir),
        "predictions_jsonl": str(predictions_jsonl),
        "prediction_report_field": selected_field,
        "missing_predictions": missing_predictions,
        "weights": WEIGHTS,
        "metrics": {
            "overall": summarize(per_sample),
            "clean": summarize(clean_rows),
            "context_flag": summarize(context_rows),
        },
    }

    write_jsonl(output_dir / "report_micro_perception_per_sample.jsonl", per_sample)
    write_json(output_dir / "report_micro_perception_summary.json", summary)


if __name__ == "__main__":
    main()
