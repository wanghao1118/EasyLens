from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LLAVA_ENV_PY = "/path/to/llava-med/bin/python"
LLAVA_ROOT = "/path/to/LLaVA-Med"
HULU_DEFAULT_MODEL = "/path/to/Hulu-Med-7B"
MEDGEMMA_DEFAULT_MODEL = "/path/to/medgemma-1.5-4b-it"
RADFM_QUICK_DEMO_ROOT = "/path/to/RadFM/Quick_demo"
RADFM_ENV_PY = "/path/to/radfm/bin/python"
RADFM_CHECKPOINT = "/path/to/RadFM/pytorch_model.bin"

TASK_TO_JSON = {
    "yesno": "vqa_yesno_llava.json",
    "loc": "vqa_loc_llava.json",
    "multiselect": "vqa_multiselect_llava.json",
    "count": "vqa_count_llava.json",
}

YESNO_TASK = "micro-lesion-vqa-yesno"
LOC_TASK = "micro-lesion-vqa-localization"
MULTISELECT_TASK = "micro-lesion-vqa-multiselect"
COUNT_TASK = "micro-lesion-vqa-count"


@dataclass
class TaskRecord:
    sample_id: str
    image_rel: str
    image_path: Path
    prompt: str
    task: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_prompt(prompt: str) -> str:
    prompt = prompt.replace("\r\n", "\n")
    if prompt.startswith("<image>\n"):
        return prompt.split("\n", 1)[1]
    return prompt.replace("<image>", "").strip()


def _infer_loc_prompt_components(prompt: str) -> tuple[str, str, str, int]:
    lower = prompt.lower()
    if "micro pulmonary nodule" in lower:
        target_plural = "micro pulmonary nodule(s)"
        target_singular = "nodule"
        slice_desc = "this chest CT slice"
    elif "micro abdominal lesion" in lower:
        target_plural = "micro abdominal lesion(s)"
        target_singular = "lesion"
        slice_desc = "this CT slice"
    else:
        target_plural = "micro pulmonary lesion(s)"
        target_singular = "lesion"
        slice_desc = "this chest CT slice"

    m = re.search(r"\bA1\s*TO\s*([A-Z])([0-9]+)\b", prompt.upper())
    if not m:
        return target_plural, target_singular, slice_desc, 8
    bottom_row = m.group(1)
    rightmost_col = int(m.group(2))
    grid_size = max(ord(bottom_row) - ord("A") + 1, rightmost_col)
    return target_plural, target_singular, slice_desc, grid_size


def strengthen_loc_prompt(prompt: str) -> str:
    if "grid cells" not in prompt.lower():
        return prompt
    target_plural, target_singular, slice_desc, grid_size = _infer_loc_prompt_components(prompt)
    bottom_row = chr(ord("A") + grid_size - 1)
    rightmost_col = grid_size
    return (
        f"Which coarse grid cells contain {target_plural} on {slice_desc}? "
        f"The image is uniformly divided into a {grid_size}x{grid_size} grid of equal-sized cells. "
        f"Rows run top-to-bottom, with A as the top row and {bottom_row} as the bottom row. "
        f"Columns run left-to-right, with 1 as the leftmost column and {rightmost_col} as the rightmost column. "
        f"Return only the cells that directly and visibly contain {target_singular}, and only those cells. "
        f"Each listed cell must itself contain visible {target_singular}. "
        f"Do not output a row, column, rectangle, or large block unless every listed cell visibly contains {target_singular}. "
        f"Extra guessed cells are wrong, so leave out any uncertain cell. "
        f"If no cell clearly contains {target_plural[:-3] if target_plural.endswith('(s)') else target_plural}, answer NONE only. "
        f"Output exactly one line using uppercase cell labels separated by commas with no spaces and no extra words. "
        f"Valid cell labels are the row-column combinations from A1 to {bottom_row}{rightmost_col}."
    )


def load_task_records(benchmark_dir: Path, task: str, limit: int | None = None) -> list[TaskRecord]:
    data = load_json(benchmark_dir / TASK_TO_JSON[task])
    rows: list[TaskRecord] = []
    for item in data:
        rows.append(
            TaskRecord(
                sample_id=str(item["id"]),
                image_rel=str(item["image"]),
                image_path=benchmark_dir / str(item["image"]),
                prompt=clean_prompt(str(item["conversations"][0]["value"])),
                task=task,
            )
        )
    if limit is not None:
        rows = rows[:limit]
    return rows


def resolve_tasks(benchmark_dir: Path, requested_tasks: list[str] | None) -> list[str]:
    available = [task for task, rel in TASK_TO_JSON.items() if (benchmark_dir / rel).exists()]
    if requested_tasks is None:
        if not available:
            raise FileNotFoundError(f"No benchmark task files found in {benchmark_dir}")
        return available
    tasks = [task for task in requested_tasks if (benchmark_dir / TASK_TO_JSON[task]).exists()]
    missing = [task for task in requested_tasks if task not in tasks]
    if missing:
        print(f"[warn] skipping missing tasks for {benchmark_dir}: {', '.join(missing)}", flush=True)
    if not tasks:
        raise FileNotFoundError(f"None of the requested tasks exist in {benchmark_dir}: {requested_tasks}")
    return tasks


def load_manifest(benchmark_dir: Path) -> dict[str, dict[str, str]]:
    """Load scoring labels from a private manifest or the public benchmark.

    Historical experiment directories use ``manifest.csv``. The public release
    intentionally omits that source-data manifest, so its immutable
    ``benchmark.jsonl`` is converted to the small scoring schema used below.
    Single-select option letters are resolved against the prompt's coordinate
    map rather than treated as grid labels.
    """

    manifest_path = benchmark_dir / "manifest.csv"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return {str(row["sample_id"]): row for row in rows}

    benchmark_path = benchmark_dir / "benchmark.jsonl"
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Neither manifest.csv nor benchmark.jsonl exists in {benchmark_dir}")

    converted: dict[str, dict[str, str]] = {}
    for item in read_jsonl(benchmark_path):
        sample_id = str(item["id"])
        task_text = str(item.get("task") or "").lower()
        prompt = str(item.get("prompt") or "")
        answer = str(item.get("answer") or "").strip()
        row = {
            "sample_id": sample_id,
            "grid_size": "3",
            "_gold_source": "benchmark.jsonl",
        }
        if "count" in task_text:
            row.update({"task": "count", "gold_count": answer})
        elif "multiselect" in task_text or "single-select" in task_text:
            option_map = parse_prompt_option_map(prompt, grid_size=3)
            gold_cell = option_map.get(answer.upper())
            if not gold_cell:
                raise ValueError(f"{benchmark_path}: cannot resolve gold option for {sample_id}")
            row.update(
                {
                    "task": "multiselect",
                    "all_grid_cells_json": json.dumps([gold_cell]),
                }
            )
        else:
            row["task"] = task_text
        converted[sample_id] = row
    return converted


def parse_bool(value: str | bool | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _extract_grid_tokens(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\b([A-Z][1-9][0-9]*)\b", text.upper())


def _extract_coord_tokens(text: str) -> list[tuple[int, int]]:
    if not text:
        return []
    out: list[tuple[int, int]] = []
    for x_text, y_text in re.findall(r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)", text):
        out.append((int(x_text), int(y_text)))
    return out


def _is_valid_grid_token(token: str, grid_size: int) -> bool:
    if not token:
        return False
    row = token[0]
    if row < "A" or row > chr(ord("A") + grid_size - 1):
        return False
    try:
        col = int(token[1:])
    except ValueError:
        return False
    return 1 <= col <= grid_size


def _coord_to_grid_token(x: int, y: int, grid_size: int) -> str | None:
    if not (1 <= x <= grid_size and 1 <= y <= grid_size):
        return None
    return f"{chr(ord('A') + y - 1)}{x}"


def has_invalid_grid_tokens(text: str, grid_size: int = 8) -> bool:
    tokens = _extract_grid_tokens(text)
    coords = _extract_coord_tokens(text)
    if any(not _is_valid_grid_token(token, grid_size) for token in tokens):
        return True
    if any(_coord_to_grid_token(x, y, grid_size) is None for x, y in coords):
        return True
    return False


def parse_grid_set(text: str, grid_size: int = 8) -> list[str]:
    if not text:
        return []
    if has_invalid_grid_tokens(text, grid_size=grid_size):
        return []
    out: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(r"\b([A-Z][1-9][0-9]*)\b|\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)")
    for match in pattern.finditer((text or "").upper()):
        grid_token = match.group(1)
        if grid_token:
            token = grid_token
        else:
            token = _coord_to_grid_token(int(match.group(2)), int(match.group(3)), grid_size)
        if token and token not in seen:
            out.append(token)
            seen.add(token)
    return out


def parse_prompt_option_map(prompt: str, grid_size: int) -> dict[str, str]:
    option_map: dict[str, str] = {}
    pattern = re.compile(r"\b([A-Z]+)\.\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)")
    for label, x_text, y_text in pattern.findall((prompt or "").upper()):
        token = _coord_to_grid_token(int(x_text), int(y_text), grid_size)
        if token:
            option_map[label] = token
    return option_map


def parse_option_set(text: str, prompt: str | None, grid_size: int = 8) -> list[str]:
    if not text or not prompt:
        return []
    option_map = parse_prompt_option_map(prompt, grid_size)
    if not option_map:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for label in re.findall(r"\b([A-Z]+)\b", (text or "").upper()):
        token = option_map.get(label)
        if token and token not in seen:
            out.append(token)
            seen.add(token)
    return out


def canonicalize_loc_answer(text: str, grid_size: int = 8, prompt: str | None = None) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    compact = re.sub(r"\s+", "", raw.upper())
    if compact == "NONE":
        return "NONE"
    cells = parse_grid_set(compact, grid_size=grid_size)
    if cells:
        return ",".join(cells)
    option_cells = parse_option_set(compact, prompt=prompt, grid_size=grid_size)
    if option_cells:
        return ",".join(option_cells)
    return raw


def should_retry_loc_answer(text: str, grid_size: int = 8, prompt: str | None = None) -> bool:
    raw = (text or "").strip()
    if not raw:
        return True
    compact = re.sub(r"\s+", "", raw.upper())
    if compact == "NONE":
        return False
    if has_invalid_grid_tokens(compact, grid_size=grid_size):
        return True
    cells = parse_grid_set(compact, grid_size=grid_size)
    if not cells:
        cells = parse_option_set(compact, prompt=prompt, grid_size=grid_size)
        if not cells:
            return True
    if compact != ",".join(cells):
        return True
    return False


def make_single_shot_loc_prompt(prompt: str) -> str:
    return strengthen_loc_prompt(prompt)


def make_strict_count_prompt(prompt: str) -> str:
    base = (prompt or "").strip()
    suffix = (
        "\n\nImportant: Answer with one integer only."
        " Do not list coordinates."
        " Do not explain your reasoning."
        " Do not output any words besides the integer."
    )
    return f"{base}{suffix}"


def make_task_prompt(task: str, prompt: str, *, strict_count_prompt: bool = False) -> str:
    if task == "loc":
        return make_single_shot_loc_prompt(prompt)
    if task == "count" and strict_count_prompt:
        return make_strict_count_prompt(prompt)
    return prompt


def parse_yesno_answer(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    return m.group(1).lower() if m else ""


COUNT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
}


def parse_count_answer(text: str) -> int | None:
    if not text:
        return None
    lower = text.strip().lower()
    matches: list[tuple[int, int]] = []
    for m in re.finditer(r"\b([0-9]+)\b", lower):
        matches.append((m.start(), int(m.group(1))))
    for word, value in COUNT_WORDS.items():
        for m in re.finditer(rf"\b{re.escape(word)}\b", lower):
            matches.append((m.start(), value))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def parse_strict_count_answer(text: str, min_value: int = 0, max_value: int = 6) -> int | None:
    """Accept only a single in-range integer, as required by the benchmark."""

    cleaned = (text or "").strip().strip("`'\" \t\r\n")
    if not re.fullmatch(r"[+-]?\d+", cleaned):
        return None
    value = int(cleaned)
    return value if min_value <= value <= max_value else None


def truth_grid_cells(row: dict[str, str]) -> list[str]:
    raw = row.get("all_grid_cells_json", "")
    if raw:
        try:
            cells = json.loads(raw)
            return [str(x).upper() for x in cells if str(x).upper() != "NONE"]
        except json.JSONDecodeError:
            pass
    answer = str(row.get("grid_answer", "")).strip()
    if not answer or answer.upper() == "NONE":
        return []
    return parse_grid_set(answer, grid_size=int(float(row.get("grid_size", 8) or 8)))


def score_yesno(rows: list[dict]) -> dict:
    total = len(rows)
    parsed = [row for row in rows if row["parsed_answer"] in {"yes", "no"}]
    yes_count = sum(1 for row in rows if row["parsed_answer"] == "yes")
    no_count = sum(1 for row in rows if row["parsed_answer"] == "no")
    correct = sum(1 for row in rows if row.get("gold_label") in {"yes", "no"} and row["parsed_answer"] == row["gold_label"])
    parsed_correct = sum(
        1 for row in parsed if row.get("gold_label") in {"yes", "no"} and row["parsed_answer"] == row["gold_label"]
    )
    positives = [row for row in rows if row.get("gold_label") == "yes"]
    negatives = [row for row in rows if row.get("gold_label") == "no"]
    true_positive = sum(1 for row in positives if row["parsed_answer"] == "yes")
    true_negative = sum(1 for row in negatives if row["parsed_answer"] == "no")
    return {
        "total": total,
        "parsed": len(parsed),
        "unparsed": total - len(parsed),
        "parsed_rate": (len(parsed) / total) if total else 0.0,
        "yes_count": yes_count,
        "no_count": no_count,
        "yes_rate": (yes_count / total) if total else 0.0,
        "accuracy": (correct / total) if total else 0.0,
        "parsed_accuracy": (parsed_correct / len(parsed)) if parsed else 0.0,
        "positive_accuracy": (true_positive / len(positives)) if positives else 0.0,
        "negative_accuracy": (true_negative / len(negatives)) if negatives else 0.0,
        "balanced_accuracy": (
            ((true_positive / len(positives)) + (true_negative / len(negatives))) / 2.0
            if positives and negatives
            else 0.0
        ),
    }


def score_count(rows: list[dict]) -> dict:
    total = len(rows)
    parsed = [row for row in rows if row.get("parsed_count") is not None]
    correct = sum(1 for row in rows if row.get("parsed_count") == row.get("gold_count"))
    parsed_correct = sum(1 for row in parsed if row.get("parsed_count") == row.get("gold_count"))
    abs_err_sum = sum(abs(int(row["parsed_count"]) - int(row["gold_count"])) for row in parsed)
    within_one = sum(1 for row in parsed if abs(int(row["parsed_count"]) - int(row["gold_count"])) <= 1)
    return {
        "total": total,
        "parsed": len(parsed),
        "unparsed": total - len(parsed),
        "parsed_rate": (len(parsed) / total) if total else 0.0,
        "accuracy": (correct / total) if total else 0.0,
        "parsed_accuracy": (parsed_correct / len(parsed)) if parsed else 0.0,
        "mean_abs_error": (abs_err_sum / len(parsed)) if parsed else 0.0,
        "within_one_accuracy": (within_one / len(parsed)) if parsed else 0.0,
    }


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def score_loc(rows: list[dict]) -> dict:
    total = len(rows)
    exact = 0
    hit_any = 0
    empty_pred = 0
    p_sum = 0.0
    r_sum = 0.0
    f1_sum = 0.0
    iou_sum = 0.0
    for row in rows:
        pred = set(row["pred_cells"])
        gold = set(row["gold_cells"])
        inter = pred & gold
        union = pred | gold
        if pred == gold:
            exact += 1
        if inter:
            hit_any += 1
        if not pred:
            empty_pred += 1
        p = _safe_div(len(inter), len(pred))
        r = _safe_div(len(inter), len(gold))
        f1 = _safe_div(2 * p * r, p + r)
        iou = _safe_div(len(inter), len(union))
        p_sum += p
        r_sum += r
        f1_sum += f1
        iou_sum += iou
    return {
        "total": total,
        "exact_match_rate": _safe_div(exact, total),
        "hit_any_rate": _safe_div(hit_any, total),
        "empty_pred_rate": _safe_div(empty_pred, total),
        "mean_precision": _safe_div(p_sum, total),
        "mean_recall": _safe_div(r_sum, total),
        "mean_f1": _safe_div(f1_sum, total),
        "mean_iou": _safe_div(iou_sum, total),
    }


def group_rows(rows: list[dict], task: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {"overall": rows}
    if not rows:
        return groups
    if "clean" in rows[0]:
        groups["clean"] = [r for r in rows if r.get("clean") is True]
    if "context_flag" in rows[0]:
        groups["context_flag"] = [r for r in rows if r.get("context_flag") is True]
    if "single_nodule_slice" in rows[0]:
        groups["single_nodule_slice"] = [r for r in rows if r.get("single_nodule_slice") is True]
        groups["multi_nodule_slice"] = [r for r in rows if r.get("multi_nodule_slice") is True]
    return groups


def score_predictions(
    *,
    benchmark_dir: Path,
    task: str,
    predictions_path: Path,
    output_dir: Path,
    model_kind: str,
    sample_ids: set[str] | None = None,
) -> dict:
    manifest = load_manifest(benchmark_dir)
    preds = read_jsonl(predictions_path)
    pred_map: dict[str, dict] = {}
    for row in preds:
        sid = str(row.get("sample_id") or row.get("question_id") or row.get("id"))
        pred_map[sid] = row

    scored_rows: list[dict] = []
    for sample_id, mrow in manifest.items():
        if sample_ids is not None and sample_id not in sample_ids:
            continue
        manifest_task = str(mrow.get("task_name") or mrow.get("task") or "").strip().lower()
        if manifest_task and manifest_task != task:
            continue
        pred = pred_map.get(sample_id, {})
        answer_text = str(pred.get("text") or pred.get("answer") or "").strip()
        base = {
            "sample_id": sample_id,
            "task": task,
            "answer_text": answer_text,
        }
        if "clean" in mrow:
            base["clean"] = parse_bool(mrow.get("clean"))
        if "context_flag" in mrow:
            base["context_flag"] = parse_bool(mrow.get("context_flag"))
        if "single_nodule_slice" in mrow:
            base["single_nodule_slice"] = parse_bool(mrow.get("single_nodule_slice"))
            base["multi_nodule_slice"] = parse_bool(mrow.get("multi_nodule_slice"))
        if task == "yesno":
            base["parsed_answer"] = parse_yesno_answer(answer_text)
            gold_label = str(mrow.get("label", "")).strip().lower()
            if gold_label in {"yes", "no"}:
                base["gold_label"] = gold_label
        elif task == "count":
            if mrow.get("_gold_source") == "benchmark.jsonl":
                base["parsed_count"] = parse_strict_count_answer(answer_text)
            else:
                base["parsed_count"] = parse_count_answer(answer_text)
            gold_count = mrow.get("gold_count")
            if gold_count not in {None, ""}:
                base["gold_count"] = int(float(gold_count))
        else:
            base["gold_cells"] = truth_grid_cells(mrow)
            base["pred_cells"] = parse_grid_set(answer_text, grid_size=int(float(mrow.get("grid_size", 8) or 8)))
        scored_rows.append(base)

    groups = group_rows(scored_rows, task)
    summary = {
        "model_kind": model_kind,
        "benchmark_dir": str(benchmark_dir),
        "task": task,
        "predictions_path": str(predictions_path),
        "gold_source": (
            "benchmark.jsonl"
            if any(row.get("_gold_source") == "benchmark.jsonl" for row in manifest.values())
            else "manifest.csv"
        ),
        "metrics": {},
    }
    for name, rows in groups.items():
        if task == "yesno":
            summary["metrics"][name] = score_yesno(rows)
        elif task == "count":
            summary["metrics"][name] = score_count(rows)
        else:
            summary["metrics"][name] = score_loc(rows)

    ensure_dir(output_dir)
    write_json(output_dir / f"{task}_summary.json", summary)
    with (output_dir / f"{task}_sample_scores.csv").open("w", encoding="utf-8", newline="") as f:
        if scored_rows:
            writer = csv.DictWriter(f, fieldnames=list(scored_rows[0].keys()))
            writer.writeheader()
            writer.writerows(scored_rows)
    return summary


def run_llava_med(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    model_path: str,
    tasks: list[str],
    limit: int | None,
    strict_count_prompt: bool,
) -> None:
    import importlib.metadata as importlib_metadata
    import torch
    from PIL import Image
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    def infer_conv_mode(model_name: str) -> str:
        name = model_name.lower()
        if "mistral" in name:
            return "mistral_instruct"
        if "chatml" in name:
            return "chatml_direct"
        if "v1" in name:
            return "llava_v1"
        if "mpt" in name:
            return "mpt"
        return "llava_v0"

    def candidate_conv_modes(model_name: str) -> list[str]:
        ordered: list[str] = []
        for mode in [
            infer_conv_mode(model_name),
            "mistral_instruct",
            "llava_v1",
            "chatml_direct",
            "llava_v0",
        ]:
            if mode in conv_templates and mode not in ordered:
                ordered.append(mode)
        return ordered

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    try:
        importlib_metadata.version("bitsandbytes")
        llava_load_4bit = True
    except importlib_metadata.PackageNotFoundError:
        llava_load_4bit = False

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        model_name,
        load_4bit=llava_load_4bit,
        device_map="auto",
    )
    ensure_dir(output_dir)
    manifest = load_manifest(benchmark_dir)
    for task in tasks:
        records = load_task_records(benchmark_dir, task, limit)
        sample_ids = {rec.sample_id for rec in records}
        pred_rows: list[dict] = []
        for idx, rec in enumerate(records, start=1):
            image = Image.open(rec.image_path).convert("RGB")
            image_size = image.size
            image_tensor = process_images([image], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            def generate_one(prompt_text: str) -> tuple[str, str]:
                qs = prompt_text
                if getattr(model.config, "mm_use_im_start_end", False):
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

                best_text = ""
                best_mode = ""
                for conv_mode in candidate_conv_modes(model_name):
                    conv = conv_templates[conv_mode].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    input_ids = input_ids.unsqueeze(0).to(model.device)
                    stop_str = conv.sep2 if conv.sep_style in {SeparatorStyle.TWO, SeparatorStyle.LLAMA_2} else conv.sep
                    stopping_criteria = None
                    if stop_str:
                        stopping_criteria = [KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)]
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=image_tensor,
                            image_sizes=[image_size],
                            do_sample=False,
                            temperature=0.0,
                            max_new_tokens=96,
                            use_cache=True,
                            stopping_criteria=stopping_criteria,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    sliced = tokenizer.batch_decode(output_ids[:, input_ids.shape[1] :], skip_special_tokens=True)[0].strip()
                    full = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    for text_candidate in [sliced, full]:
                        candidate = text_candidate
                        if stop_str and candidate.endswith(stop_str):
                            candidate = candidate[: -len(stop_str)]
                        candidate = candidate.strip()
                        if candidate:
                            return candidate, conv_mode
                    if not best_mode:
                        best_mode = conv_mode
                        best_text = sliced.strip()
                return best_text, best_mode

            prompt_text = make_task_prompt(task, rec.prompt, strict_count_prompt=strict_count_prompt)
            raw_answer, used_conv_mode = generate_one(prompt_text)
            repair_used = False
            repair_answer = ""
            answer = raw_answer
            if task in {"loc", "multiselect"}:
                row = manifest.get(rec.sample_id, {})
                grid_size = int(float(row.get("grid_size", 8) or 8))
                answer = canonicalize_loc_answer(answer, grid_size=grid_size, prompt=prompt_text)
            pred_rows.append(
                {
                    "sample_id": rec.sample_id,
                    "task": task,
                    "image": rec.image_rel,
                    "prompt": prompt_text,
                    "text": answer,
                    "raw_text": raw_answer,
                    "repair_used": repair_used,
                    "repair_text": repair_answer,
                    "llava_conv_mode": used_conv_mode,
                }
            )
            if idx % 25 == 0 or idx == len(records):
                print(f"[llava-med] task={task} {idx}/{len(records)}", flush=True)
        a_path = output_dir / f"{task}_answers.jsonl"
        write_jsonl(a_path, pred_rows)
        score_predictions(
            benchmark_dir=benchmark_dir,
            task=task,
            predictions_path=a_path,
            output_dir=output_dir,
            model_kind="llava-med",
            sample_ids=sample_ids,
        )


def _move_hulu_inputs(inputs: dict, device: str):
    import torch

    moved = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if key == "pixel_values":
                value = value.to(torch.bfloat16)
        moved[key] = value
    return moved


def run_hulu_med(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    model_path: str,
    tasks: list[str],
    limit: int | None,
    strict_count_prompt: bool,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    ensure_dir(output_dir)
    manifest = load_manifest(benchmark_dir)
    for task in tasks:
        records = load_task_records(benchmark_dir, task, limit)
        sample_ids = {rec.sample_id for rec in records}
        pred_rows: list[dict] = []
        for idx, rec in enumerate(records, start=1):
            def generate_one(prompt_text: str) -> str:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": {"image_path": str(rec.image_path)}},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                inputs = processor(
                    conversation=conversation,
                    add_system_prompt=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                inputs = _move_hulu_inputs(inputs, model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=96,
                        do_sample=False,
                    )
                return processor.batch_decode(output_ids, skip_special_tokens=True, use_think=False)[0].strip()

            prompt_text = make_task_prompt(task, rec.prompt, strict_count_prompt=strict_count_prompt)
            raw_answer = generate_one(prompt_text)
            repair_used = False
            repair_answer = ""
            answer = raw_answer
            if task in {"loc", "multiselect"}:
                row = manifest.get(rec.sample_id, {})
                grid_size = int(float(row.get("grid_size", 8) or 8))
                answer = canonicalize_loc_answer(answer, grid_size=grid_size, prompt=prompt_text)
            pred_rows.append(
                {
                    "sample_id": rec.sample_id,
                    "task": task,
                    "image": rec.image_rel,
                    "prompt": prompt_text,
                    "text": answer,
                    "raw_text": raw_answer,
                    "repair_used": repair_used,
                    "repair_text": repair_answer,
                }
            )
            if idx % 25 == 0 or idx == len(records):
                print(f"[hulu-med] task={task} {idx}/{len(records)}", flush=True)
        pred_path = output_dir / f"{task}_answers.jsonl"
        write_jsonl(pred_path, pred_rows)
        score_predictions(
            benchmark_dir=benchmark_dir,
            task=task,
            predictions_path=pred_path,
            output_dir=output_dir,
            model_kind="hulu-med",
            sample_ids=sample_ids,
        )


def _move_medgemma_inputs(inputs: dict, device: str):
    import torch

    moved = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_medgemma(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    model_path: str,
    tasks: list[str],
    limit: int | None,
    strict_count_prompt: bool,
) -> None:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    ensure_dir(output_dir)
    manifest = load_manifest(benchmark_dir)
    for task in tasks:
        records = load_task_records(benchmark_dir, task, limit)
        sample_ids = {rec.sample_id for rec in records}
        pred_rows: list[dict] = []
        for idx, rec in enumerate(records, start=1):
            image = Image.open(rec.image_path).convert("RGB")
            def generate_one(prompt_text: str) -> str:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = processor(images=image, text=prompt, return_tensors="pt")
                inputs = _move_medgemma_inputs(inputs, model.device)
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=96,
                        use_cache=True,
                    )
                input_len = inputs["input_ids"].shape[1]
                return processor.decode(generated[0][input_len:], skip_special_tokens=True).strip()

            prompt_text = make_task_prompt(task, rec.prompt, strict_count_prompt=strict_count_prompt)
            raw_answer = generate_one(prompt_text)
            repair_used = False
            repair_answer = ""
            answer = raw_answer
            if task in {"loc", "multiselect"}:
                row = manifest.get(rec.sample_id, {})
                grid_size = int(float(row.get("grid_size", 8) or 8))
                answer = canonicalize_loc_answer(answer, grid_size=grid_size, prompt=prompt_text)
            pred_rows.append(
                {
                    "sample_id": rec.sample_id,
                    "task": task,
                    "image": rec.image_rel,
                    "prompt": prompt_text,
                    "text": answer,
                    "raw_text": raw_answer,
                    "repair_used": repair_used,
                    "repair_text": repair_answer,
                }
            )
            if idx % 25 == 0 or idx == len(records):
                print(f"[medgemma] task={task} {idx}/{len(records)}", flush=True)
        pred_path = output_dir / f"{task}_answers.jsonl"
        write_jsonl(pred_path, pred_rows)
        score_predictions(
            benchmark_dir=benchmark_dir,
            task=task,
            predictions_path=pred_path,
            output_dir=output_dir,
            model_kind="medgemma",
            sample_ids=sample_ids,
        )


def _stable_seed(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def run_radfm(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    quick_demo_root: Path,
    checkpoint_path: Path,
    tasks: list[str],
    limit: int | None,
    strict_count_prompt: bool,
) -> None:
    import importlib.util
    import torch

    sys.path.insert(0, str(quick_demo_root))
    from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM

    spec = importlib.util.spec_from_file_location("radfm_demo", quick_demo_root / "test.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    tokenizer, image_padding_tokens = mod.get_tokenizer(tokenizer_path=str(quick_demo_root / "Language_files"))

    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "pytorch_model.bin"
    model = MultiLLaMAForCausalLM(lang_model_path=str(quick_demo_root / "Language_files")).half()
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(torch.float16).cuda()
    model.eval()

    ensure_dir(output_dir)
    manifest = load_manifest(benchmark_dir)
    for task in tasks:
        records = load_task_records(benchmark_dir, task, limit)
        sample_ids = {rec.sample_id for rec in records}
        pred_rows: list[dict] = []
        for idx, rec in enumerate(records, start=1):
            torch.manual_seed(_stable_seed(rec.sample_id))
            image_list = [{"img_path": str(rec.image_path), "position": 0}]
            def generate_one(prompt_text: str) -> str:
                text, vision_x = mod.combine_and_preprocess(prompt_text, image_list, image_padding_tokens)
                lang_x = tokenizer(text, max_length=2048, truncation=True, return_tensors="pt")["input_ids"]
                with torch.no_grad():
                    model.embedding_layer.flag = "Text"
                    input_embedding, _ = model.embedding_layer(
                        lang_x.to(torch.long).cuda(),
                        vision_x.half().cuda(),
                    )
                    generation = model.lang_model.generate(
                        inputs_embeds=input_embedding,
                        max_new_tokens=200,
                        do_sample=False,
                    )
                return tokenizer.batch_decode(generation, skip_special_tokens=True)[0].strip()

            prompt_text = make_task_prompt(task, rec.prompt, strict_count_prompt=strict_count_prompt)
            raw_answer = generate_one(prompt_text)
            repair_used = False
            repair_answer = ""
            answer = raw_answer
            if task in {"loc", "multiselect"}:
                row = manifest.get(rec.sample_id, {})
                grid_size = int(float(row.get("grid_size", 8) or 8))
                answer = canonicalize_loc_answer(answer, grid_size=grid_size, prompt=prompt_text)
            pred_rows.append(
                {
                    "sample_id": rec.sample_id,
                    "task": task,
                    "image": rec.image_rel,
                    "prompt": prompt_text,
                    "text": answer,
                    "raw_text": raw_answer,
                    "repair_used": repair_used,
                    "repair_text": repair_answer,
                }
            )
            if idx % 25 == 0 or idx == len(records):
                print(f"[radfm] task={task} {idx}/{len(records)}", flush=True)
        pred_path = output_dir / f"{task}_answers.jsonl"
        write_jsonl(pred_path, pred_rows)
        score_predictions(
            benchmark_dir=benchmark_dir,
            task=task,
            predictions_path=pred_path,
            output_dir=output_dir,
            model_kind="radfm",
            sample_ids=sample_ids,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-kind", choices=["llava-med", "hulu-med", "radfm", "medgemma"], required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=None, choices=["yesno", "loc", "multiselect", "count"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict-count-prompt", action="store_true")
    parser.add_argument("--llava-model-path", default="/path/to/llava-med-v1.5-mistral-7b")
    parser.add_argument("--hulu-model-path", default=HULU_DEFAULT_MODEL)
    parser.add_argument("--medgemma-model-path", default=MEDGEMMA_DEFAULT_MODEL)
    parser.add_argument("--radfm-quick-demo-root", default=RADFM_QUICK_DEMO_ROOT)
    parser.add_argument("--radfm-checkpoint", default=RADFM_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    tasks = resolve_tasks(benchmark_dir, args.tasks)
    if args.model_kind == "llava-med":
        run_llava_med(
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            model_path=args.llava_model_path,
            tasks=tasks,
            limit=args.limit,
            strict_count_prompt=args.strict_count_prompt,
        )
    elif args.model_kind == "hulu-med":
        run_hulu_med(
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            model_path=args.hulu_model_path,
            tasks=tasks,
            limit=args.limit,
            strict_count_prompt=args.strict_count_prompt,
        )
    elif args.model_kind == "medgemma":
        run_medgemma(
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            model_path=args.medgemma_model_path,
            tasks=tasks,
            limit=args.limit,
            strict_count_prompt=args.strict_count_prompt,
        )
    else:
        run_radfm(
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            quick_demo_root=Path(args.radfm_quick_demo_root),
            checkpoint_path=Path(args.radfm_checkpoint),
            tasks=tasks,
            limit=args.limit,
            strict_count_prompt=args.strict_count_prompt,
        )


if __name__ == "__main__":
    main()
