from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


DISALLOWED_WEIGHT_SUFFIXES = {".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}
REQUIRED_EVALUATION_FILES = (
    "code/method/easylens_core.py",
    "code/runners/easylens_official_runner.py",
    "code/runners/evaluate_medgemma_microbench_shape_selector_visibility.py",
    "code/runners/analyze_rex_report_lesion_token_probs.py",
    "code/runners/evaluate_rex_easylens_model_native.py",
    "code/runners/score_multiselect_set_exact_20260501.py",
    "code/runners/strict_rescore_count_predictions_20260501.py",
    "code/runners/score_easylens_report_rrg.py",
    "configs/official_runner_config.json",
    "configs/evaluation_parameters.md",
    "configs/benchmark_source_manifest.json",
    "requirements.txt",
)
REQUIRED_BENCHMARK_KEYS = {"id", "image", "dataset", "modality", "task", "prompt", "answer"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _benchmark_source_files(root: Path) -> list[Path]:
    """Return the immutable benchmark payload covered by the source digest.

    Human-authored README files are excluded so documentation can improve
    without changing the provenance check. Medical images are also excluded
    because the public release redistributes only metadata and sample lists.
    """

    paths = [
        path
        for path in (root / "benchmarks").glob("*/*")
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    ]
    paths.extend(
        path
        for path in (root / "benchmark_sample_ids").glob("**/*")
        if path.is_file() and path.suffix in {".csv", ".txt"}
    )
    paths.append(root / "configs" / "benchmark_counts.json")
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _benchmark_source_digest(root: Path, paths: list[Path]) -> str:
    """Hash sorted release paths and their SHA-256 values deterministically."""

    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_release_package(root: Path) -> dict[str, Any]:
    """Validate the public package boundary and return an inspectable summary.

    The check rejects model checkpoints and image-derived EasyBank tensors,
    verifies every published benchmark against its declared sample count and
    required schema, verifies the immutable Google Drive source-bundle digest,
    and confirms that the official inference and scoring entry points are
    present. Medical images and masks are intentionally not required because
    their redistribution remains governed by source licenses.
    """

    root = root.resolve()
    failures: list[str] = []
    weight_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in DISALLOWED_WEIGHT_SUFFIXES
    )
    if weight_files:
        failures.append("Disallowed weight files: " + ", ".join(weight_files))

    missing_evaluation_files = [relative for relative in REQUIRED_EVALUATION_FILES if not (root / relative).is_file()]
    if missing_evaluation_files:
        failures.append("Missing evaluation files: " + ", ".join(missing_evaluation_files))

    source_manifest_path = root / "configs" / "benchmark_source_manifest.json"
    benchmark_source_digest = ""
    benchmark_source_file_count = 0
    if source_manifest_path.is_file():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_paths = _benchmark_source_files(root)
        missing_source_paths = [path for path in source_paths if not path.is_file()]
        if missing_source_paths:
            failures.append(
                "Missing benchmark provenance files: "
                + ", ".join(path.relative_to(root).as_posix() for path in missing_source_paths)
            )
        else:
            benchmark_source_file_count = len(source_paths)
            benchmark_source_digest = _benchmark_source_digest(root, source_paths)
            expected_file_count = int(source_manifest["file_count"])
            expected_digest = str(source_manifest["aggregate_sha256"])
            if benchmark_source_file_count != expected_file_count:
                failures.append(
                    "Benchmark source file count mismatch: "
                    f"expected {expected_file_count}, found {benchmark_source_file_count}"
                )
            if benchmark_source_digest != expected_digest:
                failures.append(
                    "Benchmark source digest mismatch: "
                    f"expected {expected_digest}, found {benchmark_source_digest}"
                )

    count_path = root / "configs" / "benchmark_counts.json"
    if not count_path.is_file():
        failures.append("Missing configs/benchmark_counts.json")
        count_entries: list[dict[str, Any]] = []
    else:
        count_entries = json.loads(count_path.read_text(encoding="utf-8"))

    benchmark_summary: dict[str, int] = {}
    for entry in count_entries:
        name = str(entry["benchmark"])
        expected_count = int(entry["n"])
        benchmark_path = root / "benchmarks" / name / "benchmark.jsonl"
        sample_id_path = root / str(entry["sample_id_file"])
        if not benchmark_path.is_file():
            failures.append(f"Missing benchmark: {benchmark_path.relative_to(root)}")
            continue
        if not sample_id_path.is_file():
            failures.append(f"Missing sample list: {sample_id_path.relative_to(root)}")
        try:
            rows = _read_jsonl(benchmark_path)
        except (ValueError, json.JSONDecodeError) as error:
            failures.append(str(error))
            continue
        benchmark_summary[name] = len(rows)
        if len(rows) != expected_count:
            failures.append(f"{name}: expected {expected_count} rows, found {len(rows)}")
        benchmark_ids = [str(row.get("id", "")) for row in rows]
        if sample_id_path.is_file():
            with sample_id_path.open("r", encoding="utf-8", newline="") as handle:
                sample_ids = [str(row["sample_id"]) for row in csv.DictReader(handle)]
            if len(sample_ids) != len(set(sample_ids)) or len(benchmark_ids) != len(set(benchmark_ids)):
                failures.append(f"{name}: duplicate sample IDs detected")
            if set(sample_ids) != set(benchmark_ids):
                failures.append(f"{name}: sample_ids.csv does not match benchmark.jsonl")
        vqa_paths = sorted(benchmark_path.parent.glob("vqa_*.json"))
        if len(vqa_paths) != 1:
            failures.append(f"{name}: expected one vqa_*.json file, found {len(vqa_paths)}")
        else:
            vqa_rows = json.loads(vqa_paths[0].read_text(encoding="utf-8"))
            if not isinstance(vqa_rows, list) or len(vqa_rows) != expected_count:
                failures.append(f"{name}: VQA export does not contain {expected_count} records")
        for row_number, row in enumerate(rows, start=1):
            missing_keys = sorted(REQUIRED_BENCHMARK_KEYS.difference(row))
            if missing_keys:
                failures.append(f"{name}:{row_number} missing keys: {', '.join(missing_keys)}")
                break

    summary = {
        "status": "ok" if not failures else "failed",
        "root": str(root),
        "weight_file_count": len(weight_files),
        "benchmark_count": len(benchmark_summary),
        "benchmark_rows": benchmark_summary,
        "benchmark_source_digest": benchmark_source_digest,
        "benchmark_source_file_count": benchmark_source_file_count,
        "evaluation_file_count": len(REQUIRED_EVALUATION_FILES) - len(missing_evaluation_files),
        "failures": failures,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the weight-free EasyLens public release package.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args()


def main() -> None:
    summary = validate_release_package(parse_args().root)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
