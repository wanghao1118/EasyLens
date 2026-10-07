from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "tools"))
sys.path.insert(0, str(ROOT / "code" / "dependencies"))

from evaluate_server_vlm_microbench import canonicalize_loc_answer, score_predictions
from validate_release_package import validate_release_package


class ReleasePackageTest(unittest.TestCase):
    def test_release_contains_benchmarks_and_evaluation_code_without_weights(self):
        summary = validate_release_package(ROOT)

        self.assertEqual(summary["status"], "ok", summary["failures"])
        self.assertEqual(summary["weight_file_count"], 0)
        self.assertEqual(summary["benchmark_count"], 9)
        self.assertEqual(sum(summary["benchmark_rows"].values()), 862)
        self.assertEqual(summary["benchmark_source_file_count"], 46)
        self.assertEqual(
            summary["benchmark_source_digest"],
            "e4beb37919854da19ec54a53daee2113ddd4ebb46c710d596287b5f8146b9d65",
        )

    def test_public_stat_and_select_exports_are_directly_scorable(self):
        for dataset in ("rex", "lidc", "abdomen"):
            for task_name, scorer_task in (("count", "count"), ("single_select", "multiselect")):
                benchmark_dir = ROOT / "benchmarks" / f"{dataset}_{task_name}"
                rows = [
                    json.loads(line)
                    for line in (benchmark_dir / "benchmark.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                predictions = []
                for row in rows:
                    answer = str(row["answer"])
                    if scorer_task == "multiselect":
                        answer = canonicalize_loc_answer(answer, grid_size=3, prompt=str(row["prompt"]))
                    predictions.append({"sample_id": str(row["id"]), "text": answer})

                with tempfile.TemporaryDirectory() as temporary_directory:
                    temporary_root = Path(temporary_directory)
                    predictions_path = temporary_root / "predictions.jsonl"
                    predictions_path.write_text(
                        "".join(json.dumps(row) + "\n" for row in predictions),
                        encoding="utf-8",
                    )
                    summary = score_predictions(
                        benchmark_dir=benchmark_dir,
                        task=scorer_task,
                        predictions_path=predictions_path,
                        output_dir=temporary_root / "scores",
                        model_kind="oracle",
                        sample_ids={str(row["id"]) for row in rows},
                    )

                overall = summary["metrics"]["overall"]
                metric = "accuracy" if scorer_task == "count" else "exact_match_rate"
                self.assertEqual(summary["gold_source"], "benchmark.jsonl")
                self.assertEqual(overall["total"], len(rows))
                self.assertEqual(overall[metric], 1.0)


if __name__ == "__main__":
    unittest.main()
