from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach locally held ReX target descriptions for the Appendix B.2 report probe."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_target_descriptions(path: Path) -> dict[str, str]:
    targets: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "target_texts_json"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row in reader:
            values = json.loads(row["target_texts_json"])
            if values:
                targets[str(row["sample_id"])] = "; ".join(str(value).strip() for value in values)
    return targets


def main() -> None:
    """Create local ReX report metadata without modifying the public metadata.

    ReX target descriptions are required by the paper's sample-specific probe,
    but are not redistributed with EasyLens. This command joins descriptions
    from an authorized local source manifest and writes an evaluation-ready
    metadata copy. Image files remain external and can be linked under the
    resulting output directory by the user.
    """

    args = parse_args()
    targets = load_target_descriptions(args.source_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vqa_path = args.benchmark_dir / "vqa_report_llava.json"
    vqa_records = json.loads(vqa_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    for record in vqa_records:
        sample_id = str(record["id"])
        description = targets.get(sample_id.removesuffix("__report"))
        if not description:
            missing.append(sample_id)
        else:
            record["target_description"] = description
    if missing:
        raise RuntimeError(f"Missing target descriptions for {len(missing)} samples: {missing}")
    (args.output_dir / "vqa_report_llava.json").write_text(
        json.dumps(vqa_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    jsonl_path = args.benchmark_dir / "benchmark.jsonl"
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        row["target_description"] = targets[str(row["id"]).removesuffix("__report")]
    (args.output_dir / "benchmark.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"records": len(vqa_records), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
