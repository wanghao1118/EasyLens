# Benchmark Files

The public package includes all nine text-side benchmark definitions used by the runners: ReXGroundingCT, LIDC-IDRI, and AbdomenAtlas across count, single-select, and report tasks.

The data files are copied byte-for-byte from the May 2026 Google Drive evaluation bundle. `configs/benchmark_source_manifest.json` records a deterministic SHA-256 digest over the 46 published benchmark and sample-list files, and `code/tools/validate_release_package.py` verifies it.

- `benchmark.jsonl`: normalized examples with `id`, `image`, `dataset`, `modality`, `task`, `prompt`, and `answer`.
- `vqa_*.json`: conversation-format prompts used by VQA-style runners.
- `answers.json`: answer records when available in the original task export.

Image files are not included. The `image` field keeps the relative path expected by the evaluation runners.

The actual Stat. prompts count positives among six candidate regions within the 3x3 grid. The Sel. prompts present four candidate regions and require exactly one option. These prompt-level details are more specific than the high-level task summary in Appendix C.1 and are preserved unchanged from the evaluation bundle.

The expected split sizes are recorded in `configs/benchmark_counts.json` and are checked by `code/tools/validate_release_package.py`. Across the nine splits, the package contains 862 evaluation records.
