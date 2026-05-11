# Benchmark Files

Each benchmark directory contains the text-side task definition used by the runners.

- `benchmark.jsonl`: normalized examples with `id`, `image`, `dataset`, `modality`, `task`, `prompt`, and `answer`.
- `vqa_*.json`: conversation-format prompts used by VQA-style runners.
- `answers.json`: answer records when available in the original task export.

Image files are not included. The `image` field keeps the relative path expected by the evaluation runners.
