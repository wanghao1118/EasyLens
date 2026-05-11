# Evaluation Parameters

This file records the effective EasyLens evaluation settings used by the packaged runners.

## Common VQA Settings

- `vision_layer`: 1
- `support_mode`: `global_topk`
- `suppression_radius`: 1
- `margin_weight`: 0.35
- `shape_weight`: 1.0
- `score_threshold`: 0.0
- `seed_mode`: `local_peak_topk`
- `seed_peak_kernel`: 5
- `base_score_mode`: `seed_only`
- `support_score_mode`: `shape_only` unless a task-specific setting below overrides it.
- `raw_score_weight`: 1.0
- `gap_score_weight`: 0.0
- `retrieval_temperature`: 0.1
- `max_new_tokens`: 96 for count/single-select, 160 for report generation.

## 9 Subtle-Lesion Benchmarks

All compared settings use the prompt text stored in the benchmark files. The LIDC count prompt includes an integer-only response instruction as part of the benchmark question text.

The EasyLens shape banks are built from an independent bank/train split with the same task-level directory convention, not from the held-out test benchmark directories. The packaged benchmark files are used for evaluation only.

| Benchmark | Task | beta | selector_k | seed_topk | retrieval_topm | support_score_mode |
|---|---|---:|---:|---:|---:|---|
| rex_count | count | 1.25 | 384 | 32 | 128 | shape_only |
| rex_single_select | single_select | 0.75 | 256 | 16 | 64 | shape_only |
| rex_report | report | 0.50 | 256 | 16 | 64 | shape_only |
| lidc_count | count | 0.75 | 256 | 16 | 64 | shape_only |
| lidc_single_select | single_select | 0.75 | 256 | 16 | 64 | shape_only |
| lidc_report | report | 0.50 | 256 | 16 | 64 | shape_only |
| abdomen_count | count | 0.75 | 256 | 32 | 64 | shape_only |
| abdomen_single_select | single_select | 0.75 | 256 | 16 | 64 | shape_only |
| abdomen_report | report | 0.50 | 256 | 16 | 64 | shape_only |

## Normal-Lesion RRG Benchmarks

| Dataset | beta | selector_k | seed_topk | support_score_mode | Selection metric |
|---|---:|---:|---:|---|---|
| MIMIC-CXR | 0.50 | 256 | 16 | shape_only | METEOR |
| Kvasir-SEG | 1.00 | 384 | 32 | shape_only | METEOR |
| BKAI-Polyp | 2.00 | 512 | 48 | shape_only | METEOR |

## Runner Entry Points

- Subtle-lesion bank construction: `code/launch_scripts/build_subtle_lesion_banks_medgemma15.sh`
- Count/single-select VQA: `code/runners/evaluate_medgemma_microbench_shape_selector_visibility.py`
- Backbone adaptation: `code/runners/evaluate_rex_easylens_model_native.py`
- Report token-probability: `code/runners/analyze_rex_report_lesion_token_probs.py`
- Normal-lesion RRG shape pipeline: `code/runners/evaluate_rex_easylens_model_native.py`
- Exact set scorer: `code/runners/score_multiselect_set_exact_20260501.py`

## Official Runner Files

- Official parameter record: `configs/official_runner_config.json`
- Official runner: `code/runners/easylens_official_runner.py`
- Subtle-lesion bank builder script: `code/launch_scripts/build_subtle_lesion_banks_medgemma15.sh`
- Subtle-lesion entry point: `code/launch_scripts/run_subtle_lesion_medgemma15.sh`
- Normal-lesion RRG entry point: `code/launch_scripts/run_normal_rrg_medgemma15.sh`
