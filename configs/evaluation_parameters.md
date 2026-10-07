# Evaluation Parameters

This file records the effective EasyLens evaluation settings used by the packaged runners.

## Paper-Equation Method Profile

The packaged runners use `paper_equations_27_40`, an equation-literal interpretation of Appendix A.3-A.4. This label does not claim that a locally rebuilt bank or rerun reproduces the paper's tables. Where the equations and Table 4 differ, this profile follows the equations: it omits the Table 4 margin term and local-peak filter and retrieves one nearest support rather than a temperature-weighted support set.

- `vision_layer`: 1
- `prototype_topm` (M in Eqs. 27-28): 1
- `counterfactual_temperature` (tau_c in Eq. 29): 1.0
- `shape_weight` (lambda in Eq. 32): 1.0
- Seed selection: global top-K0 from Eq. 30.
- Candidate selection: global top-K over the calibrated score from Eq. 33.
- Reference retrieval: one nearest support within the recalled prototype memory, as required by Eqs. 34 and 38.
- `amplifier_threshold` (eta in Eqs. 39-40): 0.0
- `max_new_tokens`: 96 for count/single-select, 160 for report generation.
- Report probe: first 3 decoding steps with vocabulary top-50 intersection. The paper defines top-K but does not state a numeric K; 50 is the explicit release default.

The margin term, local-peak seed filter, support suppression, hybrid/raw/gap score modes, and temperature-weighted top-m support averaging from the earlier experiment scripts are not part of this profile and are not accepted by the official MedGemma runners.

## 9 Subtle-Lesion Benchmarks

All compared settings use the prompt text stored in the benchmark files. The LIDC count prompt includes an integer-only response instruction as part of the benchmark question text.

Schema-v2 EasyBank files are built from an independent bank/train split with pixel-level lesion masks, not from the held-out test benchmark directories. One bank is built per dataset and reused for its Stat., Sel., and Gen. tasks. The packaged benchmark metadata is used for evaluation only.

| Benchmark | Task | beta (alpha) | selector_k (K) | seed_topk (K0) |
|---|---|---:|---:|---:|
| rex_count | count | 1.25 | 384 | 32 |
| rex_single_select | single_select | 0.75 | 256 | 16 |
| rex_report | report | 0.50 | 256 | 16 |
| lidc_count | count | 0.75 | 256 | 16 |
| lidc_single_select | single_select | 0.75 | 256 | 16 |
| lidc_report | report | 0.50 | 256 | 16 |
| abdomen_count | count | 0.75 | 256 | 32 |
| abdomen_single_select | single_select | 0.75 | 256 | 16 |
| abdomen_report | report | 0.50 | 256 | 16 |

## Normal-Lesion RRG Benchmarks

| Dataset | beta (alpha) | selector_k (K) | seed_topk (K0) | Selection metric |
|---|---:|---:|---:|---|
| MIMIC-CXR | 0.50 | 256 | 16 | METEOR |
| Kvasir-SEG | 1.00 | 384 | 32 | METEOR |
| BKAI-Polyp | 2.00 | 512 | 48 | METEOR |

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
