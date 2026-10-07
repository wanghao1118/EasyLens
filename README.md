# EasyLens

**A Training-Free Plug-and-Play Subtle-Lesion Representation Amplifier for Medical Vision-Language Models**

Paper: [arXiv:2606.06379](https://arxiv.org/abs/2606.06379)

EasyLens is an inference-time adapter for frozen medical vision-language models. It uses an offline prototype bank to identify lesion-relevant visual patches and applies morphology-guided residual enhancement before the frozen model consumes the visual sequence. It does not update the backbone and does not require lesion masks at inference time.

<p align="center">
  <img src="assets/benchmark_distribution.png" width="72%" alt="Subtle-lesion benchmark distribution">
</p>

## Method

EasyLens has three components:

1. **EasyBank** clusters one normalized mean lesion embedding per construction image, stores patch-level lesion supports under the image's assigned prototype, estimates a morphology prior for each prototype, and clusters normal embeddings separately at every patch location.
2. **EasyTag** contrasts each inference patch with its top-M lesion prototypes and same-location normal prototypes, applies the counterfactual sigmoid score, calibrates the global top-K0 seeds with morphology priors, and retains the global top-K candidates by calibrated score.
3. **EasyAmplifier** retrieves references only from the selected prototype's support memory and applies continuous morphology-weighted residual updates.

The implementation is in `code/method/easylens_core.py`. Schema-v2 banks implement the paper formulation. Older schema-v1 banks remain readable for audit compatibility and are explicitly labeled `legacy_approximation` in run metadata.

## Repository layout

```text
.
├── benchmarks/              # 9 benchmark definitions, prompts, and answer labels
├── benchmark_sample_ids/    # Published sample lists
├── banks/                   # Bank schemas, identities, and checksums; no tensors
├── code/
│   ├── method/              # EasyLens core
│   ├── runners/             # Evaluation, scoring, and bank construction
│   ├── dependencies/        # Shared runner utilities
│   └── launch_scripts/      # Reproducible entry points
├── configs/                 # Official task settings
├── tests/                   # Synthetic method and metric tests
└── requirements.txt
```

## Public release contents

The public package includes all nine benchmark definitions, prompts, answer labels, sample lists, official evaluation runners, metric implementations, bank-construction code, task configuration, and synthetic tests.

It does not include model checkpoints, adapter weights, selector checkpoints, EasyBank `.pt` tensors, source medical images, masks, or raw reports. Weight-like extensions are blocked by `.gitignore`, and the release validator fails if any such file is present.

## Installation

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended.

```bash
git clone https://github.com/wanghao1118/easylens.git
cd easylens
python3 -m pip install -r requirements.txt
```

Model-specific backbones such as LLaVA-Med, RadFM, Lingshu, and Hulu-Med may require their upstream runtime dependencies in addition to `requirements.txt`.

## Data and model preparation

The benchmark metadata, questions, answer labels, and sample lists are included in this repository. Source medical images are not redistributed; obtain each dataset and model from its original provider under the applicable terms, then arrange authorized evaluation images under the relative paths recorded in `benchmarks/*/images/`.

The published benchmark payload is copied byte-for-byte from the May 2026 Google Drive evaluation bundle. Its 46 metadata and sample-list files are pinned by `configs/benchmark_source_manifest.json`; the release validator recomputes the aggregate SHA-256 digest so benchmark edits cannot be mistaken for the paper evaluation set.

The subtle-lesion benchmark sizes are:

| Dataset | Stat. | Sel. | Gen. |
|---|---:|---:|---:|
| ReXGroundingCT | 147 | 90 | 60 |
| LIDC-IDRI | 132 | 133 | 60 |
| AbdomenAtlas 3.0 Mini | 86 | 104 | 50 |

For report generation, the runtime needs a per-sample target description. The unchanged Google Drive exports for LIDC-IDRI and AbdomenAtlas contain synthetic benchmark target text in the answer's `Target finding:` line, which the loader extracts without editing the benchmark. The ReX export contains only a binary answer and therefore cannot reproduce the paper's lesion-token probe by itself. Users with authorized access must prepare a local evaluation copy:

```bash
python3 code/tools/prepare_rex_report_targets.py \
  --source-manifest /path/to/authorized/rex/manifest.csv \
  --benchmark-dir benchmarks/rex_report \
  --output-dir /path/to/local/benchmarks/rex_report
```

Place or link the authorized evaluation images under the resulting `images/` directory. The Appendix B.2 probe then derives a strict, per-sample lesion-token set and evaluates only the first three decoding steps.

## Build EasyBank

Use an independent construction split, not a held-out evaluation split. `BANK_SOURCE_ROOT` must contain one source directory per dataset—`rex/`, `lidc/`, and `abdomen/`—and each directory must contain a `manifest.csv`, aligned images, and pixel-level lesion supervision. Generic 2D/3D masks can be referenced by `mask_path`, `mask`, or `raw_mask`. ReX 4D finding segmentations are projected from an authorized raw root by setting `REX_RAW_ROOT`; LIDC polygons are joined through `LIDC_SOURCE_MANIFEST`. Coarse VQA grid labels are rejected by default because they are not equivalent to the lesion masks used in the paper.

```bash
export MODEL_PATH=/path/to/medgemma-1.5-4b-it
export BANK_SOURCE_ROOT=/path/to/independent_bank_construction_split
export BANK_ROOT=/path/to/output/banks/subtle_lesion_medgemma15
export REX_RAW_ROOT=/path/to/authorized/rex/raw
export LIDC_SOURCE_MANIFEST=/path/to/authorized/lidc_source_manifest.csv

bash code/launch_scripts/build_subtle_lesion_banks_medgemma15.sh
```

Useful construction controls include `NUM_LESION_PROTOTYPES`, `NUM_NORMAL_PROTOTYPES`, `MAX_LESION_TOKENS`, `MAX_BG_GLOBAL_TOKENS`, `SHAPE_RADIUS`, and `SEED`.

The builder writes schema-v2 `.pt` files containing:

- image-balanced lesion prototypes;
- cluster-conditioned lesion support memories;
- prototype-specific morphology priors normalized to `[0, 1]`;
- location-wise anatomy-aware normal prototypes;
- schema and construction metadata.

It writes one reusable bank per dataset: `rex_shape_bank_v2.pt`, `lidc_shape_bank_v2.pt`, and `abdomen_shape_bank_v2.pt`. The corresponding Stat., Sel., and Gen. tasks all consume the same dataset bank, matching the paper's construction-set definition.

## Run the subtle-lesion evaluation

```bash
export MODEL_PATH=/path/to/medgemma-1.5-4b-it
export BENCHMARK_ROOT=/path/to/benchmarks
export BANK_ROOT=/path/to/banks/subtle_lesion_medgemma15
export OUTPUT_ROOT=outputs/subtle_lesion_medgemma15

python3 code/runners/easylens_official_runner.py subtle-medgemma15 \
  --model-path "$MODEL_PATH" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --bank-root "$BANK_ROOT" \
  --output-root "$OUTPUT_ROOT"
```

Run selected tasks with `--tasks`, for example:

```bash
python3 code/runners/easylens_official_runner.py subtle-medgemma15 \
  --model-path "$MODEL_PATH" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --bank-root "$BANK_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --tasks rex_count rex_single_select rex_report
```

The packaged operating points are stored in `configs/official_runner_config.json`. The method path is identified as `paper_equations_27_40`: global top-K0 seed selection, global top-K calibrated candidate selection, and one nearest support retrieved from the recalled prototype memory. This is an equation-literal implementation, not a claim that a locally rebuilt bank reproduces the paper's table artifacts. Shared defaults include vision layer 1, prototype top-M of 1, counterfactual temperature 1.0, morphology weight 1.0, a 96-token limit for Stat./Sel., a 160-token limit for Gen., and a three-step report probe window. Dataset-task values for residual strength β (the paper's α), candidate budget K, and seed budget K0 are recorded separately in the same file.

## Report-generation probe

The Gen. metric follows Appendix B.2:

1. derive the strict lesion-token set from the sample's target description;
2. compute the next-token distribution for the first three decoding steps;
3. intersect the lesion-token set with the vocabulary top-K at each step;
4. sum the retained probability mass and report `100 ×` its three-step mean.

`--probe-vocab-topk` controls K and is set to 50 in the release configuration. The arXiv v3 equations define top-K but do not state its numeric value, so 50 is an explicit release default rather than a paper-verifiable constant. The probe is evaluation-only and does not change the prompt or generation procedure.

## Verification

Run the synthetic tests with:

```bash
python3 -m unittest discover -s tests -v
```

Validate that the public package contains all benchmarks and evaluation entry points and contains no weights:

```bash
python3 code/tools/validate_release_package.py
```

Build the validated source package with the same exclusion policy:

```bash
bash code/tools/build_release_package.sh dist/easylens-source-release.tar.gz
```

Legacy artifact hashes are retained only for audit traceability; the tensors themselves are excluded. A runtime warning and `bank_alignment=legacy_approximation` prevent separately obtained legacy results from being confused with the equation-literal schema-v2 implementation. The release does not publish the private construction samples or claim that a locally rebuilt bank is the artifact behind the paper's tables.

## Research-use notice

EasyLens is a research prototype and is not a standalone clinical diagnostic system. Users are responsible for dataset access, privacy, licensing, and clinical validation requirements.

## Citation

```bibtex
@article{easylens2026,
  title={EasyLens: A Training-Free Plug-and-Play Subtle-Lesion Representation Amplifier for Medical Vision-Language Models},
  author={Zeng, Qiwei and Wang, Hao and Lin, Jinghao and Ye, Shuchang and Yang, Yuezhe and Peng, Yige and Che, Haoyuan and Kim, Jinman and Bi, Lei},
  journal={arXiv preprint arXiv:2606.06379},
  year={2026},
  eprint={2606.06379},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  doi={10.48550/arXiv.2606.06379},
  url={https://arxiv.org/abs/2606.06379}
}
```
