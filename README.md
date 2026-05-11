# EasyLens Evaluation Package

This directory collects code, evaluation runners, parameter records, benchmark questions, and packaged feature banks for EasyLens experiments.

## Layout

- `code/method/`: EasyLens construction code, prototype/bank utilities, and related method components.
- `code/runners/`: evaluation runners and scorers for count, single-select, report token probability, and RRG tasks.
- `code/launch_scripts/`: shell entry points for the official EasyLens runner.
- `code/dependencies/`: local helper modules imported by the runners.
- `configs/`: parameter settings, benchmark counts, and entry-point records.
- `benchmarks/`: prompt/question/answer metadata for the 9 subtle-lesion benchmarks. Images are intentionally not included.
- `benchmark_sample_ids/`: sample-id lists matching the benchmark metadata.
- `banks/`: packaged EasyLens shape banks for the subtle-lesion MedGemma1.5 runner.

## Data Policy

This package includes benchmark text metadata, sample identifiers, and the EasyLens shape banks needed by the packaged subtle-lesion MedGemma1.5 runner. It does not include medical images, masks, overlays, checkpoints, logs, or generated visualizations.

## Runtime Note

For direct execution from this package, set `PYTHONPATH` to include `code/runners`, `code/method`, and `code/dependencies`.

The official runner expects external data assets to be available at the relative `image` paths referenced by the benchmark files, or through a compatible benchmark root supplied with command-line arguments.

Example:

```bash
MODEL_PATH=/path/to/medgemma-1.5-4b-it \
BANK_SOURCE_ROOT=/path/to/bank_train_split \
BANK_ROOT=/path/to/shape_banks \
bash code/launch_scripts/build_subtle_lesion_banks_medgemma15.sh

python code/runners/easylens_official_runner.py subtle-medgemma15 \
  --model-path /path/to/medgemma-1.5-4b-it \
  --benchmark-root /path/to/benchmarks_with_images \
  --bank-root banks/subtle_lesion_medgemma15 \
  --output-root outputs/subtle_lesion_medgemma15
```
