from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("method_profile") != "paper_equations_27_40":
        raise SystemExit("Official configuration must declare method_profile=paper_equations_27_40.")
    return config


def env_or_arg(value: str | None, env_name: str | None, default: Path | None = None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    if env_name and os.environ.get(env_name):
        return Path(os.environ[env_name]).expanduser().resolve()
    if default is not None:
        return default.resolve()
    name = env_name or "required path"
    raise SystemExit(f"Missing {name}. Pass an argument or set the environment variable.")


def build_pythonpath(root: Path, config: dict[str, Any]) -> str:
    parts = [str((root / item).resolve()) for item in config["runtime"]["pythonpath"]]
    current = os.environ.get("PYTHONPATH")
    if current:
        parts.append(current)
    return os.pathsep.join(parts)


def run_command(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True, env=env)


def add_common_easylens_args(command: list[str], common: dict[str, Any], device: str) -> None:
    command.extend(
        [
            "--vision-layer",
            str(common["vision_layer"]),
            "--shape-weight",
            str(common["shape_weight"]),
            "--prototype-topm",
            str(common["prototype_topm"]),
            "--counterfactual-temperature",
            str(common["counterfactual_temperature"]),
            "--amplifier-threshold",
            str(common["amplifier_threshold"]),
            "--device",
            device,
        ]
    )


def score_single_select(
    *,
    python_bin: str,
    runner_root: Path,
    benchmark_dir: Path,
    predictions_path: Path,
    output_dir: Path,
    model_kind: str,
    limit: int,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    scorer = runner_root / "code" / "runners" / "score_multiselect_set_exact_20260501.py"
    command = [
        python_bin,
        str(scorer),
        "--benchmark-dir",
        str(benchmark_dir),
        "--predictions-path",
        str(predictions_path),
        "--output-dir",
        str(output_dir),
        "--model-kind",
        model_kind,
    ]
    if limit > 0:
        command.extend(["--limit", str(limit)])
    run_command(command, env, dry_run)


def resolve_subtle_benchmark_dir(package: Path, benchmark_root: Path | None, task_cfg: dict[str, Any]) -> Path:
    rel = Path(task_cfg["benchmark_dir"])
    if benchmark_root is None:
        return package / rel
    return benchmark_root / rel.name


def run_subtle_medgemma15(args: argparse.Namespace, config: dict[str, Any], env: dict[str, str]) -> None:
    root = package_root()
    suite = config["subtle_lesion_medgemma15"]
    runtime = config["runtime"]
    common = config["common_easylens_parameters"]
    model_path = env_or_arg(args.model_path, suite.get("model_path_env"))
    benchmark_root = env_or_arg(args.benchmark_root, suite.get("benchmark_root_env"), root / "benchmarks")
    bank_root = env_or_arg(args.bank_root, suite.get("bank_root_env"))
    output_root = env_or_arg(args.output_root, suite.get("output_root_env"), root / "outputs" / "subtle_lesion_medgemma15")
    device = args.device or str(runtime["device"])

    vqa_runner = root / "code" / "runners" / "evaluate_medgemma_microbench_shape_selector_visibility.py"
    report_runner = root / "code" / "runners" / "analyze_rex_report_lesion_token_probs.py"

    selected = set(args.tasks or suite["tasks"].keys())
    unknown = selected.difference(suite["tasks"].keys())
    if unknown:
        raise SystemExit(f"Unknown subtle-lesion task(s): {', '.join(sorted(unknown))}")

    for name, task_cfg in suite["tasks"].items():
        if name not in selected:
            continue
        task = str(task_cfg["task"])
        out_dir = output_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        benchmark_dir = resolve_subtle_benchmark_dir(root, benchmark_root, task_cfg)
        bank_path = bank_root / str(task_cfg["bank_file"])

        if task in {"count", "multiselect"}:
            command = [
                args.python_bin,
                str(vqa_runner),
                "--model-path",
                str(model_path),
                "--benchmark-dir",
                str(benchmark_dir),
                "--bank-path",
                str(bank_path),
                "--output-dir",
                str(out_dir),
                "--tasks",
                task,
                "--beta",
                str(task_cfg["beta"]),
                "--selector-k",
                str(task_cfg["selector_k"]),
                "--seed-topk",
                str(task_cfg["seed_topk"]),
                "--max-new-tokens",
                str(runtime["count_select_max_new_tokens"]),
            ]
            if int(args.limit) > 0:
                command.extend(["--limit", str(args.limit)])
            add_common_easylens_args(command, common, device)
            if task_cfg.get("strict_count_prompt"):
                command.append("--strict-count-prompt")
            run_command(command, env, args.dry_run)
            if task == "multiselect":
                score_single_select(
                    python_bin=args.python_bin,
                    runner_root=root,
                    benchmark_dir=benchmark_dir,
                    predictions_path=out_dir / "multiselect_answers.jsonl",
                    output_dir=out_dir,
                    model_kind=str(suite["model_kind"]),
                    limit=int(args.limit),
                    env=env,
                    dry_run=args.dry_run,
                )
            continue
        elif task == "report":
            command = [
                args.python_bin,
                str(report_runner),
                "--model-path",
                str(model_path),
                "--benchmark-dir",
                str(benchmark_dir),
                "--output-dir",
                str(out_dir),
                "--mode",
                "shape",
                "--prompt-mode",
                str(task_cfg["prompt_mode"]),
                "--bank-path",
                str(bank_path),
                "--beta",
                str(task_cfg["beta"]),
                "--selector-k",
                str(task_cfg["selector_k"]),
                "--seed-topk",
                str(task_cfg["seed_topk"]),
                "--max-new-tokens",
                str(runtime["report_max_new_tokens"]),
                "--probe-steps",
                str(task_cfg["topk_steps"]),
                "--probe-vocab-topk",
                str(task_cfg["probe_vocab_topk"]),
            ]
            if int(args.limit) > 0:
                command.extend(["--limit", str(args.limit)])
            add_common_easylens_args(command, common, device)
        else:
            raise SystemExit(f"Unsupported task type for {name}: {task}")
        run_command(command, env, args.dry_run)


def run_normal_rrg_medgemma15(args: argparse.Namespace, config: dict[str, Any], env: dict[str, str]) -> None:
    root = package_root()
    suite = config["normal_lesion_rrg_medgemma15"]
    runtime = config["runtime"]
    model_path = env_or_arg(args.model_path, suite.get("model_path_env"))
    benchmark_root = env_or_arg(args.normal_benchmark_root, suite.get("benchmark_root_env"))
    bank_root = env_or_arg(args.bank_root, suite.get("bank_root_env"), root / "banks" / "normal_rrg")
    output_root = env_or_arg(args.output_root, suite.get("output_root_env"), root / "outputs" / "normal_rrg_medgemma15")
    device = args.device or str(runtime["device"])

    runner = root / "code" / "runners" / "evaluate_rex_easylens_model_native.py"
    scorer = root / "code" / "runners" / "score_easylens_report_rrg.py"
    bank_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    selected = set(args.datasets or suite["datasets"].keys())
    unknown = selected.difference(suite["datasets"].keys())
    if unknown:
        raise SystemExit(f"Unknown normal-lesion dataset(s): {', '.join(sorted(unknown))}")

    build_cfg = suite["bank_build"]
    eval_cfg = suite["report_eval"]
    for dataset, dataset_cfg in suite["datasets"].items():
        if dataset not in selected:
            continue
        benchmark_dir = benchmark_root / dataset
        bank_path = bank_root / f"medgemma15_{dataset}_shape_bank.pt"
        if not bank_path.exists() and not args.skip_bank_build:
            build_command = [
                args.python_bin,
                str(runner),
                "build-bank",
                "--model-kind",
                str(suite["model_kind"]),
                "--model-path",
                str(model_path),
                "--benchmark-dir",
                str(benchmark_dir),
                "--output-path",
                str(bank_path),
                "--max-bg-per-image",
                str(build_cfg["max_bg_per_image"]),
                "--max-lesion-tokens",
                str(build_cfg["max_lesion_tokens"]),
                "--max-bg-global-tokens",
                str(build_cfg["max_bg_global_tokens"]),
                "--num-normal-prototypes",
                str(build_cfg["num_normal_prototypes"]),
                "--body-intensity-threshold",
                str(build_cfg["body_intensity_threshold"]),
                "--device",
                device,
            ]
            run_command(build_command, env, args.dry_run)

        out_dir = output_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_command = [
            args.python_bin,
            str(runner),
            "eval-report",
            "--model-kind",
            str(suite["model_kind"]),
            "--model-path",
            str(model_path),
            "--benchmark-dir",
            str(benchmark_dir),
            "--output-dir",
            str(out_dir),
            "--bank-path",
            str(bank_path),
            "--prompt-mode",
            str(eval_cfg["prompt_mode"]),
            "--probe-steps",
            str(eval_cfg["topk_steps"]),
            "--probe-vocab-topk",
            str(eval_cfg["probe_vocab_topk"]),
            "--max-new-tokens",
            str(eval_cfg["max_new_tokens"]),
            "--beta",
            str(dataset_cfg["beta"]),
            "--selector-k",
            str(dataset_cfg["selector_k"]),
            "--seed-topk",
            str(dataset_cfg["seed_topk"]),
            "--device",
            device,
        ]
        if int(args.limit) > 0:
            eval_command.extend(["--limit", str(args.limit)])
        run_command(eval_command, env, args.dry_run)

        score_command = [
            args.python_bin,
            str(scorer),
            "--benchmark-dir",
            str(benchmark_dir),
            "--predictions",
            str(out_dir / "report_answers.jsonl"),
            "--output-dir",
            str(out_dir),
            "--model-kind",
            str(suite["model_kind"]),
            "--model-label",
            "EasyLens-medgemma1.5",
        ]
        run_command(score_command, env, args.dry_run)


def parse_backbone_setting(tag: str) -> dict[str, Any]:
    if tag.startswith("b") and "_k" in tag:
        parts = tag.split("_")
        out: dict[str, Any] = {"raw": tag}
        for part in parts:
            if part.startswith("b"):
                out["beta"] = float(part[1:])
            elif re.fullmatch(r"k\d+", part):
                out["selector_k"] = int(part[1:])
            elif re.fullmatch(r"s\d+", part):
                out["seed_topk"] = int(part[1:])
        return out
    raise ValueError(f"Unsupported backbone setting tag: {tag}")


def run_backbone_rex(args: argparse.Namespace, config: dict[str, Any], env: dict[str, str]) -> None:
    root = package_root()
    runtime = config["runtime"]
    common = config["common_easylens_parameters"]
    rex_cfg = config["backbone_adaptation"]["rex"]
    if not args.model_kind:
        raise SystemExit("--model-kind is required for backbone-rex")
    if args.model_kind not in rex_cfg:
        raise SystemExit(f"Unsupported backbone model: {args.model_kind}")
    model_cfg = rex_cfg[args.model_kind]
    model_path = None if args.model_kind == "radfm" and not args.model_path else env_or_arg(args.model_path, "MODEL_PATH")
    benchmark_root = env_or_arg(args.benchmark_root, config["backbone_adaptation"].get("benchmark_root_env"), root / "benchmarks")
    bank_root = env_or_arg(args.bank_root, config["backbone_adaptation"].get("bank_root_env"))
    output_root = env_or_arg(args.output_root, config["backbone_adaptation"].get("output_root_env"), root / "outputs" / "backbone_rex")
    device = args.device or str(runtime["device"])
    runner = root / "code" / "runners" / "evaluate_rex_easylens_model_native.py"

    task_map = {
        "stat": {"benchmark": "rex_count", "runner_cmd": "eval-vqa", "task": "count", "max_new_tokens": runtime["count_select_max_new_tokens"]},
        "sel": {"benchmark": "rex_single_select", "runner_cmd": "eval-vqa", "task": "multiselect", "max_new_tokens": runtime["count_select_max_new_tokens"]},
        "gen": {"benchmark": "rex_report", "runner_cmd": "eval-report", "task": "report", "max_new_tokens": runtime["report_max_new_tokens"]},
    }
    selected_tasks = set(args.tasks or task_map.keys())
    unknown_tasks = selected_tasks.difference(task_map.keys())
    if unknown_tasks:
        raise SystemExit(f"Unknown backbone-rex task(s): {', '.join(sorted(unknown_tasks))}")
    variants = ["wo", "w"] if args.variant == "both" else [args.variant]
    default_model_bank_path = Path(args.bank_path).expanduser().resolve() if args.bank_path else bank_root / str(model_cfg["bank_file"])

    for task_name, task_meta in task_map.items():
        if task_name not in selected_tasks:
            continue
        setting_ref = str(model_cfg[task_name])
        if args.model_kind == "medgemma1.5" and setting_ref in config["subtle_lesion_medgemma15"]["tasks"]:
            task_cfg = config["subtle_lesion_medgemma15"]["tasks"][setting_ref]
            setting = {
                "beta": task_cfg["beta"],
                "selector_k": task_cfg["selector_k"],
                "seed_topk": task_cfg["seed_topk"],
            }
            task_bank_path = bank_root / str(task_cfg["bank_file"])
        else:
            setting = parse_backbone_setting(setting_ref)
            task_bank_path = default_model_bank_path

        for variant in variants:
            out_dir = output_root / args.model_kind / task_name / variant
            out_dir.mkdir(parents=True, exist_ok=True)
            command = [
                args.python_bin,
                str(runner),
                str(task_meta["runner_cmd"]),
                "--model-kind",
                args.model_kind,
                "--benchmark-dir",
                str(benchmark_root / str(task_meta["benchmark"])),
                "--output-dir",
                str(out_dir),
                "--beta",
                str(setting["beta"]),
                "--selector-k",
                str(setting["selector_k"]),
                "--seed-topk",
                str(setting["seed_topk"]),
                "--max-new-tokens",
                str(task_meta["max_new_tokens"]),
            ]
            if int(args.limit) > 0:
                command.extend(["--limit", str(args.limit)])
            if model_path is not None:
                command.extend(["--model-path", str(model_path)])
            if variant == "w":
                command.extend(["--bank-path", str(task_bank_path)])
            add_common_easylens_args(command, common, device)
            if task_meta["runner_cmd"] == "eval-vqa":
                command.extend(["--tasks", str(task_meta["task"])])
            else:
                command.extend(["--prompt-mode", "benchmark_raw", "--probe-steps", "3", "--probe-vocab-topk", "50"])
            run_command(command, env, args.dry_run)
            if task_meta["runner_cmd"] == "eval-vqa" and task_meta["task"] == "multiselect":
                score_single_select(
                    python_bin=args.python_bin,
                    runner_root=root,
                    benchmark_dir=benchmark_root / str(task_meta["benchmark"]),
                    predictions_path=out_dir / "multiselect_answers.jsonl",
                    output_dir=out_dir,
                    model_kind=args.model_kind,
                    limit=int(args.limit),
                    env=env,
                    dry_run=args.dry_run,
                )


def main() -> None:
    root = package_root()
    default_config = root / "configs" / "official_runner_config.json"
    parser = argparse.ArgumentParser(description="Official EasyLens evaluation runner.")
    parser.add_argument(
        "suite",
        choices=["subtle-medgemma15", "normal-rrg-medgemma15", "backbone-rex"],
        help="Evaluation suite to run.",
    )
    parser.add_argument("--config", default=str(default_config), help="Official runner config JSON.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used to launch sub-runners.")
    parser.add_argument("--model-path", default=None, help="Model checkpoint directory.")
    parser.add_argument("--model-kind", default=None, help="Backbone model kind for backbone suites.")
    parser.add_argument("--benchmark-root", default=None, help="Root containing subtle-lesion benchmark directories.")
    parser.add_argument("--normal-benchmark-root", default=None, help="Root containing normal-lesion RRG benchmark directories.")
    parser.add_argument("--bank-root", default=None, help="Directory containing or receiving shape-bank files.")
    parser.add_argument("--bank-path", default=None, help="Optional explicit bank file for backbone suites.")
    parser.add_argument("--output-root", default=None, help="Output directory.")
    parser.add_argument("--device", default=None, help="Execution device, e.g. cuda or cpu.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional subset for subtle-lesion suite.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset subset for normal-lesion suite.")
    parser.add_argument("--variant", choices=["w", "wo", "both"], default="w", help="Backbone EasyLens variant to run.")
    parser.add_argument("--limit", type=int, default=0, help="Optional example limit passed to sub-runners.")
    parser.add_argument("--skip-bank-build", action="store_true", help="For normal RRG, use existing banks only.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    config = load_config(Path(args.config).expanduser().resolve())
    env = os.environ.copy()
    env["PYTHONPATH"] = build_pythonpath(root, config)

    if args.suite == "subtle-medgemma15":
        run_subtle_medgemma15(args, config, env)
    elif args.suite == "normal-rrg-medgemma15":
        run_normal_rrg_medgemma15(args, config, env)
    elif args.suite == "backbone-rex":
        run_backbone_rex(args, config, env)
    else:
        raise SystemExit(f"Unsupported suite: {args.suite}")


if __name__ == "__main__":
    main()
