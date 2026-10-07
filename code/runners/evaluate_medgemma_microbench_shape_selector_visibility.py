from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor, Gemma3ForConditionalGeneration

from evaluate_medgemma_microbench_bank_selector_visibility import (
    canonicalize_answer,
    move_inputs,
)
from evaluate_server_vlm_microbench import ensure_dir, load_task_records, make_task_prompt, score_predictions
from prototype_guided_fundus import extract_layer_tokens, patched_vision_layer_tokens
from easylens_core import EasyLensConfig, compute_easylens, load_easylens_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MedGemma on visibility tasks using the Rex shape selector for support plus reference-bank token injection."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--bank-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["multiselect", "count", "yesno"], choices=["multiselect", "count", "yesno"])
    parser.add_argument("--vision-layer", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.75)
    parser.add_argument("--selector-k", type=int, default=256)
    parser.add_argument("--shape-weight", type=float, default=1.0)
    parser.add_argument("--seed-topk", type=int, default=32)
    parser.add_argument("--prototype-topm", type=int, default=1)
    parser.add_argument("--counterfactual-temperature", type=float, default=1.0)
    parser.add_argument("--amplifier-threshold", type=float, default=0.0)
    parser.add_argument("--similarity-chunk", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--strict-count-prompt", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    image_processor = AutoImageProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if device == "cuda" else torch.float32),
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    bank = load_easylens_bank(args.bank_path, torch.device(model.device))
    easylens_config = EasyLensConfig(
        selector_k=int(args.selector_k),
        seed_topk=int(args.seed_topk),
        prototype_topm=int(args.prototype_topm),
        counterfactual_temperature=float(args.counterfactual_temperature),
        morphology_weight=float(args.shape_weight),
        amplifier_threshold=float(args.amplifier_threshold),
        similarity_chunk=int(args.similarity_chunk),
    )

    run_meta = {
        "model_path": str(args.model_path),
        "benchmark_dir": str(args.benchmark_dir),
        "bank_path": str(args.bank_path),
        "tasks": list(args.tasks),
        "vision_layer": int(args.vision_layer),
        "beta": float(args.beta),
        "selector_k": int(args.selector_k),
        "shape_weight": float(args.shape_weight),
        "seed_topk": int(args.seed_topk),
        "prototype_topm": int(args.prototype_topm),
        "counterfactual_temperature": float(args.counterfactual_temperature),
        "amplifier_threshold": float(args.amplifier_threshold),
        "method_profile": "paper_equations_27_40",
        "seed_selection": "global_topk",
        "candidate_selection": "global_topk_calibrated_score",
        "reference_retrieval": "cluster_nearest_neighbor",
        "residual_update": "z_plus_alpha_w_reference_delta",
        "bank_alignment": bank["alignment"],
        "bank_schema_version": int(bank["schema_version"]),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_by_task: dict[str, dict] = {}
    for task in args.tasks:
        records = load_task_records(args.benchmark_dir, task, limit=(args.limit if args.limit > 0 else None))
        pred_rows: list[dict] = []
        for idx, rec in enumerate(records, start=1):
            image = Image.open(rec.image_path).convert("RGB")
            prompt_text = make_task_prompt(task, rec.prompt, strict_count_prompt=bool(args.strict_count_prompt))
            pixel_inputs = image_processor(images=image, return_tensors="pt")
            pixel_values = pixel_inputs["pixel_values"].to(
                device=model.device,
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )
            source_tokens = extract_layer_tokens(model, pixel_values, int(args.vision_layer))
            reference_tokens, alpha, method_meta = compute_easylens(source_tokens, bank, easylens_config)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = move_inputs(inputs, model.device)

            with patched_vision_layer_tokens(
                model,
                int(args.vision_layer),
                alpha,
                float(args.beta),
                "paper_residual",
                reference_tokens,
            ):
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                    )

            input_len = inputs["input_ids"].shape[1]
            raw_text = processor.decode(generated[0][input_len:], skip_special_tokens=True).strip()
            pred_rows.append(
                {
                    "sample_id": rec.sample_id,
                    "task": task,
                    "image": rec.image_rel,
                    "prompt": prompt_text,
                    "text": canonicalize_answer(task, raw_text, grid_size=3, prompt=prompt_text),
                    "raw_text": raw_text,
                    "seed_topk": int(args.seed_topk),
                    "selector_k": int(args.selector_k),
                    "beta": float(args.beta),
                    "vision_layer": int(args.vision_layer),
                    **method_meta,
                }
            )

            del image, pixel_inputs, pixel_values, source_tokens, alpha, reference_tokens, method_meta
            del messages, prompt, inputs, generated
            if idx % 8 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        pred_path = args.output_dir / f"{task}_answers.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for row in pred_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = score_predictions(
            benchmark_dir=args.benchmark_dir,
            task=task,
            predictions_path=pred_path,
            output_dir=args.output_dir / f"_rescored_{task}",
            model_kind="shape-selector",
            sample_ids={rec.sample_id for rec in records},
        )
        metrics_by_task[task] = summary["metrics"]["overall"]
        (args.output_dir / f"{task}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics_by_task, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
