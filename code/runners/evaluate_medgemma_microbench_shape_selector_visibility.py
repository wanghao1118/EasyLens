from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor, Gemma3ForConditionalGeneration

from evaluate_medgemma_microbench_bank_selector_visibility import (
    build_reference_tokens,
    canonicalize_answer,
    move_inputs,
    select_support,
)
from evaluate_rex_shape_selector_hitmap import (
    build_cluster_member_lists,
    build_shape_heatmap,
    build_support_score,
    compute_patch_score,
    compute_position_background_maxsim,
    load_shape_bank,
    route_to_cluster_members,
)
from evaluate_server_vlm_microbench import ensure_dir, load_task_records, make_task_prompt, score_predictions
from prototype_guided_fundus import extract_layer_tokens, patched_vision_layer_tokens


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
    parser.add_argument("--support-mode", choices=["global_topk", "radius_suppressed_topk"], default="global_topk")
    parser.add_argument("--suppression-radius", type=int, default=1)
    parser.add_argument("--margin-weight", type=float, default=0.35)
    parser.add_argument("--shape-weight", type=float, default=1.0)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--seed-mode", choices=["all_positive", "local_peak", "local_peak_topk"], default="all_positive")
    parser.add_argument("--seed-peak-kernel", type=int, default=5)
    parser.add_argument("--seed-topk", type=int, default=32)
    parser.add_argument("--base-score-mode", choices=["all_score", "seed_only"], default="all_score")
    parser.add_argument("--support-score-mode", choices=["shape_only", "hybrid_max", "hybrid_sum"], default="shape_only")
    parser.add_argument("--raw-score-weight", type=float, default=1.0)
    parser.add_argument("--gap-score-weight", type=float, default=0.0)
    parser.add_argument("--retrieval-topm", type=int, default=64)
    parser.add_argument("--retrieval-temperature", type=float, default=0.1)
    parser.add_argument("--prototype-query-chunk", type=int, default=512)
    parser.add_argument("--bg-bank-chunk", type=int, default=1024)
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

    bank = load_shape_bank(args.bank_path, torch.device(model.device))
    cluster_members = build_cluster_member_lists(bank)

    run_meta = {
        "model_path": str(args.model_path),
        "benchmark_dir": str(args.benchmark_dir),
        "bank_path": str(args.bank_path),
        "tasks": list(args.tasks),
        "vision_layer": int(args.vision_layer),
        "beta": float(args.beta),
        "selector_k": int(args.selector_k),
        "support_mode": args.support_mode,
        "suppression_radius": int(args.suppression_radius),
        "margin_weight": float(args.margin_weight),
        "shape_weight": float(args.shape_weight),
        "score_threshold": float(args.score_threshold),
        "seed_mode": args.seed_mode,
        "seed_peak_kernel": int(args.seed_peak_kernel),
        "seed_topk": int(args.seed_topk),
        "base_score_mode": args.base_score_mode,
        "support_score_mode": args.support_score_mode,
        "raw_score_weight": float(args.raw_score_weight),
        "gap_score_weight": float(args.gap_score_weight),
        "retrieval_topm": int(args.retrieval_topm),
        "retrieval_temperature": float(args.retrieval_temperature),
        "retrieval_mode": "reference_bank",
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
            token_norm = torch.nn.functional.normalize(source_tokens.float(), dim=-1)
            lesion_sim, best_member_idx, proto_margin = route_to_cluster_members(
                token_norm,
                bank,
                cluster_members,
                query_chunk=int(args.prototype_query_chunk),
            )
            background_sim = compute_position_background_maxsim(
                token_norm,
                bank,
                chunk_size=int(args.bg_bank_chunk),
            )
            score, score_parts = compute_patch_score(
                lesion_sim,
                background_sim,
                proto_margin,
                margin_weight=float(args.margin_weight),
            )
            heatmap = build_shape_heatmap(
                score,
                lesion_sim,
                best_member_idx,
                bank,
                shape_weight=float(args.shape_weight),
                score_threshold=float(args.score_threshold),
                seed_mode=args.seed_mode,
                seed_peak_kernel=int(args.seed_peak_kernel),
                seed_topk=int(args.seed_topk),
                base_score_mode=args.base_score_mode,
            )
            support_score = build_support_score(
                score,
                heatmap,
                lesion_sim,
                background_sim,
                mode=args.support_score_mode,
                raw_score_weight=float(args.raw_score_weight),
                gap_score_weight=float(args.gap_score_weight),
            )
            alpha = select_support(
                support_score,
                support_mode=args.support_mode,
                k=int(args.selector_k),
                suppression_radius=int(args.suppression_radius),
            ).to(device=model.device)
            reference_tokens = build_reference_tokens(
                source_tokens=source_tokens,
                alpha=alpha,
                lesion_bank_raw=bank["lesion_bank"],
                lesion_bank_norm=bank["lesion_bank_norm"],
                lesion_delta_raw=None,
                topm=int(args.retrieval_topm),
                temperature=float(args.retrieval_temperature),
                similarity_chunk=int(args.similarity_chunk),
                retrieval_mode="reference_bank",
            )

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
                "token_delta",
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
                    "selector_mode": "shape_hitmap",
                    "support_mode": args.support_mode,
                    "seed_mode": args.seed_mode,
                    "seed_peak_kernel": int(args.seed_peak_kernel),
                    "seed_topk": int(args.seed_topk),
                    "base_score_mode": args.base_score_mode,
                    "selector_k": int(args.selector_k),
                    "suppression_radius": int(args.suppression_radius),
                    "beta": float(args.beta),
                    "vision_layer": int(args.vision_layer),
                    "retrieval_topm": int(args.retrieval_topm),
                    "retrieval_temperature": float(args.retrieval_temperature),
                    "selected_patch_count": float((alpha > 0).sum().item()),
                    "selected_patch_indices": [int(x) for x in torch.nonzero(alpha > 0, as_tuple=False).view(-1).tolist()],
                    "score_max": float(score.max().item()),
                    "score_mean": float(score.mean().item()),
                    "heatmap_max": float(heatmap.max().item()),
                    "heatmap_mean": float(heatmap.mean().item()),
                    "support_score_max": float(support_score.max().item()),
                    "support_score_mean": float(support_score.mean().item()),
                    "lesion_sim_max": float(lesion_sim.max().item()),
                    "lesion_sim_mean": float(lesion_sim.mean().item()),
                    "background_sim_max": float(background_sim.max().item()),
                    "background_sim_mean": float(background_sim.mean().item()),
                    "proto_margin_max": float(score_parts["margin_raw"].max().item()),
                    "proto_margin_mean": float(score_parts["margin_raw"].mean().item()),
                }
            )

            del image, pixel_inputs, pixel_values, source_tokens, token_norm
            del lesion_sim, best_member_idx, proto_margin, background_sim, score, score_parts, heatmap, support_score
            del alpha, reference_tokens, messages, prompt, inputs, generated
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
