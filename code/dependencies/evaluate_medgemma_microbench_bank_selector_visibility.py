from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor, Gemma3ForConditionalGeneration

from evaluate_server_vlm_microbench import (
    canonicalize_loc_answer,
    ensure_dir,
    load_task_records,
    score_predictions,
)
from prototype_guided_fundus import ENCODER_GRID, extract_layer_tokens, patched_vision_layer_tokens, robust_standardize_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MedGemma on visibility tasks with bank-guided early layer delta and rule-based selectors."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--bank-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["count", "yesno"], choices=["multiselect", "count", "yesno"])
    parser.add_argument("--vision-layer", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.75)
    parser.add_argument("--selector-mode", choices=["s1_gap", "s2_peak", "s3_stability", "learned_mlp"], required=True)
    parser.add_argument("--selector-k", type=int, required=True)
    parser.add_argument("--support-mode", choices=["global_topk", "radius_suppressed_topk"], required=True)
    parser.add_argument("--suppression-radius", type=int, default=1)
    parser.add_argument("--peak-weight", type=float, default=1.0)
    parser.add_argument("--stability-weight", type=float, default=1.0)
    parser.add_argument("--peak-kernel", type=int, default=3)
    parser.add_argument("--stability-kernel", type=int, default=3)
    parser.add_argument("--selector-ckpt", type=Path, default=None)
    parser.add_argument("--retrieval-topm", type=int, default=8)
    parser.add_argument("--retrieval-temperature", type=float, default=0.05)
    parser.add_argument("--retrieval-mode", choices=["auto", "reference_bank", "delta_bank"], default="auto")
    parser.add_argument("--similarity-chunk", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def move_inputs(inputs: dict, device: str) -> dict:
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def canonicalize_answer(task: str, raw_text: str, grid_size: int, prompt: str | None = None) -> str:
    if task == "multiselect":
        return canonicalize_loc_answer(raw_text, grid_size=grid_size, prompt=prompt)
    return raw_text.strip()


def pool_token_grid(tokens: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return tokens.float()
    if kernel_size % 2 == 0:
        kernel_size += 1
    token_grid = tokens.float().view(1, ENCODER_GRID, ENCODER_GRID, -1).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(token_grid, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return pooled.permute(0, 2, 3, 1).reshape(-1, tokens.shape[-1])


def normalized_patch_coords(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(ENCODER_GRID, device=device, dtype=torch.float32)
    cols = torch.arange(ENCODER_GRID, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    denom = float(max(ENCODER_GRID - 1, 1))
    return xx.reshape(-1) / denom, yy.reshape(-1) / denom


def compute_chunked_maxsim(query_norm: torch.Tensor, bank_norm: torch.Tensor, chunk_size: int) -> torch.Tensor:
    best = torch.full((query_norm.shape[0],), -1e9, device=query_norm.device, dtype=torch.float32)
    for start in range(0, bank_norm.shape[0], chunk_size):
        chunk = bank_norm[start : start + chunk_size]
        sims = torch.matmul(query_norm, chunk.T)
        best = torch.maximum(best, sims.max(dim=1).values)
    return best


def compute_selector_feature_map(
    source_tokens: torch.Tensor,
    lesion_bank_norm: torch.Tensor,
    background_bank_norm: torch.Tensor,
    *,
    peak_kernel: int,
    stability_kernel: int,
    similarity_chunk: int,
) -> dict[str, torch.Tensor]:
    token_norm = F.normalize(source_tokens.float(), dim=-1)
    lesion_max = compute_chunked_maxsim(token_norm, lesion_bank_norm, similarity_chunk)
    background_max = compute_chunked_maxsim(token_norm, background_bank_norm, similarity_chunk)
    gap_raw = lesion_max - background_max
    gap_score = robust_standardize_torch(gap_raw)
    local_context = pool_token_grid(gap_raw.unsqueeze(-1), peak_kernel).squeeze(-1)
    peak_raw = gap_raw - local_context
    peak_score = robust_standardize_torch(peak_raw)
    stable_tokens = pool_token_grid(source_tokens, stability_kernel)
    stability_raw = F.cosine_similarity(source_tokens.float(), stable_tokens, dim=-1)
    stability_score = robust_standardize_torch(stability_raw)
    x_coord, y_coord = normalized_patch_coords(source_tokens.device)
    return {
        "token_norm": token_norm,
        "lesion_max": lesion_max,
        "background_max": background_max,
        "gap_raw": gap_raw,
        "gap_score": gap_score,
        "peak_raw": peak_raw,
        "peak_score": peak_score,
        "stability_raw": stability_raw,
        "stability_score": stability_score,
        "x_coord": x_coord,
        "y_coord": y_coord,
    }


def assemble_selector_features(
    source_tokens: torch.Tensor,
    feature_map: dict[str, torch.Tensor],
    *,
    use_token_features: bool,
) -> torch.Tensor:
    cols: list[torch.Tensor] = []
    if use_token_features:
        cols.append(feature_map["token_norm"].float())
    aux_names = [
        "lesion_max",
        "background_max",
        "gap_raw",
        "gap_score",
        "peak_raw",
        "peak_score",
        "stability_raw",
        "stability_score",
        "x_coord",
        "y_coord",
    ]
    cols.extend(feature_map[name].float().unsqueeze(-1) for name in aux_names)
    return torch.cat(cols, dim=-1)


def compute_selector_scores(
    source_tokens: torch.Tensor,
    lesion_bank_norm: torch.Tensor,
    background_bank_norm: torch.Tensor,
    *,
    selector_mode: str,
    peak_weight: float,
    stability_weight: float,
    peak_kernel: int,
    stability_kernel: int,
    similarity_chunk: int,
) -> dict[str, torch.Tensor]:
    out = compute_selector_feature_map(
        source_tokens,
        lesion_bank_norm,
        background_bank_norm,
        peak_kernel=peak_kernel,
        stability_kernel=stability_kernel,
        similarity_chunk=similarity_chunk,
    )
    gap_raw = out["gap_raw"]
    gap_score = out["gap_score"]
    if selector_mode == "s1_gap":
        out["selector_score"] = gap_raw
        return out

    if selector_mode == "s2_peak":
        out["selector_score"] = gap_score + float(peak_weight) * out["peak_score"]
        return out

    if selector_mode == "s3_stability":
        out["selector_score"] = gap_score + float(stability_weight) * out["stability_score"]
        return out

    raise ValueError(f"Unsupported selector_mode: {selector_mode}")


class LearnedSelectorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_selector_checkpoint(ckpt_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = dict(ckpt.get("meta", {}))
    input_dim = int(meta["input_dim"])
    hidden_dim = int(meta["hidden_dim"])
    model = LearnedSelectorMLP(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    ckpt["model"] = model
    return ckpt


def idx_to_row_col(idx: int) -> tuple[int, int]:
    return divmod(int(idx), ENCODER_GRID)


def select_global_topk(score: torch.Tensor, k: int) -> torch.Tensor:
    k = max(1, min(int(k), int(score.numel())))
    positive_idx = torch.nonzero(score > 0, as_tuple=False).view(-1)
    candidate_idx = positive_idx if positive_idx.numel() > 0 else torch.arange(score.numel(), device=score.device)
    candidate_scores = score[candidate_idx]
    keep = candidate_idx[torch.topk(candidate_scores, k=min(k, int(candidate_idx.numel())), dim=0).indices]
    alpha = torch.zeros_like(score, dtype=torch.float32)
    alpha[keep] = 1.0
    return alpha


def select_radius_suppressed_topk(score: torch.Tensor, k: int, radius: int) -> torch.Tensor:
    k = max(1, min(int(k), int(score.numel())))
    radius = max(0, int(radius))
    alpha = torch.zeros_like(score, dtype=torch.float32)
    suppressed = torch.zeros_like(score, dtype=torch.bool)
    positive_idx = torch.nonzero(score > 0, as_tuple=False).view(-1)
    work_score = score.clone()
    if positive_idx.numel() == 0:
        positive_idx = torch.arange(score.numel(), device=score.device)
    picks = 0
    while picks < k:
        masked_score = work_score.masked_fill(suppressed, -1e9)
        best_idx = int(torch.argmax(masked_score).item())
        if masked_score[best_idx] <= -1e8:
            break
        alpha[best_idx] = 1.0
        picks += 1
        row, col = idx_to_row_col(best_idx)
        row_min = max(0, row - radius)
        row_max = min(ENCODER_GRID - 1, row + radius)
        col_min = max(0, col - radius)
        col_max = min(ENCODER_GRID - 1, col + radius)
        for rr in range(row_min, row_max + 1):
            base = rr * ENCODER_GRID
            suppressed[base + col_min : base + col_max + 1] = True
    if alpha.sum() <= 0:
        alpha[int(torch.argmax(score).item())] = 1.0
    return alpha


def select_support(score: torch.Tensor, *, support_mode: str, k: int, suppression_radius: int) -> torch.Tensor:
    if support_mode == "global_topk":
        return select_global_topk(score, k)
    if support_mode == "radius_suppressed_topk":
        return select_radius_suppressed_topk(score, k, suppression_radius)
    raise ValueError(f"Unsupported support_mode: {support_mode}")


def chunked_topm_single(query_norm: torch.Tensor, bank_norm: torch.Tensor, topm: int, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    top_vals = torch.empty(0, device=query_norm.device, dtype=torch.float32)
    top_idx = torch.empty(0, device=query_norm.device, dtype=torch.long)
    for start in range(0, bank_norm.shape[0], chunk_size):
        chunk = bank_norm[start : start + chunk_size]
        sims = torch.matmul(query_norm, chunk.T).view(-1)
        local_k = min(int(topm), int(sims.numel()))
        vals, idx = torch.topk(sims, k=local_k, dim=0)
        idx = idx + start
        top_vals = torch.cat([top_vals, vals], dim=0)
        top_idx = torch.cat([top_idx, idx], dim=0)
        keep_k = min(int(topm), int(top_vals.numel()))
        keep = torch.topk(top_vals, k=keep_k, dim=0).indices
        top_vals = top_vals[keep]
        top_idx = top_idx[keep]
    order = torch.argsort(top_vals, descending=True)
    return top_vals[order], top_idx[order]


def build_reference_tokens(
    source_tokens: torch.Tensor,
    alpha: torch.Tensor,
    lesion_bank_raw: torch.Tensor,
    lesion_bank_norm: torch.Tensor,
    lesion_delta_raw: torch.Tensor | None = None,
    *,
    topm: int,
    temperature: float,
    similarity_chunk: int,
    retrieval_mode: str = "reference_bank",
) -> torch.Tensor:
    reference_tokens = source_tokens.float().clone()
    selected_idx = torch.nonzero(alpha > 0, as_tuple=False).view(-1)
    if selected_idx.numel() == 0:
        return reference_tokens
    temp = max(float(temperature), 1e-6)
    for idx in selected_idx.tolist():
        query_norm = F.normalize(source_tokens[idx : idx + 1].float(), dim=-1)
        vals, nn_idx = chunked_topm_single(query_norm, lesion_bank_norm, topm=max(1, int(topm)), chunk_size=similarity_chunk)
        weights = torch.softmax(vals / temp, dim=0)
        if retrieval_mode == "delta_bank":
            if lesion_delta_raw is None:
                raise ValueError("lesion_delta_raw is required for retrieval_mode=delta_bank.")
            retrieved_delta = (weights.unsqueeze(-1) * lesion_delta_raw[nn_idx].float()).sum(dim=0)
            reference_tokens[idx] = source_tokens[idx].float() + retrieved_delta
        else:
            reference_tokens[idx] = (weights.unsqueeze(-1) * lesion_bank_raw[nn_idx].float()).sum(dim=0)
    return reference_tokens


def load_bank_tensors(bank_path: Path, device: torch.device) -> dict[str, torch.Tensor | str | bool | None]:
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    lesion_key = "lesion_bank" if ("lesion_bank" in bank or "lesion_delta_bank" in bank) else "lesion_prototypes"
    background_key = "background_bank" if "background_bank" in bank else "background_prototypes"
    if lesion_key == "lesion_bank" and "lesion_bank" not in bank:
        raise ValueError("Bank has lesion_delta_bank but no aligned lesion_bank for retrieval.")
    lesion_raw = torch.as_tensor(bank[lesion_key], dtype=torch.float32, device=device)
    background_raw = torch.as_tensor(bank[background_key], dtype=torch.float32, device=device)
    lesion_delta_raw = None
    if "lesion_delta_bank" in bank:
        lesion_delta_raw = torch.as_tensor(bank["lesion_delta_bank"], dtype=torch.float32, device=device)
        if lesion_delta_raw.shape != lesion_raw.shape:
            raise ValueError(
                f"lesion_delta_bank shape {tuple(lesion_delta_raw.shape)} does not match lesion_bank shape {tuple(lesion_raw.shape)}"
            )
    return {
        "lesion_raw": lesion_raw,
        "background_raw": background_raw,
        "lesion_norm": F.normalize(lesion_raw.float(), dim=-1),
        "background_norm": F.normalize(background_raw.float(), dim=-1),
        "lesion_delta_raw": lesion_delta_raw,
        "retrieval_source": lesion_key,
        "has_delta_bank": bool(lesion_delta_raw is not None),
    }


def resolve_retrieval_mode(requested_mode: str, bank: dict[str, torch.Tensor | str | bool | None]) -> str:
    if requested_mode == "auto":
        return "delta_bank" if bool(bank.get("has_delta_bank")) else "reference_bank"
    if requested_mode == "delta_bank" and not bool(bank.get("has_delta_bank")):
        raise ValueError("retrieval_mode=delta_bank was requested, but bank has no lesion_delta_bank.")
    return requested_mode


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

    bank = load_bank_tensors(args.bank_path, torch.device(model.device))
    retrieval_mode = resolve_retrieval_mode(args.retrieval_mode, bank)
    learned_selector = None
    if args.selector_mode == "learned_mlp":
        if args.selector_ckpt is None:
            raise ValueError("--selector-ckpt is required when --selector-mode learned_mlp")
        learned_selector = load_selector_checkpoint(args.selector_ckpt, torch.device(model.device))
    metrics_by_task: dict[str, dict] = {}

    run_meta = {
        "model_path": str(args.model_path),
        "benchmark_dir": str(args.benchmark_dir),
        "bank_path": str(args.bank_path),
        "tasks": list(args.tasks),
        "vision_layer": int(args.vision_layer),
        "beta": float(args.beta),
        "selector_mode": args.selector_mode,
        "support_mode": args.support_mode,
        "selector_k": int(args.selector_k),
        "suppression_radius": int(args.suppression_radius),
        "peak_weight": float(args.peak_weight),
        "stability_weight": float(args.stability_weight),
        "selector_ckpt": (str(args.selector_ckpt) if args.selector_ckpt is not None else None),
        "retrieval_topm": int(args.retrieval_topm),
        "retrieval_temperature": float(args.retrieval_temperature),
        "retrieval_source": bank["retrieval_source"],
        "retrieval_mode": retrieval_mode,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for task in args.tasks:
        records = load_task_records(args.benchmark_dir, task, limit=(args.limit if args.limit > 0 else None))
        pred_rows: list[dict] = []
        for idx, rec in enumerate(records, start=1):
            image = Image.open(rec.image_path).convert("RGB")
            pixel_inputs = image_processor(images=image, return_tensors="pt")
            pixel_values = pixel_inputs["pixel_values"].to(
                device=model.device,
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )
            source_tokens = extract_layer_tokens(model, pixel_values, int(args.vision_layer))
            if args.selector_mode == "learned_mlp":
                feature_map = compute_selector_feature_map(
                    source_tokens,
                    bank["lesion_norm"],
                    bank["background_norm"],
                    peak_kernel=int(args.peak_kernel),
                    stability_kernel=int(args.stability_kernel),
                    similarity_chunk=int(args.similarity_chunk),
                )
                features = assemble_selector_features(
                    source_tokens,
                    feature_map,
                    use_token_features=bool(learned_selector["meta"].get("use_token_features", True)),
                )
                selector_score = learned_selector["model"](features).float()
                score_outputs = dict(feature_map)
                score_outputs["selector_score"] = selector_score
            else:
                score_outputs = compute_selector_scores(
                    source_tokens,
                    bank["lesion_norm"],
                    bank["background_norm"],
                    selector_mode=args.selector_mode,
                    peak_weight=float(args.peak_weight),
                    stability_weight=float(args.stability_weight),
                    peak_kernel=int(args.peak_kernel),
                    stability_kernel=int(args.stability_kernel),
                    similarity_chunk=int(args.similarity_chunk),
                )
                selector_score = score_outputs["selector_score"]
            alpha = select_support(
                selector_score,
                support_mode=args.support_mode,
                k=int(args.selector_k),
                suppression_radius=int(args.suppression_radius),
            ).to(device=model.device)
            reference_tokens = build_reference_tokens(
                source_tokens=source_tokens,
                alpha=alpha,
                lesion_bank_raw=bank["lesion_raw"],
                lesion_bank_norm=bank["lesion_norm"],
                lesion_delta_raw=bank["lesion_delta_raw"],
                topm=int(args.retrieval_topm),
                temperature=float(args.retrieval_temperature),
                similarity_chunk=int(args.similarity_chunk),
                retrieval_mode=retrieval_mode,
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": rec.prompt},
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
                    "prompt": rec.prompt,
                    "text": canonicalize_answer(task, raw_text, grid_size=3, prompt=rec.prompt),
                    "raw_text": raw_text,
                    "selector_mode": args.selector_mode,
                    "support_mode": args.support_mode,
                    "selector_k": int(args.selector_k),
                    "suppression_radius": int(args.suppression_radius),
                    "beta": float(args.beta),
                    "vision_layer": int(args.vision_layer),
                    "retrieval_topm": int(args.retrieval_topm),
                    "retrieval_source": bank["retrieval_source"],
                    "retrieval_mode": retrieval_mode,
                    "selected_patch_count": float((alpha > 0).sum().item()),
                    "selected_patch_indices": [int(x) for x in torch.nonzero(alpha > 0, as_tuple=False).view(-1).tolist()],
                    "selector_score_max": float(selector_score.max().item()),
                    "selector_score_mean": float(selector_score.mean().item()),
                    "gap_score_max": float(score_outputs["gap_raw"].max().item()),
                    "gap_score_mean": float(score_outputs["gap_raw"].mean().item()),
                }
            )

            del image, pixel_inputs, pixel_values, source_tokens, score_outputs, selector_score, alpha, reference_tokens
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
            model_kind=f"bank-selector-{args.selector_mode}",
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
