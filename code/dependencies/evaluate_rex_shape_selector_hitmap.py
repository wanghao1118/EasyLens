from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, Gemma3ForConditionalGeneration

from microbench_prototype_utils import (
    lesion_background_patch_masks,
    load_benchmark_image,
    load_benchmark_manifest,
    parse_grid_cells,
)
from prototype_guided_fundus import (
    ENCODER_GRID,
    extract_layer_tokens,
    local_peak_mask_torch,
    resize_image,
    robust_standardize_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Rex-only shape-aware selector and report hit-rate metrics.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--bank-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vision-layer", type=int, default=1)
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
    parser.add_argument("--body-intensity-threshold", type=float, default=5.0)
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--prototype-query-chunk", type=int, default=512)
    parser.add_argument("--bg-bank-chunk", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-heatmaps", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_shape_bank(bank_path: Path, device: torch.device) -> dict[str, torch.Tensor | int | float]:
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    lesion_bank = torch.as_tensor(bank["lesion_bank"], dtype=torch.float32, device=device)
    lesion_prototypes = torch.as_tensor(bank["lesion_prototypes"], dtype=torch.float32, device=device)
    lesion_shape_templates = torch.as_tensor(bank["lesion_shape_templates"], dtype=torch.float32, device=device)
    cluster_member_order = bank.get("lesion_cluster_member_order")
    cluster_offsets = bank.get("lesion_cluster_offsets")
    if cluster_member_order is None or cluster_offsets is None:
        cluster_ids = np.asarray(bank["lesion_cluster_ids"], dtype=np.int32)
        cluster_member_order = np.argsort(cluster_ids, kind="stable").astype(np.int32)
        num_clusters = int(len(lesion_prototypes))
        cluster_offsets_np = np.zeros(num_clusters + 1, dtype=np.int32)
        cluster_ids_sorted = cluster_ids[cluster_member_order]
        for cluster_idx in range(num_clusters):
            cluster_offsets_np[cluster_idx + 1] = cluster_offsets_np[cluster_idx] + int((cluster_ids_sorted == cluster_idx).sum())
        cluster_offsets = cluster_offsets_np
    background_position_bank = torch.as_tensor(bank["background_position_bank"], dtype=torch.float32, device=device)
    background_position_counts = torch.as_tensor(bank["background_position_counts"], dtype=torch.int64, device=device)
    background_position_offsets = torch.as_tensor(bank["background_position_offsets"], dtype=torch.int64, device=device)
    background_global_bank = torch.as_tensor(bank["background_global_bank"], dtype=torch.float32, device=device)
    return {
        "lesion_bank": lesion_bank,
        "lesion_bank_norm": F.normalize(lesion_bank, dim=-1),
        "lesion_prototypes": lesion_prototypes,
        "lesion_prototypes_norm": F.normalize(lesion_prototypes, dim=-1),
        "lesion_shape_templates": lesion_shape_templates,
        "cluster_member_order": torch.as_tensor(cluster_member_order, dtype=torch.int64, device=device),
        "cluster_offsets": torch.as_tensor(cluster_offsets, dtype=torch.int64, device=device),
        "background_position_bank": background_position_bank,
        "background_position_bank_norm": F.normalize(background_position_bank, dim=-1) if background_position_bank.numel() > 0 else background_position_bank,
        "background_position_counts": background_position_counts,
        "background_position_offsets": background_position_offsets,
        "background_global_bank": background_global_bank,
        "background_global_bank_norm": F.normalize(background_global_bank, dim=-1),
        "shape_radius": int(bank.get("shape_radius", 2)),
        "ring_weight": float(bank.get("ring_weight", 0.5)),
    }


def build_cluster_member_lists(bank: dict[str, torch.Tensor | int | float]) -> list[torch.Tensor]:
    cluster_member_order = bank["cluster_member_order"]
    cluster_offsets = bank["cluster_offsets"]
    out: list[torch.Tensor] = []
    for cluster_idx in range(int(cluster_offsets.numel()) - 1):
        start = int(cluster_offsets[cluster_idx].item())
        end = int(cluster_offsets[cluster_idx + 1].item())
        out.append(cluster_member_order[start:end])
    return out


def compute_position_background_maxsim(
    token_norm: torch.Tensor,
    bank: dict[str, torch.Tensor | int | float],
    *,
    chunk_size: int,
) -> torch.Tensor:
    out = torch.empty(token_norm.shape[0], device=token_norm.device, dtype=torch.float32)
    pos_bank_norm = bank["background_position_bank_norm"]
    pos_counts = bank["background_position_counts"]
    pos_offsets = bank["background_position_offsets"]
    global_bank_norm = bank["background_global_bank_norm"]
    for pos_idx in range(token_norm.shape[0]):
        count = int(pos_counts[pos_idx].item())
        if count > 0:
            start = int(pos_offsets[pos_idx].item())
            end = int(pos_offsets[pos_idx + 1].item())
            cur_bank = pos_bank_norm[start:end]
        else:
            cur_bank = global_bank_norm
        query = token_norm[pos_idx : pos_idx + 1]
        best = torch.full((1,), -1e9, device=token_norm.device, dtype=torch.float32)
        for start in range(0, cur_bank.shape[0], chunk_size):
            sims = torch.matmul(query, cur_bank[start : start + chunk_size].T).view(-1)
            if sims.numel() > 0:
                best = torch.maximum(best, sims.max().view(1))
        out[pos_idx] = best[0]
    return out


def route_to_cluster_members(
    token_norm: torch.Tensor,
    bank: dict[str, torch.Tensor | int | float],
    cluster_members: list[torch.Tensor],
    *,
    query_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    proto_sims = torch.matmul(token_norm, bank["lesion_prototypes_norm"].T)
    topk_proto = min(2, int(proto_sims.shape[1]))
    top_proto_vals, top_proto_idx = torch.topk(proto_sims, k=topk_proto, dim=1)
    best_proto = top_proto_idx[:, 0]
    proto_margin = torch.zeros(token_norm.shape[0], device=token_norm.device, dtype=torch.float32)
    if topk_proto > 1:
        proto_margin = top_proto_vals[:, 0] - top_proto_vals[:, 1]

    best_member_sim = torch.empty(token_norm.shape[0], device=token_norm.device, dtype=torch.float32)
    best_member_idx = torch.empty(token_norm.shape[0], device=token_norm.device, dtype=torch.int64)
    for cluster_idx, member_idx in enumerate(cluster_members):
        sample_idx = torch.nonzero(best_proto == cluster_idx, as_tuple=False).view(-1)
        if sample_idx.numel() == 0:
            continue
        if member_idx.numel() == 0:
            best_member_sim[sample_idx] = -1e9
            best_member_idx[sample_idx] = 0
            continue
        members = bank["lesion_bank_norm"][member_idx]
        for start in range(0, sample_idx.numel(), query_chunk):
            chunk_idx = sample_idx[start : start + query_chunk]
            sims = torch.matmul(token_norm[chunk_idx], members.T)
            vals, local_idx = sims.max(dim=1)
            best_member_sim[chunk_idx] = vals
            best_member_idx[chunk_idx] = member_idx[local_idx]
    return best_member_sim, best_member_idx, proto_margin


def compute_patch_score(
    lesion_sim: torch.Tensor,
    background_sim: torch.Tensor,
    proto_margin: torch.Tensor,
    *,
    margin_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    gap_raw = lesion_sim - background_sim
    gap_score = robust_standardize_torch(gap_raw)
    margin_score = robust_standardize_torch(proto_margin) if proto_margin.numel() > 1 else torch.zeros_like(proto_margin)
    score = gap_score + float(margin_weight) * margin_score
    return score, {
        "gap_raw": gap_raw,
        "gap_score": gap_score,
        "margin_raw": proto_margin,
        "margin_score": margin_score,
    }


def add_template_inplace(heatmap: torch.Tensor, template: torch.Tensor, center_idx: int, amplitude: float) -> None:
    if amplitude == 0.0:
        return
    radius = template.shape[0] // 2
    row, col = divmod(int(center_idx), ENCODER_GRID)
    row_min = max(0, row - radius)
    row_max = min(ENCODER_GRID, row + radius + 1)
    col_min = max(0, col - radius)
    col_max = min(ENCODER_GRID, col + radius + 1)
    tr0 = row_min - (row - radius)
    tc0 = col_min - (col - radius)
    heatmap[row_min:row_max, col_min:col_max] += amplitude * template[tr0 : tr0 + (row_max - row_min), tc0 : tc0 + (col_max - col_min)]


def select_shape_seeds(
    score: torch.Tensor,
    *,
    seed_mode: str,
    score_threshold: float,
    peak_kernel: int,
    seed_topk: int,
) -> torch.Tensor:
    positive = score > float(score_threshold)
    if seed_mode == "all_positive":
        return positive
    peak_mask = local_peak_mask_torch(score, int(peak_kernel))
    seeds = positive & peak_mask
    if seed_mode == "local_peak":
        return seeds
    if seed_mode == "local_peak_topk":
        seed_idx = torch.nonzero(seeds, as_tuple=False).view(-1)
        if seed_idx.numel() == 0:
            return seeds
        keep_k = min(max(1, int(seed_topk)), int(seed_idx.numel()))
        keep_idx = seed_idx[torch.topk(score[seed_idx], k=keep_k, dim=0).indices]
        out = torch.zeros_like(seeds, dtype=torch.bool)
        out[keep_idx] = True
        return out
    raise ValueError(f"Unsupported seed mode: {seed_mode}")


def build_shape_heatmap(
    score: torch.Tensor,
    lesion_sim: torch.Tensor,
    best_member_idx: torch.Tensor,
    bank: dict[str, torch.Tensor | int | float],
    *,
    shape_weight: float,
    score_threshold: float,
    seed_mode: str,
    seed_peak_kernel: int,
    seed_topk: int,
    base_score_mode: str,
) -> torch.Tensor:
    seed_mask = select_shape_seeds(
        score,
        seed_mode=seed_mode,
        score_threshold=float(score_threshold),
        peak_kernel=int(seed_peak_kernel),
        seed_topk=int(seed_topk),
    )
    if base_score_mode == "seed_only":
        base_score = torch.zeros_like(score)
        base_score[seed_mask] = torch.relu(score[seed_mask])
    else:
        base_score = torch.relu(score)
    base = base_score.view(ENCODER_GRID, ENCODER_GRID).clone()
    if float(shape_weight) == 0.0:
        return base.view(-1)
    templates = bank["lesion_shape_templates"]
    candidate_idx = torch.nonzero(seed_mask, as_tuple=False).view(-1)
    for patch_idx in candidate_idx.tolist():
        sim_gain = float(torch.clamp((lesion_sim[patch_idx] + 1.0) * 0.5, min=0.0, max=1.0).item())
        amplitude = float(torch.relu(score[patch_idx]).item()) * float(shape_weight) * sim_gain
        template = templates[int(best_member_idx[patch_idx].item())]
        add_template_inplace(base, template, patch_idx, amplitude)
    return base.view(-1)


def build_support_score(
    score: torch.Tensor,
    heatmap: torch.Tensor,
    lesion_sim: torch.Tensor,
    background_sim: torch.Tensor,
    *,
    mode: str,
    raw_score_weight: float,
    gap_score_weight: float,
) -> torch.Tensor:
    base_pos = torch.relu(score)
    gap_pos = torch.relu(robust_standardize_torch(lesion_sim - background_sim))
    if mode == "shape_only":
        return heatmap
    if mode == "hybrid_max":
        aux = float(raw_score_weight) * base_pos + float(gap_score_weight) * gap_pos
        return torch.maximum(heatmap, aux)
    if mode == "hybrid_sum":
        return heatmap + float(raw_score_weight) * base_pos + float(gap_score_weight) * gap_pos
    raise ValueError(f"Unsupported support-score mode: {mode}")


def summarize_hits(rows: list[dict], topk_list: list[int]) -> dict[str, float | int | list[int]]:
    summary: dict[str, float | int | list[int]] = {
        "num_samples": len(rows),
        "topk": topk_list,
    }
    for k in topk_list:
        values = [float(row[f"hit_at_{k}"]) for row in rows]
        summary[f"hit_rate_at_{k}"] = float(np.mean(values)) if values else 0.0
    return summary


def unique_image_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique_rows: list[dict[str, str]] = []
    for row in rows:
        key = str(row.get("source_sample_id") or row.get("sample_id") or row.get("output_image") or "")
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topk_list = sorted({max(1, int(x)) for x in args.topk})
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    bank = load_shape_bank(args.bank_path, device)
    cluster_members = build_cluster_member_lists(bank)
    processor = AutoImageProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if device.type == "cuda" else torch.float32),
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    rows = unique_image_rows(load_benchmark_manifest(args.benchmark_dir))
    if args.limit > 0:
        rows = rows[: args.limit]

    heatmap_dir = args.output_dir / "heatmaps"
    if args.save_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    pred_rows: list[dict] = []
    for idx, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id", ""))
        image = load_benchmark_image(args.benchmark_dir, row)
        image_resized = resize_image(image, mode=Image.Resampling.BILINEAR)
        pixel_inputs = processor(images=image_resized, return_tensors="pt")
        pixel_values = pixel_inputs["pixel_values"].to(
            device=model.device,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        source_tokens = extract_layer_tokens(model, pixel_values, int(args.vision_layer))
        token_norm = F.normalize(source_tokens.float(), dim=-1)
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

        grid_cells = parse_grid_cells(row)
        grid_size = int(float(row.get("grid_size", 0) or 0))
        lesion_mask, _background_mask, _active_mask = lesion_background_patch_masks(
            grid_cells=grid_cells,
            grid_size=grid_size,
            image_resized=image_resized,
            body_intensity_threshold=float(args.body_intensity_threshold),
        )
        lesion_mask_t = torch.as_tensor(lesion_mask.astype(np.bool_), device=heatmap.device)
        topk_max = min(int(max(topk_list)), int(support_score.numel()))
        top_vals, top_idx = torch.topk(support_score, k=topk_max, dim=0)

        record = {
            "sample_id": sample_id,
            "score_max": float(score.max().item()),
            "score_mean": float(score.mean().item()),
            "heatmap_max": float(heatmap.max().item()),
            "heatmap_mean": float(heatmap.mean().item()),
            "gap_raw_max": float(score_parts["gap_raw"].max().item()),
            "gap_raw_mean": float(score_parts["gap_raw"].mean().item()),
            "positive_patch_count": int(lesion_mask_t.sum().item()),
            "top_indices": [int(x) for x in top_idx.tolist()],
            "top_values": [float(x) for x in top_vals.tolist()],
        }
        for k in topk_list:
            cur_idx = top_idx[:k]
            hit = bool(lesion_mask_t[cur_idx].any().item())
            record[f"hit_at_{k}"] = hit
        pred_rows.append(record)

        if args.save_heatmaps:
            torch.save(
                {
                    "sample_id": sample_id,
                    "heatmap": heatmap.view(ENCODER_GRID, ENCODER_GRID).cpu(),
                    "support_score": support_score.view(ENCODER_GRID, ENCODER_GRID).cpu(),
                    "score": score.view(ENCODER_GRID, ENCODER_GRID).cpu(),
                    "lesion_sim": lesion_sim.view(ENCODER_GRID, ENCODER_GRID).cpu(),
                    "background_sim": background_sim.view(ENCODER_GRID, ENCODER_GRID).cpu(),
                    "best_member_idx": best_member_idx.view(ENCODER_GRID, ENCODER_GRID).cpu(),
                    "lesion_mask": lesion_mask_t.view(ENCODER_GRID, ENCODER_GRID).cpu(),
                },
                heatmap_dir / f"{sample_id}.pt",
            )

        print(
            json.dumps(
                {
                    "phase": "selector_eval",
                    "index": idx,
                    "total": len(rows),
                    "sample_id": sample_id,
                    "score_max": record["score_max"],
                    "heatmap_max": record["heatmap_max"],
                    f"hit_at_{topk_list[0]}": record[f"hit_at_{topk_list[0]}"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        del image, image_resized, pixel_inputs, pixel_values, source_tokens, token_norm
        del lesion_sim, best_member_idx, proto_margin, background_sim, score, score_parts, heatmap
        del lesion_mask_t, top_vals, top_idx, support_score
        if idx % 8 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize_hits(pred_rows, topk_list)
    run_config = {
        "model_path": str(args.model_path),
        "benchmark_dir": str(args.benchmark_dir),
        "bank_path": str(args.bank_path),
        "vision_layer": int(args.vision_layer),
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
        "topk": topk_list,
        "prototype_query_chunk": int(args.prototype_query_chunk),
        "bg_bank_chunk": int(args.bg_bank_chunk),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "selector_predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in pred_rows) + ("\n" if pred_rows else ""),
        encoding="utf-8",
    )
    (args.output_dir / "selector_hit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
