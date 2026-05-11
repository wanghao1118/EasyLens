from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from evaluate_rex_report_baseline_token_probs import (
    DENIAL_PHRASES,
    LESION_PHRASES,
    HuluRunner,
    LingshuRunner,
    LlavaRunner,
    MedGemmaRunner,
    RadFMRunner,
    analyze_scores,
    candidate_first_token_info,
    get_tokenizer,
    load_records as load_report_records,
    summarize as summarize_report_probs,
)
from evaluate_server_vlm_microbench import (
    canonicalize_loc_answer,
    ensure_dir,
    load_manifest,
    load_task_records,
    make_task_prompt,
    score_predictions,
    write_json,
    write_jsonl,
)
from microbench_prototype_utils import load_benchmark_image, load_benchmark_manifest, parse_grid_cells
from prototype_guided_fundus import extract_layer_tokens, patched_vision_layer_tokens, run_kmeans


LLAVA_MODEL = Path("/path/to/llava-med-v1.5-mistral-7b")
HULU_MODEL = Path("/path/to/Hulu-Med-7B")
LINGSHU_MODEL = Path("/path/to/Lingshu-7B")
MEDGEMMA15_MODEL = Path("/path/to/medgemma-1.5-4b-it")
MEDGEMMA4_MODEL = Path("/path/to/medgemma-4b-it")
RADFM_QUICK_DEMO_ROOT = Path("/path/to/RadFM/Quick_demo")
RADFM_CHECKPOINT = Path("/path/to/RadFM/pytorch_model.bin")


def safe_load_sample_manifest(benchmark_dir: Path) -> dict[str, dict[str, str]]:
    manifest_path = benchmark_dir / "manifest.csv"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "sample_id" not in reader.fieldnames:
            return {}
        return {str(row["sample_id"]): row for row in reader}


@dataclass
class EasyLensParams:
    beta: float
    selector_k: int
    support_mode: str
    suppression_radius: int
    margin_weight: float
    shape_weight: float
    score_threshold: float
    seed_mode: str
    seed_peak_kernel: int
    seed_topk: int
    base_score_mode: str
    support_score_mode: str
    raw_score_weight: float
    gap_score_weight: float
    retrieval_topm: int
    retrieval_temperature: float
    query_chunk: int
    bg_bank_chunk: int
    similarity_chunk: int
    selector_override: str
    amplifier_mode: str
    selector_ckpt: str | None


def parse_jsonish(value: Any, default: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def sample_indices(num_rows: int, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if max_rows <= 0 or num_rows <= max_rows:
        return np.arange(num_rows, dtype=np.int64)
    return np.sort(rng.choice(num_rows, size=max_rows, replace=False)).astype(np.int64)


def infer_square_grid(num_tokens: int) -> tuple[int, int, int]:
    """Return (prefix_tokens, grid_side, suffix_tokens) for a square patch sequence."""
    root = int(math.isqrt(int(num_tokens)))
    if root * root == int(num_tokens):
        return 0, root, 0
    root = int(math.isqrt(int(num_tokens) - 1))
    if root * root == int(num_tokens) - 1:
        return 1, root, 0
    for prefix in range(0, min(8, int(num_tokens))):
        remain = int(num_tokens) - prefix
        root = int(math.isqrt(remain))
        if root * root == remain:
            return prefix, root, 0
    raise ValueError(f"Cannot infer a square visual-token grid from {num_tokens} tokens.")


def split_patch_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if tokens.ndim != 2:
        raise ValueError(f"Expected [N,D] tokens, got shape={tuple(tokens.shape)}")
    prefix, grid_side, suffix = infer_square_grid(tokens.shape[0])
    end = tokens.shape[0] - suffix
    return tokens[:prefix], tokens[prefix:end], tokens[end:], grid_side


def normalize_feature_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        tensor = output
    elif isinstance(output, (tuple, list)):
        tensor = next((x for x in output if isinstance(x, torch.Tensor)), None)
        if tensor is None:
            raise ValueError(f"No tensor found in feature output type={type(output)}")
    elif hasattr(output, "last_hidden_state"):
        tensor = output.last_hidden_state
    else:
        raise ValueError(f"Unsupported feature output type={type(output)}")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected visual features [B,N,D], got shape={tuple(tensor.shape)}")
    return tensor


def replace_feature_output(output: Any, new_tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return new_tensor
    if isinstance(output, tuple):
        values = list(output)
        for idx, value in enumerate(values):
            if isinstance(value, torch.Tensor):
                values[idx] = new_tensor
                return tuple(values)
        return output
    if isinstance(output, list):
        values = list(output)
        for idx, value in enumerate(values):
            if isinstance(value, torch.Tensor):
                values[idx] = new_tensor
                return values
        return output
    if hasattr(output, "last_hidden_state"):
        output.last_hidden_state = new_tensor
        return output
    return output


def dynamic_patch_mean_intensity(image: Image.Image, grid_side: int) -> np.ndarray:
    gray = np.asarray(image.convert("L").resize((grid_side, grid_side), Image.Resampling.BILINEAR), dtype=np.float32)
    return gray.reshape(-1)


def dynamic_grid_labels(grid_size: int, token_grid: int) -> np.ndarray:
    y_edges = np.linspace(0, token_grid, grid_size + 1, dtype=int)
    x_edges = np.linspace(0, token_grid, grid_size + 1, dtype=int)
    labels: list[str] = []
    for row in range(grid_size):
        y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
        for yy in range(y0, y1):
            del yy
            for col in range(grid_size):
                x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
                labels.extend([f"{chr(ord('A') + row)}{col + 1}"] * max(0, x1 - x0))
    if len(labels) != token_grid * token_grid:
        raise RuntimeError(f"Bad label map size: {len(labels)} vs {token_grid * token_grid}")
    return np.asarray(labels, dtype=object)


def build_supervision_masks(row: dict[str, str], image: Image.Image, token_grid: int, body_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    mask_text = str(row.get("mask_path") or row.get("mask") or "").strip()
    if mask_text:
        mask_path = Path(mask_text)
        mask = Image.open(mask_path).convert("L")
        mask_small = mask.resize((token_grid, token_grid), Image.Resampling.BILINEAR)
        lesion = (np.asarray(mask_small, dtype=np.float32).reshape(-1) > 0.0)
        active = dynamic_patch_mean_intensity(image, token_grid) > float(body_threshold)
        background = (~lesion) & active
        if not np.any(background):
            background = ~lesion
        return lesion.astype(bool), background.astype(bool)

    grid_size = int(float(row.get("grid_size", 3) or 3))
    cells = parse_grid_cells(row)
    labels = dynamic_grid_labels(grid_size=grid_size, token_grid=token_grid)
    lesion = np.isin(labels, np.asarray(cells, dtype=object))
    active = dynamic_patch_mean_intensity(image, token_grid) > float(body_threshold)
    background = (~lesion) & active
    if not np.any(background):
        background = ~lesion
    return lesion.astype(bool), background.astype(bool)


def dilate_binary(window: np.ndarray) -> np.ndarray:
    padded = np.pad(window.astype(bool), 1, mode="constant", constant_values=False)
    out = np.zeros_like(window, dtype=bool)
    height, width = window.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out |= padded[1 + dr : 1 + dr + height, 1 + dc : 1 + dc + width]
    return out


def build_local_shape_template(mask_flat: np.ndarray, center_idx: int, grid_side: int, radius: int, ring_weight: float, ring_mode: str) -> np.ndarray:
    radius = max(0, int(radius))
    size = 2 * radius + 1
    mask_grid = mask_flat.reshape(grid_side, grid_side).astype(bool)
    row, col = divmod(int(center_idx), grid_side)
    template_pos = np.zeros((size, size), dtype=np.float32)
    row_min = max(0, row - radius)
    row_max = min(grid_side, row + radius + 1)
    col_min = max(0, col - radius)
    col_max = min(grid_side, col + radius + 1)
    rr0 = row_min - (row - radius)
    cc0 = col_min - (col - radius)
    template_pos[rr0 : rr0 + (row_max - row_min), cc0 : cc0 + (col_max - col_min)] = mask_grid[row_min:row_max, col_min:col_max]
    template_pos = (template_pos > 0).astype(np.float32)
    if template_pos[radius, radius] <= 0:
        template_pos[radius, radius] = 1.0
    ring = dilate_binary(template_pos > 0) & ~(template_pos > 0)
    if ring_mode == "boost":
        return (template_pos + float(ring_weight) * ring.astype(np.float32)).astype(np.float32)
    return (template_pos - float(ring_weight) * ring.astype(np.float32)).astype(np.float32)


def robust_standardize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.numel() <= 1:
        return torch.zeros_like(x)
    median = x.median()
    mad = (x - median).abs().median()
    scale = torch.clamp(1.4826 * mad, min=1e-6)
    return (x - median) / scale


def local_peak_mask(score: torch.Tensor, grid_side: int, kernel: int) -> torch.Tensor:
    kernel = max(1, int(kernel))
    if kernel <= 1:
        return torch.ones_like(score, dtype=torch.bool)
    if kernel % 2 == 0:
        kernel += 1
    grid = score.float().view(1, 1, grid_side, grid_side)
    pooled = torch.nn.functional.max_pool2d(grid, kernel_size=kernel, stride=1, padding=kernel // 2)
    return (grid == pooled).view(-1)


def compute_position_background_maxsim(token_norm: torch.Tensor, bank: dict[str, Any], chunk_size: int) -> torch.Tensor:
    out = torch.empty(token_norm.shape[0], device=token_norm.device, dtype=torch.float32)
    pos_bank = bank["background_position_bank_norm"]
    pos_counts = bank["background_position_counts"]
    pos_offsets = bank["background_position_offsets"]
    global_bank = bank["background_global_bank_norm"]
    for pos_idx in range(token_norm.shape[0]):
        count = int(pos_counts[pos_idx].item()) if pos_idx < int(pos_counts.numel()) else 0
        if count > 0:
            start = int(pos_offsets[pos_idx].item())
            end = int(pos_offsets[pos_idx + 1].item())
            cur_bank = pos_bank[start:end]
        else:
            cur_bank = global_bank
        query = token_norm[pos_idx : pos_idx + 1]
        best = torch.full((1,), -1e9, device=token_norm.device)
        for start in range(0, cur_bank.shape[0], max(1, int(chunk_size))):
            sims = torch.matmul(query, cur_bank[start : start + int(chunk_size)].T).view(-1)
            if sims.numel():
                best = torch.maximum(best, sims.max().view(1))
        out[pos_idx] = best[0]
    return out


def build_cluster_member_lists(bank: dict[str, Any]) -> list[torch.Tensor]:
    order = bank["cluster_member_order"]
    offsets = bank["cluster_offsets"]
    out: list[torch.Tensor] = []
    for idx in range(int(offsets.numel()) - 1):
        out.append(order[int(offsets[idx].item()) : int(offsets[idx + 1].item())])
    return out


def route_to_cluster_members(token_norm: torch.Tensor, bank: dict[str, Any], cluster_members: list[torch.Tensor], query_chunk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    proto_sims = torch.matmul(token_norm, bank["lesion_prototypes_norm"].T)
    topk = min(2, int(proto_sims.shape[1]))
    top_vals, top_idx = torch.topk(proto_sims, k=topk, dim=1)
    best_proto = top_idx[:, 0]
    proto_margin = torch.zeros(token_norm.shape[0], device=token_norm.device, dtype=torch.float32)
    if topk > 1:
        proto_margin = top_vals[:, 0] - top_vals[:, 1]
    best_sim = torch.full((token_norm.shape[0],), -1e9, device=token_norm.device, dtype=torch.float32)
    best_idx = torch.zeros((token_norm.shape[0],), device=token_norm.device, dtype=torch.int64)
    for cluster_idx, member_idx in enumerate(cluster_members):
        sample_idx = torch.nonzero(best_proto == cluster_idx, as_tuple=False).view(-1)
        if sample_idx.numel() == 0 or member_idx.numel() == 0:
            continue
        members = bank["lesion_bank_norm"][member_idx]
        for start in range(0, sample_idx.numel(), max(1, int(query_chunk))):
            chunk_idx = sample_idx[start : start + int(query_chunk)]
            sims = torch.matmul(token_norm[chunk_idx], members.T)
            vals, local_idx = sims.max(dim=1)
            best_sim[chunk_idx] = vals
            best_idx[chunk_idx] = member_idx[local_idx]
    return best_sim, best_idx, proto_margin


def add_template(heatmap: torch.Tensor, template: torch.Tensor, center_idx: int, amplitude: float, grid_side: int) -> None:
    if float(amplitude) == 0.0:
        return
    radius = int(template.shape[0] // 2)
    row, col = divmod(int(center_idx), grid_side)
    row_min = max(0, row - radius)
    row_max = min(grid_side, row + radius + 1)
    col_min = max(0, col - radius)
    col_max = min(grid_side, col + radius + 1)
    tr0 = row_min - (row - radius)
    tc0 = col_min - (col - radius)
    heatmap[row_min:row_max, col_min:col_max] += float(amplitude) * template[tr0 : tr0 + row_max - row_min, tc0 : tc0 + col_max - col_min]


def build_shape_heatmap(score: torch.Tensor, lesion_sim: torch.Tensor, best_member_idx: torch.Tensor, bank: dict[str, Any], params: EasyLensParams) -> torch.Tensor:
    grid_side = int(bank["grid_side"])
    positive = score > float(params.score_threshold)
    if params.seed_mode == "all_positive":
        seeds = positive
    else:
        peaks = local_peak_mask(score, grid_side, int(params.seed_peak_kernel))
        seeds = positive & peaks
        if params.seed_mode == "local_peak_topk":
            seed_idx = torch.nonzero(seeds, as_tuple=False).view(-1)
            out = torch.zeros_like(seeds, dtype=torch.bool)
            if seed_idx.numel() > 0:
                k = min(max(1, int(params.seed_topk)), int(seed_idx.numel()))
                keep = seed_idx[torch.topk(score[seed_idx], k=k).indices]
                out[keep] = True
            seeds = out
    if params.base_score_mode == "seed_only":
        base_score = torch.zeros_like(score)
        base_score[seeds] = torch.relu(score[seeds])
    else:
        base_score = torch.relu(score)
    base = base_score.view(grid_side, grid_side).clone()
    if float(params.shape_weight) == 0.0:
        return base.view(-1)
    templates = bank["lesion_shape_templates"]
    for patch_idx in torch.nonzero(seeds, as_tuple=False).view(-1).tolist():
        member_idx = int(best_member_idx[patch_idx].item())
        template = templates[member_idx]
        amplitude = float(torch.relu(lesion_sim[patch_idx]).item()) * float(params.shape_weight)
        add_template(base, template, patch_idx, amplitude, grid_side)
    return base.view(-1)


def build_support_score(score: torch.Tensor, heatmap: torch.Tensor, lesion_sim: torch.Tensor, background_sim: torch.Tensor, params: EasyLensParams) -> torch.Tensor:
    if params.support_score_mode == "shape_only":
        return heatmap
    gap = lesion_sim - background_sim
    if params.support_score_mode == "hybrid_max":
        return torch.maximum(heatmap, score)
    if params.support_score_mode == "hybrid_sum":
        return float(params.raw_score_weight) * heatmap + score + float(params.gap_score_weight) * robust_standardize(gap)
    raise ValueError(f"Bad support score mode: {params.support_score_mode}")


def select_support(score: torch.Tensor, params: EasyLensParams, grid_side: int) -> torch.Tensor:
    k = max(1, min(int(params.selector_k), int(score.numel())))
    alpha = torch.zeros_like(score, dtype=torch.float32)
    candidate = torch.nonzero(score > 0, as_tuple=False).view(-1)
    if candidate.numel() == 0:
        candidate = torch.arange(score.numel(), device=score.device)
    if params.support_mode == "global_topk":
        keep = candidate[torch.topk(score[candidate], k=min(k, int(candidate.numel()))).indices]
        alpha[keep] = 1.0
        return alpha
    if params.support_mode != "radius_suppressed_topk":
        raise ValueError(f"Bad support mode: {params.support_mode}")
    suppressed = torch.zeros_like(score, dtype=torch.bool)
    work_score = score.clone()
    picks = 0
    radius = max(0, int(params.suppression_radius))
    while picks < k:
        masked = work_score.masked_fill(suppressed, -1e9)
        best = int(torch.argmax(masked).item())
        if masked[best] <= -1e8:
            break
        alpha[best] = 1.0
        picks += 1
        row, col = divmod(best, grid_side)
        for rr in range(max(0, row - radius), min(grid_side, row + radius + 1)):
            c0 = max(0, col - radius)
            c1 = min(grid_side, col + radius + 1)
            suppressed[rr * grid_side + c0 : rr * grid_side + c1] = True
    if alpha.sum() <= 0:
        alpha[int(torch.argmax(score).item())] = 1.0
    return alpha


def chunked_topm(query_norm: torch.Tensor, bank_norm: torch.Tensor, topm: int, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    vals_all = torch.empty(0, device=query_norm.device, dtype=torch.float32)
    idx_all = torch.empty(0, device=query_norm.device, dtype=torch.long)
    for start in range(0, bank_norm.shape[0], max(1, int(chunk_size))):
        sims = torch.matmul(query_norm, bank_norm[start : start + int(chunk_size)].T).view(-1)
        if sims.numel() == 0:
            continue
        vals, idx = torch.topk(sims, k=min(max(1, int(topm)), int(sims.numel())))
        vals_all = torch.cat([vals_all, vals])
        idx_all = torch.cat([idx_all, idx + start])
        keep = torch.topk(vals_all, k=min(max(1, int(topm)), int(vals_all.numel()))).indices
        vals_all = vals_all[keep]
        idx_all = idx_all[keep]
    order = torch.argsort(vals_all, descending=True)
    return vals_all[order], idx_all[order]


def build_reference_tokens(source_tokens: torch.Tensor, alpha: torch.Tensor, bank: dict[str, Any], params: EasyLensParams) -> torch.Tensor:
    reference = source_tokens.float().clone()
    selected = torch.nonzero(alpha > 0, as_tuple=False).view(-1)
    if selected.numel() == 0:
        return reference
    temp = max(float(params.retrieval_temperature), 1e-6)
    lesion_bank = bank["lesion_bank"]
    lesion_bank_norm = bank["lesion_bank_norm"]
    for idx in selected.tolist():
        query = F.normalize(source_tokens[idx : idx + 1].float(), dim=-1)
        vals, nn_idx = chunked_topm(query, lesion_bank_norm, int(params.retrieval_topm), int(params.similarity_chunk))
        weights = torch.softmax(vals / temp, dim=0)
        reference[idx] = (weights.unsqueeze(-1) * lesion_bank[nn_idx].float()).sum(dim=0)
    return reference


def compute_mlp_selector_alpha(source_tokens: torch.Tensor, bank: dict[str, Any], params: EasyLensParams) -> torch.Tensor:
    selector = bank.get("_selector_checkpoint")
    if selector is None:
        raise ValueError("MLP selector requires --selector-ckpt.")
    from evaluate_medgemma_microbench_bank_selector_visibility import assemble_selector_features, compute_selector_feature_map

    feature_map = compute_selector_feature_map(
        source_tokens.float(),
        bank["lesion_bank_norm"],
        bank["background_global_bank_norm"],
        peak_kernel=int(params.seed_peak_kernel),
        stability_kernel=int(params.seed_peak_kernel),
        similarity_chunk=int(params.similarity_chunk),
    )
    features = assemble_selector_features(
        source_tokens.float(),
        feature_map,
        use_token_features=bool(selector["meta"].get("use_token_features", True)),
    )
    with torch.no_grad():
        selector_score = selector["model"](features).float()
    return select_support(selector_score, params, int(bank["grid_side"]))


def compute_easylens_tokens(
    source_tokens: torch.Tensor,
    bank: dict[str, Any],
    params: EasyLensParams,
    alpha_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if int(source_tokens.shape[0]) != int(bank["grid_side"]) ** 2:
        raise ValueError(f"Token-grid mismatch: got {source_tokens.shape[0]} tokens, bank grid={bank['grid_side']}")
    if int(source_tokens.shape[-1]) != int(bank["feature_dim"]):
        raise ValueError(f"Feature dim mismatch: got {source_tokens.shape[-1]}, bank dim={bank['feature_dim']}")
    token_norm = F.normalize(source_tokens.float(), dim=-1)
    cluster_members = build_cluster_member_lists(bank)
    lesion_sim, best_member_idx, proto_margin = route_to_cluster_members(token_norm, bank, cluster_members, int(params.query_chunk))
    background_sim = compute_position_background_maxsim(token_norm, bank, int(params.bg_bank_chunk))
    gap = lesion_sim - background_sim
    score = robust_standardize(gap) + float(params.margin_weight) * robust_standardize(proto_margin)
    heatmap = build_shape_heatmap(score, lesion_sim, best_member_idx, bank, params)
    support_score = build_support_score(score, heatmap, lesion_sim, background_sim, params)
    if alpha_override is None and params.selector_override == "mlp":
        alpha = compute_mlp_selector_alpha(source_tokens, bank, params).to(device=source_tokens.device)
        selector_used = "mlp"
    elif alpha_override is None:
        alpha = select_support(support_score, params, int(bank["grid_side"])).to(device=source_tokens.device)
        selector_used = "easytag"
    else:
        alpha = alpha_override.to(device=source_tokens.device, dtype=torch.float32)
        selector_used = str(params.selector_override)
    reference_tokens = build_reference_tokens(source_tokens, alpha, bank, params)
    meta = {
        "selector_override": selector_used,
        "selected_patch_count": int((alpha > 0).sum().item()),
        "selected_patch_indices": [int(x) for x in torch.nonzero(alpha > 0, as_tuple=False).view(-1).tolist()],
        "score_max": float(score.max().item()),
        "score_mean": float(score.mean().item()),
        "heatmap_max": float(heatmap.max().item()),
        "support_score_max": float(support_score.max().item()),
        "lesion_sim_max": float(lesion_sim.max().item()),
        "background_sim_max": float(background_sim.max().item()),
    }
    return reference_tokens, alpha, meta


def build_gt_alpha(row: dict[str, Any], image: Image.Image, grid_side: int, device: torch.device) -> torch.Tensor:
    lesion_mask, _background_mask = build_supervision_masks(row, image, int(grid_side), body_threshold=5.0)
    alpha = torch.as_tensor(lesion_mask.astype(np.float32), device=device)
    if alpha.numel() != int(grid_side) * int(grid_side):
        raise ValueError(f"GT alpha size mismatch: got {alpha.numel()}, grid={grid_side}")
    if float(alpha.sum().item()) <= 0:
        raise ValueError("GT mask selector produced no positive patches.")
    return alpha


def load_native_bank(path: Path, device: torch.device) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    bank: dict[str, Any] = dict(raw)
    if "grid_side" not in bank:
        if "background_position_counts" in bank:
            bank["grid_side"] = int(math.isqrt(len(bank["background_position_counts"])))
        else:
            bank["grid_side"] = int(math.isqrt(int(np.asarray(bank["lesion_patch_positions"]).max()) + 1))
    if "feature_dim" not in bank:
        bank["feature_dim"] = int(np.asarray(bank["lesion_bank"]).shape[-1])
    for key in ["lesion_bank", "lesion_prototypes", "lesion_shape_templates", "background_global_bank", "background_position_bank"]:
        bank[key] = torch.as_tensor(bank[key], dtype=torch.float32, device=device)
    bank["lesion_bank_norm"] = F.normalize(bank["lesion_bank"], dim=-1)
    bank["lesion_prototypes_norm"] = F.normalize(bank["lesion_prototypes"], dim=-1)
    bank["background_global_bank_norm"] = F.normalize(bank["background_global_bank"], dim=-1)
    if bank["background_position_bank"].numel() > 0:
        bank["background_position_bank_norm"] = F.normalize(bank["background_position_bank"], dim=-1)
    else:
        bank["background_position_bank_norm"] = bank["background_position_bank"]
    bank["background_position_counts"] = torch.as_tensor(bank["background_position_counts"], dtype=torch.int64, device=device)
    bank["background_position_offsets"] = torch.as_tensor(bank["background_position_offsets"], dtype=torch.int64, device=device)
    bank["cluster_member_order"] = torch.as_tensor(bank["lesion_cluster_member_order"], dtype=torch.int64, device=device)
    bank["cluster_offsets"] = torch.as_tensor(bank["lesion_cluster_offsets"], dtype=torch.int64, device=device)
    return bank


def maybe_load_selector_checkpoint(bank: dict[str, Any] | None, params: EasyLensParams, device: torch.device) -> dict[str, Any] | None:
    if bank is None or params.selector_override != "mlp":
        return bank
    if not params.selector_ckpt:
        raise ValueError("--selector-ckpt is required when --selector-override mlp")
    from evaluate_medgemma_microbench_bank_selector_visibility import load_selector_checkpoint

    bank["_selector_checkpoint"] = load_selector_checkpoint(Path(params.selector_ckpt), device)
    return bank


def build_runner(model_kind: str, model_path: Path | None, device: str):
    if model_kind == "llava-med":
        return LlavaRunner(model_path or LLAVA_MODEL, device)
    if model_kind == "hulu-med-7b":
        return HuluRunner(model_path or HULU_MODEL, device)
    if model_kind == "lingshu":
        return LingshuRunner(model_path or LINGSHU_MODEL, device)
    if model_kind in {"medgemma", "medgemma1.5"}:
        default = MEDGEMMA15_MODEL if model_kind == "medgemma1.5" else MEDGEMMA4_MODEL
        return MedGemmaRunner(model_path or default, device)
    if model_kind == "radfm":
        return RadFMRunner(RADFM_QUICK_DEMO_ROOT, RADFM_CHECKPOINT, device)
    raise ValueError(f"Unsupported model kind: {model_kind}")


class AlignedMedGemmaVqaRunner:
    """MedGemma VQA runner matched to evaluate_server_vlm_microbench.py."""

    def __init__(self, model_path: Path, device: str) -> None:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() and device != "cpu" else None,
        )
        if device == "cpu" or not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        self.model.eval()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def generate_with_scores(
        self,
        image: Image.Image,
        image_path: Path,
        prompt_text: str,
        max_new_tokens: int,
    ) -> tuple[str, Any, torch.Tensor]:
        del image_path
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        generated = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        input_len = inputs["input_ids"].shape[1]
        text = self.processor.decode(generated[0][input_len:], skip_special_tokens=True).strip()
        return text, None, generated[0][input_len:]


def build_vqa_runner(model_kind: str, model_path: Path | None, device: str):
    if model_kind in {"medgemma", "medgemma1.5"}:
        default = MEDGEMMA15_MODEL if model_kind == "medgemma1.5" else MEDGEMMA4_MODEL
        return AlignedMedGemmaVqaRunner(model_path or default, device)
    return build_runner(model_kind, model_path, device)


def runner_device(runner) -> torch.device:
    device = runner.device
    return device if isinstance(device, torch.device) else torch.device(device)


def extract_llava_raw_features(runner: LlavaRunner, image: Image.Image) -> torch.Tensor:
    from llava.mm_utils import process_images

    image_tensor = process_images([image], runner.image_processor, runner.model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(runner.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(runner.device, dtype=torch.float16)
    return extract_raw_from_llava_like_model(runner.model, image_tensor)


def extract_hulu_raw_features(runner: HuluRunner, image_path: Path) -> torch.Tensor:
    conversation = [{"role": "user", "content": [{"type": "image", "image": {"image_path": str(image_path)}}, {"type": "text", "text": "Describe the image."}]}]
    inputs = runner.processor(conversation=conversation, add_system_prompt=True, add_generation_prompt=True, return_tensors="pt")
    if "pixel_values" not in inputs:
        raise ValueError(f"Hulu processor did not return pixel_values; keys={list(inputs.keys())}")
    pixel_values = inputs["pixel_values"].to(runner.device)
    if pixel_values.is_floating_point():
        pixel_values = pixel_values.to(torch.bfloat16)
    vision_tower = find_vision_tower(runner.model)
    kwargs = {}
    for key in ["grid_sizes", "merge_sizes"]:
        if key in inputs:
            kwargs[key] = inputs[key].to(runner.device)
    with torch.no_grad():
        raw = vision_tower(pixel_values, **kwargs)
    return normalize_feature_output(raw)


def extract_medgemma_layer_features(runner: MedGemmaRunner, image: Image.Image, layer_idx: int) -> torch.Tensor:
    image_processor = getattr(runner.processor, "image_processor", None)
    if image_processor is None:
        from transformers import AutoImageProcessor

        image_processor = AutoImageProcessor.from_pretrained(str(getattr(runner, "model_path", MEDGEMMA15_MODEL)), trust_remote_code=True)
    pixel_inputs = image_processor(images=image, return_tensors="pt")
    dtype = torch.bfloat16 if runner_device(runner).type == "cuda" else torch.float32
    pixel_values = pixel_inputs["pixel_values"].to(device=runner.device, dtype=dtype)
    return extract_layer_tokens(runner.model, pixel_values, int(layer_idx)).detach()


def capture_radfm_vision_features(runner: RadFMRunner, image_path: Path, prompt: str = "Describe the image.") -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(normalize_feature_output(output).detach())
        return output

    handle = runner.model.embedding_layer.vision_encoder.register_forward_hook(hook)
    try:
        image_list = [{"img_path": str(image_path), "position": 0}]
        text, vision_x = runner.demo.combine_and_preprocess(prompt, image_list, runner.image_padding_tokens)
        lang_x = runner.tokenizer(text, max_length=2048, truncation=True, return_tensors="pt")["input_ids"].to(torch.long).to(runner.device)
        vision_x = vision_x.to(runner.device)
        if runner_device(runner).type == "cuda":
            vision_x = vision_x.half()
        with torch.no_grad():
            runner.model.embedding_layer.flag = "Text"
            runner.model.embedding_layer(lang_x, vision_x)
    finally:
        handle.remove()
    if not captured:
        raise ValueError("RadFM vision_encoder hook did not capture features.")
    return captured[-1][0]


def extract_runner_features(runner, model_kind: str, image: Image.Image, image_path: Path, vision_layer: int) -> tuple[torch.Tensor, int]:
    if model_kind == "llava-med":
        raw = extract_llava_raw_features(runner, image)[0]
    elif model_kind == "hulu-med-7b":
        raw = extract_hulu_raw_features(runner, image_path)[0]
    elif model_kind == "lingshu":
        raw = capture_lingshu_vision_features(runner, image)
    elif model_kind in {"medgemma", "medgemma1.5"}:
        raw = extract_medgemma_layer_features(runner, image, int(vision_layer))
    elif model_kind == "radfm":
        raw = capture_radfm_vision_features(runner, image_path)
    else:
        raise ValueError(model_kind)
    _prefix, patch_tokens, _suffix, grid_side = split_patch_tokens(raw.float())
    return patch_tokens.detach(), grid_side


def find_vision_tower(model):
    candidates = [model]
    if hasattr(model, "get_model"):
        try:
            candidates.append(model.get_model())
        except Exception:
            pass
    for obj in candidates:
        if hasattr(obj, "get_vision_tower"):
            vt = obj.get_vision_tower()
            if vt is not None:
                return vt
        for name in ["vision_tower", "vision_encoder", "vision_model", "visual", "visual_encoder"]:
            if hasattr(obj, name):
                vt = getattr(obj, name)
                if isinstance(vt, (list, tuple)):
                    vt = vt[0]
                if vt is not None:
                    return vt
    raise AttributeError("Could not find vision tower.")


def capture_lingshu_vision_features(runner: LingshuRunner, image: Image.Image) -> torch.Tensor:
    module = find_vision_tower(runner.model)
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(normalize_feature_output(output).detach())

    handle = module.register_forward_hook(hook)
    try:
        # Keep this path aligned with LingshuRunner.generate_with_scores.
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe the image."}]}]
        text = runner.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = runner.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                value = value.to(runner.device)
                if value.is_floating_point() and key in {"pixel_values", "image_grid_thw", "pixel_values_videos"}:
                    value = value.to(torch.bfloat16)
            moved[key] = value
        with torch.no_grad():
            runner.model.generate(
                **moved,
                do_sample=False,
                max_new_tokens=1,
                use_cache=True,
                pad_token_id=runner.tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    if not captured:
        raise ValueError("Lingshu vision tower hook did not capture features.")
    raw = captured[-1]
    if raw.ndim == 3:
        return raw[0]
    return raw


def find_projector(model):
    candidates = [model]
    if hasattr(model, "get_model"):
        try:
            candidates.append(model.get_model())
        except Exception:
            pass
    for obj in candidates:
        for name in ["mm_projector", "multi_modal_projector", "vision_projector", "projector"]:
            if hasattr(obj, name):
                return getattr(obj, name)
    raise AttributeError("Could not find multimodal projector.")


def extract_raw_from_llava_like_model(model, images) -> torch.Tensor:
    vt = find_vision_tower(model)
    with torch.no_grad():
        raw = vt(images)
    return normalize_feature_output(raw)


@contextmanager
def patched_llava_like_encode_images(model, bank: dict[str, Any], params: EasyLensParams):
    target = model if hasattr(model, "encode_images") else None
    if target is None and hasattr(model, "get_model"):
        inner = model.get_model()
        if hasattr(inner, "encode_images"):
            target = inner
    if target is None:
        raise AttributeError("Could not find encode_images for LLaVA-like model.")
    original = target.encode_images
    projector = find_projector(model)

    def override(images):
        raw = extract_raw_from_llava_like_model(model, images)
        modified_batches = []
        for batch_idx in range(raw.shape[0]):
            prefix, patch_tokens, suffix, grid_side = split_patch_tokens(raw[batch_idx])
            if int(grid_side) != int(bank["grid_side"]):
                raise ValueError(f"Runtime grid {grid_side} does not match bank grid {bank['grid_side']}")
            ref, alpha, _meta = compute_easylens_tokens(patch_tokens, bank, params)
            modified_patch = patch_tokens.float() + float(params.beta) * alpha.to(patch_tokens.device).unsqueeze(-1) * (ref.to(patch_tokens.device) - patch_tokens.float())
            rebuilt = torch.cat([prefix.float(), modified_patch, suffix.float()], dim=0).to(device=raw.device, dtype=raw.dtype)
            modified_batches.append(rebuilt)
        modified = torch.stack(modified_batches, dim=0)
        return projector(modified)

    target.encode_images = override
    try:
        yield
    finally:
        target.encode_images = original


@contextmanager
def patched_radfm_vision_encoder(runner: RadFMRunner, bank: dict[str, Any], params: EasyLensParams):
    module = runner.model.embedding_layer.vision_encoder

    def hook(_module, _inputs, output):
        raw = normalize_feature_output(output)
        modified_batches = []
        for batch_idx in range(raw.shape[0]):
            prefix, patch_tokens, suffix, grid_side = split_patch_tokens(raw[batch_idx])
            if int(grid_side) != int(bank["grid_side"]):
                raise ValueError(f"Runtime RadFM grid {grid_side} does not match bank grid {bank['grid_side']}")
            ref, alpha, _meta = compute_easylens_tokens(patch_tokens, bank, params)
            modified_patch = patch_tokens.float() + float(params.beta) * alpha.to(patch_tokens.device).unsqueeze(-1) * (ref.to(patch_tokens.device) - patch_tokens.float())
            rebuilt = torch.cat([prefix.float(), modified_patch, suffix.float()], dim=0).to(device=raw.device, dtype=raw.dtype)
            modified_batches.append(rebuilt)
        return replace_feature_output(output, torch.stack(modified_batches, dim=0))

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def patched_hulu_vision_encoder(runner: HuluRunner, bank: dict[str, Any], params: EasyLensParams):
    module = find_vision_tower(runner.model)

    def hook(_module, _inputs, output):
        raw = normalize_feature_output(output)
        modified_batches = []
        for batch_idx in range(raw.shape[0]):
            prefix, patch_tokens, suffix, grid_side = split_patch_tokens(raw[batch_idx])
            if int(grid_side) != int(bank["grid_side"]):
                raise ValueError(f"Runtime Hulu grid {grid_side} does not match bank grid {bank['grid_side']}")
            ref, alpha, _meta = compute_easylens_tokens(patch_tokens, bank, params)
            modified_patch = patch_tokens.float() + float(params.beta) * alpha.to(patch_tokens.device).unsqueeze(-1) * (ref.to(patch_tokens.device) - patch_tokens.float())
            rebuilt = torch.cat([prefix.float(), modified_patch, suffix.float()], dim=0).to(device=raw.device, dtype=raw.dtype)
            modified_batches.append(rebuilt)
        modified = torch.stack(modified_batches, dim=0)
        if isinstance(output, torch.Tensor) and output.ndim == 2:
            return modified[0].to(device=output.device, dtype=output.dtype)
        return replace_feature_output(output, modified)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def easylens_context(
    runner,
    model_kind: str,
    image: Image.Image,
    image_path: Path,
    bank: dict[str, Any] | None,
    params: EasyLensParams,
    vision_layer: int,
    row: dict[str, Any] | None = None,
):
    del image_path
    if bank is None:
        yield {}
        return
    if abs(float(params.beta)) <= 1e-12:
        yield {"easylens_noop": True}
        return
    if model_kind == "llava-med":
        with patched_llava_like_encode_images(runner.model, bank, params):
            yield {}
        return
    if model_kind == "hulu-med-7b":
        with patched_hulu_vision_encoder(runner, bank, params):
            yield {}
        return
    if model_kind == "lingshu":
        with patched_hulu_vision_encoder(runner, bank, params):
            yield {}
        return
    if model_kind == "radfm":
        with patched_radfm_vision_encoder(runner, bank, params):
            yield {}
        return
    if model_kind in {"medgemma", "medgemma1.5"}:
        source_tokens, grid_side = extract_runner_features(runner, model_kind, image, Path("unused"), int(vision_layer))
        if int(grid_side) != int(bank["grid_side"]):
            raise ValueError(f"Runtime MedGemma grid {grid_side} does not match bank grid {bank['grid_side']}")
        alpha_override = None
        if params.selector_override == "gt_mask":
            if row is None:
                raise ValueError("GT mask selector requires a manifest row.")
            alpha_override = build_gt_alpha(row, image, int(grid_side), runner.device)
        ref, alpha, meta = compute_easylens_tokens(source_tokens.to(runner.device), bank, params, alpha_override=alpha_override)
        intervention_mode = "token_replace" if params.amplifier_mode == "token_replace" else "token_delta"
        with patched_vision_layer_tokens(runner.model, int(vision_layer), alpha, float(params.beta), intervention_mode, ref):
            yield meta
        return
    raise ValueError(model_kind)


def build_bank(args: argparse.Namespace) -> None:
    ensure_dir(args.output_path.parent)
    rng = np.random.default_rng(int(args.seed))
    runner = build_runner(args.model_kind, args.model_path, args.device)
    rows = load_benchmark_manifest(args.benchmark_dir)
    if args.limit > 0:
        rows = rows[: args.limit]
    lesion_tokens: list[np.ndarray] = []
    lesion_templates: list[np.ndarray] = []
    lesion_positions: list[int] = []
    lesion_sample_ids: list[str] = []
    background_global: list[np.ndarray] = []
    background_by_position: list[list[np.ndarray]] | None = None
    used_sample_ids: list[str] = []
    grid_side_seen: int | None = None
    feature_dim_seen: int | None = None

    for idx, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id") or row.get("id") or idx)
        try:
            image = load_benchmark_image(args.benchmark_dir, row)
            image_text = row.get("output_image") or row.get("image") or row.get("image_path") or ""
            image_path = Path(image_text)
            if not image_path.is_absolute():
                image_path = args.benchmark_dir / image_path
            patch_tokens, grid_side = extract_runner_features(runner, args.model_kind, image, image_path, int(args.vision_layer))
            patch_np = patch_tokens.float().cpu().numpy()
            if grid_side_seen is None:
                grid_side_seen = int(grid_side)
                feature_dim_seen = int(patch_np.shape[-1])
                background_by_position = [[] for _ in range(grid_side_seen * grid_side_seen)]
            if grid_side_seen != int(grid_side) or feature_dim_seen != int(patch_np.shape[-1]):
                raise ValueError(f"Feature shape changed: grid={grid_side}, dim={patch_np.shape[-1]}")
            lesion_mask, background_mask = build_supervision_masks(row, image, int(grid_side), float(args.body_intensity_threshold))
            if not np.any(lesion_mask) or not np.any(background_mask):
                print(f"[skip] {sample_id}: empty lesion/background mask", flush=True)
                continue
            lesion_idx = np.flatnonzero(lesion_mask)
            lesion_keep = lesion_idx[sample_indices(len(lesion_idx), int(args.max_lesion_per_image), rng)]
            for patch_idx in sorted(int(x) for x in lesion_keep.tolist()):
                lesion_tokens.append(patch_np[patch_idx].astype(np.float32, copy=False))
                lesion_templates.append(
                    build_local_shape_template(
                        lesion_mask,
                        patch_idx,
                        int(grid_side),
                        int(args.shape_radius),
                        float(args.ring_weight),
                        args.ring_mode,
                    )
                )
                lesion_positions.append(patch_idx)
                lesion_sample_ids.append(sample_id)
            bg_idx = np.flatnonzero(background_mask)
            bg_keep = bg_idx[sample_indices(len(bg_idx), int(args.max_bg_per_image), rng)]
            assert background_by_position is not None
            for patch_idx in sorted(int(x) for x in bg_keep.tolist()):
                tok = patch_np[patch_idx].astype(np.float16, copy=False)
                background_global.append(tok)
                background_by_position[patch_idx].append(tok)
            used_sample_ids.append(sample_id)
            print(f"[bank:{args.model_kind}] {idx}/{len(rows)} {sample_id} grid={grid_side} lesion={len(lesion_keep)} bg={len(bg_keep)}", flush=True)
        except Exception as exc:
            print(f"[skip] {sample_id}: {exc}", flush=True)
        if idx % 8 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not lesion_tokens or not background_global or background_by_position is None or grid_side_seen is None or feature_dim_seen is None:
        raise ValueError("No usable model-native bank tokens were collected.")

    lesion_bank = np.stack(lesion_tokens, axis=0).astype(np.float32, copy=False)
    keep = sample_indices(len(lesion_bank), int(args.max_lesion_tokens), rng)
    lesion_bank = lesion_bank[keep]
    lesion_templates_arr = np.stack(lesion_templates, axis=0).astype(np.float16, copy=False)[keep]
    lesion_positions_arr = np.asarray(lesion_positions, dtype=np.int32)[keep]
    lesion_sample_ids_keep = np.asarray(lesion_sample_ids, dtype=object)[keep].tolist()
    num_centers = min(int(args.num_lesion_prototypes), int(len(lesion_bank)))
    lesion_prototypes = run_kmeans(lesion_bank.astype(np.float32, copy=False), num_centers, int(args.seed), int(args.kmeans_iters))
    lesion_bank_norm = lesion_bank / np.clip(np.linalg.norm(lesion_bank, axis=1, keepdims=True), 1e-6, None)
    proto_norm = lesion_prototypes / np.clip(np.linalg.norm(lesion_prototypes, axis=1, keepdims=True), 1e-6, None)
    cluster_ids = np.argmax(lesion_bank_norm @ proto_norm.T, axis=1).astype(np.int16)
    member_order = np.argsort(cluster_ids, kind="stable").astype(np.int32)
    member_ids = cluster_ids[member_order]
    offsets = np.zeros(num_centers + 1, dtype=np.int32)
    for cluster_idx in range(num_centers):
        offsets[cluster_idx + 1] = offsets[cluster_idx] + int((member_ids == cluster_idx).sum())

    bg_global = np.stack(background_global, axis=0).astype(np.float16, copy=False)
    bg_keep = sample_indices(len(bg_global), int(args.max_bg_global_tokens), rng)
    bg_global = bg_global[bg_keep]
    position_counts = np.zeros(grid_side_seen * grid_side_seen, dtype=np.int32)
    parts: list[np.ndarray] = []
    for pos_idx, rows_at_pos in enumerate(background_by_position):
        if not rows_at_pos:
            continue
        arr = np.stack(rows_at_pos, axis=0).astype(np.float16, copy=False)
        if int(args.max_bg_per_position) > 0:
            arr = arr[sample_indices(len(arr), int(args.max_bg_per_position), rng)]
        position_counts[pos_idx] = int(len(arr))
        parts.append(arr)
    position_offsets = np.zeros(grid_side_seen * grid_side_seen + 1, dtype=np.int64)
    position_offsets[1:] = np.cumsum(position_counts, dtype=np.int64)
    if parts:
        bg_position = np.concatenate(parts, axis=0).astype(np.float16, copy=False)
    else:
        bg_position = np.zeros((0, feature_dim_seen), dtype=np.float16)

    bank = {
        "model_kind": args.model_kind,
        "model_path": str(args.model_path) if args.model_path else None,
        "benchmark_dir": str(args.benchmark_dir),
        "feature_site": "medgemma_vision_layer" if args.model_kind in {"medgemma", "medgemma1.5"} else "pre_projector_or_vision_encoder",
        "vision_layer": int(args.vision_layer),
        "grid_side": int(grid_side_seen),
        "feature_dim": int(feature_dim_seen),
        "num_samples_used": int(len(used_sample_ids)),
        "used_sample_ids": used_sample_ids,
        "num_lesion_tokens": int(len(lesion_bank)),
        "num_lesion_prototypes": int(len(lesion_prototypes)),
        "lesion_bank": lesion_bank.astype(np.float16),
        "lesion_prototypes": lesion_prototypes.astype(np.float16),
        "lesion_cluster_ids": cluster_ids,
        "lesion_cluster_member_order": member_order,
        "lesion_cluster_offsets": offsets,
        "lesion_shape_templates": lesion_templates_arr,
        "lesion_patch_positions": lesion_positions_arr,
        "lesion_sample_ids": lesion_sample_ids_keep,
        "background_global_bank": bg_global,
        "background_position_bank": bg_position,
        "background_position_counts": position_counts,
        "background_position_offsets": position_offsets,
        "body_intensity_threshold": float(args.body_intensity_threshold),
        "shape_radius": int(args.shape_radius),
        "ring_weight": float(args.ring_weight),
        "ring_mode": args.ring_mode,
        "seed": int(args.seed),
        "kmeans_iters": int(args.kmeans_iters),
    }
    torch.save(bank, args.output_path)
    print(json.dumps({k: bank[k] for k in ["model_kind", "grid_side", "feature_dim", "num_samples_used", "num_lesion_tokens", "num_lesion_prototypes"]}, indent=2), flush=True)


def parse_params(args: argparse.Namespace) -> EasyLensParams:
    return EasyLensParams(
        beta=float(args.beta),
        selector_k=int(args.selector_k),
        support_mode=args.support_mode,
        suppression_radius=int(args.suppression_radius),
        margin_weight=float(args.margin_weight),
        shape_weight=float(args.shape_weight),
        score_threshold=float(args.score_threshold),
        seed_mode=args.seed_mode,
        seed_peak_kernel=int(args.seed_peak_kernel),
        seed_topk=int(args.seed_topk),
        base_score_mode=args.base_score_mode,
        support_score_mode=args.support_score_mode,
        raw_score_weight=float(args.raw_score_weight),
        gap_score_weight=float(args.gap_score_weight),
        retrieval_topm=int(args.retrieval_topm),
        retrieval_temperature=float(args.retrieval_temperature),
        query_chunk=int(args.prototype_query_chunk),
        bg_bank_chunk=int(args.bg_bank_chunk),
        similarity_chunk=int(args.similarity_chunk),
        selector_override=args.selector_override,
        amplifier_mode=args.amplifier_mode,
        selector_ckpt=(str(args.selector_ckpt) if args.selector_ckpt else None),
    )


def eval_vqa(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    params = parse_params(args)
    runner = build_vqa_runner(args.model_kind, args.model_path, args.device)
    bank = load_native_bank(args.bank_path, runner_device(runner)) if args.bank_path else None
    bank = maybe_load_selector_checkpoint(bank, params, runner_device(runner))
    manifest = safe_load_sample_manifest(args.benchmark_dir)
    all_runtime: list[dict[str, Any]] = []
    for task in args.tasks:
        records = load_task_records(args.benchmark_dir, task, limit=(args.limit if args.limit > 0 else None))
        sample_ids = {rec.sample_id for rec in records}
        pred_rows: list[dict[str, Any]] = []
        for idx, rec in enumerate(records, start=1):
            image = Image.open(rec.image_path).convert("RGB")
            prompt_text = make_task_prompt(task, rec.prompt, strict_count_prompt=bool(args.strict_count_prompt))
            start = time.perf_counter()
            meta: dict[str, Any] = {}
            try:
                row = manifest.get(rec.sample_id, {})
                with easylens_context(runner, args.model_kind, image, rec.image_path, bank, params, int(args.vision_layer), row=row) as ctx_meta:
                    meta = ctx_meta
                    raw_text, _outputs, _generated = runner.generate_with_scores(image, rec.image_path, prompt_text, int(args.max_new_tokens))
                error = ""
            except Exception as exc:
                raw_text = ""
                error = repr(exc)
            elapsed = time.perf_counter() - start
            answer = raw_text
            if task in {"loc", "multiselect"}:
                row = manifest.get(rec.sample_id, {})
                grid_size = int(float(row.get("grid_size", 3) or 3))
                answer = canonicalize_loc_answer(raw_text, grid_size=grid_size, prompt=prompt_text)
            pred_rows.append(
                {
                    "sample_id": rec.sample_id,
                    "task": task,
                    "image": rec.image_rel,
                    "prompt": prompt_text,
                    "text": answer,
                    "raw_text": raw_text,
                    "error": error,
                    "elapsed_sec": elapsed,
                    **meta,
                }
            )
            all_runtime.append({"sample_id": rec.sample_id, "task": task, "elapsed_sec": elapsed, "error": error})
            if idx % 10 == 0 or idx == len(records):
                print(f"[{args.model_kind} {'w' if bank else 'wo'}] {task} {idx}/{len(records)}", flush=True)
        pred_path = args.output_dir / f"{task}_answers.jsonl"
        write_jsonl(pred_path, pred_rows)
        if (args.benchmark_dir / "manifest.csv").exists():
            score_predictions(
                benchmark_dir=args.benchmark_dir,
                task=task,
                predictions_path=pred_path,
                output_dir=args.output_dir,
                model_kind=args.model_kind,
                sample_ids=sample_ids,
            )
    write_runtime_summary(args.output_dir, args.model_kind, bool(bank), all_runtime)


def eval_report(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)
    params = parse_params(args)
    runner = build_runner(args.model_kind, args.model_path, args.device)
    bank = load_native_bank(args.bank_path, runner_device(runner)) if args.bank_path else None
    bank = maybe_load_selector_checkpoint(bank, params, runner_device(runner))
    manifest = safe_load_sample_manifest(args.benchmark_dir)
    records = load_report_records(args.benchmark_dir, args.prompt_mode, limit=(args.limit if args.limit > 0 else None))
    tokenizer = get_tokenizer(runner.tokenizer)
    lesion_ids, lesion_phrase_map, lesion_token_names = candidate_first_token_info(tokenizer, LESION_PHRASES, min_alpha_len=3)
    denial_ids, denial_phrase_map, denial_token_names = candidate_first_token_info(tokenizer, DENIAL_PHRASES, min_alpha_len=2)
    write_json(
        args.output_dir / "run_config.json",
        {
            "model_kind": args.model_kind,
            "with_easylens": bool(bank),
            "model_path": str(args.model_path),
            "benchmark_dir": str(args.benchmark_dir),
            "bank_path": str(args.bank_path) if args.bank_path else None,
            "prompt_mode": args.prompt_mode,
            "max_new_tokens": int(args.max_new_tokens),
            "topk_steps": int(args.topk_steps),
            "selector_override": args.selector_override,
            "amplifier_mode": args.amplifier_mode,
            "selector_ckpt": str(args.selector_ckpt) if args.selector_ckpt else None,
            "lesion_phrases": lesion_phrase_map,
            "lesion_first_token_ids": lesion_ids,
            "lesion_first_tokens": {str(k): v for k, v in lesion_token_names.items()},
            "denial_phrases": denial_phrase_map,
            "denial_first_token_ids": denial_ids,
            "denial_first_tokens": {str(k): v for k, v in denial_token_names.items()},
        },
    )
    pred_rows: list[dict[str, Any]] = []
    prob_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(records, start=1):
        image = Image.open(rec.image_path).convert("RGB")
        start = time.perf_counter()
        meta: dict[str, Any] = {}
        try:
            row = manifest.get(rec.sample_id, {})
            with easylens_context(runner, args.model_kind, image, rec.image_path, bank, params, int(args.vision_layer), row=row) as ctx_meta:
                meta = ctx_meta
                text, outputs, generated = runner.generate_with_scores(image, rec.image_path, rec.prompt, int(args.max_new_tokens))
            prob_meta = analyze_scores(outputs, generated, tokenizer, lesion_ids, denial_ids, int(args.topk_steps))
            error = ""
        except Exception as exc:
            text = ""
            prob_meta = {}
            error = repr(exc)
        elapsed = time.perf_counter() - start
        base = {
            "sample_id": rec.sample_id,
            "image": rec.image_rel,
            "prompt": rec.prompt,
            "report_text": text,
            "error": error,
            "elapsed_sec": elapsed,
            **meta,
        }
        pred_rows.append(base)
        prob_rows.append({**base, **prob_meta})
        runtime_rows.append({"sample_id": rec.sample_id, "task": "report", "elapsed_sec": elapsed, "error": error})
        if idx % 10 == 0 or idx == len(records):
            print(f"[{args.model_kind} {'w' if bank else 'wo'}] report {idx}/{len(records)}", flush=True)
    metric_rows = [row for row in prob_rows if not row.get("error")]
    summary = {
        "benchmark_dir": str(args.benchmark_dir),
        "model_kind": args.model_kind,
        "with_easylens": bool(bank),
        "metrics": summarize_report_probs(metric_rows),
    }
    write_jsonl(args.output_dir / "report_answers.jsonl", pred_rows)
    write_jsonl(args.output_dir / "lesion_token_prob_per_sample.jsonl", prob_rows)
    write_json(args.output_dir / "lesion_token_prob_summary.json", summary)
    write_runtime_summary(args.output_dir, args.model_kind, bool(bank), runtime_rows)


def write_runtime_summary(output_dir: Path, model_kind: str, with_easylens: bool, rows: list[dict[str, Any]]) -> None:
    valid = [r for r in rows if not r.get("error")]
    by_task: dict[str, list[float]] = {}
    for row in valid:
        by_task.setdefault(str(row["task"]), []).append(float(row["elapsed_sec"]))
    summary = {
        "model_kind": model_kind,
        "with_easylens": bool(with_easylens),
        "total": len(rows),
        "valid": len(valid),
        "mean_sec_per_sample": float(sum(float(r["elapsed_sec"]) for r in valid) / len(valid)) if valid else 0.0,
        "by_task": {
            task: {
                "n": len(vals),
                "mean_sec_per_sample": float(sum(vals) / len(vals)) if vals else 0.0,
            }
            for task, vals in sorted(by_task.items())
        },
    }
    write_jsonl(output_dir / "runtime_per_sample.jsonl", rows)
    write_json(output_dir / "runtime_summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model-native EasyLens bank building and ReX evaluation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common_model = argparse.ArgumentParser(add_help=False)
    common_model.add_argument("--model-kind", choices=["llava-med", "hulu-med-7b", "lingshu", "medgemma", "medgemma1.5", "radfm"], required=True)
    common_model.add_argument("--model-path", type=Path, default=None)
    common_model.add_argument("--radfm-quick-demo-root", type=Path, default=RADFM_QUICK_DEMO_ROOT)
    common_model.add_argument("--radfm-checkpoint", type=Path, default=RADFM_CHECKPOINT)
    common_model.add_argument("--device", default="cuda")
    common_model.add_argument("--vision-layer", type=int, default=1)

    bank = sub.add_parser("build-bank", parents=[common_model])
    bank.add_argument("--benchmark-dir", type=Path, required=True)
    bank.add_argument("--output-path", type=Path, required=True)
    bank.add_argument("--num-lesion-prototypes", type=int, default=8)
    bank.add_argument("--max-lesion-per-image", type=int, default=256)
    bank.add_argument("--max-bg-per-image", type=int, default=0)
    bank.add_argument("--max-lesion-tokens", type=int, default=30000)
    bank.add_argument("--max-bg-per-position", type=int, default=0)
    bank.add_argument("--max-bg-global-tokens", type=int, default=60000)
    bank.add_argument("--body-intensity-threshold", type=float, default=5.0)
    bank.add_argument("--shape-radius", type=int, default=2)
    bank.add_argument("--ring-weight", type=float, default=0.5)
    bank.add_argument("--ring-mode", choices=["suppress", "boost"], default="suppress")
    bank.add_argument("--kmeans-iters", type=int, default=50)
    bank.add_argument("--seed", type=int, default=42)
    bank.add_argument("--limit", type=int, default=0)

    eval_parent = argparse.ArgumentParser(add_help=False)
    eval_parent.add_argument("--benchmark-dir", type=Path, required=True)
    eval_parent.add_argument("--output-dir", type=Path, required=True)
    eval_parent.add_argument("--bank-path", type=Path, default=None)
    eval_parent.add_argument("--beta", type=float, default=0.75)
    eval_parent.add_argument("--selector-k", type=int, default=256)
    eval_parent.add_argument("--support-mode", choices=["global_topk", "radius_suppressed_topk"], default="global_topk")
    eval_parent.add_argument("--suppression-radius", type=int, default=1)
    eval_parent.add_argument("--margin-weight", type=float, default=0.35)
    eval_parent.add_argument("--shape-weight", type=float, default=1.0)
    eval_parent.add_argument("--score-threshold", type=float, default=0.0)
    eval_parent.add_argument("--seed-mode", choices=["all_positive", "local_peak", "local_peak_topk"], default="local_peak_topk")
    eval_parent.add_argument("--seed-peak-kernel", type=int, default=5)
    eval_parent.add_argument("--seed-topk", type=int, default=16)
    eval_parent.add_argument("--base-score-mode", choices=["all_score", "seed_only"], default="seed_only")
    eval_parent.add_argument("--support-score-mode", choices=["shape_only", "hybrid_max", "hybrid_sum"], default="shape_only")
    eval_parent.add_argument("--raw-score-weight", type=float, default=1.0)
    eval_parent.add_argument("--gap-score-weight", type=float, default=0.0)
    eval_parent.add_argument("--retrieval-topm", type=int, default=64)
    eval_parent.add_argument("--retrieval-temperature", type=float, default=0.1)
    eval_parent.add_argument("--prototype-query-chunk", type=int, default=512)
    eval_parent.add_argument("--bg-bank-chunk", type=int, default=1024)
    eval_parent.add_argument("--similarity-chunk", type=int, default=4096)
    eval_parent.add_argument("--selector-override", choices=["easytag", "gt_mask", "mlp"], default="easytag")
    eval_parent.add_argument("--selector-ckpt", type=Path, default=None)
    eval_parent.add_argument("--amplifier-mode", choices=["residual_delta", "token_replace"], default="residual_delta")
    eval_parent.add_argument("--limit", type=int, default=0)
    eval_parent.add_argument("--max-new-tokens", type=int, default=96)

    vqa = sub.add_parser("eval-vqa", parents=[common_model, eval_parent])
    vqa.add_argument("--tasks", nargs="+", default=["count", "multiselect"], choices=["count", "multiselect", "yesno", "loc"])
    vqa.add_argument("--strict-count-prompt", action="store_true")

    report = sub.add_parser("eval-report", parents=[common_model, eval_parent])
    report.add_argument("--prompt-mode", choices=["benchmark_raw", "micro_focus_v3"], default="benchmark_raw")
    report.add_argument("--topk-steps", type=int, default=3)
    report.set_defaults(max_new_tokens=160)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.cmd == "build-bank":
        build_bank(args)
    elif args.cmd == "eval-vqa":
        eval_vqa(args)
    elif args.cmd == "eval-report":
        eval_report(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
