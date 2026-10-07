from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


IMAGE_SIZE = 896
PATCH_SIZE = 14
ENCODER_GRID = IMAGE_SIZE // PATCH_SIZE


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fundus_image(image_path: Path, green_channel: bool) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if green_channel:
        image_np = np.array(image)
        green = image_np[..., 1]
        image = Image.fromarray(np.stack([green] * 3, axis=-1))
    return image


def load_binary_mask(mask_path: Path | None, threshold: int) -> Image.Image | None:
    if mask_path is None or not mask_path.exists():
        return None
    mask_np = np.array(Image.open(mask_path).convert("L"))
    mask_bin = (mask_np >= threshold).astype(np.uint8) * 255
    return Image.fromarray(mask_bin)


def apply_lesion_only(image: Image.Image, mask_img: Image.Image | None) -> Image.Image:
    image_np = np.array(image)
    kept = np.zeros_like(image_np)
    if mask_img is None:
        return Image.fromarray(kept)
    mask_np = np.array(mask_img.convert("L")) > 0
    kept[mask_np] = image_np[mask_np]
    return Image.fromarray(kept)


def resize_image(image: Image.Image, mode=Image.Resampling.LANCZOS) -> Image.Image:
    return image.resize((IMAGE_SIZE, IMAGE_SIZE), mode)


def patch_overlap_from_mask(mask_img: Image.Image, grid_side: int) -> np.ndarray:
    mask_np = np.array(mask_img, dtype=np.float32) / 255.0
    patch_size = IMAGE_SIZE // grid_side
    overlaps = []
    for row in range(grid_side):
        for col in range(grid_side):
            patch = mask_np[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size]
            overlaps.append(float(patch.mean()))
    return np.asarray(overlaps, dtype=np.float32)


def build_patch_labels(overlaps: np.ndarray, tau_pos: float, tau_neg: float) -> np.ndarray:
    labels = np.full(overlaps.shape, -1, dtype=np.int64)
    labels[overlaps > tau_pos] = 1
    labels[overlaps < tau_neg] = 0
    if not np.any(labels == 1):
        nonzero_idx = np.where(overlaps > 0)[0]
        if len(nonzero_idx):
            best_idx = nonzero_idx[np.argmax(overlaps[nonzero_idx])]
            labels[best_idx] = 1
    return labels


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_true = y_true[order].astype(np.float64)
    tp = np.cumsum(y_true)
    precision = tp / np.maximum(np.arange(1, len(y_true) + 1), 1e-12)
    recall = tp / np.maximum(tp[-1], 1e-12)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_ranks = ranks[y_true == 1].sum()
    auc = (pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def compute_localization_metrics(overlaps: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    pos_mask = labels == 1
    lesion_mask = overlaps > 0
    background_mask = labels == 0
    lesion_mass = float(overlaps.sum())
    num_pos = int(pos_mask.sum())
    order = np.argsort(-scores)
    lesion_ranks = np.flatnonzero(lesion_mask[order])
    lesion_score_mean = float(scores[lesion_mask].mean()) if np.any(lesion_mask) else float("nan")
    background_score_mean = float(scores[background_mask].mean()) if np.any(background_mask) else float("nan")
    metrics: Dict[str, float] = {
        "pointing_game": float(lesion_mask[order[0]]) if len(order) else 0.0,
        "num_positive_patches": float(num_pos),
        "num_lesion_patches": float(lesion_mask.sum()),
        "num_background_patches": float(background_mask.sum()),
        "best_lesion_patch_rank": float(lesion_ranks[0] + 1) if len(lesion_ranks) else float("nan"),
        "best_lesion_patch_rank_frac": float((lesion_ranks[0] + 1) / max(len(scores), 1)) if len(lesion_ranks) else float("nan"),
        "lesion_score_mean": lesion_score_mean,
        "background_score_mean": background_score_mean,
        "lesion_vs_background_score_gap": (
            float(lesion_score_mean - background_score_mean)
            if np.isfinite(lesion_score_mean) and np.isfinite(background_score_mean)
            else float("nan")
        ),
    }
    for k in (1, 5, 10):
        topk = order[: min(k, len(order))]
        metrics[f"top{k}_lesion_hit"] = float(np.any(lesion_mask[topk])) if len(topk) else 0.0
        metrics[f"top{k}_lesion_recall"] = float(pos_mask[topk].sum() / max(num_pos, 1)) if len(topk) else 0.0
        metrics[f"top{k}_lesion_coverage"] = float(overlaps[topk].sum() / max(lesion_mass, 1e-8)) if len(topk) else 0.0
    if num_pos > 0:
        top_pos = order[:num_pos]
        metrics["top_pos_lesion_recall"] = float(pos_mask[top_pos].sum() / num_pos)
        metrics["top_pos_lesion_coverage"] = float(overlaps[top_pos].sum() / max(lesion_mass, 1e-8))
    else:
        metrics["top_pos_lesion_recall"] = 0.0
        metrics["top_pos_lesion_coverage"] = 0.0
    return metrics


def evaluate_score_map(scores: np.ndarray, overlaps: np.ndarray, tau_pos: float, tau_neg: float) -> Dict[str, float]:
    labels = build_patch_labels(overlaps, tau_pos, tau_neg)
    usable = labels >= 0
    usable_labels = labels[usable]
    usable_scores = scores[usable]
    if (usable_labels == 1).sum() == 0 or (usable_labels == 0).sum() == 0:
        raise ValueError("Not enough usable positive/negative patches for prototype score evaluation.")
    pred = (usable_scores >= 0.0).astype(np.int64)
    metrics = {
        "score_mode": "prototype_score",
        "val_acc": float((pred == usable_labels).mean()),
        "val_auroc": roc_auc_score(usable_labels, usable_scores),
        "val_auprc": average_precision_score(usable_labels, usable_scores),
        "patch_auroc": roc_auc_score(usable_labels, usable_scores),
        "patch_ap": average_precision_score(usable_labels, usable_scores),
        "num_val": int(len(usable_labels)),
        "num_pos": int((usable_labels == 1).sum()),
        "num_neg": int((usable_labels == 0).sum()),
    }
    metrics.update(compute_localization_metrics(overlaps, labels, scores))
    return metrics


def normalize_source_layers(source_layers: Sequence[int] | int) -> list[int]:
    if isinstance(source_layers, int):
        layers = [int(source_layers)]
    else:
        layers = [int(layer) for layer in source_layers]
    if not layers:
        raise ValueError("source_layers must contain at least one layer.")
    deduped = list(dict.fromkeys(layers))
    return deduped


def format_source_layer_tag(source_layers: Sequence[int] | int) -> str:
    layers = normalize_source_layers(source_layers)
    return "l" + "_".join(str(layer) for layer in layers)


class SelectedLayersTap:
    def __init__(self, model, layer_indices: Sequence[int]):
        self.layer_indices = normalize_source_layers(layer_indices)
        self.hidden_by_layer: Dict[int, torch.Tensor] = {}
        self.handles = []
        encoder_layers = get_vision_encoder_layers(model)
        for layer_idx in self.layer_indices:
            layer = encoder_layers[layer_idx]

            def hook(_module, _inputs, output, *, _layer_idx=layer_idx):
                hidden = output[0] if isinstance(output, tuple) else output
                self.hidden_by_layer[_layer_idx] = hidden.detach()

            self.handles.append(layer.register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()


def fuse_source_tokens(source_tokens_by_layer: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(source_tokens_by_layer) == 1:
        return source_tokens_by_layer[0]
    normalized = [F.normalize(tokens.float(), dim=-1) for tokens in source_tokens_by_layer]
    fused = torch.stack(normalized, dim=0).mean(dim=0)
    return F.normalize(fused, dim=-1)


def get_vision_tower(model):
    if hasattr(model, "vision_tower"):
        return model.vision_tower
    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        return model.model.vision_tower
    raise AttributeError("Could not find vision_tower on model.")


def get_vision_encoder_layers(model):
    vision_tower = get_vision_tower(model)
    if hasattr(vision_tower, "vision_model") and hasattr(vision_tower.vision_model, "encoder"):
        return vision_tower.vision_model.encoder.layers
    if hasattr(vision_tower, "encoder"):
        return vision_tower.encoder.layers
    raise AttributeError("Could not find vision encoder layers on model.")


def get_multi_modal_projector(model):
    if hasattr(model, "multi_modal_projector"):
        return model.multi_modal_projector
    if hasattr(model, "model") and hasattr(model.model, "multi_modal_projector"):
        return model.model.multi_modal_projector
    raise AttributeError("Could not find multi_modal_projector on model.")


@torch.no_grad()
def extract_source_and_final_tokens(model, pixel_values: torch.Tensor, source_layers: Sequence[int] | int):
    source_layers = normalize_source_layers(source_layers)
    encoder_layers = get_vision_encoder_layers(model)
    num_layers = len(encoder_layers)
    for layer_idx in source_layers:
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"source_layer={layer_idx} is out of range for vision encoder with {num_layers} layers.")
    tap = SelectedLayersTap(model, source_layers)
    try:
        vision_outputs = get_vision_tower(model)(pixel_values=pixel_values, return_dict=True)
    finally:
        tap.close()

    missing = [layer_idx for layer_idx in source_layers if layer_idx not in tap.hidden_by_layer]
    if missing:
        raise RuntimeError(f"Failed to capture source layer activations for layers {missing}.")

    fused_source_tokens = fuse_source_tokens([tap.hidden_by_layer[layer_idx][0] for layer_idx in source_layers])
    final_tokens = vision_outputs.last_hidden_state[0].detach()
    return fused_source_tokens, final_tokens


def l2_normalize_rows_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def l2_normalize_rows_torch(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def topk_mean_torch(x: torch.Tensor, k: int) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("topk_mean_torch expects a 2D tensor.")
    k = max(1, min(int(k), x.shape[1]))
    return torch.topk(x, k=k, dim=1).values.mean(dim=1)


def robust_standardize_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    center = x.median()
    mad = (x - center).abs().median().clamp_min(1e-6)
    scale = 1.4826 * mad
    return (x - center) / scale


def maybe_smooth_token_scores(x: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0 or x.numel() != ENCODER_GRID * ENCODER_GRID:
        return x.float()
    score_grid = x.float().view(1, 1, ENCODER_GRID, ENCODER_GRID)
    smoothed = F.avg_pool2d(score_grid, kernel_size=3, stride=1, padding=1).view(-1)
    return (1.0 - weight) * x.float() + weight * smoothed


def local_peak_mask_torch(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1 or x.numel() != ENCODER_GRID * ENCODER_GRID:
        return torch.ones_like(x, dtype=torch.bool)
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    score_grid = x.float().view(1, 1, ENCODER_GRID, ENCODER_GRID)
    pooled = F.max_pool2d(score_grid, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (score_grid >= pooled - 1e-6).view(-1)


def dilate_mask_torch(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0 or mask.numel() != ENCODER_GRID * ENCODER_GRID:
        return mask.bool()
    kernel_size = 2 * int(radius) + 1
    mask_grid = mask.float().view(1, 1, ENCODER_GRID, ENCODER_GRID)
    dilated = F.max_pool2d(mask_grid, kernel_size=kernel_size, stride=1, padding=radius)
    return (dilated > 0).view(-1)


def sample_rows_np(features: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if len(features) <= max_rows:
        return features
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(features), size=max_rows, replace=False)
    return features[idx]


def run_kmeans(features: np.ndarray, num_centers: int, seed: int, max_iters: int) -> np.ndarray:
    if len(features) == 0:
        raise ValueError("Cannot run k-means with zero features.")
    features = l2_normalize_rows_np(features)
    num_centers = max(1, min(int(num_centers), len(features)))
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(len(features), size=num_centers, replace=False)
    centers = features[init_idx].copy()

    for _ in range(max_iters):
        similarity = features @ centers.T
        assignment = similarity.argmax(axis=1)
        new_centers = []
        for center_idx in range(num_centers):
            cluster = features[assignment == center_idx]
            if len(cluster) == 0:
                new_centers.append(features[rng.integers(len(features))])
            else:
                centroid = cluster.mean(axis=0)
                centroid /= max(np.linalg.norm(centroid), 1e-8)
                new_centers.append(centroid.astype(np.float32))
        new_centers = np.stack(new_centers, axis=0)
        if np.allclose(new_centers, centers, atol=1e-5):
            centers = new_centers
            break
        centers = new_centers
    return l2_normalize_rows_np(centers)


def compute_prototype_scores(
    source_tokens: torch.Tensor,
    lesion_prototypes: torch.Tensor,
    background_prototypes: torch.Tensor,
    selector_mode: str = "legacy",
    lesion_topk: int = 1,
    background_topk: int = 1,
    smooth_weight: float = 0.0,
    local_contrast_weight: float = 0.0,
    agreement_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    source_norm = l2_normalize_rows_torch(source_tokens)
    lesion_sim = torch.matmul(source_norm, lesion_prototypes.T)
    background_sim = torch.matmul(source_norm, background_prototypes.T)
    lesion_max = lesion_sim.max(dim=1).values
    background_max = background_sim.max(dim=1).values
    lesion_topk_mean = topk_mean_torch(lesion_sim, lesion_topk)
    background_topk_mean = topk_mean_torch(background_sim, background_topk)
    raw_margin = lesion_topk_mean - background_topk_mean
    local_context = maybe_smooth_token_scores(raw_margin, 1.0)
    local_contrast = raw_margin - local_context
    lesion_agreement = lesion_topk_mean - lesion_max

    if selector_mode == "legacy":
        lesion_score = lesion_max - background_max
    elif selector_mode == "contrastive_topk":
        lesion_score = robust_standardize_torch(raw_margin)
        lesion_score = maybe_smooth_token_scores(lesion_score, smooth_weight)
    elif selector_mode == "contrastive_peak":
        lesion_score = robust_standardize_torch(raw_margin)
        if local_contrast_weight != 0.0:
            lesion_score = lesion_score + float(local_contrast_weight) * robust_standardize_torch(local_contrast)
        if agreement_weight != 0.0 and lesion_topk > 1:
            lesion_score = lesion_score + float(agreement_weight) * robust_standardize_torch(lesion_agreement)
        lesion_score = maybe_smooth_token_scores(lesion_score, smooth_weight)
    else:
        raise ValueError(f"Unsupported selector_mode: {selector_mode}")

    return {
        "lesion_score": lesion_score,
        "selector_score_raw": raw_margin,
        "selector_local_contrast_raw": local_contrast,
        "selector_lesion_agreement_raw": lesion_agreement,
        "lesion_sim": lesion_sim,
        "background_sim": background_sim,
        "lesion_max": lesion_max,
        "background_max": background_max,
        "lesion_topk_mean": lesion_topk_mean,
        "background_topk_mean": background_topk_mean,
    }


def build_alpha_from_scores(
    lesion_score: torch.Tensor,
    temperature: float,
    alpha_mode: str = "softmax",
    alpha_positive_threshold: float = 0.0,
    alpha_power: float = 1.0,
    alpha_topk: int | None = None,
    alpha_local_peak_kernel: int = 1,
    alpha_peak_expand_radius: int = 0,
) -> Dict[str, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    score = lesion_score.float() / temperature
    peak_mask = local_peak_mask_torch(score, alpha_local_peak_kernel)
    support_mask = dilate_mask_torch(peak_mask, alpha_peak_expand_radius)
    if alpha_mode == "softmax":
        use_mask = support_mask if alpha_local_peak_kernel > 1 or alpha_peak_expand_radius > 0 else None
        masked_score = score.masked_fill(~use_mask, float("-inf")) if use_mask is not None else score
        if not torch.isfinite(masked_score).any():
            masked_score = score
        alpha = torch.softmax(masked_score, dim=0)
        if alpha_power != 1.0:
            alpha = alpha.pow(float(alpha_power))
            alpha = alpha / alpha.sum().clamp_min(1e-8)
        if alpha_topk is not None and alpha_topk > 0:
            keep_k = min(int(alpha_topk), int(alpha.numel()))
            keep_idx = torch.topk(alpha, k=keep_k, dim=0).indices
            keep_mask = torch.zeros_like(alpha, dtype=torch.bool)
            keep_mask[keep_idx] = True
            alpha = alpha * keep_mask.to(dtype=alpha.dtype)
            alpha = alpha / alpha.sum().clamp_min(1e-8)
        positive_support = alpha > 0
    elif alpha_mode == "sparse_positive":
        positive_mass = torch.relu(score - float(alpha_positive_threshold))
        positive_mass = positive_mass * support_mask.to(dtype=positive_mass.dtype)
        if alpha_power != 1.0:
            positive_mass = positive_mass.pow(float(alpha_power))
        positive_support = positive_mass > 0
        if alpha_topk is not None and alpha_topk > 0 and positive_support.any():
            keep_k = min(int(alpha_topk), int(positive_support.sum().item()))
            keep_idx = torch.topk(positive_mass, k=keep_k, dim=0).indices
            keep_mask = torch.zeros_like(positive_support, dtype=torch.bool)
            keep_mask[keep_idx] = True
            positive_mass = positive_mass * keep_mask.to(dtype=positive_mass.dtype)
            positive_support = positive_mass > 0
        if positive_support.any():
            alpha = positive_mass / positive_mass.sum().clamp_min(1e-8)
        else:
            alpha = torch.zeros_like(score)
            use_mask = support_mask if alpha_local_peak_kernel > 1 or alpha_peak_expand_radius > 0 else None
            fallback_score = score.masked_fill(~use_mask, float("-inf")) if use_mask is not None else score
            if not torch.isfinite(fallback_score).any():
                fallback_score = score
            alpha[int(torch.argmax(fallback_score).item())] = 1.0
            positive_support = alpha > 0
    else:
        raise ValueError(f"Unsupported alpha_mode: {alpha_mode}")
    return {
        "alpha": alpha,
        "alpha_positive_support": positive_support,
        "alpha_peak_support": peak_mask,
        "alpha_support_mask": support_mask,
    }


def build_modified_final_tokens_from_alpha(
    final_tokens: torch.Tensor,
    alpha: torch.Tensor,
    beta: float,
    mode: str,
    reference_tokens: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    if alpha.ndim != 1 or alpha.shape[0] != final_tokens.shape[0]:
        raise ValueError("alpha must be a 1D tensor with the same token length as final_tokens.")
    final_float = final_tokens.float()
    raw_alpha = alpha.float()
    raw_alpha_max = raw_alpha.max()
    if raw_alpha_max > 0:
        alpha_gate = raw_alpha / raw_alpha_max.clamp_min(1e-6)
    else:
        alpha_gate = torch.zeros_like(raw_alpha)
    alpha_sum = raw_alpha.sum()
    if alpha_sum > 0:
        alpha = raw_alpha / alpha_sum
    else:
        alpha = torch.zeros_like(raw_alpha)
    alpha_scaled = alpha * alpha.numel()
    alpha_centered = alpha_scaled - alpha_scaled.mean()
    alpha_focus = torch.relu(alpha_centered)
    if alpha_focus.max() > 0:
        alpha_focus = alpha_focus / alpha_focus.max().clamp_min(1e-6)

    if alpha_sum > 0:
        summary = (alpha.unsqueeze(-1) * final_float).sum(dim=0)
        summary = F.normalize(summary, dim=0)
        token_norm = final_float.norm(dim=-1).mean().clamp_min(1e-6)
        summary = summary * token_norm
    else:
        summary = torch.zeros(final_float.shape[-1], device=final_float.device, dtype=final_float.dtype)

    if mode == "none":
        modified = final_float
    elif mode == "residual_summary":
        modified = final_float + beta * alpha_focus.unsqueeze(-1) * summary.unsqueeze(0)
    elif mode == "token_reweight":
        modified = final_float * (1.0 + beta * alpha_gate.unsqueeze(-1))
    elif mode == "token_amplify":
        modified = final_float * (1.0 + beta * alpha_focus.unsqueeze(-1))
    elif mode == "token_spotlight":
        focus = alpha_focus.unsqueeze(-1)
        focus_sum = focus.sum().clamp_min(1e-6)
        bg_focus = 1.0 - focus
        bg_sum = bg_focus.sum().clamp_min(1e-6)
        fg_center = (focus * final_float).sum(dim=0) / focus_sum
        bg_center = (bg_focus * final_float).sum(dim=0) / bg_sum
        contrast_dir = fg_center - bg_center
        contrast_norm = contrast_dir.norm().clamp_min(1e-6)
        token_norm = final_float.norm(dim=-1).mean().clamp_min(1e-6)
        contrast_dir = contrast_dir / contrast_norm * token_norm
        modified = final_float * (1.0 + beta * focus) + beta * focus * contrast_dir.unsqueeze(0)
    elif mode == "token_contrast":
        contrast_scale = 1.0 + beta * (2.0 * alpha_focus.unsqueeze(-1) - 1.0)
        contrast_scale = torch.clamp(contrast_scale, min=0.1)
        modified = final_float * contrast_scale
    elif mode == "token_replace":
        if reference_tokens is None:
            raise ValueError("reference_tokens is required for token_replace mode.")
        ref_float = reference_tokens.float()
        replace_mask = alpha_focus > 0
        modified = final_float.clone()
        modified[replace_mask] = ref_float[replace_mask]
    elif mode == "paper_residual":
        if reference_tokens is None:
            raise ValueError("reference_tokens is required for paper_residual mode.")
        ref_float = reference_tokens.float()
        modified = final_float + beta * raw_alpha.unsqueeze(-1) * (ref_float - final_float)
    elif mode == "token_delta":
        if reference_tokens is None:
            raise ValueError("reference_tokens is required for token_delta mode.")
        ref_float = reference_tokens.float()
        modified = final_float + beta * alpha_focus.unsqueeze(-1) * (ref_float - final_float)
    else:
        raise ValueError(f"Unsupported intervention mode: {mode}")

    delta = modified - final_float
    if alpha_sum > 0 and alpha.numel() > 1:
        entropy = float((-(alpha * torch.log(alpha.clamp_min(1e-8))).sum() / math.log(alpha.numel())).item())
    else:
        entropy = 0.0
    topk = min(8, alpha.numel())
    top_alpha, top_idx = torch.topk(alpha, k=topk)
    return {
        "alpha": alpha,
        "alpha_scaled": alpha_scaled,
        "alpha_centered": alpha_centered,
        "alpha_focus": alpha_focus,
        "summary": summary,
        "modified_final_tokens": modified.to(dtype=final_tokens.dtype),
        "delta_l2_mean": delta.norm(dim=-1).mean(),
        "delta_l2_max": delta.norm(dim=-1).max(),
        "alpha_entropy": torch.tensor(entropy, device=final_tokens.device),
        "alpha_top_indices": top_idx,
        "alpha_top_weights": top_alpha,
    }


def build_modified_final_tokens(
    final_tokens: torch.Tensor,
    lesion_score: torch.Tensor,
    temperature: float,
    beta: float,
    mode: str,
    alpha_mode: str = "softmax",
    alpha_positive_threshold: float = 0.0,
    alpha_power: float = 1.0,
    alpha_topk: int | None = None,
    alpha_local_peak_kernel: int = 1,
    alpha_peak_expand_radius: int = 0,
) -> Dict[str, torch.Tensor]:
    alpha_outputs = build_alpha_from_scores(
        lesion_score=lesion_score,
        temperature=temperature,
        alpha_mode=alpha_mode,
        alpha_positive_threshold=alpha_positive_threshold,
        alpha_power=alpha_power,
        alpha_topk=alpha_topk,
        alpha_local_peak_kernel=alpha_local_peak_kernel,
        alpha_peak_expand_radius=alpha_peak_expand_radius,
    )
    outputs = build_modified_final_tokens_from_alpha(final_tokens, alpha_outputs["alpha"], beta, mode)
    outputs["alpha_positive_count"] = alpha_outputs["alpha_positive_support"].sum().to(dtype=torch.float32)
    outputs["alpha_positive_fraction"] = (
        alpha_outputs["alpha_positive_support"].sum().to(dtype=torch.float32) / max(alpha_outputs["alpha"].numel(), 1)
    )
    outputs["alpha_peak_count"] = alpha_outputs["alpha_peak_support"].sum().to(dtype=torch.float32)
    outputs["alpha_peak_fraction"] = (
        alpha_outputs["alpha_peak_support"].sum().to(dtype=torch.float32) / max(alpha_outputs["alpha"].numel(), 1)
    )
    outputs["alpha_support_count"] = alpha_outputs["alpha_support_mask"].sum().to(dtype=torch.float32)
    outputs["alpha_support_fraction"] = (
        alpha_outputs["alpha_support_mask"].sum().to(dtype=torch.float32) / max(alpha_outputs["alpha"].numel(), 1)
    )
    return outputs


def build_oracle_alpha_from_overlaps(
    overlaps: np.ndarray,
    tau_pos: float,
    tau_neg: float,
) -> tuple[np.ndarray, Dict[str, float]]:
    labels = build_patch_labels(overlaps, tau_pos, tau_neg)
    positive_mask = labels == 1
    alpha = positive_mask.astype(np.float32)
    if alpha.sum() == 0 and np.any(overlaps > 0):
        alpha[int(np.argmax(overlaps))] = 1.0
        positive_mask = alpha > 0
    stats = {
        "oracle_num_positive_patches": float(positive_mask.sum()),
        "oracle_positive_patch_fraction": float(positive_mask.mean()) if len(positive_mask) else 0.0,
        "oracle_positive_overlap_mass": float(overlaps[positive_mask].sum()) if np.any(positive_mask) else 0.0,
        "oracle_positive_overlap_mean": float(overlaps[positive_mask].mean()) if np.any(positive_mask) else 0.0,
    }
    return alpha, stats


@torch.no_grad()
def extract_final_tokens(model, pixel_values: torch.Tensor) -> torch.Tensor:
    vision_outputs = get_vision_tower(model)(pixel_values=pixel_values, return_dict=True)
    return vision_outputs.last_hidden_state[0].detach()


@torch.no_grad()
def extract_layer_tokens(model, pixel_values: torch.Tensor, layer_idx: int) -> torch.Tensor:
    source_tokens, _ = extract_source_and_final_tokens(model, pixel_values, [int(layer_idx)])
    return source_tokens.detach()


@torch.no_grad()
def prepare_oracle_intervention_features(
    model,
    pixel_values: torch.Tensor,
    oracle_alpha: torch.Tensor,
    beta: float,
    mode: str,
    reference_pixel_values: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    final_tokens = extract_final_tokens(model, pixel_values)
    reference_tokens = None
    if reference_pixel_values is not None:
        reference_tokens = extract_final_tokens(model, reference_pixel_values)
    intervention = build_modified_final_tokens_from_alpha(
        final_tokens,
        oracle_alpha.to(device=final_tokens.device),
        beta,
        mode,
        reference_tokens=reference_tokens,
    )
    projected = get_multi_modal_projector(model)(intervention["modified_final_tokens"].unsqueeze(0))
    return {
        "final_tokens": final_tokens,
        "reference_tokens": reference_tokens,
        "projected_features": projected,
        **intervention,
    }


@torch.no_grad()
def prepare_oracle_layer_intervention(
    model,
    pixel_values: torch.Tensor,
    reference_pixel_values: torch.Tensor | None,
    oracle_alpha: torch.Tensor,
    beta: float,
    mode: str,
    layer_idx: int,
) -> Dict[str, torch.Tensor]:
    base_tokens = extract_layer_tokens(model, pixel_values, layer_idx)
    reference_tokens = None
    if reference_pixel_values is not None:
        reference_tokens = extract_layer_tokens(model, reference_pixel_values, layer_idx)
    intervention = build_modified_final_tokens_from_alpha(
        base_tokens,
        oracle_alpha.to(device=base_tokens.device),
        beta,
        mode,
        reference_tokens=reference_tokens,
    )
    return {
        "layer_tokens": base_tokens,
        "reference_tokens": reference_tokens,
        "layer_idx": int(layer_idx),
        **intervention,
    }


@torch.no_grad()
def prepare_prototype_guided_features(
    model,
    pixel_values: torch.Tensor,
    lesion_prototypes: torch.Tensor,
    background_prototypes: torch.Tensor,
    source_layers: Sequence[int] | int,
    temperature: float,
    beta: float,
    mode: str,
    selector_mode: str = "legacy",
    selector_lesion_topk: int = 1,
    selector_background_topk: int = 1,
    selector_smooth_weight: float = 0.0,
    selector_local_contrast_weight: float = 0.0,
    selector_agreement_weight: float = 0.0,
    selector_alpha_mode: str = "softmax",
    selector_alpha_positive_threshold: float = 0.0,
    selector_alpha_power: float = 1.0,
    selector_alpha_topk: int | None = None,
    selector_alpha_local_peak_kernel: int = 1,
    selector_alpha_peak_expand_radius: int = 0,
) -> Dict[str, torch.Tensor]:
    source_tokens, final_tokens = extract_source_and_final_tokens(model, pixel_values, source_layers)
    score_outputs = compute_prototype_scores(
        source_tokens,
        lesion_prototypes,
        background_prototypes,
        selector_mode=selector_mode,
        lesion_topk=selector_lesion_topk,
        background_topk=selector_background_topk,
        smooth_weight=selector_smooth_weight,
        local_contrast_weight=selector_local_contrast_weight,
        agreement_weight=selector_agreement_weight,
    )
    intervention = build_modified_final_tokens(
        final_tokens,
        score_outputs["lesion_score"],
        temperature,
        beta,
        mode,
        alpha_mode=selector_alpha_mode,
        alpha_positive_threshold=selector_alpha_positive_threshold,
        alpha_power=selector_alpha_power,
        alpha_topk=selector_alpha_topk,
        alpha_local_peak_kernel=selector_alpha_local_peak_kernel,
        alpha_peak_expand_radius=selector_alpha_peak_expand_radius,
    )
    projected = get_multi_modal_projector(model)(intervention["modified_final_tokens"].unsqueeze(0))
    return {
        "source_tokens": source_tokens,
        "final_tokens": final_tokens,
        "projected_features": projected,
        **score_outputs,
        **intervention,
    }


def load_bank(bank_path: Path, device: torch.device) -> Dict[str, torch.Tensor | int | float | str]:
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    bank["lesion_prototypes"] = torch.as_tensor(bank["lesion_prototypes"], dtype=torch.float32, device=device)
    bank["background_prototypes"] = torch.as_tensor(bank["background_prototypes"], dtype=torch.float32, device=device)
    if "source_layers" in bank:
        bank["source_layers"] = normalize_source_layers(bank["source_layers"])
    else:
        bank["source_layers"] = normalize_source_layers(int(bank["source_layer"]))
    bank["source_layer_tag"] = format_source_layer_tag(bank["source_layers"])
    bank["source_layer"] = int(bank["source_layers"][0])
    return bank


@contextmanager
def patched_image_features(model, image_features: torch.Tensor):
    projector = get_multi_modal_projector(model)
    original = projector.forward

    def override(vision_outputs: torch.Tensor):
        return image_features.to(device=vision_outputs.device, dtype=vision_outputs.dtype)

    projector.forward = override
    try:
        yield
    finally:
        projector.forward = original


@contextmanager
def patched_vision_layer_tokens(
    model,
    layer_idx: int,
    oracle_alpha: torch.Tensor,
    beta: float,
    mode: str,
    reference_tokens: torch.Tensor | None,
):
    layer = get_vision_encoder_layers(model)[int(layer_idx)]

    def override(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            return output
        intervention = build_modified_final_tokens_from_alpha(
            hidden[0],
            oracle_alpha.to(device=hidden.device),
            beta,
            mode,
            reference_tokens=(
                reference_tokens.to(device=hidden.device, dtype=hidden.dtype)
                if reference_tokens is not None
                else None
            ),
        )
        modified = intervention["modified_final_tokens"].unsqueeze(0).to(device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    handle = layer.register_forward_hook(override)
    try:
        yield
    finally:
        handle.remove()
