from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PAPER_BANK_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class EasyLensConfig:
    """Inference parameters for the paper-defined EasyTag and EasyAmplifier flow.

    The implementation follows Eqs. (27)-(40): top-M prototype comparison,
    counterfactual scoring, morphology-prior calibration, top-k candidate
    selection, cluster-conditioned support retrieval, and score-weighted
    residual enhancement. The host-model hook applies the paper's residual
    coefficient alpha after this module returns the reference and weight.
    """

    selector_k: int
    seed_topk: int
    prototype_topm: int = 1
    counterfactual_temperature: float = 1.0
    morphology_weight: float = 1.0
    amplifier_threshold: float = 0.0
    similarity_chunk: int = 4096


def _as_float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _normalize_rows(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1, eps=1e-8)


def _legacy_morphology_priors(raw: dict[str, Any]) -> np.ndarray:
    templates = np.asarray(raw["lesion_shape_templates"], dtype=np.float32)
    cluster_ids = np.asarray(raw["lesion_cluster_ids"], dtype=np.int64)
    num_clusters = int(np.asarray(raw["lesion_prototypes"]).shape[0])
    priors: list[np.ndarray] = []
    for cluster_idx in range(num_clusters):
        rows = templates[cluster_ids == cluster_idx]
        prior = rows.mean(axis=0) if len(rows) else np.zeros(templates.shape[1:], dtype=np.float32)
        prior = np.maximum(prior, 0.0)
        scale = float(prior.max())
        if scale > 0:
            prior /= scale
        priors.append(prior.astype(np.float32))
    return np.stack(priors, axis=0)


def load_easylens_bank(path: Path | str, device: torch.device) -> dict[str, Any]:
    """Load an EasyBank for the equation-literal schema-v2 implementation.

    Schema-v2 banks contain image-level lesion prototypes, prototype-specific
    support memories, per-prototype morphology priors, and location-wise normal
    prototypes. Legacy release banks remain readable so published artifacts do
    not become unusable, but the returned metadata labels them as an
    approximation because they were built from patch-level clustering.
    """

    raw = dict(torch.load(Path(path), map_location="cpu", weights_only=False))
    schema_version = int(raw.get("schema_version", 1))
    if schema_version < PAPER_BANK_SCHEMA_VERSION:
        warnings.warn(
            "Loading a legacy EasyBank. Rebuild with the schema-v2 builder for the equation-literal method profile.",
            RuntimeWarning,
            stacklevel=2,
        )
        raw["morphology_priors"] = _legacy_morphology_priors(raw)

    lesion_bank = _as_float_tensor(raw["lesion_bank"], device)
    lesion_prototypes = _as_float_tensor(raw["lesion_prototypes"], device)
    lesion_cluster_ids = torch.as_tensor(raw["lesion_cluster_ids"], dtype=torch.long, device=device)

    member_order_value = raw.get("lesion_cluster_member_order")
    offsets_value = raw.get("lesion_cluster_offsets")
    if member_order_value is None or offsets_value is None:
        order = torch.argsort(lesion_cluster_ids, stable=True)
        counts = torch.bincount(lesion_cluster_ids, minlength=lesion_prototypes.shape[0])
        offsets = torch.zeros(lesion_prototypes.shape[0] + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(counts, dim=0)
    else:
        order = torch.as_tensor(member_order_value, dtype=torch.long, device=device)
        offsets = torch.as_tensor(offsets_value, dtype=torch.long, device=device)

    normal_position = _as_float_tensor(raw["background_position_bank"], device)
    normal_global = _as_float_tensor(raw["background_global_bank"], device)
    normal_counts = torch.as_tensor(raw["background_position_counts"], dtype=torch.long, device=device)
    normal_offsets = torch.as_tensor(raw["background_position_offsets"], dtype=torch.long, device=device)

    grid_side = int(raw.get("grid_side", math.isqrt(int(normal_counts.numel()))))
    feature_dim = int(raw.get("feature_dim", lesion_bank.shape[-1]))
    if grid_side * grid_side != int(normal_counts.numel()):
        raise ValueError("EasyBank position table does not match a square patch grid.")
    if feature_dim != int(lesion_bank.shape[-1]):
        raise ValueError("EasyBank feature_dim does not match lesion support embeddings.")

    return {
        "schema_version": schema_version,
        "alignment": "equation_literal_v3" if schema_version >= PAPER_BANK_SCHEMA_VERSION else "legacy_approximation",
        "grid_side": grid_side,
        "feature_dim": feature_dim,
        "lesion_bank": lesion_bank,
        "lesion_bank_norm": _normalize_rows(lesion_bank),
        "lesion_prototypes": lesion_prototypes,
        "lesion_prototypes_norm": _normalize_rows(lesion_prototypes),
        "lesion_cluster_ids": lesion_cluster_ids,
        "cluster_member_order": order,
        "cluster_offsets": offsets,
        "morphology_priors": _as_float_tensor(raw["morphology_priors"], device).clamp_(0.0, 1.0),
        "normal_position_bank": normal_position,
        "normal_position_bank_norm": _normalize_rows(normal_position) if normal_position.numel() else normal_position,
        "normal_position_counts": normal_counts,
        "normal_position_offsets": normal_offsets,
        "normal_global_bank": normal_global,
        "normal_global_bank_norm": _normalize_rows(normal_global),
        "metadata": {key: value for key, value in raw.items() if not isinstance(value, (np.ndarray, torch.Tensor))},
    }


def _topm_mean(similarity: torch.Tensor, topm: int) -> torch.Tensor:
    if similarity.shape[-1] == 0:
        raise ValueError("Cannot retrieve from an empty prototype space.")
    k = min(max(1, int(topm)), int(similarity.shape[-1]))
    return torch.topk(similarity, k=k, dim=-1).values.mean(dim=-1)


def _location_normal_scores(token_norm: torch.Tensor, bank: dict[str, Any], topm: int) -> torch.Tensor:
    scores = torch.empty(token_norm.shape[0], dtype=torch.float32, device=token_norm.device)
    for position in range(token_norm.shape[0]):
        count = int(bank["normal_position_counts"][position].item())
        if count:
            start = int(bank["normal_position_offsets"][position].item())
            end = int(bank["normal_position_offsets"][position + 1].item())
            references = bank["normal_position_bank_norm"][start:end]
        else:
            references = bank["normal_global_bank_norm"]
        similarity = torch.matmul(token_norm[position : position + 1], references.T)
        scores[position] = _topm_mean(similarity, topm)[0]
    return scores


def _topk_indices(score: torch.Tensor, k: int) -> torch.Tensor:
    k = min(max(1, int(k)), int(score.numel()))
    return torch.topk(score, k=k).indices


def _propagate_morphology(
    source_score: torch.Tensor,
    source_indices: torch.Tensor,
    source_clusters: torch.Tensor,
    priors: torch.Tensor,
    grid_side: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    support = torch.zeros(grid_side * grid_side, dtype=torch.float32, device=source_score.device)
    owner = torch.full((grid_side * grid_side,), -1, dtype=torch.long, device=source_score.device)
    radius = int(priors.shape[-1] // 2)
    for source_index, cluster_index in zip(source_indices.tolist(), source_clusters.tolist()):
        row, col = divmod(int(source_index), grid_side)
        prior = priors[int(cluster_index)]
        amplitude = source_score[int(source_index)]
        for delta_row in range(-radius, radius + 1):
            target_row = row + delta_row
            if not 0 <= target_row < grid_side:
                continue
            for delta_col in range(-radius, radius + 1):
                target_col = col + delta_col
                if not 0 <= target_col < grid_side:
                    continue
                value = amplitude * prior[delta_row + radius, delta_col + radius]
                target = target_row * grid_side + target_col
                if value > support[target]:
                    support[target] = value
                    owner[target] = int(cluster_index)
    return support, owner


def _retrieve_cluster_reference(
    query: torch.Tensor,
    cluster_index: int,
    bank: dict[str, Any],
    chunk_size: int,
) -> torch.Tensor:
    """Return the single closest support from the recalled prototype memory.

    This is the literal argmax retrieval defined by Eqs. (34) and (38). The
    search is chunked only to bound memory; chunking does not alter the nearest
    neighbor result.
    """

    start = int(bank["cluster_offsets"][cluster_index].item())
    end = int(bank["cluster_offsets"][cluster_index + 1].item())
    member_indices = bank["cluster_member_order"][start:end]
    if member_indices.numel() == 0:
        return bank["lesion_prototypes"][cluster_index]

    query_norm = _normalize_rows(query.view(1, -1))
    best_value = torch.tensor(-torch.inf, dtype=torch.float32, device=query.device)
    best_index = member_indices[0]
    for offset in range(0, int(member_indices.numel()), max(1, int(chunk_size))):
        current_indices = member_indices[offset : offset + int(chunk_size)]
        similarities = torch.matmul(query_norm, bank["lesion_bank_norm"][current_indices].T).view(-1)
        local_value, local_index = similarities.max(dim=0)
        if local_value > best_value:
            best_value = local_value
            best_index = current_indices[local_index]
    return bank["lesion_bank"][best_index]


def compute_easylens(
    source_tokens: torch.Tensor,
    bank: dict[str, Any],
    config: EasyLensConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Run EasyTag and EasyAmplifier while leaving the host VLM frozen.

    First, every patch is contrasted against global lesion prototypes and
    same-location normal prototypes. The global top-K0 patches seed
    morphology-prior calibration, after which the global top-K candidates are
    retained. A second morphology pass assigns every supported patch a
    prototype owner and continuous residual weight. Finally, each supported
    patch retrieves a reference only from that prototype's support memory. The
    returned ``residual_weight`` is intended for the host hook as
    ``z + beta * residual_weight * (reference - z)``.
    """

    if source_tokens.ndim != 2:
        raise ValueError(f"Expected [P,D] source tokens, got {tuple(source_tokens.shape)}")
    expected_patches = int(bank["grid_side"]) ** 2
    if int(source_tokens.shape[0]) != expected_patches:
        raise ValueError(f"Patch count {source_tokens.shape[0]} does not match EasyBank grid {bank['grid_side']}.")
    if int(source_tokens.shape[1]) != int(bank["feature_dim"]):
        raise ValueError(f"Feature size {source_tokens.shape[1]} does not match EasyBank size {bank['feature_dim']}.")

    token_norm = _normalize_rows(source_tokens)
    prototype_similarity = torch.matmul(token_norm, bank["lesion_prototypes_norm"].T)
    lesion_score = _topm_mean(prototype_similarity, config.prototype_topm)
    best_prototype = prototype_similarity.argmax(dim=-1)
    normal_score = _location_normal_scores(token_norm, bank, config.prototype_topm)

    counterfactual_logit = lesion_score - normal_score
    initial_score = torch.sigmoid(counterfactual_logit / max(float(config.counterfactual_temperature), 1e-6))

    seed_indices = _topk_indices(initial_score, int(config.seed_topk))
    seed_support, _ = _propagate_morphology(
        initial_score,
        seed_indices,
        best_prototype[seed_indices],
        bank["morphology_priors"],
        int(bank["grid_side"]),
    )
    calibrated_score = initial_score + float(config.morphology_weight) * seed_support

    candidates = _topk_indices(calibrated_score, int(config.selector_k))
    residual_weight, owner = _propagate_morphology(
        calibrated_score,
        candidates,
        best_prototype[candidates],
        bank["morphology_priors"],
        int(bank["grid_side"]),
    )
    active = torch.nonzero(residual_weight > float(config.amplifier_threshold), as_tuple=False).view(-1)
    reference_tokens = source_tokens.float().clone()
    for patch_index in active.tolist():
        cluster_index = int(owner[patch_index].item())
        if cluster_index < 0:
            continue
        reference_tokens[patch_index] = _retrieve_cluster_reference(
            source_tokens[patch_index].float(),
            cluster_index,
            bank,
            int(config.similarity_chunk),
        )

    metadata = {
        "method_profile": "paper_equations_27_40",
        "seed_selection": "global_topk",
        "candidate_selection": "global_topk_calibrated_score",
        "reference_retrieval": "cluster_nearest_neighbor",
        "residual_update": "z_plus_alpha_w_reference_delta",
        "bank_alignment": bank["alignment"],
        "bank_schema_version": int(bank["schema_version"]),
        "seed_patch_count": int(seed_indices.numel()),
        "seed_patch_indices": [int(value) for value in seed_indices.tolist()],
        "selected_patch_count": int(candidates.numel()),
        "selected_patch_indices": [int(value) for value in candidates.tolist()],
        "amplified_patch_count": int(active.numel()),
        "amplified_patch_indices": [int(value) for value in active.tolist()],
        "initial_score_max": float(initial_score.max().item()),
        "initial_score_mean": float(initial_score.mean().item()),
        "calibrated_score_max": float(calibrated_score.max().item()),
        "calibrated_score_mean": float(calibrated_score.mean().item()),
        "residual_weight_max": float(residual_weight.max().item()),
        "residual_weight_mean": float(residual_weight.mean().item()),
        "lesion_similarity_max": float(lesion_score.max().item()),
        "normal_similarity_max": float(normal_score.max().item()),
    }
    return reference_tokens, residual_weight, metadata
