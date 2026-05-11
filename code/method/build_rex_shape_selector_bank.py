from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Gemma3ForConditionalGeneration

from microbench_prototype_utils import (
    lesion_background_patch_masks,
    load_benchmark_image,
    load_benchmark_manifest,
    parse_grid_cells,
    patch_mean_intensity,
)
from micro_lesion_vqa_utils import load_nifti_array, project_mask_to_native_canvas
from prototype_guided_fundus import ENCODER_GRID, extract_layer_tokens, patch_overlap_from_mask, resize_image, run_kmeans, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a ReX-specific shape-aware selector bank.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--source-layer", type=int, default=1)
    parser.add_argument("--num-lesion-prototypes", type=int, default=8)
    parser.add_argument("--max-lesion-per-image", type=int, default=256)
    parser.add_argument("--max-bg-per-image", type=int, default=0, help="0 means keep all background tokens from each image before final bank sampling.")
    parser.add_argument("--max-lesion-tokens", type=int, default=30000)
    parser.add_argument("--max-bg-per-position", type=int, default=0, help="0 means keep all position-specific background tokens.")
    parser.add_argument("--max-bg-global-tokens", type=int, default=60000)
    parser.add_argument("--body-intensity-threshold", type=float, default=5.0)
    parser.add_argument("--shape-radius", type=int, default=2, help="Local window radius used to store lesion shape templates.")
    parser.add_argument("--ring-weight", type=float, default=0.5, help="Negative weight assigned to the one-ring suppression region.")
    parser.add_argument("--ring-mode", choices=["suppress", "boost"], default="suppress")
    parser.add_argument("--kmeans-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--mask-overlap-threshold", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_jsonish(value, default):
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def parse_boolish(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def sample_indices(num_rows: int, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if max_rows <= 0 or num_rows <= max_rows:
        return np.arange(num_rows, dtype=np.int64)
    idx = np.sort(rng.choice(num_rows, size=max_rows, replace=False))
    return idx.astype(np.int64)


def build_grid_cell_mask_image(grid_cells: list[str], grid_size: int, image_size: tuple[int, int]) -> Image.Image:
    width, height = image_size
    cell_mask = np.zeros((height, width), dtype=np.uint8)
    y_edges = np.linspace(0, height, grid_size + 1, dtype=int)
    x_edges = np.linspace(0, width, grid_size + 1, dtype=int)
    for cell in grid_cells:
        cell = str(cell).strip().upper()
        if len(cell) < 2:
            continue
        row = ord(cell[0]) - ord("A")
        try:
            col = int(cell[1:]) - 1
        except ValueError:
            continue
        if not (0 <= row < grid_size and 0 <= col < grid_size):
            continue
        y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
        x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
        cell_mask[y0:y1, x0:x1] = 255
    return Image.fromarray(cell_mask, mode="L")


def dilate_binary(window: np.ndarray) -> np.ndarray:
    padded = np.pad(window.astype(bool), 1, mode="constant", constant_values=False)
    out = np.zeros_like(window, dtype=bool)
    height, width = window.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out |= padded[1 + dr : 1 + dr + height, 1 + dc : 1 + dc + width]
    return out


def build_local_shape_template(
    lesion_mask_flat: np.ndarray,
    center_idx: int,
    radius: int,
    ring_weight: float,
    ring_mode: str,
) -> np.ndarray:
    radius = max(0, int(radius))
    size = 2 * radius + 1
    lesion_grid = lesion_mask_flat.reshape(ENCODER_GRID, ENCODER_GRID).astype(bool)
    row, col = divmod(int(center_idx), ENCODER_GRID)
    template_pos = np.zeros((size, size), dtype=np.float32)
    row_min = max(0, row - radius)
    row_max = min(ENCODER_GRID, row + radius + 1)
    col_min = max(0, col - radius)
    col_max = min(ENCODER_GRID, col + radius + 1)
    rr0 = row_min - (row - radius)
    cc0 = col_min - (col - radius)
    template_pos[rr0 : rr0 + (row_max - row_min), cc0 : cc0 + (col_max - col_min)] = lesion_grid[row_min:row_max, col_min:col_max]
    template_pos = (template_pos > 0).astype(np.float32)
    if template_pos[radius, radius] <= 0:
        template_pos[radius, radius] = 1.0
    ring = dilate_binary(template_pos > 0) & ~(template_pos > 0)
    if ring_mode == "boost":
        template = template_pos + float(ring_weight) * ring.astype(np.float32)
    else:
        template = template_pos - float(ring_weight) * ring.astype(np.float32)
    return template.astype(np.float32)


def build_supervision_masks(
    *,
    row: dict[str, str],
    image_resized: Image.Image,
    raw_root: Path | None,
    mask_overlap_threshold: float,
    body_intensity_threshold: float,
    last_case_name: str | None,
    last_seg_4d: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, Image.Image | None, str, str | None, np.ndarray | None]:
    use_mask_supervision = bool(
        raw_root is not None
        and row.get("case_name")
        and row.get("slice_index") not in {None, ""}
        and row.get("target_finding_keys_json") not in {None, ""}
        and row.get("body_bbox_xyxy") not in {None, ""}
    )
    if use_mask_supervision:
        case_name = str(row["case_name"])
        seg_4d = last_seg_4d if case_name == last_case_name else None
        if seg_4d is None:
            seg_path = raw_root / "segmentations" / case_name
            if not seg_path.exists():
                raise FileNotFoundError(f"Missing segmentation for {case_name}: {seg_path}")
            seg_4d = load_nifti_array(seg_path)
        if seg_4d.ndim != 4:
            raise ValueError(f"Bad segmentation shape for {case_name}: {seg_4d.shape}")
        slice_index = int(float(row["slice_index"]))
        target_finding_keys = [int(x) for x in parse_jsonish(row.get("target_finding_keys_json"), [])]
        union_mask = np.zeros(seg_4d.shape[1:3], dtype=np.uint8)
        for finding_key in target_finding_keys:
            if 0 <= finding_key < seg_4d.shape[0]:
                union_mask |= np.asarray(seg_4d[finding_key, :, :, slice_index] > 0, dtype=np.uint8)
        body_bbox = tuple(int(v) for v in parse_jsonish(row.get("body_bbox_xyxy"), [0, 0, 0, 0]))
        target_size = int(float(row.get("target_size", 512) or 512))
        upscale = parse_boolish(row.get("native_upscale", False))
        mask_canvas = project_mask_to_native_canvas(union_mask, body_bbox, target_size=target_size, upscale=upscale)
        mask_img = Image.fromarray((mask_canvas > 0).astype(np.uint8) * 255, mode="L")
        mask_resized = resize_image(mask_img, mode=Image.Resampling.NEAREST)
        overlaps = patch_overlap_from_mask(mask_resized, ENCODER_GRID)
        lesion_mask = overlaps > float(mask_overlap_threshold)
        active_mask = patch_mean_intensity(image_resized=image_resized, encoder_grid=ENCODER_GRID) > float(body_intensity_threshold)
        background_mask = (~lesion_mask) & active_mask
        if not np.any(background_mask):
            background_mask = ~lesion_mask
        return lesion_mask, background_mask, mask_resized, "raw_mask_projection", case_name, seg_4d

    grid_size = int(float(row.get("grid_size", 0) or 0))
    grid_cells = parse_grid_cells(row)
    lesion_mask, background_mask, _active_mask = lesion_background_patch_masks(
        grid_cells=grid_cells,
        grid_size=grid_size,
        image_resized=image_resized,
        body_intensity_threshold=body_intensity_threshold,
    )
    lesion_mask_img = build_grid_cell_mask_image(grid_cells, grid_size, image_resized.size)
    return lesion_mask, background_mask, lesion_mask_img, "benchmark_grid_cells", last_case_name, last_seg_4d


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_benchmark_manifest(args.benchmark_dir)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No rows found in {args.benchmark_dir / 'manifest.csv'}")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        str(args.model_path),
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    lesion_tokens: list[np.ndarray] = []
    lesion_templates: list[np.ndarray] = []
    lesion_patch_positions: list[int] = []
    lesion_sample_ids: list[str] = []
    background_global_tokens: list[np.ndarray] = []
    background_position_tokens: list[list[np.ndarray]] = [[] for _ in range(ENCODER_GRID * ENCODER_GRID)]
    used_sample_ids: list[str] = []
    last_case_name: str | None = None
    last_seg_4d: np.ndarray | None = None

    for idx, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id", ""))
        image = load_benchmark_image(args.benchmark_dir, row)
        image_resized = resize_image(image, mode=Image.Resampling.BILINEAR)
        try:
            lesion_mask, background_mask, _mask_img, supervision_tag, new_last_case_name, new_last_seg_4d = build_supervision_masks(
                row=row,
                image_resized=image_resized,
                raw_root=args.raw_root,
                mask_overlap_threshold=float(args.mask_overlap_threshold),
                body_intensity_threshold=float(args.body_intensity_threshold),
                last_case_name=last_case_name,
                last_seg_4d=last_seg_4d,
            )
        except Exception as exc:
            print(f"[skip] {sample_id}: {exc}", flush=True)
            continue

        last_case_name = new_last_case_name
        last_seg_4d = new_last_seg_4d
        if not np.any(lesion_mask) or not np.any(background_mask):
            print(f"[skip] {sample_id}: empty lesion/background mask", flush=True)
            continue

        inputs = processor(images=image_resized, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
        source_tokens = extract_layer_tokens(model, pixel_values, int(args.source_layer))
        source_np = source_tokens.float().cpu().numpy()

        lesion_idx = np.flatnonzero(lesion_mask)
        lesion_keep = lesion_idx[sample_indices(len(lesion_idx), int(args.max_lesion_per_image), rng)]
        lesion_keep = np.asarray(sorted(int(x) for x in lesion_keep), dtype=np.int64)
        for patch_idx in lesion_keep.tolist():
            lesion_tokens.append(source_np[patch_idx].astype(np.float32, copy=False))
            lesion_templates.append(
                build_local_shape_template(
                    lesion_mask.astype(np.bool_),
                    patch_idx,
                    radius=int(args.shape_radius),
                    ring_weight=float(args.ring_weight),
                    ring_mode=str(args.ring_mode),
                )
            )
            lesion_patch_positions.append(int(patch_idx))
            lesion_sample_ids.append(sample_id)

        bg_idx = np.flatnonzero(background_mask)
        bg_keep = bg_idx[sample_indices(len(bg_idx), int(args.max_bg_per_image), rng)]
        bg_keep = np.asarray(sorted(int(x) for x in bg_keep), dtype=np.int64)
        for patch_idx in bg_keep.tolist():
            tok = source_np[patch_idx].astype(np.float16, copy=False)
            background_global_tokens.append(tok)
            background_position_tokens[int(patch_idx)].append(tok)

        if len(lesion_keep) > 0 and len(bg_keep) > 0:
            used_sample_ids.append(sample_id)

        print(
            f"[{idx}/{len(rows)}] {sample_id} supervision={supervision_tag} lesion_kept={len(lesion_keep)} bg_tokens={len(bg_keep)}",
            flush=True,
        )
        del image, image_resized, inputs, pixel_values, source_tokens, source_np
        if idx % 8 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not lesion_tokens:
        raise ValueError("No lesion tokens were collected.")
    if not background_global_tokens:
        raise ValueError("No background tokens were collected.")

    lesion_bank = np.stack(lesion_tokens, axis=0).astype(np.float32, copy=False)
    lesion_shape_templates = np.stack(lesion_templates, axis=0).astype(np.float16, copy=False)
    lesion_patch_positions_arr = np.asarray(lesion_patch_positions, dtype=np.int32)
    lesion_sample_ids_arr = np.asarray(lesion_sample_ids, dtype=object)

    lesion_keep = sample_indices(len(lesion_bank), int(args.max_lesion_tokens), rng)
    lesion_bank = lesion_bank[lesion_keep]
    lesion_shape_templates = lesion_shape_templates[lesion_keep]
    lesion_patch_positions_arr = lesion_patch_positions_arr[lesion_keep]
    lesion_sample_ids_arr = lesion_sample_ids_arr[lesion_keep]

    num_centers = min(int(args.num_lesion_prototypes), int(len(lesion_bank)))
    lesion_prototypes = run_kmeans(lesion_bank.astype(np.float32, copy=False), num_centers, int(args.seed), int(args.kmeans_iters))
    lesion_bank_norm = lesion_bank / np.clip(np.linalg.norm(lesion_bank, axis=1, keepdims=True), 1e-6, None)
    lesion_proto_norm = lesion_prototypes / np.clip(np.linalg.norm(lesion_prototypes, axis=1, keepdims=True), 1e-6, None)
    lesion_cluster_ids = np.argmax(lesion_bank_norm @ lesion_proto_norm.T, axis=1).astype(np.int16)
    lesion_cluster_member_order = np.argsort(lesion_cluster_ids, kind="stable").astype(np.int32)
    lesion_cluster_member_ids = lesion_cluster_ids[lesion_cluster_member_order]
    lesion_cluster_offsets = np.zeros(num_centers + 1, dtype=np.int32)
    for cluster_idx in range(num_centers):
        lesion_cluster_offsets[cluster_idx + 1] = lesion_cluster_offsets[cluster_idx] + int((lesion_cluster_member_ids == cluster_idx).sum())

    background_global_bank = np.stack(background_global_tokens, axis=0).astype(np.float16, copy=False)
    global_keep = sample_indices(len(background_global_bank), int(args.max_bg_global_tokens), rng)
    background_global_bank = background_global_bank[global_keep]

    position_counts = np.zeros(ENCODER_GRID * ENCODER_GRID, dtype=np.int32)
    position_parts: list[np.ndarray] = []
    for pos_idx, pos_rows in enumerate(background_position_tokens):
        if not pos_rows:
            continue
        arr = np.stack(pos_rows, axis=0).astype(np.float16, copy=False)
        if int(args.max_bg_per_position) > 0:
            keep = sample_indices(len(arr), int(args.max_bg_per_position), rng)
            arr = arr[keep]
        position_counts[pos_idx] = int(len(arr))
        position_parts.append(arr)
    position_offsets = np.zeros(ENCODER_GRID * ENCODER_GRID + 1, dtype=np.int64)
    position_offsets[1:] = np.cumsum(position_counts, dtype=np.int64)
    if position_parts:
        background_position_bank = np.concatenate(position_parts, axis=0).astype(np.float16, copy=False)
    else:
        feature_dim = int(lesion_bank.shape[1])
        background_position_bank = np.zeros((0, feature_dim), dtype=np.float16)

    bank = {
        "benchmark_dir": str(args.benchmark_dir),
        "model_path": str(args.model_path),
        "source_layer": int(args.source_layer),
        "token_supervision": ("raw_mask_projection" if args.raw_root is not None else "benchmark_grid_cells"),
        "raw_root": str(args.raw_root) if args.raw_root is not None else None,
        "mask_overlap_threshold": float(args.mask_overlap_threshold),
        "body_intensity_threshold": float(args.body_intensity_threshold),
        "shape_radius": int(args.shape_radius),
        "ring_weight": float(args.ring_weight),
        "ring_mode": str(args.ring_mode),
        "num_samples_used": int(len(used_sample_ids)),
        "used_sample_ids": used_sample_ids,
        "num_lesion_tokens": int(len(lesion_bank)),
        "num_lesion_prototypes": int(len(lesion_prototypes)),
        "lesion_bank": lesion_bank.astype(np.float16),
        "lesion_prototypes": lesion_prototypes.astype(np.float16),
        "lesion_cluster_ids": lesion_cluster_ids,
        "lesion_cluster_member_order": lesion_cluster_member_order,
        "lesion_cluster_offsets": lesion_cluster_offsets,
        "lesion_shape_templates": lesion_shape_templates,
        "lesion_patch_positions": lesion_patch_positions_arr,
        "lesion_sample_ids": lesion_sample_ids_arr.tolist(),
        "background_global_bank": background_global_bank,
        "background_position_bank": background_position_bank,
        "background_position_counts": position_counts,
        "background_position_offsets": position_offsets,
        "background_position_num_banks": int(len(position_counts)),
        "seed": int(args.seed),
        "kmeans_iters": int(args.kmeans_iters),
        "max_bg_per_position": int(args.max_bg_per_position),
        "max_bg_per_image": int(args.max_bg_per_image),
        "max_bg_global_tokens": int(args.max_bg_global_tokens),
    }
    torch.save(bank, args.output_path)
    print(
        json.dumps(
            {
                "output_path": str(args.output_path),
                "num_samples_used": bank["num_samples_used"],
                "num_lesion_tokens": bank["num_lesion_tokens"],
                "num_lesion_prototypes": bank["num_lesion_prototypes"],
                "background_position_nonempty": int((position_counts > 0).sum()),
                "background_position_total_tokens": int(position_counts.sum()),
                "background_global_tokens": int(len(background_global_bank)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
