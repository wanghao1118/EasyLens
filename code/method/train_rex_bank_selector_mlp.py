from __future__ import annotations

import argparse
import csv
import gc
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoImageProcessor, Gemma3ForConditionalGeneration

from evaluate_medgemma_microbench_bank_selector_visibility import (
    LearnedSelectorMLP,
    assemble_selector_features,
    compute_selector_feature_map,
    load_bank_tensors,
    select_support,
)
from evaluate_server_vlm_microbench import load_manifest
from microbench_prototype_utils import lesion_background_patch_masks, parse_grid_cells
from micro_lesion_vqa_utils import load_nifti_array, project_mask_to_native_canvas
from prototype_guided_fundus import (
    ENCODER_GRID,
    build_oracle_alpha_from_overlaps,
    extract_layer_tokens,
    patch_overlap_from_mask,
    resize_image,
)


@dataclass
class SampleRecord:
    sample_id: str
    feature_path: str
    num_patches: int
    positive_fraction: float
    feature_dim: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small learned selector on micro-lesion patch labels.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-csv", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--bank-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vision-layer", type=int, default=1)
    parser.add_argument("--selector-mode", choices=["s1_gap", "s2_peak", "s3_stability"], default="s1_gap")
    parser.add_argument("--peak-weight", type=float, default=1.0)
    parser.add_argument("--stability-weight", type=float, default=1.0)
    parser.add_argument("--peak-kernel", type=int, default=3)
    parser.add_argument("--stability-kernel", type=int, default=3)
    parser.add_argument("--similarity-chunk", type=int, default=4096)
    parser.add_argument("--tau-pos", type=float, default=0.0)
    parser.add_argument("--tau-neg", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-k", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--body-intensity-threshold", type=float, default=5.0)
    parser.add_argument("--use-token-features", action="store_true")
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


def load_source_case_map(source_manifest_csv: Path) -> dict[str, dict[str, str]]:
    with source_manifest_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    case_map: dict[str, dict[str, str]] = {}
    for row in rows:
        case_map.setdefault(str(row["case_name"]), row)
    return case_map


def build_sample_feature_cache(
    *,
    args: argparse.Namespace,
    model: Gemma3ForConditionalGeneration,
    image_processor: AutoImageProcessor,
    bank: dict,
) -> list[SampleRecord]:
    benchmark_manifest = load_manifest(args.benchmark_dir)
    source_manifest = None
    source_case_map = None
    summary_path = args.benchmark_dir / "summary.json"
    if args.source_manifest_csv is not None and args.raw_root is not None and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_benchmark_dir = Path(str(summary.get("source_benchmark_dir") or ""))
        if source_benchmark_dir.exists():
            source_manifest = load_manifest(source_benchmark_dir)
            source_case_map = load_source_case_map(args.source_manifest_csv)

    rows = list(benchmark_manifest.values())
    if args.limit > 0:
        rows = rows[: args.limit]

    samples: list[SampleRecord] = []
    cache_dir = args.output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    last_case_name: str | None = None
    last_case_payload: tuple[np.ndarray, np.ndarray] | None = None
    use_token_features = bool(args.use_token_features)

    with torch.inference_mode():
        for idx, row in enumerate(rows, start=1):
            image = Image.open(args.benchmark_dir / str(row["output_image"])).convert("RGB")
            image_resized = resize_image(image, mode=Image.Resampling.BILINEAR)
            pixel_inputs = image_processor(images=image, return_tensors="pt")
            pixel_values = pixel_inputs["pixel_values"].to(
                device=model.device,
                dtype=torch.bfloat16 if model.device.type == "cuda" else torch.float32,
            )
            source_tokens = extract_layer_tokens(model, pixel_values, int(args.vision_layer))
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
                use_token_features=use_token_features,
            ).cpu().to(dtype=torch.float16)

            use_raw_supervision = bool(
                source_manifest is not None
                and source_case_map is not None
                and row.get("source_sample_id")
            )
            if use_raw_supervision:
                source_sample_id = str(row["source_sample_id"])
                source_row = source_manifest.get(source_sample_id)
                if not source_row or not source_row.get("case_name"):
                    use_raw_supervision = False

            if use_raw_supervision:
                case_name = str(source_row["case_name"])
                slice_index = int(float(source_row["slice_index"]))
                target_finding_keys = [int(x) for x in parse_jsonish(source_row.get("target_finding_keys_json"), [])]
                body_bbox = tuple(int(v) for v in parse_jsonish(source_row.get("body_bbox_xyxy"), [0, 0, 0, 0]))
                target_size = int(float(source_row.get("target_size", 512) or 512))
                upscale = parse_boolish(source_row.get("native_upscale", False))

                case_payload = last_case_payload if case_name == last_case_name else None
                if case_payload is None:
                    source_case = source_case_map[case_name]
                    image_path = args.raw_root / source_case["image_relpath"]
                    mask_path = args.raw_root / source_case["mask_relpath"]
                    case_payload = (load_nifti_array(image_path), load_nifti_array(mask_path))
                    last_case_name = case_name
                    last_case_payload = case_payload
                image_3d, seg_4d = case_payload

                union_mask = np.zeros(image_3d.shape[:2], dtype=np.uint8)
                for finding_key in target_finding_keys:
                    if 0 <= finding_key < seg_4d.shape[0]:
                        union_mask |= np.asarray(seg_4d[finding_key, :, :, slice_index] > 0, dtype=np.uint8)

                mask_canvas = project_mask_to_native_canvas(union_mask, body_bbox, target_size=target_size, upscale=upscale)
                mask_img = Image.fromarray((mask_canvas > 0).astype(np.uint8) * 255, mode="L")
                mask_resized = resize_image(mask_img, mode=Image.Resampling.NEAREST)
                overlaps = patch_overlap_from_mask(mask_resized, ENCODER_GRID)
                oracle_alpha_np, _ = build_oracle_alpha_from_overlaps(
                    overlaps,
                    tau_pos=float(args.tau_pos),
                    tau_neg=float(args.tau_neg),
                )
                labels = torch.as_tensor(oracle_alpha_np, dtype=torch.float32).cpu()
            else:
                grid_cells = parse_grid_cells(row)
                grid_size = int(float(row.get("grid_size", 0) or 0))
                if grid_size <= 0:
                    raise ValueError(f"Invalid grid_size for sample_id={row.get('sample_id', '')}: {row.get('grid_size')}")
                lesion_mask, _background_mask, _active_mask = lesion_background_patch_masks(
                    grid_cells=grid_cells,
                    grid_size=grid_size,
                    image_resized=image_resized,
                    body_intensity_threshold=float(args.body_intensity_threshold),
                )
                labels = torch.as_tensor(lesion_mask.astype(np.float32), dtype=torch.float32).cpu()
            sample_id = str(row["sample_id"])
            feature_path = cache_dir / f"{sample_id}.pt"
            torch.save(
                {
                    "sample_id": sample_id,
                    "features": features.contiguous(),
                    "labels": labels.contiguous(),
                },
                feature_path,
            )
            samples.append(
                SampleRecord(
                    sample_id=sample_id,
                    feature_path=str(feature_path),
                    num_patches=int(labels.numel()),
                    positive_fraction=float(labels.mean().item()),
                    feature_dim=int(features.shape[1]),
                )
            )
            if idx % 8 == 0 or idx == len(rows):
                pos_frac = float(np.mean([s.positive_fraction for s in samples])) if samples else 0.0
                print(
                    json.dumps(
                        {
                            "phase": "feature_cache",
                            "processed_samples": idx,
                            "total_samples": len(rows),
                            "avg_positive_fraction": pos_frac,
                            "feature_dim": int(features.shape[1]),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            del image, image_resized, pixel_inputs, pixel_values, source_tokens, feature_map, features, labels
            if idx % 8 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    manifest = [record.__dict__ for record in samples]
    (args.output_dir / "feature_cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return samples


def split_samples(samples: list[SampleRecord], train_frac: float, seed: int) -> tuple[list[SampleRecord], list[SampleRecord]]:
    rng = random.Random(seed)
    sample_ids = [s.sample_id for s in samples]
    rng.shuffle(sample_ids)
    train_count = max(1, int(round(len(sample_ids) * train_frac)))
    if train_count >= len(sample_ids):
        train_count = max(1, len(sample_ids) - 1)
    train_ids = set(sample_ids[:train_count])
    train_samples = [s for s in samples if s.sample_id in train_ids]
    val_samples = [s for s in samples if s.sample_id not in train_ids]
    return train_samples, val_samples


def load_cached_sample(record: SampleRecord) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(record.feature_path, map_location="cpu", weights_only=False)
    return payload["features"], payload["labels"]


def flatten_samples(samples: list[SampleRecord]) -> tuple[torch.Tensor, torch.Tensor]:
    features_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for record in samples:
        features, labels = load_cached_sample(record)
        features_list.append(features)
        labels_list.append(labels)
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return features, labels


def evaluate_val(
    model: LearnedSelectorMLP,
    samples: list[SampleRecord],
    *,
    device: torch.device,
    support_k: int,
) -> dict[str, float]:
    if not samples:
        return {
            "patch_loss": 0.0,
            "mean_positive_logit": 0.0,
            "mean_negative_logit": 0.0,
            "hit_rate_at_k": 0.0,
            "mean_recall_at_k": 0.0,
        }
    patch_losses: list[float] = []
    pos_logits: list[float] = []
    neg_logits: list[float] = []
    hit_rows: list[float] = []
    recall_rows: list[float] = []
    with torch.inference_mode():
        for sample in samples:
            features_cpu, labels_cpu = load_cached_sample(sample)
            feats = features_cpu.to(device=device, dtype=torch.float32)
            labels = labels_cpu.to(device=device, dtype=torch.float32)
            logits = model(feats)
            patch_losses.append(float(F.binary_cross_entropy_with_logits(logits, labels).item()))
            pos_mask = labels > 0.5
            neg_mask = ~pos_mask
            if pos_mask.any():
                pos_logits.append(float(logits[pos_mask].mean().item()))
            if neg_mask.any():
                neg_logits.append(float(logits[neg_mask].mean().item()))
            alpha = select_support(logits.float(), support_mode="global_topk", k=support_k, suppression_radius=1)
            selected = alpha > 0
            tp = int((selected & pos_mask).sum().item())
            pos_count = int(pos_mask.sum().item())
            hit_rows.append(float(tp > 0))
            recall_rows.append(float(tp / max(pos_count, 1)))
            del features_cpu, labels_cpu, feats, labels, logits, alpha, selected
    return {
        "patch_loss": float(np.mean(patch_losses)),
        "mean_positive_logit": float(np.mean(pos_logits)) if pos_logits else 0.0,
        "mean_negative_logit": float(np.mean(neg_logits)) if neg_logits else 0.0,
        "hit_rate_at_k": float(np.mean(hit_rows)),
        "mean_recall_at_k": float(np.mean(recall_rows)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    image_processor = AutoImageProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if device.type == "cuda" else torch.float32),
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    bank = load_bank_tensors(args.bank_path, torch.device(model.device))
    samples = build_sample_feature_cache(args=args, model=model, image_processor=image_processor, bank=bank)
    train_samples, val_samples = split_samples(samples, float(args.train_frac), int(args.seed))
    train_x, train_y = flatten_samples(train_samples)
    val_x, val_y = flatten_samples(val_samples) if val_samples else (torch.empty(0), torch.empty(0))

    feature_dim = int(train_x.shape[1])
    selector = LearnedSelectorMLP(input_dim=feature_dim, hidden_dim=int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    pos_count = float(train_y.sum().item())
    neg_count = float(train_y.numel() - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, drop_last=False)

    history: list[dict] = []
    best_state = None
    best_metric = -1e9

    run_meta = {
        "model_path": str(args.model_path),
        "benchmark_dir": str(args.benchmark_dir),
        "source_manifest_csv": str(args.source_manifest_csv) if args.source_manifest_csv is not None else None,
        "raw_root": str(args.raw_root) if args.raw_root is not None else None,
        "bank_path": str(args.bank_path),
        "vision_layer": int(args.vision_layer),
        "selector_mode": args.selector_mode,
        "use_token_features": bool(args.use_token_features),
        "feature_dim": feature_dim,
        "hidden_dim": int(args.hidden_dim),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "train_frac": float(args.train_frac),
        "val_k": int(args.val_k),
        "seed": int(args.seed),
        "num_samples_total": len(samples),
        "num_samples_train": len(train_samples),
        "num_samples_val": len(val_samples),
        "num_patches_train": int(train_x.shape[0]),
        "num_patches_val": int(val_x.shape[0]) if val_samples else 0,
        "positive_fraction_train": float(train_y.mean().item()),
        "positive_fraction_val": float(val_y.mean().item()) if val_samples else 0.0,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    for epoch in range(1, int(args.epochs) + 1):
        selector.train()
        train_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = selector(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        selector.eval()
        val_metrics = evaluate_val(selector, val_samples, device=device, support_k=int(args.val_k))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            **val_metrics,
        }
        history.append(row)
        (args.output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

        score = float(val_metrics["mean_recall_at_k"])
        if score > best_metric:
            best_metric = score
            best_state = {
                "state_dict": selector.state_dict(),
                "meta": {
                    **run_meta,
                    "input_dim": feature_dim,
                    "best_epoch": epoch,
                    "best_val_metric_name": f"mean_recall_at_{int(args.val_k)}",
                    "best_val_metric": score,
                    "val_metrics": val_metrics,
                },
            }
            torch.save(best_state, args.output_dir / "selector_best.pt")

        print(json.dumps(row, ensure_ascii=False), flush=True)

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint")

    summary = {
        "best_metric": best_metric,
        "best_epoch": best_state["meta"]["best_epoch"],
        "best_val_metric_name": best_state["meta"]["best_val_metric_name"],
        "best_val_metrics": best_state["meta"]["val_metrics"],
        "run_meta": run_meta,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
