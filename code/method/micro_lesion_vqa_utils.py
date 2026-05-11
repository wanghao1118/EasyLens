from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_nifti_array(path: Path | str) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata())


def window_ct(slice_2d: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    arr = np.clip((slice_2d - low) / (high - low + 1e-8), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def choose_best_slice(mask_3d: np.ndarray) -> int:
    sums = mask_3d.sum(axis=(0, 1))
    return int(np.argmax(sums))


def bbox_from_mask(mask_2d: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        h, w = mask_2d.shape
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: Sequence[int],
    margin: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    h, w = image_shape
    return (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(w - 1, x2 + margin),
        min(h - 1, y2 + margin),
    )


def crop_box(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return arr[y1 : y2 + 1, x1 : x2 + 1]


def resize_gray(arr: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray(arr).convert("L").resize((size, size), Image.Resampling.BILINEAR)


def make_ct_montage(
    slice_2d: np.ndarray,
    mask_2d: np.ndarray,
    center: float,
    width: float,
    crop_margin: int = 24,
    panel_size: int = 384,
) -> tuple[Image.Image, Image.Image, dict]:
    full_u8 = window_ct(slice_2d, center=center, width=width)
    bbox = expand_bbox(bbox_from_mask(mask_2d), full_u8.shape, crop_margin)
    crop_u8 = crop_box(full_u8, bbox)
    mask_crop = crop_box(mask_2d.astype(np.uint8), bbox)

    full_img = resize_gray(full_u8, panel_size)
    crop_img = resize_gray(crop_u8, panel_size)

    canvas = Image.new("RGB", (panel_size * 2, panel_size), color=(0, 0, 0))
    canvas.paste(full_img.convert("RGB"), (0, 0))
    canvas.paste(crop_img.convert("RGB"), (panel_size, 0))

    overlay = canvas.copy()
    draw = ImageDraw.Draw(overlay)

    fx1 = bbox[0] * panel_size / full_u8.shape[1]
    fy1 = bbox[1] * panel_size / full_u8.shape[0]
    fx2 = bbox[2] * panel_size / full_u8.shape[1]
    fy2 = bbox[3] * panel_size / full_u8.shape[0]
    draw.rectangle((fx1, fy1, fx2, fy2), outline=(255, 64, 64), width=3)

    if mask_crop.sum() > 0:
        my, mx = np.where(mask_crop > 0)
        if len(mx) > 0:
            cx1 = panel_size + mx.min() * panel_size / mask_crop.shape[1]
            cy1 = my.min() * panel_size / mask_crop.shape[0]
            cx2 = panel_size + mx.max() * panel_size / mask_crop.shape[1]
            cy2 = my.max() * panel_size / mask_crop.shape[0]
            draw.rectangle((cx1, cy1, cx2, cy2), outline=(64, 255, 64), width=3)

    meta = {
        "bbox_xyxy": [int(v) for v in bbox],
        "crop_margin": int(crop_margin),
        "panel_size": int(panel_size),
    }
    return canvas, overlay, meta


def body_bbox_from_hu(
    slice_2d: np.ndarray,
    threshold_hu: float = -950.0,
    min_foreground_pixels: int = 1024,
) -> tuple[int, int, int, int]:
    fg = slice_2d > threshold_hu
    ys, xs = np.where(fg)
    h, w = slice_2d.shape
    if len(xs) < min_foreground_pixels:
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def fit_square_gray(
    arr: np.ndarray,
    target_size: int,
    *,
    upscale: bool = False,
    resample: int = Image.Resampling.BILINEAR,
) -> tuple[np.ndarray, dict]:
    src_h, src_w = arr.shape
    max_src = max(src_h, src_w)
    if max_src <= target_size and not upscale:
        scale = 1.0
        out_h, out_w = src_h, src_w
        resized = arr
    else:
        scale = target_size / float(max_src)
        out_h = max(1, int(round(src_h * scale)))
        out_w = max(1, int(round(src_w * scale)))
        resized = np.asarray(
            Image.fromarray(arr).resize((out_w, out_h), resample),
            dtype=np.uint8,
        )
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    off_y = (target_size - out_h) // 2
    off_x = (target_size - out_w) // 2
    canvas[off_y : off_y + out_h, off_x : off_x + out_w] = resized
    meta = {
        "scale": float(scale),
        "offset_x": int(off_x),
        "offset_y": int(off_y),
        "out_w": int(out_w),
        "out_h": int(out_h),
    }
    return canvas, meta


def project_mask_to_native_canvas(
    mask_2d: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    *,
    target_size: int,
    upscale: bool = False,
) -> np.ndarray:
    mask_crop = crop_box((mask_2d > 0).astype(np.uint8) * 255, crop_bbox)
    mask_fit, _ = fit_square_gray(
        mask_crop,
        target_size,
        upscale=upscale,
        resample=Image.Resampling.NEAREST,
    )
    return mask_fit > 0


def grid_cell_label(row: int, col: int) -> str:
    return f"{chr(ord('A') + row)}{col + 1}"


def grid_cell_xy(row: int, col: int) -> tuple[int, int]:
    return col + 1, row + 1


def grid_cell_coord_text(row: int, col: int) -> str:
    x, y = grid_cell_xy(row, col)
    return f"({x},{y})"


def option_label(index: int) -> str:
    if index < 0:
        raise ValueError("index must be non-negative")
    chars: list[str] = []
    value = index
    while True:
        value, rem = divmod(value, 26)
        chars.append(chr(ord("A") + rem))
        if value == 0:
            break
        value -= 1
    return "".join(reversed(chars))


def draw_grid(
    image: Image.Image,
    grid_size: int,
    *,
    line_color: tuple[int, int, int] = (255, 255, 0),
    line_width: int = 1,
) -> Image.Image:
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    x_edges = np.linspace(0, w, grid_size + 1, dtype=int)
    y_edges = np.linspace(0, h, grid_size + 1, dtype=int)
    for x in x_edges:
        draw.line((int(x), 0, int(x), h), fill=line_color, width=line_width)
    for y in y_edges:
        draw.line((0, int(y), w, int(y)), fill=line_color, width=line_width)
    return canvas


def draw_labeled_grid(
    image: Image.Image,
    grid_size: int,
    *,
    line_color: tuple[int, int, int] = (255, 255, 0),
    text_color: tuple[int, int, int] = (255, 255, 0),
    box_fill: tuple[int, int, int] = (0, 0, 0),
    line_width: int = 1,
    margin: int = 4,
) -> Image.Image:
    canvas = draw_grid(
        image,
        grid_size,
        line_color=line_color,
        line_width=line_width,
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    w, h = canvas.size
    x_edges = np.linspace(0, w, grid_size + 1, dtype=int)
    y_edges = np.linspace(0, h, grid_size + 1, dtype=int)
    for r in range(grid_size):
        for c in range(grid_size):
            label = grid_cell_label(r, c)
            x0, y0 = int(x_edges[c]), int(y_edges[r])
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = int(bbox[2] - bbox[0])
            th = int(bbox[3] - bbox[1])
            tx = x0 + margin
            ty = y0 + margin
            draw.rectangle((tx - 1, ty - 1, tx + tw + 1, ty + th + 1), fill=box_fill)
            draw.text((tx, ty), label, fill=text_color, font=font)
    return canvas


def mask_to_grid_cells(mask_canvas: np.ndarray, grid_size: int) -> tuple[list[str], str | None]:
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    ys, xs = np.where(mask_canvas > 0)
    if len(xs) == 0:
        return [], None

    h, w = mask_canvas.shape
    y_edges = np.linspace(0, h, grid_size + 1, dtype=int)
    x_edges = np.linspace(0, w, grid_size + 1, dtype=int)

    labels: list[str] = []
    best_label: str | None = None
    best_count = -1
    for r in range(grid_size):
        y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
        for c in range(grid_size):
            x0, x1 = int(x_edges[c]), int(x_edges[c + 1])
            count = int(mask_canvas[y0:y1, x0:x1].sum())
            if count <= 0:
                continue
            label = grid_cell_label(r, c)
            labels.append(label)
            if count > best_count:
                best_count = count
                best_label = label
    return labels, best_label


def make_ct_native_slice(
    prev_slice_2d: np.ndarray,
    center_slice_2d: np.ndarray,
    next_slice_2d: np.ndarray,
    mask_2d: np.ndarray,
    *,
    center: float,
    width: float,
    body_margin: int = 8,
    target_size: int = 512,
    threshold_hu: float = -950.0,
    upscale: bool = False,
) -> tuple[Image.Image, Image.Image, dict]:
    crop_bbox = expand_bbox(
        body_bbox_from_hu(center_slice_2d, threshold_hu=threshold_hu),
        center_slice_2d.shape,
        body_margin,
    )
    prev_u8 = crop_box(window_ct(prev_slice_2d, center=center, width=width), crop_bbox)
    center_u8 = crop_box(window_ct(center_slice_2d, center=center, width=width), crop_bbox)
    next_u8 = crop_box(window_ct(next_slice_2d, center=center, width=width), crop_bbox)
    mask_crop = crop_box(mask_2d.astype(np.uint8), crop_bbox)

    prev_fit, fit_meta = fit_square_gray(prev_u8, target_size, upscale=upscale)
    center_fit, _ = fit_square_gray(center_u8, target_size, upscale=upscale)
    next_fit, _ = fit_square_gray(next_u8, target_size, upscale=upscale)

    rgb = np.stack([prev_fit, center_fit, next_fit], axis=-1)
    image = Image.fromarray(rgb, mode="RGB")

    overlay = image.copy()
    if mask_crop.sum() > 0:
        mask_fit, _ = fit_square_gray(
            (mask_crop > 0).astype(np.uint8) * 255,
            target_size,
            upscale=upscale,
            resample=Image.Resampling.NEAREST,
        )
        mask_pos = mask_fit > 0
        arr = np.asarray(overlay, dtype=np.uint8).copy()
        arr[mask_pos] = (
            0.55 * arr[mask_pos] + 0.45 * np.asarray([255, 64, 64], dtype=np.float32)
        ).astype(np.uint8)
        overlay = Image.fromarray(arr, mode="RGB")
        my, mx = np.where(mask_pos)
        if len(mx) > 0:
            draw = ImageDraw.Draw(overlay)
            draw.rectangle(
                (int(mx.min()), int(my.min()), int(mx.max()), int(my.max())),
                outline=(64, 255, 64),
                width=2,
            )

    meta = {
        "body_bbox_xyxy": [int(v) for v in crop_bbox],
        "body_margin": int(body_margin),
        "target_size": int(target_size),
        "threshold_hu": float(threshold_hu),
        "native_upscale": bool(upscale),
        **fit_meta,
    }
    return image, overlay, meta


def make_ct_native_single_slice(
    center_slice_2d: np.ndarray,
    mask_2d: np.ndarray,
    *,
    center: float,
    width: float,
    body_margin: int = 8,
    target_size: int = 512,
    threshold_hu: float = -950.0,
    upscale: bool = False,
) -> tuple[Image.Image, Image.Image, dict]:
    crop_bbox = expand_bbox(
        body_bbox_from_hu(center_slice_2d, threshold_hu=threshold_hu),
        center_slice_2d.shape,
        body_margin,
    )
    center_u8 = crop_box(window_ct(center_slice_2d, center=center, width=width), crop_bbox)
    mask_crop = crop_box(mask_2d.astype(np.uint8), crop_bbox)

    center_fit, fit_meta = fit_square_gray(center_u8, target_size, upscale=upscale)
    image = Image.fromarray(center_fit, mode="L")

    overlay = image.convert("RGB")
    if mask_crop.sum() > 0:
        mask_fit, _ = fit_square_gray(
            (mask_crop > 0).astype(np.uint8) * 255,
            target_size,
            upscale=upscale,
            resample=Image.Resampling.NEAREST,
        )
        mask_pos = mask_fit > 0
        arr = np.asarray(overlay, dtype=np.uint8).copy()
        arr[mask_pos] = (
            0.55 * arr[mask_pos] + 0.45 * np.asarray([255, 64, 64], dtype=np.float32)
        ).astype(np.uint8)
        overlay = Image.fromarray(arr, mode="RGB")
        my, mx = np.where(mask_pos)
        if len(mx) > 0:
            draw = ImageDraw.Draw(overlay)
            draw.rectangle(
                (int(mx.min()), int(my.min()), int(mx.max()), int(my.max())),
                outline=(64, 255, 64),
                width=2,
            )

    meta = {
        "body_bbox_xyxy": [int(v) for v in crop_bbox],
        "body_margin": int(body_margin),
        "target_size": int(target_size),
        "threshold_hu": float(threshold_hu),
        "native_upscale": bool(upscale),
        **fit_meta,
    }
    return image, overlay, meta


def save_llava_records(records: Iterable[dict], path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(list(records), ensure_ascii=False, indent=2), encoding="utf-8")


def make_llava_record(
    *,
    sample_id: str,
    image_rel: str,
    dataset: str,
    modality: str,
    task: str,
    prompt: str,
    answer: str,
) -> dict:
    return {
        "id": sample_id,
        "image": image_rel,
        "dataset": dataset,
        "modality": modality,
        "task": task,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
    }


def save_vqa_variants(
    *,
    yesno_records: Iterable[dict] | None,
    loc_records: Iterable[dict],
    output_dir: Path | str,
) -> None:
    output_dir = Path(output_dir)
    yesno_records = list(yesno_records or [])
    loc_records = list(loc_records)
    yesno_path = output_dir / "vqa_yesno_llava.json"
    if yesno_records:
        save_llava_records(yesno_records, yesno_path)
    elif yesno_path.exists():
        yesno_path.unlink()
    save_llava_records(loc_records, output_dir / "vqa_loc_llava.json")
    # Backward compatibility: the single-file export now tracks the loc-only benchmark.
    save_llava_records(loc_records, output_dir / "vqa_llava.json")


def clean_text(text: str) -> str:
    text = " ".join(str(text).split())
    if not text:
        return text
    return text[0].upper() + text[1:]


def float_or_none(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f
