from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from micro_lesion_vqa_utils import grid_cell_label
from prototype_guided_fundus import ENCODER_GRID, IMAGE_SIZE


def load_benchmark_manifest(benchmark_dir: Path) -> list[dict[str, str]]:
    with (benchmark_dir / "manifest.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_grid_cells(row: dict[str, str]) -> list[str]:
    raw = (row.get("all_grid_cells_json") or "").strip()
    if raw:
        try:
            cells = json.loads(raw)
            return [str(cell).upper() for cell in cells if str(cell).upper() != "NONE"]
        except json.JSONDecodeError:
            pass
    answer = str(row.get("grid_answer") or "").strip()
    if not answer or answer.upper() == "NONE":
        return []
    return [part.strip().upper() for part in answer.split(",") if part.strip()]


def load_benchmark_image(benchmark_dir: Path, row: dict[str, str]) -> Image.Image:
    rel = row.get("output_image") or row.get("image") or row.get("image_path") or ""
    if not rel:
        raise ValueError(f"Missing image path for sample_id={row.get('sample_id', '')}")
    path = Path(rel)
    if not path.is_absolute():
        path = benchmark_dir / path
    return Image.open(path).convert("RGB")


def encoder_patch_grid_labels(grid_size: int, encoder_grid: int = ENCODER_GRID) -> np.ndarray:
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    y_edges = np.linspace(0, encoder_grid, grid_size + 1, dtype=int)
    x_edges = np.linspace(0, encoder_grid, grid_size + 1, dtype=int)
    labels: list[str] = []
    for r in range(grid_size):
        y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
        for yy in range(y0, y1):
            for c in range(grid_size):
                x0, x1 = int(x_edges[c]), int(x_edges[c + 1])
                for _xx in range(x0, x1):
                    labels.append(grid_cell_label(r, c))
    if len(labels) != encoder_grid * encoder_grid:
        raise RuntimeError(
            f"encoder_patch_grid_labels produced {len(labels)} labels, expected {encoder_grid * encoder_grid}"
        )
    return np.asarray(labels, dtype=object)


def patch_mean_intensity(image_resized: Image.Image, encoder_grid: int = ENCODER_GRID) -> np.ndarray:
    gray = np.asarray(image_resized.convert("L"), dtype=np.float32)
    if gray.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected resized image {(IMAGE_SIZE, IMAGE_SIZE)}, got {gray.shape}")
    patch_size = IMAGE_SIZE // encoder_grid
    means: list[float] = []
    for r in range(encoder_grid):
        y0 = r * patch_size
        y1 = (r + 1) * patch_size
        for c in range(encoder_grid):
            x0 = c * patch_size
            x1 = (c + 1) * patch_size
            means.append(float(gray[y0:y1, x0:x1].mean()))
    return np.asarray(means, dtype=np.float32)


def lesion_background_patch_masks(
    *,
    grid_cells: list[str],
    grid_size: int,
    image_resized: Image.Image,
    body_intensity_threshold: float = 5.0,
    encoder_grid: int = ENCODER_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = encoder_patch_grid_labels(grid_size=grid_size, encoder_grid=encoder_grid)
    lesion_mask = np.isin(labels, np.asarray(grid_cells, dtype=object))
    patch_mean = patch_mean_intensity(image_resized=image_resized, encoder_grid=encoder_grid)
    active_mask = patch_mean > float(body_intensity_threshold)
    background_mask = (~lesion_mask) & active_mask
    if not np.any(background_mask):
        background_mask = ~lesion_mask
    return lesion_mask, background_mask, active_mask
