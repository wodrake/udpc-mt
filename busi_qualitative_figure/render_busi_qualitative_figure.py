from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a BUSI qualitative comparison figure.")
    parser.add_argument("--asset-root", required=True, help="Asset directory with images, masks, and prediction folders.")
    parser.add_argument("--output", required=True, help="Output figure path, e.g. figure7_busi_4methods.png")
    return parser.parse_args()


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    return (load_gray(path) > 127).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgb = np.stack([image, image, image], axis=-1).astype(np.float32)
    alpha = 0.45
    region = mask > 0
    rgb[region] = (1.0 - alpha) * rgb[region] + alpha * np.array(color, dtype=np.float32)

    # Add a thin boundary for easier reading in print.
    padded = np.pad(mask, 1, mode="constant")
    core = padded[1:-1, 1:-1]
    neighbors = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    boundary = (core > 0) & (neighbors < 4)
    rgb[boundary] = np.array([255, 255, 0], dtype=np.float32)
    return rgb.clip(0, 255).astype(np.uint8)


def pretty_case_label(case_id: str) -> str:
    parts = case_id.split("_")
    return f"{parts[0].title()} {parts[1].title()} {parts[2]}"


def main() -> None:
    args = parse_args()
    asset_root = Path(args.asset_root)
    output_path = Path(args.output)

    manifest = json.loads((asset_root / "manifest.json").read_text())
    cases: list[str] = manifest["cases"]
    methods = [
        ("unet", "UNet"),
        ("mt", "MT"),
        ("pccs", "PCCS"),
        ("udpc_mt", "UDPC-MT"),
    ]
    columns = [("input", "Image"), ("gt", "GT")] + methods

    n_rows = len(cases)
    n_cols = len(columns)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.2 * n_cols, 2.25 * n_rows + 0.2),
        constrained_layout=True,
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, case_id in enumerate(cases):
        image = load_gray(asset_root / "images" / f"{case_id}.png")
        gt_mask = load_mask(asset_root / "masks" / f"{case_id}.png")

        panels = {
            "input": np.stack([image, image, image], axis=-1),
            "gt": overlay_mask(image, gt_mask, (60, 220, 80)),
        }
        for method_key, _ in methods:
            pred_mask = load_mask(asset_root / "preds" / method_key / f"{case_id}_pred.png")
            panels[method_key] = overlay_mask(image, pred_mask, (255, 80, 80))

        for col_idx, (column_key, title) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            ax.imshow(panels[column_key], cmap=None)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(title, fontsize=11, pad=6)
            if col_idx == 0:
                ax.set_ylabel(pretty_case_label(case_id), fontsize=10, rotation=90, labelpad=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
