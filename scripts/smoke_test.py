from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from udpc_mt.engine.trainer import run_training
from udpc_mt.utils.config import load_config
from udpc_mt.utils.misc import PROJECT_ROOT, ensure_dir


def make_toy_dataset(root: Path, image_size: int = 192, seed: int = 123) -> None:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    image_dir = ensure_dir(root / "images")
    mask_dir = ensure_dir(root / "masks")
    rows = []

    specs = [("benign", 12), ("malignant", 12), ("normal", 8)]
    for pathology, count in specs:
        for index in range(count):
            case_id = f"{pathology}_{index:03d}"
            image = Image.fromarray(np.clip(np_rng.normal(90, 18, (image_size, image_size)), 0, 255).astype(np.uint8))
            image = image.filter(ImageFilter.GaussianBlur(radius=1.2))
            mask = Image.new("L", (image_size, image_size), 0)
            if pathology != "normal":
                draw = ImageDraw.Draw(mask)
                lesions = 1 if pathology == "benign" else 2
                for _ in range(lesions):
                    x0 = rng.randint(20, image_size // 2)
                    y0 = rng.randint(20, image_size // 2)
                    x1 = min(image_size - 15, x0 + rng.randint(30, 70))
                    y1 = min(image_size - 15, y0 + rng.randint(25, 65))
                    draw.ellipse((x0, y0, x1, y1), fill=255)
                mask_array = np.array(mask, dtype=np.uint8)
                image_array = np.array(image, dtype=np.uint8)
                lesion_boost = 70 if pathology == "benign" else 95
                image_array = np.clip(image_array + (mask_array > 0) * lesion_boost, 0, 255)
                image = Image.fromarray(image_array.astype(np.uint8))

            image_path = image_dir / f"{case_id}.png"
            mask_path = mask_dir / f"{case_id}.png"
            image.save(image_path)
            mask.save(mask_path)
            rows.append(
                {
                    "case_id": case_id,
                    "pathology": pathology,
                    "image_path": str(image_path.relative_to(root)),
                    "mask_path": str(mask_path.relative_to(root)),
                }
            )

    with (root / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "pathology", "image_path", "mask_path"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal end-to-end training smoke test.")
    parser.add_argument(
        "--mode",
        default="udpc_mt",
        choices=["baseline", "mean_teacher", "udpc_mt"],
        help="Training mode to test.",
    )
    args = parser.parse_args()

    dataset_root = PROJECT_ROOT / "data" / "toy_busi"
    ensure_dir(dataset_root)
    make_toy_dataset(dataset_root)

    config = load_config(PROJECT_ROOT / "configs" / "busi_udpc_mt.yaml")
    config["project_name"] = f"smoke_{args.mode}"
    config["mode"] = args.mode
    config["device"] = "cpu"
    config["dataset"]["root"] = str(dataset_root)
    config["dataset"]["image_size"] = 128
    config["dataset"]["labeled_fraction"] = 0.5
    config["dataset"]["val_fraction"] = 0.2
    config["dataset"]["test_fraction"] = 0.1
    config["train"]["epochs"] = 2
    config["train"]["batch_size"] = 4
    config["train"]["labeled_batch_size"] = 2 if args.mode != "baseline" else 4
    config["train"]["max_steps_per_epoch"] = 2
    config["loss"]["prototype_weight"] = 0.1
    if args.mode == "baseline":
        config["uncertainty"]["enabled"] = False
        config["prototype"]["enabled"] = False
    elif args.mode == "mean_teacher":
        config["uncertainty"]["enabled"] = False
        config["prototype"]["enabled"] = False
    else:
        config["uncertainty"]["enabled"] = True
        config["prototype"]["enabled"] = True
        config["uncertainty"]["mc_passes"] = 2

    run_dir = PROJECT_ROOT / "outputs" / "runs" / f"smoke_{args.mode}"
    summary = run_training(config, run_dir=run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
