from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from udpc_mt.utils.misc import PROJECT_ROOT, ensure_dir


HF_REPO_ID = "veyselozdemir/nnUNet-Breast-Cancer-Ultrasound"


def sanitize_identifier(text: str) -> str:
    text = text.lower()
    text = re.sub(r"_0000$", "", text)
    text = re.sub(r"_mask(?:_\d+)?$", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def infer_pathology(path: Path) -> str | None:
    lowered_parts = [part.lower() for part in path.parts]
    for pathology in ("benign", "malignant", "normal"):
        if pathology in lowered_parts or any(pathology in part for part in lowered_parts):
            return pathology
    return None


def discover_original_pairs(raw_dir: Path) -> list[dict]:
    images: dict[str, dict] = {}
    masks: defaultdict[str, list[Path]] = defaultdict(list)

    for png_path in raw_dir.rglob("*.png"):
        pathology = infer_pathology(png_path)
        if pathology is None:
            continue
        stem = png_path.stem.lower()
        case_id = sanitize_identifier(f"{pathology}_{stem}")
        parent_name = png_path.parent.name.lower()
        is_mask = "mask" in parent_name or "_mask" in stem
        if is_mask:
            masks[case_id].append(png_path)
        else:
            images[case_id] = {
                "case_id": case_id,
                "image_path": png_path,
                "pathology": pathology,
            }

    discovered = []
    for case_id, image_info in sorted(images.items()):
        discovered.append({**image_info, "mask_paths": sorted(masks.get(case_id, []))})
    return discovered


def process_dataset(raw_dir: Path, output_dir: Path, limit: int | None = None) -> int:
    image_dir = ensure_dir(output_dir / "images")
    mask_dir = ensure_dir(output_dir / "masks")
    pairs = discover_original_pairs(raw_dir)
    if not pairs:
        raise RuntimeError(f"No BUSI image/mask pairs discovered under {raw_dir}")

    if limit is not None:
        pairs = pairs[:limit]

    rows = []
    for pair in pairs:
        image = Image.open(pair["image_path"]).convert("L")
        image_array = np.array(image, dtype=np.uint8)
        if pair["mask_paths"]:
            combined_mask = np.zeros_like(image_array, dtype=np.uint8)
            for mask_path in pair["mask_paths"]:
                mask_array = (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.uint8) * 255
                combined_mask = np.maximum(combined_mask, mask_array)
        else:
            combined_mask = np.zeros_like(image_array, dtype=np.uint8)

        image_output = image_dir / f"{pair['case_id']}.png"
        mask_output = mask_dir / f"{pair['case_id']}.png"
        Image.fromarray(image_array).save(image_output)
        Image.fromarray(combined_mask).save(mask_output)
        rows.append(
            {
                "case_id": pair["case_id"],
                "pathology": pair["pathology"],
                "image_path": str(image_output.relative_to(output_dir)),
                "mask_path": str(mask_output.relative_to(output_dir)),
            }
        )

    with (output_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "pathology", "image_path", "mask_path"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare BUSI for this project.")
    parser.add_argument("--repo-id", default=HF_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument(
        "--raw-dir",
        default=str(PROJECT_ROOT / "data" / "raw" / "busi_hf"),
        help="Local directory for the raw download cache.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "busi"),
        help="Processed output directory.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional item limit for quick debugging.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(raw_dir)
    ensure_dir(output_dir)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=raw_dir,
        allow_patterns=["**/*.png", "**/*.md", "**/*.json"],
    )
    count = process_dataset(raw_dir, output_dir, limit=args.limit)
    print(f"Prepared {count} BUSI samples in {output_dir}")


if __name__ == "__main__":
    main()
