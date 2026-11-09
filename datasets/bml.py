from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from datasets.base import SegmentationDatasetAdapter


def _find_split_dirs(dataset_root: Path) -> list[tuple[str, Path, Path]]:
    candidates = []
    for split in ("train", "val", "test"):
        image_dir = None
        mask_dir = None
        for image_name in ("image", "images"):
            path = dataset_root / split / image_name
            if path.exists():
                image_dir = path
                break
        for mask_name in ("mask", "masks"):
            path = dataset_root / split / mask_name
            if path.exists():
                mask_dir = path
                break
        if image_dir is not None and mask_dir is not None:
            candidates.append((split, image_dir, mask_dir))
    return candidates


def _infer_bml_stratify_key(case_id: str) -> str:
    lowered = case_id.lower()
    for tag in ("benign", "malignant", "normal"):
        if tag in lowered:
            return tag
    return "lesion"


class BMLDatasetAdapter(SegmentationDatasetAdapter):
    name = "bml"
    modality = "mri"
    task_type = "binary"
    default_num_classes = 2
    default_input_channels = 1
    dimension_mode = "2d"
    slice_mode = "single_slice"
    class_names = ("background", "lesion")

    def _discover_from_metadata(self, dataset_root: Path) -> list[dict[str, Any]]:
        metadata_path = dataset_root / "metadata.csv"
        with metadata_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            records = []
            for row in reader:
                split = row.get("split")
                case_id = row["case_id"]
                mask_path = dataset_root / row["mask_path"] if row.get("mask_path") else None
                records.append(
                    {
                        "sample_id": case_id,
                        "case_id": case_id,
                        "group_id": case_id,
                        "image_path": str(dataset_root / row["image_path"]),
                        "mask_paths": [str(mask_path)] if mask_path is not None else [],
                        "split": split,
                        "stratify_key": row.get("pathology", _infer_bml_stratify_key(case_id)),
                        "source_layout": "metadata",
                    }
                )
        return records

    def _discover_from_split_dirs(self, dataset_root: Path) -> list[dict[str, Any]]:
        records = []
        for split, image_dir, mask_dir in _find_split_dirs(dataset_root):
            for image_path in sorted(image_dir.iterdir()):
                if not image_path.is_file():
                    continue
                mask_path = mask_dir / image_path.name
                if not mask_path.exists():
                    alt_mask = mask_dir / f"{image_path.stem}_mask{image_path.suffix}"
                    if alt_mask.exists():
                        mask_path = alt_mask
                    else:
                        continue
                case_id = image_path.stem
                records.append(
                    {
                        "sample_id": case_id,
                        "case_id": case_id,
                        "group_id": case_id,
                        "image_path": str(image_path),
                        "mask_paths": [str(mask_path)],
                        "split": split,
                        "stratify_key": _infer_bml_stratify_key(case_id),
                        "source_layout": "split_dirs",
                    }
                )
        return records

    def discover_records(self, dataset_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
        metadata_path = dataset_root / "metadata.csv"
        if metadata_path.exists():
            records = self._discover_from_metadata(dataset_root)
        else:
            records = self._discover_from_split_dirs(dataset_root)
        if not records:
            raise FileNotFoundError(
                f"Could not discover BML samples under {dataset_root}. "
                "Expected metadata.csv or split folders like train/image and train/mask."
            )
        return records

    def load_image(self, record: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
        image = np.array(Image.open(record["image_path"]).convert("L"), dtype=np.float32)
        return self.normalize_loaded_image(image, config)

    def load_mask(
        self,
        record: dict[str, Any],
        image_shape: tuple[int, int] | tuple[int, int, int],
        config: dict[str, Any],
    ) -> np.ndarray:
        if not record.get("mask_paths"):
            spatial_shape = image_shape[:2] if len(image_shape) == 3 else image_shape
            return np.zeros(spatial_shape, dtype=np.uint8)
        mask = np.array(Image.open(record["mask_paths"][0]).convert("L"))
        return ((mask > 0).astype(np.uint8) * 255)
