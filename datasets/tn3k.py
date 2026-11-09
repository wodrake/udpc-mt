from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from datasets.base import SegmentationDatasetAdapter


SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SUPPORTED_SPLIT_FILES = ("tg3k-trainval.json", "tn3k-trainval.json", "trainval.json", "splits.json")


def _is_tn3k_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "thyroid-image").is_dir()
        and (path / "thyroid-mask").is_dir()
    )


def _resolve_tn3k_root(dataset_root: Path) -> Path:
    candidates = [
        dataset_root,
        dataset_root / "tg3k",
        dataset_root / "Thyroid Dataset" / "tg3k",
    ]
    for candidate in candidates:
        if _is_tn3k_root(candidate):
            return candidate

    for candidate in sorted(dataset_root.rglob("tg3k")):
        if _is_tn3k_root(candidate):
            return candidate

    for candidate in sorted(dataset_root.rglob("*")):
        if _is_tn3k_root(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not discover TN3K/TG3K under {dataset_root}. "
        "Expected a directory containing thyroid-image/ and thyroid-mask/."
    )


def _find_split_file(dataset_root: Path) -> Path | None:
    for name in SUPPORTED_SPLIT_FILES:
        candidate = dataset_root / name
        if candidate.exists():
            return candidate
    return None


def _normalize_split_case_id(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return f"{int(stripped):04d}"
        return Path(stripped).stem
    if isinstance(value, (int, np.integer)):
        return f"{int(value):04d}"
    if isinstance(value, float) and float(value).is_integer():
        return f"{int(value):04d}"
    return str(value)


def _load_split_lookup(dataset_root: Path) -> dict[str, str]:
    split_file = _find_split_file(dataset_root)
    if split_file is None:
        return {}

    with split_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    split_lookup: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for sample_id in payload.get(split_name, []):
            split_lookup[_normalize_split_case_id(sample_id)] = split_name
    return split_lookup


def _resolve_mask_path(mask_dir: Path, stem: str) -> Path | None:
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = mask_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=512)
def _load_grayscale_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


class TN3KDatasetAdapter(SegmentationDatasetAdapter):
    name = "tn3k"
    modality = "ultrasound"
    task_type = "binary"
    default_num_classes = 2
    default_input_channels = 1
    dimension_mode = "2d"
    slice_mode = "single_slice"
    class_names = ("background", "lesion")

    def discover_records(self, dataset_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
        resolved_root = _resolve_tn3k_root(dataset_root)
        image_dir = resolved_root / "thyroid-image"
        mask_dir = resolved_root / "thyroid-mask"
        split_lookup = _load_split_lookup(resolved_root)

        records: list[dict[str, Any]] = []
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            case_id = image_path.stem
            mask_path = _resolve_mask_path(mask_dir, case_id)
            if mask_path is None:
                raise FileNotFoundError(f"Missing TN3K mask for {image_path.name} under {mask_dir}")
            records.append(
                {
                    "sample_id": case_id,
                    "case_id": case_id,
                    "group_id": case_id,
                    "image_path": str(image_path),
                    "mask_paths": [str(mask_path)],
                    "split": split_lookup.get(case_id),
                    "stratify_key": "thyroid_nodule",
                    "source_layout": "tg3k_official" if split_lookup else "tg3k_image_mask",
                }
            )

        if not records:
            raise FileNotFoundError(
                f"Could not discover TN3K/TG3K samples under {resolved_root}. "
                "Expected matching files under thyroid-image/ and thyroid-mask/."
            )
        return records

    def load_image(self, record: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
        image = _load_grayscale_image(record["image_path"])
        return self.normalize_loaded_image(image, config)

    def load_mask(
        self,
        record: dict[str, Any],
        image_shape: tuple[int, int] | tuple[int, int, int],
        config: dict[str, Any],
    ) -> np.ndarray:
        mask = np.asarray(Image.open(record["mask_paths"][0]).convert("L"), dtype=np.uint8)
        threshold = int(config["dataset"].get("mask_threshold", 128))
        return ((mask >= threshold).astype(np.uint8) * 255)
