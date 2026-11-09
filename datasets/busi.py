from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from datasets.base import (
    SegmentationDatasetAdapter,
    SegmentationEvalDataset,
    SegmentationLabeledDataset,
    SegmentationUnlabeledDataset,
    resolve_dataset_root,
)
from datasets.factory import create_dataloaders as create_generic_dataloaders
from datasets.semi_split import (
    apply_busi_split_assignments,
    build_explicit_or_stratified_splits,
    filter_busi_subset,
    load_busi_split_file_assignments,
    read_busi_manifest,
    split_labeled_unlabeled,
)


def read_grayscale_image(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def read_binary_mask(mask_paths: list[str] | list[Path], image_shape: tuple[int, int]) -> np.ndarray:
    if not mask_paths:
        return np.zeros(image_shape, dtype=np.uint8)

    combined_mask = np.zeros(image_shape, dtype=np.uint8)
    for mask_path in mask_paths:
        mask_array = (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.uint8) * 255
        combined_mask = np.maximum(combined_mask, mask_array)
    return combined_mask


class BUSIDatasetAdapter(SegmentationDatasetAdapter):
    name = "busi"
    modality = "ultrasound"
    task_type = "binary"
    default_num_classes = 2
    default_input_channels = 1
    dimension_mode = "2d"
    slice_mode = "single_slice"
    class_names = ("background", "lesion")

    def discover_records(self, dataset_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
        dataset_cfg = config["dataset"]
        items = read_busi_manifest(dataset_root)
        items = filter_busi_subset(items, subset=str(dataset_cfg.get("subset", "all")))
        split_assignments = load_busi_split_file_assignments(dataset_cfg.get("split_files"))
        items = apply_busi_split_assignments(items, split_assignments)
        records = []
        for item in items:
            records.append(
                {
                    **item,
                    "sample_id": item["case_id"],
                    "group_id": item.get("group_id", item["case_id"]),
                    "stratify_key": item.get("stratify_key", item["pathology"]),
                }
            )
        return records

    def load_image(self, record: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
        image = read_grayscale_image(record["image_path"])
        return self.normalize_loaded_image(image, config)

    def load_mask(
        self,
        record: dict[str, Any],
        image_shape: tuple[int, int] | tuple[int, int, int],
        config: dict[str, Any],
    ) -> np.ndarray:
        spatial_shape = image_shape[:2] if len(image_shape) == 3 else image_shape
        return read_binary_mask(record["mask_paths"], spatial_shape)


class BUSILabeledDataset(SegmentationLabeledDataset):
    def __init__(self, items: list[dict[str, Any]], image_size: int) -> None:
        config = {
            "dataset": {
                "name": "busi",
                "image_size": image_size,
                "task_type": "binary",
                "augmentation_profile": "ultrasound",
                "preprocessing": {"normalization": "minmax"},
            },
            "model": {"num_classes": 1, "in_channels": 1},
        }
        super().__init__(items, adapter=BUSIDatasetAdapter(), config=config)


class BUSIUnlabeledDataset(SegmentationUnlabeledDataset):
    def __init__(self, items: list[dict[str, Any]], image_size: int) -> None:
        config = {
            "dataset": {
                "name": "busi",
                "image_size": image_size,
                "task_type": "binary",
                "augmentation_profile": "ultrasound",
                "preprocessing": {"normalization": "minmax"},
            },
            "model": {"num_classes": 1, "in_channels": 1},
        }
        super().__init__(items, adapter=BUSIDatasetAdapter(), config=config)


class BUSIEvalDataset(SegmentationEvalDataset):
    def __init__(self, items: list[dict[str, Any]], image_size: int) -> None:
        config = {
            "dataset": {
                "name": "busi",
                "image_size": image_size,
                "task_type": "binary",
                "augmentation_profile": "ultrasound",
                "preprocessing": {"normalization": "minmax"},
            },
            "model": {"num_classes": 1, "in_channels": 1},
        }
        super().__init__(items, adapter=BUSIDatasetAdapter(), config=config)


def build_busi_splits(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    dataset_cfg = config["dataset"]
    dataset_root = resolve_dataset_root(dataset_cfg["root"])
    items = BUSIDatasetAdapter().discover_records(dataset_root, config)
    if dataset_cfg.get("sample_limit") is not None:
        items = items[: int(dataset_cfg["sample_limit"])]

    split_items = build_explicit_or_stratified_splits(
        items,
        val_fraction=dataset_cfg["val_fraction"],
        test_fraction=dataset_cfg["test_fraction"],
        seed=config["seed"],
        split_source=str(dataset_cfg.get("split_source", "auto")),
    )
    train_items = split_items["train"]
    explicit_label_flags = ["is_labeled" in item for item in train_items]
    if any(explicit_label_flags) and not all(explicit_label_flags):
        raise ValueError(
            "Train split mixes explicit and implicit BUSI label assignments. "
            "Use labeled/unlabeled split files consistently."
        )
    if train_items and all(explicit_label_flags):
        labeled_items = [item for item in train_items if bool(item["is_labeled"])]
        unlabeled_items = [item for item in train_items if not bool(item["is_labeled"])]
    else:
        labeled_items, unlabeled_items = split_labeled_unlabeled(
            train_items,
            labeled_ratio=dataset_cfg["labeled_fraction"],
            include_normal_as_unlabeled=dataset_cfg["include_normal_as_unlabeled"],
            seed=config["seed"],
        )
    return {
        "labeled_train": labeled_items,
        "unlabeled_train": unlabeled_items,
        "val": split_items["val"],
        "test": split_items["test"],
    }


def create_dataloaders(config: dict[str, Any]):
    return create_generic_dataloaders(config)
