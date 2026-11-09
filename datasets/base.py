from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from datasets.transforms import (
    build_eval_spatial_transform,
    build_strong_transform,
    build_train_spatial_transform,
    build_weak_transform,
    image_to_tensor,
    mask_to_tensor,
)
from udpc_mt.utils.misc import PROJECT_ROOT


def resolve_dataset_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    if root.is_absolute():
        return root
    return PROJECT_ROOT / root


def normalize_image(
    image: np.ndarray,
    method: str = "minmax",
    clip_percentiles: tuple[float, float] | None = None,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if clip_percentiles is not None:
        low, high = np.percentile(image, clip_percentiles)
        image = np.clip(image, low, high)

    if method == "none":
        return image.astype(np.float32)

    if method == "minmax":
        min_value = float(image.min())
        max_value = float(image.max())
        if max_value - min_value < 1e-8:
            return np.zeros_like(image, dtype=np.float32)
        return ((image - min_value) / (max_value - min_value)).astype(np.float32)

    if method == "zscore":
        mean_value = float(image.mean())
        std_value = float(image.std())
        if std_value < 1e-8:
            return np.zeros_like(image, dtype=np.float32)
        image = (image - mean_value) / std_value
        min_value = float(image.min())
        max_value = float(image.max())
        if max_value - min_value < 1e-8:
            return np.zeros_like(image, dtype=np.float32)
        return ((image - min_value) / (max_value - min_value)).astype(np.float32)

    raise ValueError(f"Unsupported normalization method: {method}")


class SegmentationDatasetAdapter(ABC):
    name = "generic"
    modality = "generic"
    task_type = "binary"
    default_num_classes = 2
    default_input_channels = 1
    dimension_mode = "2d"
    slice_mode = "single_slice"
    class_names = ("background", "foreground")

    def build_dataset_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_type": self.task_type,
            "num_classes": self.default_num_classes,
            "input_channels": self.default_input_channels,
            "dimension_mode": self.dimension_mode,
            "slice_mode": self.slice_mode,
            "class_names": list(self.class_names),
            "modality": self.modality,
        }

    def get_augmentation_profile(self, config: dict[str, Any]) -> str:
        return str(config["dataset"].get("augmentation_profile", self.modality))

    def normalize_loaded_image(self, image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        preprocessing_cfg = config["dataset"].get("preprocessing", {})
        normalization = str(preprocessing_cfg.get("normalization", "minmax"))
        clip_percentiles = preprocessing_cfg.get("clip_percentiles")
        if clip_percentiles is not None:
            clip_percentiles = (float(clip_percentiles[0]), float(clip_percentiles[1]))
        return normalize_image(image, method=normalization, clip_percentiles=clip_percentiles)

    @abstractmethod
    def discover_records(self, dataset_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def load_image(self, record: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def load_mask(
        self,
        record: dict[str, Any],
        image_shape: tuple[int, int] | tuple[int, int, int],
        config: dict[str, Any],
    ) -> np.ndarray:
        raise NotImplementedError


class SegmentationLabeledDataset(Dataset):
    def __init__(
        self,
        items: list[dict[str, Any]],
        adapter: SegmentationDatasetAdapter,
        config: dict[str, Any],
    ) -> None:
        self.items = items
        self.adapter = adapter
        self.config = config
        self.task_type = str(config["dataset"].get("task_type", adapter.task_type))
        self.image_size = int(config["dataset"]["image_size"])
        self.spatial_transform = build_train_spatial_transform(self.image_size)
        self.appearance_transform = build_weak_transform(adapter.get_augmentation_profile(config))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = self.adapter.load_image(item, self.config)
        mask = self.adapter.load_mask(item, image.shape, self.config)
        transformed = self.spatial_transform(image=image, mask=mask)
        image_aug = self.appearance_transform(image=transformed["image"])["image"]
        return {
            "image": image_to_tensor(image_aug),
            "mask": mask_to_tensor(transformed["mask"], task_type=self.task_type),
            "case_id": item["case_id"],
            "group_id": item["group_id"],
            "dataset_name": self.adapter.name,
            "split": item.get("split", "train"),
            "is_labeled": True,
        }


class SegmentationUnlabeledDataset(Dataset):
    def __init__(
        self,
        items: list[dict[str, Any]],
        adapter: SegmentationDatasetAdapter,
        config: dict[str, Any],
    ) -> None:
        self.items = items
        self.adapter = adapter
        self.config = config
        self.image_size = int(config["dataset"]["image_size"])
        self.spatial_transform = build_train_spatial_transform(self.image_size)
        profile = adapter.get_augmentation_profile(config)
        self.weak_transform = build_weak_transform(profile)
        self.strong_transform = build_strong_transform(profile)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = self.adapter.load_image(item, self.config)
        transformed = self.spatial_transform(image=image)
        shared_view = transformed["image"]
        return {
            "teacher_image": image_to_tensor(self.weak_transform(image=shared_view)["image"]),
            "student_image": image_to_tensor(self.strong_transform(image=shared_view)["image"]),
            "case_id": item["case_id"],
            "group_id": item["group_id"],
            "dataset_name": self.adapter.name,
            "split": item.get("split", "train"),
            "is_labeled": False,
            "has_gt": bool(item.get("mask_paths")),
        }


class SegmentationEvalDataset(Dataset):
    def __init__(
        self,
        items: list[dict[str, Any]],
        adapter: SegmentationDatasetAdapter,
        config: dict[str, Any],
    ) -> None:
        self.items = items
        self.adapter = adapter
        self.config = config
        self.task_type = str(config["dataset"].get("task_type", adapter.task_type))
        self.image_size = int(config["dataset"]["image_size"])
        self.spatial_transform = build_eval_spatial_transform(self.image_size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = self.adapter.load_image(item, self.config)
        mask = self.adapter.load_mask(item, image.shape, self.config)
        transformed = self.spatial_transform(image=image, mask=mask)
        return {
            "image": image_to_tensor(transformed["image"]),
            "mask": mask_to_tensor(transformed["mask"], task_type=self.task_type),
            "case_id": item["case_id"],
            "group_id": item["group_id"],
            "dataset_name": self.adapter.name,
            "split": item.get("split", "eval"),
            "image_path": item["image_path"],
        }
