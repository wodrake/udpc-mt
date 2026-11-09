from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import torch


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32)
    if image.ndim == 2:
        return torch.from_numpy(image).unsqueeze(0)
    if image.ndim == 3:
        return torch.from_numpy(np.moveaxis(image, -1, 0))
    raise ValueError(f"Unsupported image shape for tensor conversion: {image.shape}")


def mask_to_tensor(mask: np.ndarray, task_type: str = "binary") -> torch.Tensor:
    if task_type == "binary":
        return torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
    if task_type == "multiclass":
        return torch.from_numpy(mask.astype(np.int64))
    raise ValueError(f"Unsupported task_type for mask conversion: {task_type}")


def build_train_spatial_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.15),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.92, 1.08),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
        ]
    )


def build_eval_spatial_transform(image_size: int) -> A.Compose:
    return A.Compose([A.Resize(image_size, image_size)])


def build_weak_ultrasound_transform() -> A.Compose:
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.08,
                contrast_limit=0.08,
                p=0.25,
            ),
            A.GaussNoise(p=0.1),
        ]
    )


def build_strong_ultrasound_transform() -> A.Compose:
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.18,
                contrast_limit=0.18,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.25,
            ),
            A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.2),
            A.GaussNoise(p=0.3),
        ]
    )


def build_weak_mri_transform() -> A.Compose:
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.05,
                contrast_limit=0.08,
                p=0.2,
            ),
            A.GaussNoise(p=0.08),
        ]
    )


def build_strong_mri_transform() -> A.Compose:
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.12,
                contrast_limit=0.15,
                p=0.4,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.15,
            ),
            A.GaussNoise(p=0.15),
        ]
    )


def build_weak_transform(profile: str) -> A.Compose:
    normalized = profile.lower()
    if normalized == "ultrasound":
        return build_weak_ultrasound_transform()
    if normalized == "mri":
        return build_weak_mri_transform()
    raise ValueError(f"Unsupported augmentation profile: {profile}")


def build_strong_transform(profile: str) -> A.Compose:
    normalized = profile.lower()
    if normalized == "ultrasound":
        return build_strong_ultrasound_transform()
    if normalized == "mri":
        return build_strong_mri_transform()
    raise ValueError(f"Unsupported augmentation profile: {profile}")
