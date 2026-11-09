from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

try:
    import nibabel as nib
except ModuleNotFoundError:  # pragma: no cover - optional test dependency
    nib = None

from datasets import create_dataloaders
from trainers.trainer import run_training


def _make_toy_bml(root: Path, image_size: int = 48) -> None:
    for split in ("train", "val"):
        (root / split / "image").mkdir(parents=True, exist_ok=True)
        (root / split / "mask").mkdir(parents=True, exist_ok=True)

    for split, count in (("train", 6), ("val", 2)):
        for index in range(count):
            case_id = f"lesion_{split}_{index:03d}"
            image = np.linspace(0, 255, image_size * image_size, dtype=np.uint8).reshape(image_size, image_size)
            image = np.roll(image, index, axis=0)
            mask = np.zeros((image_size, image_size), dtype=np.uint8)
            mask[10:28, 12:30] = 255
            Image.fromarray(image).save(root / split / "image" / f"{case_id}.png")
            Image.fromarray(mask).save(root / split / "mask" / f"{case_id}.png")


def _make_toy_acdc(root: Path, image_size: int = 40) -> None:
    if nib is None:
        pytest.skip("nibabel is required for ACDC toy dataset tests")
    root.mkdir(parents=True, exist_ok=True)
    group_cycle = ["NOR", "MINF", "DCM", "HCM"]
    for patient_index in range(4):
        patient_id = f"patient{patient_index + 1:03d}"
        patient_dir = root / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        info = [
            f"Group: {group_cycle[patient_index % len(group_cycle)]}",
            "ED: 1",
            "ES: 2",
        ]
        (patient_dir / "Info.cfg").write_text("\n".join(info), encoding="utf-8")
        for frame_index in (1, 2):
            image = np.zeros((image_size, image_size, 3), dtype=np.float32)
            mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            for slice_index in range(3):
                image[:, :, slice_index] = 0.2 + 0.1 * frame_index + 0.05 * slice_index
                mask[4:18, 4:18, slice_index] = 1
                mask[12:24, 12:24, slice_index] = 2
                mask[20:32, 20:32, slice_index] = 3
            nib.save(
                nib.Nifti1Image(image, affine=np.eye(4)),
                patient_dir / f"{patient_id}_frame{frame_index:02d}.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(mask.astype(np.float32), affine=np.eye(4)),
                patient_dir / f"{patient_id}_frame{frame_index:02d}_gt.nii.gz",
            )


def _make_toy_tn3k(root: Path, image_size: int = 48) -> None:
    image_dir = root / "thyroid-image"
    mask_dir = root / "thyroid-mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    split_payload = {"train": list(range(8)), "val": list(range(8, 10)), "test": []}
    (root / "tg3k-trainval.json").write_text(json.dumps(split_payload), encoding="utf-8")

    for index in range(10):
        case_id = f"{index:04d}"
        image = np.zeros((image_size, image_size), dtype=np.uint8)
        image[:] = 40 + index * 8
        image[8:36, 10:34] = min(255, 120 + index * 10)

        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        mask[14:30, 16:32] = 255

        Image.fromarray(image).save(image_dir / f"{case_id}.jpg", quality=100, subsampling=0)
        Image.fromarray(mask).save(mask_dir / f"{case_id}.jpg", quality=100, subsampling=0)


def test_bml_explicit_split_dataloaders(tmp_path: Path) -> None:
    _make_toy_bml(tmp_path)
    config = {
        "seed": 7,
        "mode": "mean_teacher",
        "dataset": {
            "name": "bml",
            "root": str(tmp_path),
            "image_size": 32,
            "val_fraction": 0.2,
            "test_fraction": 0.1,
            "labeled_fraction": 0.5,
            "num_workers": 0,
            "sample_limit": None,
            "split_source": "auto",
            "task_type": "binary",
            "augmentation_profile": "mri",
            "preprocessing": {"normalization": "minmax"},
        },
        "train": {"batch_size": 4, "labeled_batch_size": 2},
        "model": {"in_channels": 1, "num_classes": 1},
    }
    loaders, split_items = create_dataloaders(config)
    assert len(split_items["val"]) == 2
    assert len(split_items["test"]) == 2
    labeled_batch = next(iter(loaders["labeled_train"]))
    unlabeled_batch = next(iter(loaders["unlabeled_train"]))
    assert labeled_batch["image"].shape[1:] == (1, 32, 32)
    assert labeled_batch["mask"].shape[1:] == (1, 32, 32)
    assert unlabeled_batch["teacher_image"].shape[1:] == (1, 32, 32)


def test_acdc_multiclass_dataloader_shapes(tmp_path: Path) -> None:
    _make_toy_acdc(tmp_path)
    config = {
        "seed": 11,
        "mode": "mean_teacher",
        "dataset": {
            "name": "acdc",
            "root": str(tmp_path),
            "image_size": 32,
            "val_fraction": 0.25,
            "test_fraction": 0.25,
            "labeled_fraction": 0.5,
            "num_workers": 0,
            "sample_limit": None,
            "split_source": "internal",
            "task_type": "multiclass",
            "num_classes": 4,
            "input_channels": 1,
            "augmentation_profile": "mri",
            "preprocessing": {"normalization": "minmax"},
        },
        "train": {"batch_size": 4, "labeled_batch_size": 2},
        "model": {"in_channels": 1, "num_classes": 4},
    }
    loaders, _ = create_dataloaders(config)
    labeled_batch = next(iter(loaders["labeled_train"]))
    unlabeled_batch = next(iter(loaders["unlabeled_train"]))
    assert labeled_batch["image"].shape[1:] == (1, 32, 32)
    assert labeled_batch["mask"].ndim == 3
    assert unlabeled_batch["teacher_image"].shape[1:] == (1, 32, 32)


def test_tn3k_official_split_dataloaders(tmp_path: Path) -> None:
    _make_toy_tn3k(tmp_path)
    config = {
        "seed": 17,
        "mode": "mean_teacher",
        "dataset": {
            "name": "tn3k",
            "root": str(tmp_path),
            "image_size": 32,
            "val_fraction": 0.0,
            "test_fraction": 0.0,
            "labeled_fraction": 0.5,
            "num_workers": 0,
            "sample_limit": None,
            "split_source": "official",
            "task_type": "binary",
            "augmentation_profile": "ultrasound",
            "mask_threshold": 128,
            "preprocessing": {"normalization": "minmax"},
        },
        "train": {"batch_size": 4, "labeled_batch_size": 2},
        "model": {"in_channels": 1, "num_classes": 1},
    }
    loaders, split_items = create_dataloaders(config)
    assert len(split_items["train"]) == 8
    assert len(split_items["val"]) == 2
    assert len(split_items["test"]) == 2
    labeled_batch = next(iter(loaders["labeled_train"]))
    unlabeled_batch = next(iter(loaders["unlabeled_train"]))
    assert labeled_batch["image"].shape[1:] == (1, 32, 32)
    assert labeled_batch["mask"].shape[1:] == (1, 32, 32)
    assert unlabeled_batch["teacher_image"].shape[1:] == (1, 32, 32)


def test_acdc_mean_teacher_smoke_training(tmp_path: Path) -> None:
    dataset_root = tmp_path / "acdc"
    output_root = tmp_path / "runs"
    _make_toy_acdc(dataset_root)
    config = {
        "project_name": "toy_acdc_mt",
        "mode": "mean_teacher",
        "seed": 13,
        "device": "cpu",
        "output_root": str(output_root),
        "dataset": {
            "name": "acdc",
            "root": str(dataset_root),
            "image_size": 32,
            "val_fraction": 0.25,
            "test_fraction": 0.25,
            "labeled_fraction": 0.5,
            "num_workers": 0,
            "sample_limit": 12,
            "split_source": "internal",
            "task_type": "multiclass",
            "num_classes": 4,
            "input_channels": 1,
            "augmentation_profile": "mri",
            "preprocessing": {"normalization": "minmax"},
        },
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "labeled_batch_size": 2,
            "max_steps_per_epoch": 1,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "ema_decay": 0.99,
            "unsupervised_weight": 0.5,
            "save_best_metric": "dice",
        },
        "model": {
            "in_channels": 1,
            "num_classes": 4,
            "base_channels": 16,
            "dropout": 0.1,
        },
        "loss": {
            "bce_weight": 0.5,
            "dice_weight": 0.5,
        },
        "uncertainty": {
            "enabled": False,
            "mc_passes": 1,
            "threshold": 0.35,
        },
    }
    summary = run_training(config, run_dir=output_root / "toy_acdc_mt")
    assert Path(summary["best_checkpoint"]).exists()
    assert Path(summary["last_checkpoint"]).exists()
