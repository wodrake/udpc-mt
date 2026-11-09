from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from datasets.busi import BUSILabeledDataset, BUSIUnlabeledDataset, create_dataloaders
from datasets.semi_split import (
    count_by_pathology,
    normalize_labeled_ratio,
    read_busi_manifest,
    split_labeled_unlabeled,
    stratified_train_val_test_split,
)


def _make_toy_busi(root: Path, image_size: int = 96, seed: int = 7) -> None:
    rng = random.Random(seed)
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    specs = [("benign", 10), ("malignant", 10), ("normal", 10)]
    for pathology, count in specs:
        for index in range(count):
            case_id = f"{pathology}_{index:03d}"
            image = np.full((image_size, image_size), 80, dtype=np.uint8)
            image += np.random.default_rng(seed + index).integers(0, 30, size=(image_size, image_size), dtype=np.uint8)
            mask = Image.new("L", (image_size, image_size), 0)
            if pathology != "normal":
                draw = ImageDraw.Draw(mask)
                x0 = rng.randint(10, 30)
                y0 = rng.randint(10, 30)
                x1 = rng.randint(50, 80)
                y1 = rng.randint(50, 80)
                draw.ellipse((x0, y0, x1, y1), fill=255)
                image = np.clip(image + (np.array(mask) > 0) * 50, 0, 255).astype(np.uint8)
            Image.fromarray(image).save(image_dir / f"{case_id}.png")
            mask.save(mask_dir / f"{case_id}.png")
            rows.append(
                {
                    "case_id": case_id,
                    "pathology": pathology,
                    "image_path": f"images/{case_id}.png",
                    "mask_path": f"masks/{case_id}.png",
                }
            )

    with (root / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "pathology", "image_path", "mask_path"])
        writer.writeheader()
        writer.writerows(rows)


def _make_raw_style_busi(root: Path, image_size: int = 96) -> None:
    for pathology in ("benign", "malignant", "normal"):
        (root / pathology).mkdir(parents=True, exist_ok=True)
        (root / f"{pathology}_masks").mkdir(parents=True, exist_ok=True)

    sample_specs = [("benign", 1), ("malignant", 2), ("normal", 3)]
    for pathology, index in sample_specs:
        image = np.full((image_size, image_size), 100, dtype=np.uint8)
        mask = Image.new("L", (image_size, image_size), 0)
        if pathology != "normal":
            draw = ImageDraw.Draw(mask)
            draw.ellipse((20, 20, 70, 70), fill=255)
            image = np.clip(image + (np.array(mask) > 0) * 40, 0, 255).astype(np.uint8)

        image_name = f"breast_{index:03d}_0000.png"
        mask_name = f"breast_{index:03d}.png"
        Image.fromarray(image).save(root / pathology / image_name)
        mask.save(root / f"{pathology}_masks" / mask_name)


def _make_custom_metadata_busi(root: Path, image_size: int = 96) -> None:
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    specs = [
        ("clip_a_000", "train", "video_a"),
        ("clip_a_001", "train", "video_a"),
        ("clip_b_000", "train", "video_b"),
        ("clip_b_001", "train", "video_b"),
        ("clip_c_000", "val", "video_c"),
        ("clip_d_000", "test", "video_d"),
    ]
    for index, (case_id, split, group_id) in enumerate(specs):
        image = np.full((image_size, image_size), 60 + index * 10, dtype=np.uint8)
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        mask[20:60, 24:68] = 255
        Image.fromarray(image).save(image_dir / f"{case_id}.png")
        Image.fromarray(mask).save(mask_dir / f"{case_id}.png")
        rows.append(
            {
                "case_id": case_id,
                "pathology": "laa",
                "image_path": f"images/{case_id}.png",
                "mask_path": f"masks/{case_id}.png",
                "split": split,
                "group_id": group_id,
                "stratify_key": "laa",
            }
        )

    with (root / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "pathology", "image_path", "mask_path", "split", "group_id", "stratify_key"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _build_config(dataset_root: Path, labeled_fraction: float = 0.2, mode: str = "mean_teacher") -> dict:
    return {
        "seed": 123,
        "mode": mode,
        "dataset": {
            "root": str(dataset_root),
            "image_size": 64,
            "val_fraction": 0.2,
            "test_fraction": 0.1,
            "labeled_fraction": labeled_fraction,
            "include_normal_as_unlabeled": True,
            "num_workers": 0,
            "sample_limit": None,
        },
        "train": {
            "batch_size": 4,
            "labeled_batch_size": 2,
        },
    }


def test_manifest_and_split_pipeline(tmp_path: Path) -> None:
    _make_toy_busi(tmp_path)
    items = read_busi_manifest(tmp_path)
    assert len(items) == 30
    assert count_by_pathology(items) == {"benign": 10, "malignant": 10, "normal": 10}

    train_items, val_items, test_items = stratified_train_val_test_split(
        items,
        val_fraction=0.2,
        test_fraction=0.1,
        seed=123,
    )
    assert len(train_items) + len(val_items) + len(test_items) == len(items)
    assert set(item["case_id"] for item in train_items).isdisjoint(item["case_id"] for item in val_items)
    assert set(item["case_id"] for item in train_items).isdisjoint(item["case_id"] for item in test_items)

    labeled_items, unlabeled_items = split_labeled_unlabeled(
        train_items,
        labeled_ratio=20,
        include_normal_as_unlabeled=True,
        seed=123,
    )
    labeled_counts = count_by_pathology(labeled_items)
    unlabeled_counts = count_by_pathology(unlabeled_items)
    assert labeled_counts["benign"] >= 1
    assert labeled_counts["malignant"] >= 1
    assert "normal" not in labeled_counts
    assert unlabeled_counts["normal"] > 0
    assert normalize_labeled_ratio(10) == 0.1
    assert normalize_labeled_ratio(0.5) == 0.5


def test_raw_style_busi_manifest_discovery(tmp_path: Path) -> None:
    _make_raw_style_busi(tmp_path)
    items = read_busi_manifest(tmp_path)
    assert len(items) == 3
    assert count_by_pathology(items) == {"benign": 1, "malignant": 1, "normal": 1}
    assert all("mask_paths" in item for item in items)
    assert all(isinstance(item["mask_paths"], list) for item in items)


def test_metadata_manifest_preserves_optional_split_and_group_fields(tmp_path: Path) -> None:
    _make_custom_metadata_busi(tmp_path)
    items = read_busi_manifest(tmp_path)
    assert len(items) == 6
    assert {item["split"] for item in items} == {"train", "val", "test"}
    assert {item["group_id"] for item in items if item["split"] == "train"} == {"video_a", "video_b"}
    assert all(item["stratify_key"] == "laa" for item in items)


def test_dataset_items_and_batch_shapes(tmp_path: Path) -> None:
    _make_toy_busi(tmp_path)
    config = _build_config(tmp_path, labeled_fraction=0.5, mode="mean_teacher")
    loaders, split_items = create_dataloaders(config)

    labeled_dataset = BUSILabeledDataset(split_items["labeled_train"], image_size=64)
    labeled_item = labeled_dataset[0]
    assert labeled_item["image"].shape == (1, 64, 64)
    assert labeled_item["mask"].shape == (1, 64, 64)

    unlabeled_dataset = BUSIUnlabeledDataset(split_items["unlabeled_train"], image_size=64)
    unlabeled_item = unlabeled_dataset[0]
    assert unlabeled_item["teacher_image"].shape == (1, 64, 64)
    assert unlabeled_item["student_image"].shape == (1, 64, 64)
    assert "mask" not in unlabeled_item

    labeled_batch = next(iter(loaders["labeled_train"]))
    unlabeled_batch = next(iter(loaders["unlabeled_train"]))
    assert labeled_batch["image"].shape == (2, 1, 64, 64)
    assert labeled_batch["mask"].shape == (2, 1, 64, 64)
    assert unlabeled_batch["teacher_image"].shape == (2, 1, 64, 64)
    assert unlabeled_batch["student_image"].shape == (2, 1, 64, 64)


def test_multiple_labeled_ratios(tmp_path: Path) -> None:
    _make_toy_busi(tmp_path)
    items = read_busi_manifest(tmp_path)
    train_items, _, _ = stratified_train_val_test_split(items, 0.2, 0.1, seed=123)
    lesion_total = sum(item["pathology"] != "normal" for item in train_items)
    for ratio in (0.1, 0.2, 0.5):
        labeled_items, unlabeled_items = split_labeled_unlabeled(
            train_items,
            labeled_ratio=ratio,
            include_normal_as_unlabeled=True,
            seed=123,
        )
    assert len(labeled_items) > 0
    assert len(labeled_items) < lesion_total
    assert len(unlabeled_items) > 0


def test_official_metadata_split_and_group_level_ssl_split(tmp_path: Path) -> None:
    _make_custom_metadata_busi(tmp_path)
    config = _build_config(tmp_path, labeled_fraction=0.5, mode="mean_teacher")
    config["dataset"]["split_source"] = "official"
    loaders, split_items = create_dataloaders(config)

    assert loaders["labeled_train"] is not None
    assert loaders["unlabeled_train"] is not None
    assert {item["case_id"] for item in split_items["val"]} == {"clip_c_000"}
    assert {item["case_id"] for item in split_items["test"]} == {"clip_d_000"}
    assert {item["group_id"] for item in split_items["labeled_train"]}.isdisjoint(
        item["group_id"] for item in split_items["unlabeled_train"]
    )
    assert len(split_items["labeled_train"]) == 2
    assert len(split_items["unlabeled_train"]) == 2


def test_busi_lesion_only_subset_excludes_normal_cases(tmp_path: Path) -> None:
    _make_toy_busi(tmp_path)
    config = _build_config(tmp_path, labeled_fraction=0.5, mode="mean_teacher")
    config["dataset"]["subset"] = "lesion_only"
    loaders, split_items = create_dataloaders(config)

    assert loaders["labeled_train"] is not None
    assert loaders["unlabeled_train"] is not None
    for split_name in ("labeled_train", "unlabeled_train", "val", "test"):
        assert split_items[split_name]
        assert all(item["pathology"] != "normal" for item in split_items[split_name])


def test_busi_external_split_files_override_auto_ssl_split(tmp_path: Path) -> None:
    _make_toy_busi(tmp_path)
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "labeled.txt").write_text("benign_000.png\nmalignant_000.png\n", encoding="utf-8")
    (split_dir / "unlabeled.txt").write_text("benign_001.png\nmalignant_001.png\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("benign_002.png\nmalignant_002.png\n", encoding="utf-8")

    config = _build_config(tmp_path, labeled_fraction=0.9, mode="mean_teacher")
    config["dataset"]["subset"] = "lesion_only"
    config["dataset"]["split_source"] = "official"
    config["dataset"]["split_files"] = {
        "labeled": str(split_dir / "labeled.txt"),
        "unlabeled": str(split_dir / "unlabeled.txt"),
        "test": str(split_dir / "test.txt"),
    }

    loaders, split_items = create_dataloaders(config)

    assert loaders["labeled_train"] is not None
    assert loaders["unlabeled_train"] is not None
    assert {item["case_id"] for item in split_items["labeled_train"]} == {"benign_000", "malignant_000"}
    assert {item["case_id"] for item in split_items["unlabeled_train"]} == {"benign_001", "malignant_001"}
    assert {item["case_id"] for item in split_items["test"]} == {"benign_002", "malignant_002"}
    assert {item["case_id"] for item in split_items["val"]} == {"benign_002", "malignant_002"}


def test_busi_external_split_files_accept_direct_case_ids(tmp_path: Path) -> None:
    _make_custom_metadata_busi(tmp_path)
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "labeled.txt").write_text("clip_a_000\nclip_a_001\n", encoding="utf-8")
    (split_dir / "unlabeled.txt").write_text("clip_b_000\nclip_b_001\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("clip_c_000\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("clip_d_000\n", encoding="utf-8")

    config = _build_config(tmp_path, labeled_fraction=0.9, mode="mean_teacher")
    config["dataset"]["split_source"] = "official"
    config["dataset"]["split_files"] = {
        "labeled": str(split_dir / "labeled.txt"),
        "unlabeled": str(split_dir / "unlabeled.txt"),
        "val": str(split_dir / "val.txt"),
        "test": str(split_dir / "test.txt"),
    }

    loaders, split_items = create_dataloaders(config)

    assert loaders["labeled_train"] is not None
    assert loaders["unlabeled_train"] is not None
    assert {item["case_id"] for item in split_items["labeled_train"]} == {"clip_a_000", "clip_a_001"}
    assert {item["case_id"] for item in split_items["unlabeled_train"]} == {"clip_b_000", "clip_b_001"}
    assert {item["case_id"] for item in split_items["val"]} == {"clip_c_000"}
    assert {item["case_id"] for item in split_items["test"]} == {"clip_d_000"}
