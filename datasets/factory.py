from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from datasets.base import (
    SegmentationEvalDataset,
    SegmentationLabeledDataset,
    SegmentationUnlabeledDataset,
    resolve_dataset_root,
)
from datasets.semi_split import (
    build_explicit_or_stratified_splits,
    split_labeled_unlabeled_records,
)


def build_dataset_adapter(dataset_name: str):
    normalized = dataset_name.lower()
    if normalized == "busi":
        from datasets.busi import BUSIDatasetAdapter

        return BUSIDatasetAdapter()
    if normalized == "acdc":
        try:
            from datasets.acdc import ACDCDatasetAdapter
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ACDC dataset support requires optional dependency 'nibabel'. "
                "Install nibabel or select a different dataset.name."
            ) from exc
        return ACDCDatasetAdapter()
    if normalized == "bml":
        from datasets.bml import BMLDatasetAdapter

        return BMLDatasetAdapter()
    if normalized in {"tn3k", "tg3k", "tn3k2021"}:
        from datasets.tn3k import TN3KDatasetAdapter

        return TN3KDatasetAdapter()
    raise ValueError(f"Unsupported dataset adapter: {dataset_name}")


def prepare_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    dataset_cfg = dict(config["dataset"])
    dataset_name = str(dataset_cfg.get("name", "busi")).lower()
    adapter = build_dataset_adapter(dataset_name)
    info = adapter.build_dataset_info()

    dataset_cfg.setdefault("name", info["name"])
    dataset_cfg.setdefault("task_type", info["task_type"])
    dataset_cfg.setdefault("num_classes", info["num_classes"])
    dataset_cfg.setdefault("input_channels", info["input_channels"])
    dataset_cfg.setdefault("dimension_mode", info["dimension_mode"])
    dataset_cfg.setdefault("slice_mode", info["slice_mode"])
    dataset_cfg.setdefault("class_names", info["class_names"])
    dataset_cfg.setdefault("augmentation_profile", info["modality"])
    dataset_cfg.setdefault("subset", "all")
    dataset_cfg.setdefault("split_source", "auto")
    dataset_cfg.setdefault("preprocessing", {})

    model_cfg = dict(config.get("model", {}))
    expected_in_channels = int(dataset_cfg["input_channels"])
    expected_output_channels = 1 if dataset_cfg["task_type"] == "binary" else int(dataset_cfg["num_classes"])
    model_cfg.setdefault("in_channels", expected_in_channels)
    model_cfg.setdefault("num_classes", expected_output_channels)
    model_in_channels = int(model_cfg["in_channels"])
    model_num_classes = int(model_cfg["num_classes"])
    if model_in_channels != expected_in_channels:
        raise ValueError(
            f"model.in_channels={model_in_channels} does not match dataset.input_channels={expected_in_channels}"
        )
    if model_num_classes != expected_output_channels:
        raise ValueError(
            f"model.num_classes={model_num_classes} does not match expected output channels={expected_output_channels}"
        )

    if "prototype" in config:
        prototype_cfg = dict(config["prototype"])
        prototype_cfg.setdefault("num_classes", int(dataset_cfg["num_classes"]))
        config["prototype"] = prototype_cfg

    config["dataset"] = dataset_cfg
    config["model"] = model_cfg
    return config


def create_dataloaders(config: dict[str, Any]) -> tuple[dict[str, DataLoader | None], dict[str, list[dict[str, Any]]]]:
    config = prepare_dataset_config(config)
    dataset_cfg = config["dataset"]
    adapter = build_dataset_adapter(dataset_cfg["name"])
    dataset_root = resolve_dataset_root(dataset_cfg["root"])
    records = adapter.discover_records(dataset_root, config)
    if dataset_cfg.get("sample_limit") is not None:
        records = records[: int(dataset_cfg["sample_limit"])]

    split_items = build_explicit_or_stratified_splits(
        records,
        val_fraction=float(dataset_cfg["val_fraction"]),
        test_fraction=float(dataset_cfg["test_fraction"]),
        seed=int(config["seed"]),
        split_source=str(dataset_cfg.get("split_source", "auto")),
    )

    train_items = split_items["train"]
    explicit_label_flags = ["is_labeled" in item for item in train_items]
    if any(explicit_label_flags) and not all(explicit_label_flags):
        raise ValueError(
            "Train split mixes explicit and implicit BUSI label assignments. "
            "Use labeled/unlabeled split files consistently."
        )
    has_explicit_train_labels = bool(train_items) and all(explicit_label_flags)
    if has_explicit_train_labels:
        labeled_items = [item for item in train_items if bool(item["is_labeled"])]
        unlabeled_items = [item for item in train_items if not bool(item["is_labeled"])]
    else:
        include_as_unlabeled = dataset_cfg.get("include_groups_as_unlabeled")
        if include_as_unlabeled is None and dataset_cfg.get("include_normal_as_unlabeled", False):
            include_as_unlabeled = ["normal"]

        labeled_items, unlabeled_items = split_labeled_unlabeled_records(
            train_items,
            labeled_ratio=dataset_cfg["labeled_fraction"],
            seed=int(config["seed"]),
            include_groups_as_unlabeled=include_as_unlabeled or [],
        )
    split_items["labeled_train"] = labeled_items
    split_items["unlabeled_train"] = unlabeled_items

    if not labeled_items:
        raise ValueError("No labeled training samples were created. Check labeled_fraction and split settings.")

    num_workers = int(dataset_cfg["num_workers"])
    batch_size = int(config["train"]["batch_size"])
    labeled_batch_size = min(int(config["train"]["labeled_batch_size"]), len(labeled_items))
    unlabeled_batch_size = max(batch_size - labeled_batch_size, 1)

    labeled_loader = DataLoader(
        SegmentationLabeledDataset(labeled_items, adapter=adapter, config=config),
        batch_size=labeled_batch_size if config["mode"] != "baseline" else min(batch_size, len(labeled_items)),
        shuffle=True,
        drop_last=len(labeled_items) > 1,
        num_workers=num_workers,
    )

    unlabeled_loader = None
    if config["mode"] != "baseline":
        if not unlabeled_items:
            raise ValueError("Semi-supervised mode requires at least one unlabeled sample.")
        unlabeled_loader = DataLoader(
            SegmentationUnlabeledDataset(unlabeled_items, adapter=adapter, config=config),
            batch_size=min(unlabeled_batch_size, len(unlabeled_items)),
            shuffle=True,
            drop_last=len(unlabeled_items) > 1,
            num_workers=num_workers,
        )

    val_items = split_items["val"]
    test_items = split_items["test"]
    val_loader = DataLoader(
        SegmentationEvalDataset(val_items, adapter=adapter, config=config),
        batch_size=min(batch_size, max(len(val_items), 1)),
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        SegmentationEvalDataset(test_items, adapter=adapter, config=config),
        batch_size=min(batch_size, max(len(test_items), 1)),
        shuffle=False,
        num_workers=num_workers,
    )

    return {
        "labeled_train": labeled_loader,
        "unlabeled_train": unlabeled_loader,
        "val": val_loader,
        "test": test_loader,
    }, split_items
