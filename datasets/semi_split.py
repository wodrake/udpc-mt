from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from udpc_mt.utils.misc import PROJECT_ROOT


PATHOLOGIES = ("benign", "malignant", "normal")


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == "":
        return None
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean metadata value: {value}")


def normalize_labeled_ratio(ratio: float | int) -> float:
    normalized = float(ratio)
    if normalized > 1.0:
        normalized = normalized / 100.0
    if not 0.0 < normalized <= 1.0:
        raise ValueError(f"labeled_ratio must be in (0, 1] or percentage form, got {ratio}")
    return normalized


def count_by_pathology(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(item["pathology"] for item in items))


def _normalize_identifier(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\((\d+)\)", r"_\1", text)
    text = re.sub(r"_0000$", "", text)
    text = re.sub(r"_mask(?:_\d+)?$", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _infer_pathology_from_path(path: Path) -> str | None:
    lowered_parts = [part.lower() for part in path.parts]
    for pathology in PATHOLOGIES:
        if pathology in lowered_parts or any(pathology in part for part in lowered_parts):
            return pathology
    return None


def _is_mask_path(path: Path) -> bool:
    parent_name = path.parent.name.lower()
    stem = path.stem.lower()
    return "mask" in parent_name or "_mask" in stem


def _build_case_id(pathology: str, stem: str) -> str:
    normalized_stem = _normalize_identifier(stem)
    if normalized_stem.startswith(pathology):
        return normalized_stem
    return f"{pathology}_{normalized_stem}"


def _read_manifest_from_metadata(dataset_root: Path) -> list[dict[str, Any]]:
    metadata_path = dataset_root / "metadata.csv"
    with metadata_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        items = []
        for row in reader:
            mask_path = dataset_root / row["mask_path"] if row.get("mask_path") else None
            item = {
                "case_id": row["case_id"],
                "pathology": row["pathology"],
                "image_path": str(dataset_root / row["image_path"]),
                "mask_paths": [str(mask_path)] if mask_path is not None else [],
                "source_layout": "processed",
            }
            if row.get("split"):
                item["split"] = row["split"]
            if row.get("group_id"):
                item["group_id"] = row["group_id"]
            if row.get("stratify_key"):
                item["stratify_key"] = row["stratify_key"]
            is_labeled = _parse_optional_bool(row.get("is_labeled"))
            if is_labeled is not None:
                item["is_labeled"] = is_labeled
            items.append(item)
    return items


def _scan_original_busi_layout(dataset_root: Path) -> list[dict[str, Any]]:
    images: dict[str, dict[str, Any]] = {}
    masks: defaultdict[str, list[str]] = defaultdict(list)
    for png_path in dataset_root.rglob("*.png"):
        pathology = _infer_pathology_from_path(png_path)
        if pathology is None:
            continue
        case_id = _build_case_id(pathology, png_path.stem)
        if _is_mask_path(png_path):
            masks[case_id].append(str(png_path))
        else:
            images[case_id] = {
                "case_id": case_id,
                "pathology": pathology,
                "image_path": str(png_path),
                "source_layout": "original",
            }
    items = []
    for case_id, image_item in sorted(images.items()):
        items.append({**image_item, "mask_paths": sorted(masks.get(case_id, []))})
    return items


def read_busi_manifest(dataset_root: str | Path) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    metadata_path = root / "metadata.csv"
    if metadata_path.exists():
        items = _read_manifest_from_metadata(root)
    else:
        items = _scan_original_busi_layout(root)
    if not items:
        raise FileNotFoundError(
            f"Could not discover BUSI samples under {root}. Expected metadata.csv or the original BUSI folder layout."
        )
    return items


def filter_busi_subset(items: list[dict[str, Any]], subset: str = "all") -> list[dict[str, Any]]:
    normalized_subset = str(subset).lower()
    if normalized_subset in {"all", "full"}:
        return items
    if normalized_subset == "lesion_only":
        return [item for item in items if str(item["pathology"]).lower() != "normal"]
    raise ValueError(f"Unsupported BUSI subset: {subset}")


def _resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _parse_busi_case_id_from_reference(reference: str) -> str:
    path = Path(reference.strip())
    pathology = _infer_pathology_from_path(path)
    if pathology is None:
        stem_or_name = path.stem if path.suffix else path.name
        case_id = _normalize_identifier(stem_or_name)
        if not case_id:
            raise ValueError(f"Could not parse case_id from split entry: {reference}")
        return case_id
    return _build_case_id(pathology, path.stem)


def _normalize_busi_split_key(split_name: str) -> tuple[str, bool | None]:
    normalized = str(split_name).strip().lower()
    if normalized in {"labeled", "labeled_train"}:
        return "train", True
    if normalized in {"unlabeled", "unlabeled_train"}:
        return "train", False
    if normalized == "train":
        return "train", None
    if normalized in {"val", "valid", "validation", "dev"}:
        return "val", None
    if normalized == "test":
        return "test", None
    raise ValueError(f"Unsupported BUSI split file key: {split_name}")


def load_busi_split_file_assignments(
    split_files: dict[str, str | Path] | None,
) -> dict[str, dict[str, Any]]:
    if not split_files:
        return {}

    assignments: dict[str, dict[str, Any]] = {}
    for split_name, file_path in split_files.items():
        if file_path in (None, ""):
            continue
        target_split, is_labeled = _normalize_busi_split_key(split_name)
        resolved_path = _resolve_config_path(file_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"BUSI split file not found: {resolved_path}")
        with resolved_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                entry = raw_line.strip()
                if not entry or entry.startswith("#"):
                    continue
                case_id = _parse_busi_case_id_from_reference(entry)
                assignment = {"split": target_split}
                if is_labeled is not None:
                    assignment["is_labeled"] = is_labeled
                existing = assignments.get(case_id)
                if existing is not None and existing != assignment:
                    raise ValueError(
                        f"Conflicting BUSI split assignment for {case_id}: {existing} vs {assignment} "
                        f"(from {resolved_path}:{line_number})"
                    )
                assignments[case_id] = assignment

    if not assignments:
        raise ValueError("No BUSI split assignments were loaded from dataset.split_files.")
    return assignments


def apply_busi_split_assignments(
    items: list[dict[str, Any]],
    assignments: dict[str, dict[str, Any]],
    *,
    drop_unassigned: bool = True,
) -> list[dict[str, Any]]:
    if not assignments:
        return items

    item_by_case_id = {str(item["case_id"]): item for item in items}
    missing_case_ids = sorted(case_id for case_id in assignments if case_id not in item_by_case_id)
    if missing_case_ids:
        preview = ", ".join(missing_case_ids[:5])
        suffix = " ..." if len(missing_case_ids) > 5 else ""
        raise ValueError(
            f"BUSI split files referenced {len(missing_case_ids)} missing cases: {preview}{suffix}"
        )

    assigned_items: list[dict[str, Any]] = []
    for item in items:
        case_id = str(item["case_id"])
        assignment = assignments.get(case_id)
        if assignment is None:
            if not drop_unassigned:
                assigned_items.append(item)
            continue
        assigned_items.append({**item, **assignment})
    return assigned_items


def stratified_train_val_test_split(
    items: list[dict[str, Any]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be >= 0 and sum to less than 1")

    rng = random.Random(seed)
    train_items: list[dict[str, Any]] = []
    val_items: list[dict[str, Any]] = []
    test_items: list[dict[str, Any]] = []
    by_class: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_class[item["pathology"]].append(item)

    for class_items in by_class.values():
        class_items = class_items[:]
        rng.shuffle(class_items)
        total = len(class_items)
        n_test = int(round(total * test_fraction))
        n_val = int(round(total * val_fraction))
        while total - n_test - n_val < 1 and (n_test > 0 or n_val > 0):
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
        test_items.extend(class_items[:n_test])
        val_items.extend(class_items[n_test : n_test + n_val])
        train_items.extend(class_items[n_test + n_val :])

    return train_items, val_items, test_items


def split_labeled_unlabeled(
    train_items: list[dict[str, Any]],
    labeled_ratio: float | int,
    include_normal_as_unlabeled: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_ratio = normalize_labeled_ratio(labeled_ratio)
    rng = random.Random(seed)
    labeled_items: list[dict[str, Any]] = []
    unlabeled_items: list[dict[str, Any]] = []
    lesion_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in train_items:
        if item["pathology"] == "normal":
            if include_normal_as_unlabeled:
                unlabeled_items.append(item)
            else:
                labeled_items.append(item)
            continue
        lesion_groups[item["pathology"]].append(item)

    for class_items in lesion_groups.values():
        class_items = class_items[:]
        rng.shuffle(class_items)
        n_labeled = max(1, int(round(len(class_items) * normalized_ratio)))
        labeled_items.extend(class_items[:n_labeled])
        unlabeled_items.extend(class_items[n_labeled:])

    return labeled_items, unlabeled_items


def stratified_group_train_val_test_split(
    records: list[dict[str, Any]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be >= 0 and sum to less than 1")

    rng = random.Random(seed)
    group_to_records: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    group_to_stratum: dict[str, str] = {}
    for record in records:
        group_id = str(record["group_id"])
        group_to_records[group_id].append(record)
        group_to_stratum[group_id] = str(record.get("stratify_key", "all"))

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []
    by_stratum: defaultdict[str, list[str]] = defaultdict(list)
    for group_id, stratum in group_to_stratum.items():
        by_stratum[stratum].append(group_id)

    for group_ids in by_stratum.values():
        group_ids = group_ids[:]
        rng.shuffle(group_ids)
        total = len(group_ids)
        n_test = int(round(total * test_fraction))
        n_val = int(round(total * val_fraction))
        while total - n_test - n_val < 1 and (n_test > 0 or n_val > 0):
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
        test_groups = set(group_ids[:n_test])
        val_groups = set(group_ids[n_test : n_test + n_val])
        train_groups = set(group_ids[n_test + n_val :])
        for group_id in train_groups:
            train_records.extend(group_to_records[group_id])
        for group_id in val_groups:
            val_records.extend(group_to_records[group_id])
        for group_id in test_groups:
            test_records.extend(group_to_records[group_id])

    needs_global_fallback = (
        (val_fraction > 0 and not val_records) or (test_fraction > 0 and not test_records)
    ) and len(group_to_records) >= 3
    if needs_global_fallback:
        all_group_ids = list(group_to_records.keys())
        rng.shuffle(all_group_ids)
        total = len(all_group_ids)
        n_test = int(round(total * test_fraction))
        n_val = int(round(total * val_fraction))
        if test_fraction > 0 and n_test == 0:
            n_test = 1
        if val_fraction > 0 and n_val == 0:
            n_val = 1
        while total - n_test - n_val < 1 and (n_test > 0 or n_val > 0):
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1

        test_groups = set(all_group_ids[:n_test])
        val_groups = set(all_group_ids[n_test : n_test + n_val])
        train_groups = set(all_group_ids[n_test + n_val :])

        train_records = []
        val_records = []
        test_records = []
        for group_id in train_groups:
            train_records.extend(group_to_records[group_id])
        for group_id in val_groups:
            val_records.extend(group_to_records[group_id])
        for group_id in test_groups:
            test_records.extend(group_to_records[group_id])

    return train_records, val_records, test_records


def split_labeled_unlabeled_records(
    train_records: list[dict[str, Any]],
    labeled_ratio: float | int,
    seed: int,
    include_groups_as_unlabeled: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_ratio = normalize_labeled_ratio(labeled_ratio)
    rng = random.Random(seed)
    include_groups_as_unlabeled = {str(name).lower() for name in include_groups_as_unlabeled or []}

    group_to_records: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    group_to_stratum: dict[str, str] = {}
    for record in train_records:
        group_id = str(record["group_id"])
        group_to_records[group_id].append(record)
        group_to_stratum[group_id] = str(record.get("stratify_key", "all")).lower()

    labeled_records: list[dict[str, Any]] = []
    unlabeled_records: list[dict[str, Any]] = []
    stratum_groups: defaultdict[str, list[str]] = defaultdict(list)
    forced_unlabeled_groups: list[str] = []
    for group_id, stratum in group_to_stratum.items():
        if stratum in include_groups_as_unlabeled:
            forced_unlabeled_groups.append(group_id)
        else:
            stratum_groups[stratum].append(group_id)

    for group_id in forced_unlabeled_groups:
        unlabeled_records.extend(group_to_records[group_id])

    for group_ids in stratum_groups.values():
        group_ids = group_ids[:]
        rng.shuffle(group_ids)
        n_labeled = max(1, int(round(len(group_ids) * normalized_ratio)))
        labeled_group_ids = set(group_ids[:n_labeled])
        for group_id in group_ids:
            if group_id in labeled_group_ids:
                labeled_records.extend(group_to_records[group_id])
            else:
                unlabeled_records.extend(group_to_records[group_id])

    candidate_group_ids = [
        group_id for group_id, stratum in group_to_stratum.items() if stratum not in include_groups_as_unlabeled
    ]
    needs_global_fallback = normalized_ratio < 1.0 and not unlabeled_records and len(candidate_group_ids) >= 2
    if needs_global_fallback:
        shuffled_group_ids = candidate_group_ids[:]
        rng.shuffle(shuffled_group_ids)
        n_labeled = max(1, int(round(len(shuffled_group_ids) * normalized_ratio)))
        n_labeled = min(n_labeled, len(shuffled_group_ids) - 1)
        labeled_group_ids = set(shuffled_group_ids[:n_labeled])

        labeled_records = []
        unlabeled_records = []
        for group_id in forced_unlabeled_groups:
            unlabeled_records.extend(group_to_records[group_id])
        for group_id in shuffled_group_ids:
            if group_id in labeled_group_ids:
                labeled_records.extend(group_to_records[group_id])
            else:
                unlabeled_records.extend(group_to_records[group_id])

    return labeled_records, unlabeled_records


def build_explicit_or_stratified_splits(
    records: list[dict[str, Any]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
    split_source: str = "auto",
) -> dict[str, list[dict[str, Any]]]:
    normalized_split_source = split_source.lower()
    explicit_split_present = any(record.get("split") for record in records)
    if normalized_split_source == "official" or (normalized_split_source == "auto" and explicit_split_present):
        split_items = {"train": [], "val": [], "test": []}
        for record in records:
            split = str(record.get("split", "train")).lower()
            if split not in split_items:
                continue
            split_items[split].append(record)
        if split_items["train"] and split_items["val"] and not split_items["test"]:
            split_items["test"] = list(split_items["val"])
        if split_items["train"] and split_items["test"] and not split_items["val"]:
            split_items["val"] = list(split_items["test"])
        if not split_items["train"] or not split_items["val"] or not split_items["test"]:
            raise ValueError(
                "Official split mode requires discoverable train/val/test records, or at least train/val "
                "so val can be reused as test."
            )
        return split_items

    train_records, val_records, test_records = stratified_group_train_val_test_split(
        records,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    return {"train": train_records, "val": val_records, "test": test_records}
