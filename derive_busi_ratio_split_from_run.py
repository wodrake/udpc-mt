#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.busi import BUSIDatasetAdapter
from datasets.semi_split import split_labeled_unlabeled_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive BUSI labeled/unlabeled split files from a saved run split.")
    parser.add_argument("--dataset-root", required=True, help="BUSI processed dataset root.")
    parser.add_argument("--base-splits", required=True, help="Path to an existing run's splits.json.")
    parser.add_argument("--ratio", type=float, required=True, help="Target labeled ratio within the train split.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--output-dir", required=True, help="Directory to write labeled/unlabeled/val/test files.")
    return parser.parse_args()


def build_records(dataset_root: str) -> list[dict]:
    config = {
        "seed": 42,
        "dataset": {
            "name": "busi",
            "root": dataset_root,
            "image_size": 256,
            "subset": "all",
            "task_type": "binary",
            "augmentation_profile": "ultrasound",
            "preprocessing": {"normalization": "minmax"},
        },
        "model": {"num_classes": 1, "in_channels": 1},
    }
    adapter = BUSIDatasetAdapter()
    return adapter.discover_records(Path(dataset_root), config)


def write_case_ids(path: Path, case_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{case_id}\n" for case_id in case_ids), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_splits_path = Path(args.base_splits)
    output_dir = Path(args.output_dir)

    with base_splits_path.open("r", encoding="utf-8") as handle:
        split_payload = json.load(handle)

    records = build_records(args.dataset_root)
    record_by_case = {str(record["case_id"]): record for record in records}

    train_case_ids = [str(case_id) for case_id in split_payload["train"]]
    val_case_ids = [str(case_id) for case_id in split_payload["val"]]
    test_case_ids = [str(case_id) for case_id in split_payload["test"]]

    missing = sorted(case_id for case_id in train_case_ids + val_case_ids + test_case_ids if case_id not in record_by_case)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(f"Missing BUSI case ids from dataset root: {preview}{suffix}")

    train_records = [record_by_case[case_id] for case_id in train_case_ids]
    labeled_records, unlabeled_records = split_labeled_unlabeled_records(
        train_records,
        labeled_ratio=args.ratio,
        seed=args.seed,
        include_groups_as_unlabeled=[],
    )

    labeled_case_ids = sorted({str(record["case_id"]) for record in labeled_records})
    unlabeled_case_ids = sorted({str(record["case_id"]) for record in unlabeled_records})
    val_case_ids = sorted(set(val_case_ids))
    test_case_ids = sorted(set(test_case_ids))

    write_case_ids(output_dir / "labeled.txt", labeled_case_ids)
    write_case_ids(output_dir / "unlabeled.txt", unlabeled_case_ids)
    write_case_ids(output_dir / "val.txt", val_case_ids)
    write_case_ids(output_dir / "test.txt", test_case_ids)
    write_case_ids(output_dir / "train.txt", sorted(set(train_case_ids)))

    summary = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "base_splits": str(base_splits_path.resolve()),
        "ratio": args.ratio,
        "seed": args.seed,
        "train_count": len(set(train_case_ids)),
        "labeled_count": len(labeled_case_ids),
        "unlabeled_count": len(unlabeled_case_ids),
        "val_count": len(val_case_ids),
        "test_count": len(test_case_ids),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
