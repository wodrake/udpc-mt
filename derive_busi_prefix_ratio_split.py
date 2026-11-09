#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive BUSI labeled/unlabeled split files from the PCCS bridge train order."
    )
    parser.add_argument(
        "--bridge-summary",
        required=True,
        help="Path to PCCS BUSI split_bridge_summary.json containing the ordered train filenames.",
    )
    parser.add_argument(
        "--base-splits",
        required=True,
        help="Path to the original udpc-mt splits.json for val/test case ids.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        required=True,
        help="Target labeled ratio within the shared train split, expressed in [0, 1].",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write labeled/unlabeled/train/val/test split files.",
    )
    return parser.parse_args()


def parse_case_id_from_bridge_name(filename: str) -> str:
    stem = Path(filename).stem
    if "__" not in stem:
        raise ValueError(f"Expected indexed BUSI bridge filename, got: {filename}")
    return stem.split("__", 1)[1]


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bridge_summary_path = Path(args.bridge_summary).resolve()
    base_splits_path = Path(args.base_splits).resolve()
    output_dir = Path(args.output_dir).resolve()

    bridge_summary = json.loads(bridge_summary_path.read_text(encoding="utf-8"))
    base_splits = json.loads(base_splits_path.read_text(encoding="utf-8"))

    train_filenames = list(bridge_summary["files"]["train"])
    train_case_ids = [parse_case_id_from_bridge_name(name) for name in train_filenames]
    train_count = len(train_case_ids)
    labeled_count = max(1, int(round(train_count * float(args.ratio))))
    labeled_count = min(labeled_count, train_count)

    labeled_case_ids = train_case_ids[:labeled_count]
    unlabeled_case_ids = train_case_ids[labeled_count:]
    val_case_ids = [str(case_id) for case_id in base_splits["val"]]
    test_case_ids = [str(case_id) for case_id in base_splits["test"]]

    write_lines(output_dir / "labeled.txt", labeled_case_ids)
    write_lines(output_dir / "unlabeled.txt", unlabeled_case_ids)
    write_lines(output_dir / "train.txt", train_case_ids)
    write_lines(output_dir / "val.txt", val_case_ids)
    write_lines(output_dir / "test.txt", test_case_ids)

    summary = {
        "bridge_summary": str(bridge_summary_path),
        "base_splits": str(base_splits_path),
        "selection_rule": "prefix_of_bridge_train_order",
        "ratio": float(args.ratio),
        "train_count": train_count,
        "labeled_count": len(labeled_case_ids),
        "unlabeled_count": len(unlabeled_case_ids),
        "val_count": len(val_case_ids),
        "test_count": len(test_case_ids),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
