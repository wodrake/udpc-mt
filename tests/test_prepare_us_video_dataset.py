from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prepare_us_video_dataset import pair_split_files


def _paths(names: list[str], suffix: str) -> list[Path]:
    return [Path(f"{name}.{suffix}") for name in names]


def test_auto_pairing_skips_small_number_of_orphan_annotations() -> None:
    video_files = _paths(["100", "101", "102"], "npy")
    annotation_files = _paths(["100", "101", "102", "999"], "npz")

    pairs = pair_split_files(
        video_files,
        annotation_files,
        pairing_mode="auto",
        dataset_name="LAA2",
        split="train",
    )

    assert [(video_path.stem, annotation_path.stem, mode) for video_path, annotation_path, mode in pairs] == [
        ("100", "100", "by_name_partial"),
        ("101", "101", "by_name_partial"),
        ("102", "102", "by_name_partial"),
    ]


def test_by_name_pairing_remains_strict_when_orphans_exist() -> None:
    video_files = _paths(["100", "101", "102"], "npy")
    annotation_files = _paths(["100", "101", "102", "999"], "npz")

    with pytest.raises(ValueError, match="strictly by filename"):
        pair_split_files(
            video_files,
            annotation_files,
            pairing_mode="by_name",
            dataset_name="LAA2",
            split="train",
        )


def test_auto_pairing_falls_back_to_sorted_when_no_stems_match() -> None:
    video_files = _paths(["video_a", "video_b"], "npy")
    annotation_files = _paths(["anno_a", "anno_b"], "npz")

    pairs = pair_split_files(
        video_files,
        annotation_files,
        pairing_mode="auto",
        dataset_name="LAA2",
        split="train",
    )

    assert [(video_path.stem, annotation_path.stem, mode) for video_path, annotation_path, mode in pairs] == [
        ("video_a", "anno_a", "sorted"),
        ("video_b", "anno_b", "sorted"),
    ]


def test_auto_pairing_rejects_large_partial_name_mismatch() -> None:
    video_files = _paths(["100", "101", "102", "103", "104", "105"], "npy")
    annotation_files = _paths(["100", "201", "202", "203", "204", "205"], "npz")

    with pytest.raises(ValueError, match="real naming mismatch"):
        pair_split_files(
            video_files,
            annotation_files,
            pairing_mode="auto",
            dataset_name="LAA2",
            split="train",
        )
