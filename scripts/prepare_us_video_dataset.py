from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from udpc_mt.utils.misc import PROJECT_ROOT, ensure_dir, json_dump


DEFAULT_LABELED_RATIOS = (0.1, 0.2)
SUPPORTED_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class VideoAnnotationPair:
    dataset_name: str
    split: str
    video_path: Path
    annotation_path: Path
    pairing_mode_used: str

    @property
    def video_id(self) -> str:
        return self.video_path.stem


def slugify(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return normalized.strip("_")


def normalize_ratio(value: float | int) -> float:
    ratio = float(value)
    if ratio > 1.0:
        ratio = ratio / 100.0
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"labeled ratio must be in (0, 1] or percentage form, got {value}")
    return ratio


def ratio_tag(ratio: float) -> str:
    percentage = int(round(ratio * 100))
    return f"ratio_{percentage:02d}"


def dataset_has_expected_layout(dataset_root: Path) -> bool:
    return (dataset_root / "videos").exists() and (dataset_root / "annotations").exists()


def discover_dataset_names(input_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    discovered = [
        child.name
        for child in sorted(input_root.iterdir())
        if child.is_dir() and dataset_has_expected_layout(child)
    ]
    if not discovered:
        raise FileNotFoundError(f"No dataset folders with videos/ and annotations/ were found under {input_root}")
    return discovered


def pair_split_files(
    video_files: list[Path],
    annotation_files: list[Path],
    pairing_mode: str,
    dataset_name: str,
    split: str,
) -> list[tuple[Path, Path, str]]:
    if not video_files and not annotation_files:
        return []

    video_by_stem = {path.stem: path for path in video_files}
    annotation_by_stem = {path.stem: path for path in annotation_files}
    shared_stems = sorted(set(video_by_stem) & set(annotation_by_stem))
    missing_annotations = sorted(set(video_by_stem) - set(annotation_by_stem))
    missing_videos = sorted(set(annotation_by_stem) - set(video_by_stem))
    min_count = min(len(video_by_stem), len(annotation_by_stem))

    if pairing_mode in {"auto", "by_name"} and shared_stems:
        if not missing_annotations and not missing_videos:
            return [(video_by_stem[stem], annotation_by_stem[stem], "by_name") for stem in shared_stems]
        if pairing_mode == "by_name":
            raise ValueError(
                f"{dataset_name}/{split} could not be paired strictly by filename. "
                f"Missing annotations for {missing_annotations[:5]} and missing videos for {missing_videos[:5]}."
            )

        name_match_coverage = len(shared_stems) / max(min_count, 1)
        small_mismatch = len(missing_annotations) + len(missing_videos) <= 4
        if name_match_coverage >= 0.95 or small_mismatch:
            print(
                f"[prepare_us_video_dataset] warning: {dataset_name}/{split} matched {len(shared_stems)} pairs by "
                f"name and skipped {len(missing_annotations)} videos without annotations plus "
                f"{len(missing_videos)} annotations without videos. "
                f"Examples: missing_annotations={missing_annotations[:3]}, missing_videos={missing_videos[:3]}",
                file=sys.stderr,
            )
            return [(video_by_stem[stem], annotation_by_stem[stem], "by_name_partial") for stem in shared_stems]
        raise ValueError(
            f"{dataset_name}/{split} only matched {len(shared_stems)} of at least {min_count} files by stem. "
            "This looks like a real naming mismatch, not a few orphan files. "
            "Rename files to a consistent convention or rerun with --pairing-mode sorted if directory order is known "
            "to be aligned."
        )

    if pairing_mode == "by_name":
        raise ValueError(
            f"{dataset_name}/{split} could not be paired by filename. "
            f"Missing annotations for {missing_annotations[:5]} and missing videos for {missing_videos[:5]}."
        )

    if pairing_mode not in {"auto", "sorted"}:
        raise ValueError(f"Unsupported pairing mode: {pairing_mode}")

    if len(video_files) != len(annotation_files):
        raise ValueError(
            f"{dataset_name}/{split} has {len(video_files)} videos but {len(annotation_files)} annotations. "
            "Since no reliable filename matching was found, sorted pairing would be unsafe. "
            "Please remove orphan files or rename them consistently."
        )

    if pairing_mode == "auto":
        print(
            f"[prepare_us_video_dataset] warning: {dataset_name}/{split} fell back to sorted pairing because file stems "
            "did not match.",
            file=sys.stderr,
        )
    return [(video_path, annotation_path, "sorted") for video_path, annotation_path in zip(video_files, annotation_files)]


def discover_pairs(dataset_root: Path, dataset_name: str, pairing_mode: str) -> list[VideoAnnotationPair]:
    pairs: list[VideoAnnotationPair] = []
    for split in SUPPORTED_SPLITS:
        video_files = sorted((dataset_root / "videos" / split).glob("*.npy"))
        annotation_files = sorted((dataset_root / "annotations" / split).glob("*.npz"))
        for video_path, annotation_path, mode_used in pair_split_files(
            video_files,
            annotation_files,
            pairing_mode=pairing_mode,
            dataset_name=dataset_name,
            split=split,
        ):
            pairs.append(
                VideoAnnotationPair(
                    dataset_name=dataset_name,
                    split=split,
                    video_path=video_path,
                    annotation_path=annotation_path,
                    pairing_mode_used=mode_used,
                )
            )
    if not pairs:
        raise FileNotFoundError(f"No .npy/.npz pairs were discovered under {dataset_root}")
    return pairs


def load_video_array(video_path: Path) -> np.ndarray:
    array = np.load(video_path, mmap_mode="r")
    if array.ndim not in {3, 4}:
        raise ValueError(f"Unsupported video tensor shape for {video_path}: {array.shape}")
    return array


def video_num_frames(array: np.ndarray) -> int:
    if array.ndim == 3:
        return int(array.shape[0])
    if array.shape[0] in {1, 3, 4} and array.shape[1] > 4:
        return int(array.shape[1])
    if array.shape[-1] in {1, 3, 4}:
        return int(array.shape[0])
    raise ValueError(f"Unsupported 4D video tensor shape: {array.shape}")


def extract_frame(array: np.ndarray, frame_index: int) -> np.ndarray:
    if array.ndim == 3:
        return np.asarray(array[frame_index])
    if array.shape[0] in {1, 3, 4} and array.shape[1] > 4:
        return np.moveaxis(np.asarray(array[:, frame_index, :, :]), 0, -1)
    if array.shape[-1] in {1, 3, 4}:
        return np.asarray(array[frame_index])
    raise ValueError(f"Unsupported 4D video tensor shape: {array.shape}")


def to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return np.asarray(array)
    clipped = np.clip(np.asarray(array), 0, 255)
    return clipped.astype(np.uint8)


def convert_frame_to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return to_uint8(frame)
    if frame.ndim != 3:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")
    frame = to_uint8(frame)
    if frame.shape[-1] == 1:
        return frame[..., 0]
    if frame.shape[-1] in {3, 4}:
        return np.array(Image.fromarray(frame).convert("L"), dtype=np.uint8)
    return np.clip(np.mean(frame.astype(np.float32), axis=-1), 0, 255).astype(np.uint8)


def rasterize_polygon(points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Unsupported polygon shape: {points.shape}")
    canvas = Image.new("L", (image_shape[1], image_shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(int(x), int(y)) for x, y in points], fill=255)
    return np.array(canvas, dtype=np.uint8)


def load_annotation_payload(annotation_path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    payload = np.load(annotation_path, allow_pickle=True)

    mask_dict: dict[int, np.ndarray] = {}
    polygon_dict: dict[int, np.ndarray] = {}
    if "fnum_mask" in payload.files:
        raw_masks = payload["fnum_mask"].item()
        mask_dict = {int(frame_index): np.asarray(mask) for frame_index, mask in raw_masks.items()}
    elif "LAA" in payload.files:
        raw_masks = payload["LAA"].item()
        mask_dict = {int(frame_index): np.asarray(mask) for frame_index, mask in raw_masks.items()}

    if "LAAD" in payload.files:
        raw_polygons = payload["LAAD"].item()
        polygon_dict = {int(frame_index): np.asarray(polygon) for frame_index, polygon in raw_polygons.items()}
    return mask_dict, polygon_dict


def resolve_frame_index_base(frame_indices: list[int], num_frames: int, configured_base: str) -> int:
    if configured_base in {"0", "1"}:
        return int(configured_base)
    if not frame_indices:
        return 0
    if 0 in frame_indices:
        return 0
    if max(frame_indices) >= num_frames and min(frame_indices) >= 1:
        return 1
    return 0


def normalize_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape != image_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {image_shape}")
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def collect_context_indices(annotated_indices: list[int], num_frames: int, radius: int) -> list[int]:
    if radius <= 0:
        return []
    annotated = set(annotated_indices)
    context = set()
    for frame_index in annotated_indices:
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            candidate = frame_index + offset
            if 0 <= candidate < num_frames and candidate not in annotated:
                context.add(candidate)
    return sorted(context)


def build_case_id(dataset_slug: str, split: str, video_id: str, frame_index: int) -> str:
    return f"{dataset_slug}_{split}_{slugify(video_id)}_f{frame_index:04d}"


def build_group_id(dataset_slug: str, split: str, video_id: str) -> str:
    return f"{dataset_slug}_{split}_{slugify(video_id)}"


def write_list(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def generate_explicit_ssl_splits(
    split_to_case_ids: dict[str, list[str]],
    train_group_to_cases: dict[str, list[str]],
    split_root: Path,
    labeled_ratios: list[float],
    seed: int,
) -> None:
    ensure_dir(split_root)
    for split_name, case_ids in split_to_case_ids.items():
        write_list(split_root / f"{split_name}.txt", case_ids)

    train_group_ids = sorted(train_group_to_cases)
    train_case_ids = split_to_case_ids.get("train", [])
    for ratio in labeled_ratios:
        ratio_dir = ensure_dir(split_root / ratio_tag(ratio))
        write_list(ratio_dir / "train.txt", train_case_ids)
        write_list(ratio_dir / "val.txt", split_to_case_ids.get("val", []))
        write_list(ratio_dir / "test.txt", split_to_case_ids.get("test", []))

        shuffled_group_ids = train_group_ids[:]
        rng = random.Random(seed + int(round(ratio * 1000)))
        rng.shuffle(shuffled_group_ids)
        n_groups = len(shuffled_group_ids)
        if n_groups == 0:
            labeled_group_ids: set[str] = set()
        else:
            n_labeled = max(1, int(round(n_groups * ratio)))
            if ratio < 1.0 and n_groups > 1:
                n_labeled = min(n_labeled, n_groups - 1)
            labeled_group_ids = set(shuffled_group_ids[:n_labeled])

        labeled_case_ids: list[str] = []
        unlabeled_case_ids: list[str] = []
        for group_id in train_group_ids:
            target = labeled_case_ids if group_id in labeled_group_ids else unlabeled_case_ids
            target.extend(train_group_to_cases[group_id])

        write_list(ratio_dir / "labeled.txt", labeled_case_ids)
        write_list(ratio_dir / "unlabeled.txt", unlabeled_case_ids)
        if ratio < 1.0 and train_case_ids and not unlabeled_case_ids:
            print(
                f"[prepare_us_video_dataset] warning: ratio {ratio:.2f} for {ratio_dir.parent.parent.name} produced "
                "no unlabeled train samples. This usually means the current train split has fewer than two videos.",
                file=sys.stderr,
            )


def process_dataset(
    dataset_name: str,
    dataset_root: Path,
    output_dir: Path,
    pairing_mode: str,
    frame_index_base: str,
    train_context_radius: int,
    labeled_ratios: list[float],
    seed: int,
    limit_videos: int | None,
) -> dict[str, Any]:
    dataset_slug = slugify(dataset_name)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    image_dir = ensure_dir(output_dir / "images")
    mask_dir = ensure_dir(output_dir / "masks")
    split_dir = ensure_dir(output_dir / "splits")

    pairs = discover_pairs(dataset_root, dataset_name=dataset_name, pairing_mode=pairing_mode)
    if limit_videos is not None:
        pairs = pairs[:limit_videos]

    rows: list[dict[str, Any]] = []
    split_to_case_ids: dict[str, list[str]] = defaultdict(list)
    train_group_to_cases: dict[str, list[str]] = defaultdict(list)
    split_summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "videos": 0,
            "annotated_frames": 0,
            "context_frames": 0,
            "exported_samples": 0,
            "pairing_modes": [],
        }
    )

    for pair in pairs:
        video_array = load_video_array(pair.video_path)
        num_frames = video_num_frames(video_array)
        first_frame = convert_frame_to_grayscale(extract_frame(video_array, 0))
        image_shape = tuple(first_frame.shape)

        raw_masks, raw_polygons = load_annotation_payload(pair.annotation_path)
        raw_frame_indices = sorted(set(raw_masks) | set(raw_polygons))
        if not raw_frame_indices:
            raise ValueError(f"No frame annotations were found in {pair.annotation_path}")

        index_base = resolve_frame_index_base(raw_frame_indices, num_frames=num_frames, configured_base=frame_index_base)
        masks_by_frame: dict[int, np.ndarray] = {}
        for raw_index in raw_frame_indices:
            normalized_index = raw_index - index_base
            if not 0 <= normalized_index < num_frames:
                raise ValueError(
                    f"Resolved frame index {normalized_index} is out of bounds for {pair.annotation_path} "
                    f"with {num_frames} frames."
                )
            mask = raw_masks.get(raw_index)
            if mask is not None:
                masks_by_frame[normalized_index] = normalize_mask(mask, image_shape)
                continue
            polygon = raw_polygons.get(raw_index)
            if polygon is None:
                raise ValueError(f"Frame {raw_index} in {pair.annotation_path} has neither mask nor polygon data.")
            masks_by_frame[normalized_index] = rasterize_polygon(polygon, image_shape=image_shape)

        annotated_indices = sorted(masks_by_frame)
        context_indices = collect_context_indices(
            annotated_indices,
            num_frames=num_frames,
            radius=train_context_radius if pair.split == "train" else 0,
        )
        group_id = build_group_id(dataset_slug, pair.split, pair.video_id)

        split_info = split_summary[pair.split]
        split_info["videos"] += 1
        split_info["annotated_frames"] += len(annotated_indices)
        split_info["context_frames"] += len(context_indices)
        split_info["pairing_modes"].append(pair.pairing_mode_used)

        for frame_index in annotated_indices + context_indices:
            case_id = build_case_id(dataset_slug, pair.split, pair.video_id, frame_index)
            frame = convert_frame_to_grayscale(extract_frame(video_array, frame_index))
            image_output = image_dir / f"{case_id}.png"
            Image.fromarray(frame).save(image_output)

            mask_output: Path | None = None
            if frame_index in masks_by_frame:
                mask_output = mask_dir / f"{case_id}.png"
                Image.fromarray(masks_by_frame[frame_index]).save(mask_output)

            rows.append(
                {
                    "case_id": case_id,
                    "pathology": "laa",
                    "image_path": str(image_output.relative_to(output_dir)),
                    "mask_path": str(mask_output.relative_to(output_dir)) if mask_output is not None else "",
                    "split": pair.split,
                    "group_id": group_id,
                    "stratify_key": dataset_slug,
                    "source_dataset": dataset_name,
                    "source_video": pair.video_id,
                    "frame_index": frame_index,
                    "has_mask": int(mask_output is not None),
                    "is_context_frame": int(frame_index in context_indices),
                }
            )
            split_to_case_ids[pair.split].append(case_id)
            if pair.split == "train":
                train_group_to_cases[group_id].append(case_id)
            split_info["exported_samples"] += 1

    metadata_path = output_dir / "metadata.csv"
    fieldnames = [
        "case_id",
        "pathology",
        "image_path",
        "mask_path",
        "split",
        "group_id",
        "stratify_key",
        "source_dataset",
        "source_video",
        "frame_index",
        "has_mask",
        "is_context_frame",
    ]
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for split_name, case_ids in split_to_case_ids.items():
        split_to_case_ids[split_name] = sorted(case_ids)
    for group_id, case_ids in train_group_to_cases.items():
        train_group_to_cases[group_id] = sorted(case_ids)

    generate_explicit_ssl_splits(
        split_to_case_ids=split_to_case_ids,
        train_group_to_cases=train_group_to_cases,
        split_root=split_dir,
        labeled_ratios=labeled_ratios,
        seed=seed,
    )

    summary = {
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "pairing_mode": pairing_mode,
        "frame_index_base": frame_index_base,
        "train_context_radius": train_context_radius,
        "labeled_ratios": labeled_ratios,
        "splits": {
            split_name: {
                **details,
                "pairing_modes": sorted(set(details["pairing_modes"])),
                "case_ids": len(split_to_case_ids.get(split_name, [])),
            }
            for split_name, details in split_summary.items()
        },
    }
    json_dump(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ultrasound video + sparse frame annotations into a BUSI-like image dataset."
    )
    parser.add_argument(
        "--input-root",
        default=str(PROJECT_ROOT / "US_Datasets"),
        help="Root directory containing one or more datasets such as LAA1/ and LAA2/.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data" / "processed" / "us_busi_like"),
        help="Base directory for the converted datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to convert. Defaults to every subdirectory that has videos/ and annotations/.",
    )
    parser.add_argument(
        "--pairing-mode",
        choices=("auto", "by_name", "sorted"),
        default="auto",
        help="How to pair videos and annotations within each split.",
    )
    parser.add_argument(
        "--frame-index-base",
        choices=("auto", "0", "1"),
        default="auto",
        help="Interpret annotation frame numbers as zero-based, one-based, or auto-detect.",
    )
    parser.add_argument(
        "--train-context-radius",
        type=int,
        default=0,
        help="Optional number of neighboring train frames to export around each annotated frame as unlabeled context.",
    )
    parser.add_argument(
        "--labeled-ratios",
        nargs="*",
        type=float,
        default=list(DEFAULT_LABELED_RATIOS),
        help="Generate explicit labeled/unlabeled split files for these ratios.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for labeled/unlabeled video grouping.")
    parser.add_argument("--limit-videos", type=int, default=None, help="Optional video limit for local debugging.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove and recreate an existing converted output directory.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    datasets = discover_dataset_names(input_root, args.datasets)
    labeled_ratios = [normalize_ratio(value) for value in args.labeled_ratios]

    summaries = []
    for dataset_name in datasets:
        dataset_root = input_root / dataset_name
        if not dataset_has_expected_layout(dataset_root):
            raise FileNotFoundError(f"Expected videos/ and annotations/ under {dataset_root}")
        output_dir = output_root / slugify(dataset_name)
        if output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. Use --overwrite to regenerate it."
                )
            shutil.rmtree(output_dir)
        summaries.append(
            process_dataset(
                dataset_name=dataset_name,
                dataset_root=dataset_root,
                output_dir=output_dir,
                pairing_mode=args.pairing_mode,
                frame_index_base=args.frame_index_base,
                train_context_radius=args.train_context_radius,
                labeled_ratios=labeled_ratios,
                seed=args.seed,
                limit_videos=args.limit_videos,
            )
        )

    print(f"Prepared {len(summaries)} dataset(s) under {output_root}")
    for summary in summaries:
        print(
            f"- {summary['dataset']}: "
            f"{sum(split['case_ids'] for split in summary['splits'].values())} exported frames -> {summary['output_dir']}"
        )


if __name__ == "__main__":
    main()
