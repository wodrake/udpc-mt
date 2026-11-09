from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def find_dataset_root(path: Path) -> Path:
    current = path.resolve()
    while True:
        metadata_path = current / "metadata.csv"
        if metadata_path.exists():
            return current

        split_dirs = []
        for split in ("train", "val", "test"):
            if (current / split).exists():
                split_dirs.append(split)
        if split_dirs:
            return current

        child_dirs = [entry for entry in current.iterdir() if entry.is_dir()]
        if len(child_dirs) != 1:
            return current
        current = child_dirs[0]


def infer_download_name(source_url: str) -> str:
    parsed = urlparse(source_url)
    candidate = Path(parsed.path).name
    if candidate:
        return candidate
    return "bml_source"


def find_split_dirs(dataset_root: Path) -> list[tuple[str, Path, Path]]:
    candidates = []
    for split in ("train", "val", "test"):
        image_dir = None
        mask_dir = None
        for image_name in ("image", "images"):
            path = dataset_root / split / image_name
            if path.exists():
                image_dir = path
                break
        for mask_name in ("mask", "masks"):
            path = dataset_root / split / mask_name
            if path.exists():
                mask_dir = path
                break
        if image_dir is not None and mask_dir is not None:
            candidates.append((split, image_dir, mask_dir))
    return candidates


def count_bml_records(dataset_root: Path) -> int:
    metadata_path = dataset_root / "metadata.csv"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            count = 0
            for row in reader:
                if row.get("image_path"):
                    count += 1
            return count

    count = 0
    for _, image_dir, mask_dir in find_split_dirs(dataset_root):
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file():
                continue
            mask_path = mask_dir / image_path.name
            alt_mask = mask_dir / f"{image_path.stem}_mask{image_path.suffix}"
            if mask_path.exists() or alt_mask.exists():
                count += 1
    return count


def prepare_from_directory(source_dir: Path, output_dir: Path) -> int:
    dataset_root = find_dataset_root(source_dir)
    ensure_dir(output_dir.parent)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(dataset_root, output_dir)
    sample_count = count_bml_records(output_dir)
    if sample_count < 1:
        raise RuntimeError(
            "Prepared directory does not look like a supported BML layout. "
            "Expected metadata.csv or split folders like train/image and train/mask."
        )
    return sample_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare BML under data/processed/bml from an existing directory, archive, or URL. "
            "No default public BML source is bundled in this repository."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Existing local directory that already contains the BML dataset.",
    )
    parser.add_argument(
        "--archive-path",
        default=None,
        help="Local archive path (.zip, .tar, .tar.gz, .tgz, etc.) containing the BML dataset.",
    )
    parser.add_argument(
        "--source-url",
        default=None,
        help="Optional direct URL to a BML archive if you have one.",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(PROJECT_ROOT / "data" / "raw" / "bml"),
        help="Directory used for downloaded archives and extraction.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "bml"),
        help="Prepared output directory.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = ensure_dir(args.raw_dir)

    archive_path: Path | None = None
    if args.source_url is not None:
        archive_path = raw_dir / infer_download_name(args.source_url)
        print(f"Downloading BML archive from {args.source_url} to {archive_path}")
        urlretrieve(args.source_url, archive_path)
    elif args.archive_path is not None:
        archive_path = Path(args.archive_path).resolve()

    input_dir: Path | None = Path(args.input_dir).resolve() if args.input_dir is not None else None
    if archive_path is not None:
        extract_dir = raw_dir / "extracted"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        ensure_dir(extract_dir)
        shutil.unpack_archive(str(archive_path), str(extract_dir))
        input_dir = extract_dir

    if input_dir is None:
        raise SystemExit(
            "No default public BML download source is configured. "
            "Provide --input-dir, --archive-path, or --source-url."
        )

    sample_count = prepare_from_directory(input_dir, output_dir)
    print(f"Prepared {sample_count} BML samples in {output_dir}")


if __name__ == "__main__":
    main()
