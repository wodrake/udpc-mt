from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_tn3k_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "thyroid-image").is_dir()
        and (path / "thyroid-mask").is_dir()
    )


def find_tn3k_root(path: Path) -> Path:
    candidates = [
        path.resolve(),
        path.resolve() / "tg3k",
        path.resolve() / "Thyroid Dataset" / "tg3k",
    ]
    for candidate in candidates:
        if is_tn3k_root(candidate):
            return candidate

    for candidate in sorted(path.resolve().rglob("tg3k")):
        if is_tn3k_root(candidate):
            return candidate

    for candidate in sorted(path.resolve().rglob("*")):
        if is_tn3k_root(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find a TN3K/TG3K dataset root under {path}. "
        "Expected thyroid-image/ and thyroid-mask/ directories."
    )


def count_tn3k_records(dataset_root: Path) -> int:
    image_dir = dataset_root / "thyroid-image"
    mask_dir = dataset_root / "thyroid-mask"
    count = 0
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        if (mask_dir / image_path.name).exists():
            count += 1
    return count


def prepare_from_directory(source_dir: Path, output_dir: Path) -> int:
    dataset_root = find_tn3k_root(source_dir)
    ensure_dir(output_dir.parent)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(dataset_root, output_dir)
    sample_count = count_tn3k_records(output_dir)
    if sample_count < 1:
        raise RuntimeError(
            "Prepared directory does not look like a supported TN3K/TG3K layout. "
            "Expected thyroid-image/ and thyroid-mask/ with matching files."
        )
    return sample_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare TN3K/TG3K under data/processed/tn3k from an existing directory or "
            "from the official 'Thyroid Dataset.zip' archive."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Existing local directory that already contains the extracted TN3K/TG3K dataset.",
    )
    parser.add_argument(
        "--archive-path",
        default=None,
        help="Local archive path containing 'Thyroid Dataset/tg3k'.",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(PROJECT_ROOT / "data" / "raw" / "tn3k"),
        help="Directory used for temporary extraction.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "tn3k"),
        help="Prepared output directory.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = ensure_dir(args.raw_dir)

    input_dir: Path | None = Path(args.input_dir).resolve() if args.input_dir is not None else None
    if args.archive_path is not None:
        archive_path = Path(args.archive_path).resolve()
        extract_dir = raw_dir / "extracted"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        ensure_dir(extract_dir)
        shutil.unpack_archive(str(archive_path), str(extract_dir))
        input_dir = extract_dir

    if input_dir is None:
        raise SystemExit("Provide --input-dir or --archive-path to prepare TN3K/TG3K.")

    sample_count = prepare_from_directory(input_dir, output_dir)
    print(f"Prepared {sample_count} TN3K/TG3K samples in {output_dir}")


if __name__ == "__main__":
    main()
