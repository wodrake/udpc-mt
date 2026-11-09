from __future__ import annotations

import argparse
from pathlib import Path


HF_REPO_ID = "mathpluscode/ACDC"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def count_patient_dirs(root: Path) -> int:
    return len([path for path in root.rglob("patient*") if path.is_dir()])


def count_labeled_frames(root: Path) -> int:
    return len(list(root.rglob("*_gt.nii.gz")))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the ACDC dataset mirror into data/processed/acdc for this project."
    )
    parser.add_argument("--repo-id", default=HF_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "acdc"),
        help="Destination directory for the downloaded dataset.",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    output_dir = ensure_dir(args.output_dir)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=["**/*.nii.gz", "**/*.cfg", "**/*.csv", "**/*.md", "**/*.txt"],
    )

    patient_count = count_patient_dirs(output_dir)
    label_count = count_labeled_frames(output_dir)
    print(
        f"Prepared ACDC mirror in {output_dir} "
        f"(patient_dirs={patient_count}, labeled_frames={label_count})"
    )


if __name__ == "__main__":
    main()
