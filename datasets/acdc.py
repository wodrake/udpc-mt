from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from datasets.base import SegmentationDatasetAdapter


def _parse_info_cfg(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not path.exists():
        return metadata
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata


@lru_cache(maxsize=64)
def _load_nifti(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata(), dtype=np.float32)


class ACDCDatasetAdapter(SegmentationDatasetAdapter):
    name = "acdc"
    modality = "mri"
    task_type = "multiclass"
    default_num_classes = 4
    default_input_channels = 1
    dimension_mode = "2d"
    slice_mode = "single_slice"
    class_names = ("background", "rv", "myocardium", "lv")

    def discover_records(self, dataset_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        patient_dirs = sorted(path for path in dataset_root.rglob("patient*") if path.is_dir())
        for patient_dir in patient_dirs:
            info = _parse_info_cfg(patient_dir / "Info.cfg")
            diagnosis = info.get("Group", "unknown").lower()
            patient_id = patient_dir.name
            for mask_path in sorted(patient_dir.glob("*_gt.nii.gz")):
                image_name = mask_path.name.replace("_gt.nii.gz", ".nii.gz")
                image_path = patient_dir / image_name
                if not image_path.exists():
                    continue
                volume = _load_nifti(str(image_path))
                if volume.ndim != 3:
                    raise ValueError(f"Expected 3D ACDC volume, got {volume.shape} for {image_path}")
                frame_id = image_path.name.replace(".nii.gz", "")
                for slice_index in range(volume.shape[-1]):
                    records.append(
                        {
                            "sample_id": f"{frame_id}_z{slice_index:03d}",
                            "case_id": frame_id,
                            "group_id": patient_id,
                            "image_path": str(image_path),
                            "mask_paths": [str(mask_path)],
                            "slice_index": slice_index,
                            "phase": frame_id,
                            "stratify_key": diagnosis,
                            "source_layout": "acdc_raw_nifti",
                        }
                    )
        if not records:
            raise FileNotFoundError(
                f"Could not discover ACDC samples under {dataset_root}. "
                "Expected patient directories with Info.cfg and *_frameXX_gt.nii.gz files."
            )
        return records

    def load_image(self, record: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
        volume = _load_nifti(record["image_path"])
        image = volume[:, :, int(record["slice_index"])]
        return self.normalize_loaded_image(image, config)

    def load_mask(
        self,
        record: dict[str, Any],
        image_shape: tuple[int, int] | tuple[int, int, int],
        config: dict[str, Any],
    ) -> np.ndarray:
        mask_volume = _load_nifti(record["mask_paths"][0])
        mask = mask_volume[:, :, int(record["slice_index"])]
        return np.asarray(mask, dtype=np.int64)
