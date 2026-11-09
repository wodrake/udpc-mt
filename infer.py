from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.unet import build_unet
from trainers.trainer import resolve_device
from udpc_mt.models.unet import build_model as build_udpc_model
from udpc_mt.utils.config import apply_overrides, load_config
from udpc_mt.utils.misc import PROJECT_ROOT, ensure_dir
from utils.uncertainty import estimate_teacher_uncertainty

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resolve_input_path(path: str | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def load_config_from_args(checkpoint_path: Path, config_arg: str | None, overrides: list[str]) -> tuple[dict, dict]:
    state = torch.load(checkpoint_path, map_location="cpu")
    if config_arg is not None:
        config_path = resolve_input_path(config_arg)
        if config_path is None:
            raise ValueError("Failed to resolve --config path.")
        config = load_config(config_path, overrides)
    else:
        config = apply_overrides(state["config"], overrides)
    return config, state


def collect_input_images(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return [input_path]


def preprocess_image(image_path: Path, image_size: int) -> tuple[torch.Tensor, Image.Image]:
    original = Image.open(image_path).convert("L")
    resized = original.resize((image_size, image_size), Image.BILINEAR)
    image_array = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    return tensor, original


def save_grayscale_image(array: np.ndarray, destination: Path, size: tuple[int, int] | None = None) -> None:
    image = Image.fromarray(array.astype(np.uint8))
    if size is not None and image.size != size:
        image = image.resize(size, Image.NEAREST)
    image.save(destination)


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        return np.zeros_like(array, dtype=np.uint8)
    array_min = float(array.min())
    array_max = float(array.max())
    if array_max - array_min < 1e-8:
        return np.zeros_like(array, dtype=np.uint8)
    normalized = (array - array_min) / (array_max - array_min)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)


def make_prediction_overlay(original: Image.Image, pred_mask: np.ndarray, gt_mask: np.ndarray | None = None) -> Image.Image:
    original_rgb = np.stack([np.array(original)] * 3, axis=-1)
    overlay = original_rgb.astype(np.float32)
    pred_region = pred_mask > 0
    overlay[pred_region, 0] = 255
    overlay[pred_region, 1] *= 0.35
    overlay[pred_region, 2] *= 0.35
    if gt_mask is not None:
        gt_region = gt_mask > 0
        overlay[gt_region, 1] = 255
        overlay[gt_region, 0] *= 0.35
        overlay[gt_region, 2] *= 0.35
    return Image.fromarray(overlay.clip(0, 255).astype(np.uint8))


def infer_gt_path(image_path: Path, explicit_gt: Path | None = None, gt_dir: Path | None = None) -> Path | None:
    if explicit_gt is not None:
        return explicit_gt
    if gt_dir is not None:
        candidate = gt_dir / image_path.name
        if candidate.exists():
            return candidate
        alt_candidate = gt_dir / f"{image_path.stem}_mask{image_path.suffix}"
        if alt_candidate.exists():
            return alt_candidate
    if image_path.parent.name == "images":
        sibling = image_path.parent.parent / "masks" / image_path.name
        if sibling.exists():
            return sibling
    same_dir_mask = image_path.with_name(f"{image_path.stem}_mask{image_path.suffix}")
    if same_dir_mask.exists():
        return same_dir_mask
    return None


def load_gt_mask(gt_path: Path | None, original_size: tuple[int, int]) -> np.ndarray | None:
    if gt_path is None or not gt_path.exists():
        return None
    mask = Image.open(gt_path).convert("L").resize(original_size, Image.NEAREST)
    return ((np.array(mask) > 0).astype(np.uint8) * 255)


def build_inference_models(config: dict, state: dict, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module | None, str]:
    mode = config["mode"]
    if mode == "udpc_mt":
        student = build_udpc_model(config).to(device)
        teacher = build_udpc_model(config).to(device) if state.get("teacher_model") is not None else None
        model_state_key = "student_model" if "student_model" in state else "model"
        student.load_state_dict(state[model_state_key])
        if teacher is not None:
            teacher.load_state_dict(state["teacher_model"])
            teacher.eval()
        student.eval()
        return student, teacher, mode

    student = build_unet(config).to(device)
    teacher = build_unet(config).to(device) if state.get("teacher_model") is not None else None
    model_state_key = "student_model" if "student_model" in state else "model"
    student.load_state_dict(state[model_state_key])
    if teacher is not None:
        teacher.load_state_dict(state["teacher_model"])
        teacher.eval()
    student.eval()
    return student, teacher, mode


def build_teacher_forward(mode: str) -> Callable[[torch.nn.Module, torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]]:
    if mode == "udpc_mt":
        def teacher_forward(model: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
            outputs = model(images)
            return torch.sigmoid(outputs["logits"]), outputs["embeddings"]

        return teacher_forward

    def teacher_forward(model: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = model(images)
        return torch.sigmoid(logits), None

    return teacher_forward


@torch.no_grad()
def run_inference_on_image(
    image_path: Path,
    output_dir: Path,
    student: torch.nn.Module,
    teacher: torch.nn.Module | None,
    mode: str,
    config: dict,
    threshold: float,
    gt_path: Path | None = None,
    save_uncertainty: bool = True,
) -> dict[str, str | bool]:
    tensor, original = preprocess_image(image_path, config["dataset"]["image_size"])
    device = next(student.parameters()).device
    tensor = tensor.to(device)

    if mode == "udpc_mt":
        student_outputs = student(tensor)
        prediction = torch.sigmoid(student_outputs["logits"])[0, 0].cpu().numpy()
    else:
        prediction = torch.sigmoid(student(tensor))[0, 0].cpu().numpy()

    pred_mask = (prediction >= threshold).astype(np.uint8) * 255
    pred_mask_resized = np.array(
        Image.fromarray(pred_mask).resize(original.size, Image.NEAREST),
        dtype=np.uint8,
    )

    gt_mask = load_gt_mask(gt_path, original.size)
    stem = image_path.stem

    original.save(output_dir / f"{stem}_input.png")
    save_grayscale_image(pred_mask_resized, output_dir / f"{stem}_pred.png")
    save_grayscale_image(pred_mask_resized, output_dir / f"{stem}_mask.png")
    if gt_mask is not None:
        save_grayscale_image(gt_mask, output_dir / f"{stem}_gt.png")
    make_prediction_overlay(original, pred_mask_resized, gt_mask).save(output_dir / f"{stem}_overlay.png")

    saved_uncertainty = False
    uncertainty_cfg = config.get("uncertainty", {})
    if save_uncertainty and teacher is not None and bool(uncertainty_cfg.get("enabled", False)):
        estimate = estimate_teacher_uncertainty(
            teacher,
            tensor,
            stochastic_forward=build_teacher_forward(mode),
            mc_passes=int(uncertainty_cfg.get("mc_passes", 1)),
            threshold=float(uncertainty_cfg.get("threshold", 0.35)),
            normalize_feature_mean=(mode == "udpc_mt"),
        )
        uncertainty_map = estimate.uncertainty_map[0, 0].cpu().numpy()
        uncertainty_resized = np.array(
            Image.fromarray(normalize_to_uint8(uncertainty_map)).resize(original.size, Image.BILINEAR),
            dtype=np.uint8,
        )
        save_grayscale_image(uncertainty_resized, output_dir / f"{stem}_uncertainty.png")
        saved_uncertainty = True

    return {
        "image": str(image_path),
        "gt_used": gt_mask is not None,
        "uncertainty_saved": saved_uncertainty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on one image or a folder of ultrasound images.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--ckpt", "--checkpoint", dest="checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--input", required=True, help="Input image path or directory.")
    parser.add_argument("--output-dir", default="outputs/predictions", help="Directory for prediction outputs.")
    parser.add_argument("--gt", default=None, help="Optional GT mask path for single-image inference.")
    parser.add_argument("--gt-dir", default=None, help="Optional GT mask directory for batch inference.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary mask threshold.")
    parser.add_argument(
        "--no-save-uncertainty",
        action="store_true",
        help="Disable uncertainty map export even when teacher and MC Dropout are available.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config entries with key=value. Can be repeated.",
    )
    args = parser.parse_args()

    checkpoint_path = resolve_input_path(args.checkpoint)
    input_path = resolve_input_path(args.input)
    gt_path = resolve_input_path(args.gt)
    gt_dir = resolve_input_path(args.gt_dir)
    output_dir = resolve_input_path(args.output_dir)
    if checkpoint_path is None or input_path is None or output_dir is None:
        raise ValueError("Failed to resolve input, checkpoint, or output path.")

    config, state = load_config_from_args(checkpoint_path, args.config, args.override)
    device = resolve_device(config)
    student, teacher, mode = build_inference_models(config, state, device)
    ensure_dir(output_dir)

    records = []
    for image_path in collect_input_images(input_path):
        resolved_gt = infer_gt_path(image_path, explicit_gt=gt_path, gt_dir=gt_dir)
        result = run_inference_on_image(
            image_path=image_path,
            output_dir=output_dir,
            student=student,
            teacher=teacher,
            mode=mode,
            config=config,
            threshold=args.threshold,
            gt_path=resolved_gt,
            save_uncertainty=not args.no_save_uncertainty,
        )
        records.append(result)
        print(f"Saved predictions for {image_path.name} to {output_dir}")

    print(f"Inference complete: {len(records)} image(s) processed.")


if __name__ == "__main__":
    main()
