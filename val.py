from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainers.trainer import resolve_device, validate_checkpoint as validate_root_checkpoint
from udpc_mt.engine.trainer import evaluate as evaluate_udpc_checkpoint
from udpc_mt.models.unet import build_model as build_udpc_model
from udpc_mt.utils.config import apply_overrides, load_config
from udpc_mt.utils.misc import PROJECT_ROOT
from datasets import create_dataloaders


def resolve_input_path(path: str | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def load_validation_config(
    checkpoint_path: Path,
    config_arg: str | None,
    overrides: list[str],
) -> dict:
    state = torch.load(checkpoint_path, map_location="cpu")
    if config_arg is not None:
        config_path = resolve_input_path(config_arg)
        if config_path is None:
            raise ValueError("Failed to resolve --config path.")
        config = load_config(config_path, overrides)
    else:
        config = apply_overrides(state["config"], overrides)
    return config


def validate_udpc_checkpoint(
    checkpoint_path: Path,
    config: dict,
    split: str,
) -> dict[str, float]:
    device = resolve_device(config)
    model = build_udpc_model(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model_state_key = "student_model" if "student_model" in state else "model"
    model.load_state_dict(state[model_state_key])
    loaders, _ = create_dataloaders(config)
    return evaluate_udpc_checkpoint(model, loaders[split], device, config)


def validate_supervised_or_mt_checkpoint(
    checkpoint_path: Path,
    config: dict,
    split: str,
) -> dict[str, float]:
    return validate_root_checkpoint(
        checkpoint_path=checkpoint_path,
        split=split,
        config_override=config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a BUSI segmentation checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--ckpt", "--checkpoint", dest="checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Evaluation split.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config entries with key=value. Can be repeated.",
    )
    args = parser.parse_args()

    checkpoint_path = resolve_input_path(args.checkpoint)
    if checkpoint_path is None:
        raise ValueError("Failed to resolve checkpoint path.")

    config = load_validation_config(checkpoint_path, args.config, args.override)
    if config["mode"] in {"baseline", "mean_teacher"}:
        metrics = validate_supervised_or_mt_checkpoint(checkpoint_path, config, args.split)
    elif config["mode"] == "udpc_mt":
        metrics = validate_udpc_checkpoint(checkpoint_path, config, args.split)
    else:
        raise ValueError(f"Unsupported validation mode: {config['mode']}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
