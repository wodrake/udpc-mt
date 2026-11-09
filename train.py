from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainers.trainer import run_training as run_root_training
from udpc_mt.engine.trainer import run_training as run_udpc_training
from udpc_mt.utils.config import apply_overrides, load_config
from udpc_mt.utils.misc import PROJECT_ROOT


def resolve_input_path(path: str | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def load_training_config(config_arg: str | None, resume_arg: str | None, overrides: list[str]) -> tuple[dict, Path | None]:
    resume_path = resolve_input_path(resume_arg)
    if resume_path is not None:
        state = torch.load(resume_path, map_location="cpu")
        return apply_overrides(state["config"], overrides), resume_path

    config_path = resolve_input_path(config_arg or "configs/busi_udpcmt.yaml")
    if config_path is None:
        raise ValueError("Either --config or --resume must be provided.")
    return load_config(config_path, overrides), None


def dispatch_training(
    config: dict,
    run_dir: str | Path | None = None,
    resume_path: str | Path | None = None,
) -> dict:
    mode = config["mode"]
    if mode in {"baseline", "mean_teacher"}:
        return run_root_training(config, run_dir=run_dir, resume_path=resume_path)
    if mode == "udpc_mt":
        return run_udpc_training(config, run_dir=run_dir, resume_path=resume_path)
    raise ValueError(f"Unsupported training mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BUSI segmentation model from a YAML config.")
    parser.add_argument("--config", default="configs/busi_udpcmt.yaml", help="Path to a YAML config.")
    parser.add_argument("--run-dir", default=None, help="Optional explicit run directory.")
    parser.add_argument("--resume", default=None, help="Resume training from a checkpoint.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config entries with key=value. Can be repeated.",
    )
    args = parser.parse_args()

    config, resume_path = load_training_config(args.config, args.resume, args.override)
    summary = dispatch_training(config, run_dir=args.run_dir, resume_path=resume_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
