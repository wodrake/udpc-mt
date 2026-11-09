from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    config: dict[str, Any],
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
    teacher: torch.nn.Module | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": config,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }
    if teacher is None:
        payload["model"] = model.state_dict()
    else:
        payload["student_model"] = model.state_dict()
        payload["teacher_model"] = teacher.state_dict()
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    teacher: torch.nn.Module | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    model_state_key = "model" if "model" in checkpoint else "student_model"
    model.load_state_dict(checkpoint[model_state_key])
    if teacher is not None and checkpoint.get("teacher_model") is not None:
        teacher.load_state_dict(checkpoint["teacher_model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
