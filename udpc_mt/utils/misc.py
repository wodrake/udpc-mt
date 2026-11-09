from __future__ import annotations

import contextlib
import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def sigmoid_rampup(current: float, rampup_length: float) -> float:
    if rampup_length <= 0:
        return 1.0
    current = float(np.clip(current, 0.0, rampup_length))
    phase = 1.0 - current / rampup_length
    return math.exp(-5.0 * phase * phase)


class ScalarTracker:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, float] = {}

    def update(self, values: dict[str, float], weight: float = 1.0) -> None:
        for key, value in values.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * weight
            self.counts[key] = self.counts.get(key, 0.0) + weight

    def average(self) -> dict[str, float]:
        return {
            key: self.totals[key] / max(self.counts.get(key, 1.0), 1e-8)
            for key in self.totals
        }


def json_dump(payload: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


@contextlib.contextmanager
def enable_dropout(model: nn.Module):
    dropout_modules: list[nn.Module] = []
    original_states: list[bool] = []
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            dropout_modules.append(module)
            original_states.append(module.training)
            module.train()
    try:
        yield
    finally:
        for module, training in zip(dropout_modules, original_states):
            module.train(training)


@torch.no_grad()
def update_ema(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    teacher_params = dict(teacher.named_parameters())
    student_params = dict(student.named_parameters())
    for name, teacher_param in teacher_params.items():
        teacher_param.data.mul_(decay).add_(student_params[name].data, alpha=1.0 - decay)
