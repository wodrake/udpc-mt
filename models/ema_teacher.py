from __future__ import annotations

import copy

import torch
from torch import nn


@torch.no_grad()
def build_ema_teacher(student: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(student)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema_teacher(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    if not 0.0 <= float(decay) <= 1.0:
        raise ValueError(f"EMA decay must be in [0, 1], got {decay}")

    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    for name, teacher_tensor in teacher_state.items():
        student_tensor = student_state[name]
        if torch.is_floating_point(teacher_tensor):
            teacher_tensor.mul_(decay).add_(student_tensor, alpha=1.0 - decay)
        else:
            teacher_tensor.copy_(student_tensor)
