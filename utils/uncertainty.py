from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class UncertaintyEstimate:
    mean_probabilities: torch.Tensor
    uncertainty_map: torch.Tensor
    low_uncertainty_mask: torch.Tensor
    high_uncertainty_mask: torch.Tensor
    low_uncertainty_weights: torch.Tensor
    mean_features: torch.Tensor | None = None


@contextlib.contextmanager
def enable_dropout_modules(model: nn.Module):
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


def binary_entropy(probabilities: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = probabilities.clamp(eps, 1.0 - eps)
    entropy = -(probabilities * probabilities.log() + (1.0 - probabilities) * (1.0 - probabilities).log())
    return entropy / math.log(2.0)


def segmentation_entropy(probabilities: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if probabilities.shape[1] == 1:
        return binary_entropy(probabilities, eps=eps)
    probabilities = probabilities.clamp(eps, 1.0)
    entropy = -(probabilities * probabilities.log()).sum(dim=1, keepdim=True)
    return entropy / math.log(probabilities.shape[1])


def build_uncertainty_masks(
    uncertainty_map: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    threshold = max(float(threshold), 1e-6)
    low_uncertainty_mask = (uncertainty_map <= threshold).float()
    high_uncertainty_mask = 1.0 - low_uncertainty_mask
    low_uncertainty_weights = (1.0 - uncertainty_map / threshold).clamp(0.0, 1.0) * low_uncertainty_mask
    return low_uncertainty_mask, high_uncertainty_mask, low_uncertainty_weights


@torch.no_grad()
def estimate_teacher_uncertainty(
    teacher: nn.Module,
    images: torch.Tensor,
    stochastic_forward: Callable[[nn.Module, torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]],
    mc_passes: int,
    threshold: float,
    normalize_feature_mean: bool = True,
) -> UncertaintyEstimate:
    mc_passes = max(int(mc_passes), 1)
    probability_samples: list[torch.Tensor] = []
    feature_samples: list[torch.Tensor] = []

    context = enable_dropout_modules(teacher) if mc_passes > 1 else contextlib.nullcontext()
    with context:
        for _ in range(mc_passes):
            probabilities, features = stochastic_forward(teacher, images)
            probability_samples.append(probabilities)
            if features is not None:
                feature_samples.append(features)

    stacked_probabilities = torch.stack(probability_samples, dim=0)
    mean_probabilities = stacked_probabilities.mean(dim=0)
    if mc_passes > 1:
        uncertainty_map = segmentation_entropy(mean_probabilities)
    else:
        uncertainty_map = torch.zeros_like(mean_probabilities[:, :1])

    mean_features = None
    if feature_samples:
        mean_features = torch.stack(feature_samples, dim=0).mean(dim=0)
        if normalize_feature_mean:
            mean_features = F.normalize(mean_features, dim=1)

    low_uncertainty_mask, high_uncertainty_mask, low_uncertainty_weights = build_uncertainty_masks(
        uncertainty_map,
        threshold=threshold,
    )
    return UncertaintyEstimate(
        mean_probabilities=mean_probabilities,
        uncertainty_map=uncertainty_map,
        low_uncertainty_mask=low_uncertainty_mask,
        high_uncertainty_mask=high_uncertainty_mask,
        low_uncertainty_weights=low_uncertainty_weights,
        mean_features=mean_features,
    )
