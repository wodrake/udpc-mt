from __future__ import annotations

import math

import torch

from losses.supervised import supervised_segmentation_loss as shared_supervised_segmentation_loss


def supervised_segmentation_loss(logits: torch.Tensor, targets: torch.Tensor, config: dict) -> tuple[torch.Tensor, dict[str, float]]:
    loss_cfg = config["loss"]
    return shared_supervised_segmentation_loss(
        logits,
        targets,
        bce_weight=loss_cfg["bce_weight"],
        dice_weight=loss_cfg["dice_weight"],
    )


def weighted_consistency_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    from losses.consistency import masked_mse_loss
    from losses.supervised import probabilities_from_logits

    student_probs = probabilities_from_logits(student_logits)
    return masked_mse_loss(student_probs, teacher_probs, mask=weights)


def binary_entropy(probabilities: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = probabilities.clamp(eps, 1.0 - eps)
    entropy = -(probabilities * probabilities.log() + (1.0 - probabilities) * (1.0 - probabilities).log())
    return entropy / math.log(2.0)


def uncertainty_to_weights(uncertainty: torch.Tensor, threshold: float) -> torch.Tensor:
    threshold = max(float(threshold), 1e-6)
    weights = 1.0 - uncertainty / threshold
    weights = weights.clamp(0.0, 1.0)
    return torch.where(uncertainty <= threshold, weights, torch.zeros_like(weights))
