from __future__ import annotations

import torch


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must have the same shape, got {predictions.shape} and {targets.shape}"
        )
    squared_error = (predictions - targets).pow(2)
    if mask is None:
        return squared_error.mean()
    if mask.ndim == squared_error.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.ndim != squared_error.ndim:
        raise ValueError(
            f"mask must have the same ndim as predictions or one less, got {mask.shape} and {squared_error.shape}"
        )
    if mask.shape[1] == 1 and squared_error.shape[1] > 1:
        mask = mask.expand(-1, squared_error.shape[1], -1, -1)
    if mask.shape != squared_error.shape:
        raise ValueError(
            f"mask must match prediction shape, got {mask.shape} and {squared_error.shape}"
        )
    masked_error = squared_error * mask.float()
    denominator = mask.float().sum()
    if denominator.item() <= 0:
        return predictions.new_tensor(0.0)
    return masked_error.sum() / denominator


def decoupled_masked_mse_consistency_loss(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    low_uncertainty_mask: torch.Tensor | None = None,
    high_uncertainty_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    from losses.supervised import probabilities_from_logits

    student_probabilities = probabilities_from_logits(student_logits)
    pixel_consistency = masked_mse_loss(
        student_probabilities,
        teacher_probabilities,
        mask=low_uncertainty_mask,
    )
    metrics: dict[str, float] = {}
    if low_uncertainty_mask is not None:
        metrics["low_uncertainty_ratio"] = low_uncertainty_mask.float().mean().item()
    if high_uncertainty_mask is not None:
        metrics["high_uncertainty_ratio"] = high_uncertainty_mask.float().mean().item()
    return pixel_consistency, metrics


def mse_consistency_loss(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    from losses.supervised import probabilities_from_logits

    student_probabilities = probabilities_from_logits(student_logits)
    return masked_mse_loss(student_probabilities, teacher_probabilities, mask=weights)
