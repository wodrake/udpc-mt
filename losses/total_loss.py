from __future__ import annotations

from typing import Iterable

import torch


def compose_total_loss(
    supervised_loss: torch.Tensor,
    pixel_consistency_loss: torch.Tensor,
    pixel_consistency_weight: float,
    extra_unsupervised_losses: Iterable[tuple[str, torch.Tensor, float]] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weighted_pixel_consistency = float(pixel_consistency_weight) * pixel_consistency_loss
    total_unsupervised = weighted_pixel_consistency
    metrics = {
        "loss_supervised": supervised_loss.item(),
        "loss_pixel_consistency": pixel_consistency_loss.item(),
        "loss_consistency": pixel_consistency_loss.item(),
        "loss_pixel_consistency_weighted": weighted_pixel_consistency.item(),
    }

    for name, loss_value, loss_weight in extra_unsupervised_losses or []:
        weighted_loss = float(loss_weight) * loss_value
        total_unsupervised = total_unsupervised + weighted_loss
        metrics[f"loss_{name}"] = loss_value.item()
        metrics[f"loss_{name}_weighted"] = weighted_loss.item()

    total_loss = supervised_loss + total_unsupervised
    metrics["loss_unsupervised"] = total_unsupervised.item()
    metrics["loss_total"] = total_loss.item()
    return total_loss, metrics
