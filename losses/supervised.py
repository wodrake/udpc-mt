from __future__ import annotations

import torch
import torch.nn.functional as F


def is_binary_segmentation(logits: torch.Tensor) -> bool:
    return logits.shape[1] == 1


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if is_binary_segmentation(logits):
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=1)


def targets_to_label_map(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 4 and targets.shape[1] > 1:
        return targets.argmax(dim=1).long()
    if targets.ndim == 4 and targets.shape[1] == 1:
        return targets[:, 0].round().long()
    if targets.ndim == 3:
        return targets.long()
    raise ValueError(f"Unsupported target shape for label map conversion: {targets.shape}")


def targets_to_channel_label_map(targets: torch.Tensor) -> torch.Tensor:
    return targets_to_label_map(targets).unsqueeze(1)


def binary_targets_as_float(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    if targets.ndim != 4 or targets.shape[1] != 1:
        raise ValueError(f"Binary targets must be [B,1,H,W] or [B,H,W], got {targets.shape}")
    return targets.float()


def hard_labels_from_probabilities(probabilities: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if probabilities.shape[1] == 1:
        return (probabilities >= threshold).long()
    return probabilities.argmax(dim=1, keepdim=True).long()


def binary_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    targets = binary_targets_as_float(targets)
    probabilities = torch.sigmoid(logits)
    dims = (2, 3)
    intersection = (probabilities * targets).sum(dim=dims)
    cardinality = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    include_background: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    label_map = targets_to_label_map(targets)
    target_one_hot = F.one_hot(label_map, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    dims = (2, 3)
    intersection = (probabilities * target_one_hot).sum(dim=dims)
    cardinality = probabilities.sum(dim=dims) + target_one_hot.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    if logits.shape[1] > 1 and not include_background:
        dice = dice[:, 1:]
    return 1.0 - dice.mean()


def supervised_segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    if is_binary_segmentation(logits):
        targets = binary_targets_as_float(targets)
        loss_ce = F.binary_cross_entropy_with_logits(logits, targets)
        loss_dice = binary_dice_loss(logits, targets)
        total_loss = bce_weight * loss_ce + dice_weight * loss_dice
        return total_loss, {
            "loss_bce": loss_ce.item(),
            "loss_dice": loss_dice.item(),
        }

    label_map = targets_to_label_map(targets)
    loss_ce = F.cross_entropy(logits, label_map)
    loss_dice = multiclass_dice_loss(logits, label_map, include_background=False)
    total_loss = bce_weight * loss_ce + dice_weight * loss_dice
    return total_loss, {
        "loss_ce": loss_ce.item(),
        "loss_dice": loss_dice.item(),
    }
