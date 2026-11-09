from __future__ import annotations

import torch
import torch.nn.functional as F


def _resize_labels_and_mask(
    features: torch.Tensor,
    labels: torch.Tensor,
    pixel_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.ndim == 3:
        labels = labels.unsqueeze(1)
    elif labels.ndim == 4 and labels.shape[1] > 1:
        labels = labels.argmax(dim=1, keepdim=True)
    spatial_size = features.shape[-2:]
    labels = F.interpolate(labels.float(), size=spatial_size, mode="nearest").long()
    if pixel_mask is None:
        pixel_mask = torch.ones_like(labels, dtype=features.dtype)
    else:
        if pixel_mask.ndim == 3:
            pixel_mask = pixel_mask.unsqueeze(1)
        pixel_mask = F.interpolate(pixel_mask.float(), size=spatial_size, mode="nearest")
    return labels, pixel_mask


def pixel_to_prototype_contrastive_loss(
    pixel_features: torch.Tensor,
    class_targets: torch.Tensor,
    prototypes: torch.Tensor,
    initialized: torch.Tensor,
    pixel_mask: torch.Tensor | None = None,
    temperature: float = 0.1,
    normalize: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if pixel_features.ndim != 4:
        raise ValueError(f"pixel_features must be BCHW, got shape {pixel_features.shape}")

    labels, pixel_mask = _resize_labels_and_mask(pixel_features, class_targets, pixel_mask)
    valid_class_indices = torch.nonzero(initialized, as_tuple=False).flatten()
    metrics = {
        "prototype_initialized_classes": float(initialized.sum().item()),
        "prototype_contrastive_selected_pixels": 0.0,
    }
    if valid_class_indices.numel() < 2:
        return pixel_features.new_tensor(0.0), metrics

    flattened_features = pixel_features.permute(0, 2, 3, 1).reshape(-1, pixel_features.shape[1])
    flattened_labels = labels.reshape(-1)
    flattened_mask = pixel_mask.reshape(-1) > 0.5

    valid_label_mask = torch.isin(flattened_labels, valid_class_indices)
    selected_mask = flattened_mask & valid_label_mask
    metrics["prototype_contrastive_selected_pixels"] = float(selected_mask.sum().item())
    if selected_mask.sum().item() <= 0:
        return pixel_features.new_tensor(0.0), metrics

    selected_features = flattened_features[selected_mask]
    selected_labels = flattened_labels[selected_mask]
    selected_prototypes = prototypes[valid_class_indices]

    if normalize:
        selected_features = F.normalize(selected_features, dim=1)
        selected_prototypes = F.normalize(selected_prototypes, dim=1)

    temperature = max(float(temperature), 1e-6)
    logits = selected_features @ selected_prototypes.t()
    logits = logits / temperature

    target_indices = (selected_labels.unsqueeze(1) == valid_class_indices.unsqueeze(0)).long().argmax(dim=1)
    loss = F.cross_entropy(logits, target_indices)
    return loss, metrics
