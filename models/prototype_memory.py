from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PrototypeMemory(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 2,
        momentum: float = 0.95,
        min_pixels: int = 32,
        normalize: bool = True,
        init_strategy: str = "first_valid_batch",
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not 0.0 <= float(momentum) <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        if init_strategy != "first_valid_batch":
            raise ValueError(f"Unsupported init_strategy: {init_strategy}")

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.min_pixels = int(min_pixels)
        self.normalize = bool(normalize)
        self.init_strategy = init_strategy

        self.register_buffer("prototypes", torch.zeros(self.num_classes, self.feature_dim))
        self.register_buffer("initialized", torch.zeros(self.num_classes, dtype=torch.bool))

    def _normalize_vector(self, vector: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return vector
        return F.normalize(vector, dim=0)

    def _resize_targets(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if labels.ndim == 3:
            labels = labels.unsqueeze(1)
        elif labels.ndim == 4 and labels.shape[1] > 1:
            labels = labels.argmax(dim=1, keepdim=True)
        spatial_size = features.shape[-2:]
        labels = F.interpolate(labels.float(), size=spatial_size, mode="nearest").long()
        if weights is None:
            weights = torch.ones_like(labels, dtype=features.dtype)
        else:
            if weights.ndim == 3:
                weights = weights.unsqueeze(1)
            weights = F.interpolate(weights.float(), size=spatial_size, mode="nearest")
        return labels, weights

    def extract_class_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels, weights = self._resize_targets(features, labels, weights)
        class_prototypes = features.new_zeros(self.num_classes, features.shape[1])
        valid_classes = torch.zeros(self.num_classes, dtype=torch.bool, device=features.device)
        pixel_counts = features.new_zeros(self.num_classes)

        for class_index in range(self.num_classes):
            class_mask = (labels == class_index).float() * weights
            pixel_count = class_mask.sum()
            pixel_counts[class_index] = pixel_count
            if pixel_count.item() < self.min_pixels:
                continue
            prototype = (features * class_mask).sum(dim=(0, 2, 3)) / pixel_count
            class_prototypes[class_index] = self._normalize_vector(prototype)
            valid_classes[class_index] = True

        return class_prototypes, valid_classes, pixel_counts

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> dict[str, float]:
        class_prototypes, valid_classes, _ = self.extract_class_prototypes(features, labels, weights)
        updated_classes = 0
        for class_index in range(self.num_classes):
            if not valid_classes[class_index]:
                continue
            updated_classes += 1
            prototype = class_prototypes[class_index]
            if not self.initialized[class_index]:
                self.prototypes[class_index] = prototype
                self.initialized[class_index] = True
            else:
                blended = self.momentum * self.prototypes[class_index] + (1.0 - self.momentum) * prototype
                self.prototypes[class_index] = self._normalize_vector(blended)

        return {
            "prototype_initialized_classes": float(self.initialized.sum().item()),
            "prototype_updated_classes": float(updated_classes),
            "prototype_valid_classes": float(valid_classes.sum().item()),
        }

    def consistency_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        class_prototypes, valid_classes, _ = self.extract_class_prototypes(features, labels, weights)
        losses = []
        for class_index in range(self.num_classes):
            if not valid_classes[class_index] or not self.initialized[class_index]:
                continue
            similarity = F.cosine_similarity(
                class_prototypes[class_index].unsqueeze(0),
                self.prototypes[class_index].unsqueeze(0),
            ).mean()
            losses.append(1.0 - similarity)
        if not losses:
            return features.new_tensor(0.0)
        return torch.stack(losses).mean()
