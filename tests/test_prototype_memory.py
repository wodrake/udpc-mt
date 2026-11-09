from __future__ import annotations

import torch

from losses.prototype import pixel_to_prototype_contrastive_loss
from models.prototype_memory import PrototypeMemory
from udpc_mt.models.unet import UNet


def test_decoder_head_feature_extraction_for_prototypes() -> None:
    model = UNet(
        in_channels=1,
        num_classes=1,
        base_channels=16,
        prototype_dim=24,
        dropout=0.1,
    )
    outputs = model(torch.randn(2, 1, 64, 64))
    assert outputs["embeddings"].shape == (2, 24, 64, 64)


def test_prototype_initialization_and_update() -> None:
    memory = PrototypeMemory(
        feature_dim=8,
        num_classes=2,
        momentum=0.9,
        min_pixels=1,
        normalize=True,
        init_strategy="first_valid_batch",
    )
    features = torch.randn(2, 8, 16, 16)
    labels = torch.zeros(2, 1, 16, 16)
    labels[:, :, :, 8:] = 1.0

    stats = memory.update(features, labels)
    assert stats["prototype_initialized_classes"] == 2.0
    assert stats["prototype_updated_classes"] == 2.0
    assert memory.initialized.all()
    assert memory.prototypes.shape == (2, 8)

    before = memory.prototypes.clone()
    memory.update(features + 0.1, labels)
    assert not torch.allclose(before, memory.prototypes)


def test_prototype_multi_batch_updates_and_consistency_loss() -> None:
    memory = PrototypeMemory(
        feature_dim=6,
        num_classes=2,
        momentum=0.95,
        min_pixels=1,
        normalize=True,
        init_strategy="first_valid_batch",
    )

    for _ in range(3):
        features = torch.randn(2, 6, 8, 8)
        labels = torch.randint(0, 2, (2, 1, 8, 8)).float()
        weights = torch.ones(2, 1, 8, 8)
        stats = memory.update(features, labels, weights)
        assert stats["prototype_valid_classes"] >= 1.0

    loss = memory.consistency_loss(features, labels, weights)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_pixel_to_prototype_contrastive_loss_forward() -> None:
    memory = PrototypeMemory(
        feature_dim=6,
        num_classes=2,
        momentum=0.95,
        min_pixels=1,
        normalize=True,
        init_strategy="first_valid_batch",
    )
    features = torch.randn(2, 6, 8, 8)
    labels = torch.randint(0, 2, (2, 1, 8, 8)).float()
    memory.update(features, labels)

    loss, metrics = pixel_to_prototype_contrastive_loss(
        pixel_features=features,
        class_targets=labels,
        prototypes=memory.prototypes,
        initialized=memory.initialized,
        pixel_mask=torch.ones(2, 1, 8, 8),
        temperature=0.1,
        normalize=True,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert metrics["prototype_initialized_classes"] == 2.0
    assert metrics["prototype_contrastive_selected_pixels"] > 0.0


def test_pixel_to_prototype_contrastive_loss_handles_empty_mask() -> None:
    memory = PrototypeMemory(
        feature_dim=4,
        num_classes=2,
        momentum=0.95,
        min_pixels=1,
        normalize=True,
        init_strategy="first_valid_batch",
    )
    features = torch.randn(1, 4, 4, 4)
    labels = torch.randint(0, 2, (1, 1, 4, 4)).float()
    memory.update(features, labels)

    loss, metrics = pixel_to_prototype_contrastive_loss(
        pixel_features=features,
        class_targets=labels,
        prototypes=memory.prototypes,
        initialized=memory.initialized,
        pixel_mask=torch.zeros(1, 1, 4, 4),
        temperature=0.1,
        normalize=True,
    )
    assert torch.isclose(loss, torch.tensor(0.0))
    assert metrics["prototype_contrastive_selected_pixels"] == 0.0
