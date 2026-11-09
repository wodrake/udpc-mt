from __future__ import annotations

import copy

import torch

from losses.consistency import decoupled_masked_mse_consistency_loss, masked_mse_loss
from losses.total_loss import compose_total_loss
from models.ema_teacher import build_ema_teacher
from models.unet import UNet
from udpc_mt.models.unet import UNet as PrototypeUNet
from utils.uncertainty import estimate_teacher_uncertainty


def test_mc_dropout_uncertainty_map_and_masks() -> None:
    student = UNet(in_channels=1, num_classes=1, base_channels=16, dropout=0.3)
    teacher = build_ema_teacher(student)
    images = torch.randn(2, 1, 64, 64)

    def forward_fn(model: torch.nn.Module, batch: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits = model(batch)
        return torch.sigmoid(logits), None

    estimate = estimate_teacher_uncertainty(
        teacher,
        images,
        stochastic_forward=forward_fn,
        mc_passes=4,
        threshold=0.35,
        normalize_feature_mean=False,
    )

    assert estimate.mean_probabilities.shape == (2, 1, 64, 64)
    assert estimate.uncertainty_map.shape == (2, 1, 64, 64)
    assert estimate.low_uncertainty_mask.shape == (2, 1, 64, 64)
    assert estimate.high_uncertainty_mask.shape == (2, 1, 64, 64)
    assert estimate.low_uncertainty_weights.shape == (2, 1, 64, 64)
    assert torch.all((estimate.low_uncertainty_mask + estimate.high_uncertainty_mask) == 1.0)
    assert float(estimate.uncertainty_map.min()) >= 0.0
    assert float(estimate.uncertainty_map.max()) <= 1.0
    assert float(estimate.low_uncertainty_weights.min()) >= 0.0
    assert float(estimate.low_uncertainty_weights.max()) <= 1.0


def test_mc_dropout_uncertainty_supports_feature_outputs() -> None:
    student = PrototypeUNet(
        in_channels=1,
        num_classes=1,
        base_channels=16,
        prototype_dim=24,
        dropout=0.3,
    )
    teacher = copy.deepcopy(student)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.eval()
    images = torch.randn(2, 1, 64, 64)

    def forward_fn(model: torch.nn.Module, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = model(batch)
        return torch.sigmoid(outputs["logits"]), outputs["embeddings"]

    estimate = estimate_teacher_uncertainty(
        teacher,
        images,
        stochastic_forward=forward_fn,
        mc_passes=4,
        threshold=0.35,
    )

    assert estimate.mean_probabilities.shape == (2, 1, 64, 64)
    assert estimate.mean_features is not None
    assert estimate.mean_features.shape == (2, 24, 64, 64)


def test_masked_mse_loss_and_decoupled_region_logic() -> None:
    student_probabilities = torch.tensor([[[[0.2, 0.9]]]], dtype=torch.float32)
    teacher_probabilities = torch.tensor([[[[0.2, 0.1]]]], dtype=torch.float32)
    student_logits = torch.logit(student_probabilities.clamp(1e-4, 1.0 - 1e-4))
    low_mask = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    high_mask = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32)

    masked_loss = masked_mse_loss(student_probabilities, teacher_probabilities, mask=low_mask)
    decoupled_loss, metrics = decoupled_masked_mse_consistency_loss(
        student_logits,
        teacher_probabilities,
        low_uncertainty_mask=low_mask,
        high_uncertainty_mask=high_mask,
    )

    assert masked_loss.ndim == 0
    assert decoupled_loss.ndim == 0
    assert torch.isclose(masked_loss, torch.tensor(0.0))
    assert torch.isclose(decoupled_loss, torch.tensor(0.0))
    assert metrics["low_uncertainty_ratio"] == 0.5
    assert metrics["high_uncertainty_ratio"] == 0.5


def test_total_loss_breakdown_is_reported() -> None:
    supervised_loss = torch.tensor(1.5)
    pixel_consistency_loss = torch.tensor(0.4)
    prototype_loss = torch.tensor(0.2)

    total_loss, metrics = compose_total_loss(
        supervised_loss=supervised_loss,
        pixel_consistency_loss=pixel_consistency_loss,
        pixel_consistency_weight=2.0,
        extra_unsupervised_losses=[("prototype", prototype_loss, 0.5)],
    )

    assert torch.isclose(total_loss, torch.tensor(2.4))
    assert abs(metrics["loss_supervised"] - 1.5) < 1e-6
    assert abs(metrics["loss_pixel_consistency"] - 0.4) < 1e-6
    assert abs(metrics["loss_unsupervised"] - 0.9) < 1e-6
    assert abs(metrics["loss_total"] - 2.4) < 1e-6
