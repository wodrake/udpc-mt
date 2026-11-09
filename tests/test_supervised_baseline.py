from __future__ import annotations

from pathlib import Path

import torch

from models.unet import UNet
from trainers.trainer import SupervisedTrainer


def _build_test_config(dataset_root: str) -> dict:
    return {
        "project_name": "test_supervised_baseline",
        "mode": "baseline",
        "seed": 42,
        "device": "cpu",
        "output_root": "outputs/runs",
        "dataset": {
            "root": dataset_root,
            "image_size": 64,
            "val_fraction": 0.2,
            "test_fraction": 0.1,
            "labeled_fraction": 1.0,
            "include_normal_as_unlabeled": False,
            "num_workers": 0,
            "sample_limit": None,
        },
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "labeled_batch_size": 4,
            "max_steps_per_epoch": 1,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "log_interval": 10,
            "save_best_metric": "dice",
        },
        "model": {
            "in_channels": 1,
            "num_classes": 1,
            "base_channels": 16,
            "dropout": 0.1,
        },
        "loss": {
            "bce_weight": 0.5,
            "dice_weight": 0.5,
        },
    }


def test_unet_forward_shape() -> None:
    model = UNet(in_channels=1, num_classes=1, base_channels=16, dropout=0.1)
    inputs = torch.randn(2, 1, 64, 64)
    logits = model(inputs)
    assert logits.shape == (2, 1, 64, 64)


def test_one_step_supervised_training(tmp_path: Path) -> None:
    config = _build_test_config("/home/jett/code/udpc-mt/data/toy_busi")
    trainer = SupervisedTrainer(
        config,
        run_dir=tmp_path / "supervised_one_step",
        logger_name="supervised_one_step_test",
    )
    train_metrics = trainer.train_one_epoch(epoch=1)
    assert "loss_total" in train_metrics
    assert "dice" in train_metrics
