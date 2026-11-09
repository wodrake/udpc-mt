from __future__ import annotations

from pathlib import Path

import torch

from datasets.busi import create_dataloaders
from models.ema_teacher import build_ema_teacher, update_ema_teacher
from models.unet import UNet
from trainers.trainer import MeanTeacherTrainer


def _build_mt_config(dataset_root: str) -> dict:
    return {
        "project_name": "test_mean_teacher",
        "mode": "mean_teacher",
        "seed": 42,
        "device": "cpu",
        "output_root": "outputs/runs",
        "dataset": {
            "root": dataset_root,
            "image_size": 64,
            "val_fraction": 0.2,
            "test_fraction": 0.1,
            "labeled_fraction": 0.5,
            "include_normal_as_unlabeled": True,
            "num_workers": 0,
            "sample_limit": None,
        },
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "labeled_batch_size": 2,
            "max_steps_per_epoch": 1,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "ema_decay": 0.99,
            "unsupervised_weight": 1.0,
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
        "uncertainty": {
            "enabled": True,
            "mc_passes": 4,
            "threshold": 0.35,
        },
    }


def test_student_teacher_forward_and_ema_update() -> None:
    student = UNet(in_channels=1, num_classes=1, base_channels=16, dropout=0.1)
    teacher = build_ema_teacher(student)
    student_inputs = torch.randn(2, 1, 64, 64)
    teacher_inputs = torch.randn(2, 1, 64, 64)

    student_logits = student(student_inputs)
    teacher_logits = teacher(teacher_inputs)
    assert student_logits.shape == (2, 1, 64, 64)
    assert teacher_logits.shape == (2, 1, 64, 64)

    student_parameter = next(student.parameters())
    teacher_before = next(teacher.parameters()).detach().clone()
    with torch.no_grad():
        student_parameter.add_(1.0)
    distance_before = (teacher_before - student_parameter).abs().mean()
    update_ema_teacher(teacher, student, decay=0.9)
    teacher_after = next(teacher.parameters()).detach().clone()
    distance_after = (teacher_after - student_parameter).abs().mean()
    assert distance_after < distance_before


def test_unlabeled_batch_and_one_step_mean_teacher_training(tmp_path: Path) -> None:
    config = _build_mt_config("/home/jett/code/udpc-mt/data/toy_busi")
    loaders, _ = create_dataloaders(config)
    unlabeled_batch = next(iter(loaders["unlabeled_train"]))
    assert "teacher_image" in unlabeled_batch
    assert "student_image" in unlabeled_batch
    assert "mask" not in unlabeled_batch
    assert unlabeled_batch["teacher_image"].shape == unlabeled_batch["student_image"].shape

    trainer = MeanTeacherTrainer(
        config,
        run_dir=tmp_path / "mean_teacher_one_step",
        logger_name="mean_teacher_one_step_test",
    )
    train_metrics = trainer.train_one_epoch(epoch=1)
    assert "loss_total" in train_metrics
    assert "loss_supervised" in train_metrics
    assert "loss_pixel_consistency" in train_metrics
    assert "loss_unsupervised" in train_metrics
    assert "loss_consistency" in train_metrics


def test_resume_restores_teacher_ema_state(tmp_path: Path) -> None:
    base_config = _build_mt_config("/home/jett/code/udpc-mt/data/toy_busi")
    base_config["train"]["epochs"] = 1
    base_run_dir = tmp_path / "mean_teacher_resume"

    initial_trainer = MeanTeacherTrainer(
        base_config,
        run_dir=base_run_dir,
        logger_name="mean_teacher_resume_initial",
    )
    initial_summary = initial_trainer.fit()
    last_checkpoint = Path(initial_summary["last_checkpoint"])
    checkpoint_state = torch.load(last_checkpoint, map_location="cpu")
    saved_teacher_tensor = checkpoint_state["teacher_model"]["enc1.conv.block.0.weight"].clone()

    resumed_config = _build_mt_config("/home/jett/code/udpc-mt/data/toy_busi")
    resumed_config["train"]["epochs"] = 2
    resumed_trainer = MeanTeacherTrainer(
        resumed_config,
        run_dir=base_run_dir,
        resume_path=last_checkpoint,
        logger_name="mean_teacher_resume_continue",
    )

    restored_teacher_tensor = resumed_trainer.teacher.state_dict()["enc1.conv.block.0.weight"].clone()
    assert torch.allclose(restored_teacher_tensor, saved_teacher_tensor)
    assert resumed_trainer.start_epoch == 2
    assert len(resumed_trainer.history) == 1

    resumed_summary = resumed_trainer.fit()
    resumed_last_state = torch.load(resumed_summary["last_checkpoint"], map_location="cpu")
    assert resumed_last_state["epoch"] == 2
    assert len(resumed_last_state["history"]) == 2
