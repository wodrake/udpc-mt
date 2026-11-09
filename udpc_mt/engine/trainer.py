from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from losses.consistency import decoupled_masked_mse_consistency_loss
from losses.prototype import pixel_to_prototype_contrastive_loss
from losses.supervised import hard_labels_from_probabilities, probabilities_from_logits, targets_to_channel_label_map
from losses.total_loss import compose_total_loss
from datasets import create_dataloaders
from udpc_mt.engine.losses import (
    supervised_segmentation_loss,
)
from udpc_mt.engine.prototypes import PrototypeBank
from udpc_mt.models.unet import build_model
from udpc_mt.utils.config import dump_config
from udpc_mt.utils.experiment_logger import build_experiment_logger
from udpc_mt.utils.metrics import compute_segmentation_metrics, resolve_metric_names
from udpc_mt.utils.misc import (
    ScalarTracker,
    cycle_loader,
    ensure_dir,
    json_dump,
    set_seed,
    sigmoid_rampup,
    timestamp,
    update_ema,
)
from utils.uncertainty import estimate_teacher_uncertainty


def resolve_device(config: dict) -> torch.device:
    requested = str(config.get("device", "cpu"))
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _build_logger(log_path: Path, logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def teacher_forward(
    teacher: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = teacher(images)
    return probabilities_from_logits(outputs["logits"]), outputs["embeddings"]


def evaluate(model: torch.nn.Module, loader, device: torch.device, config: dict) -> dict[str, float]:
    model.eval()
    tracker = ScalarTracker()
    metric_names = resolve_metric_names(config)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)
            loss, _ = supervised_segmentation_loss(outputs["logits"], masks, config)
            metrics = compute_segmentation_metrics(outputs["logits"], masks, metric_names=metric_names)
            tracker.update({"loss": loss.item(), **metrics}, weight=images.size(0))
    metrics = tracker.average()
    if metrics:
        return metrics
    return {
        "loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "sensitivity": 0.0,
        "specificity": 0.0,
    }


def _make_run_dir(config: dict, run_dir: str | Path | None) -> Path:
    if run_dir is None:
        run_dir = Path(config["output_root"]) / f"{config['project_name']}_{timestamp()}"
    run_dir = Path(run_dir)
    ensure_dir(run_dir / "checkpoints")
    return run_dir


def _resolve_run_dir(run_dir: str | Path | None, resume_path: str | Path | None) -> str | Path | None:
    if run_dir is not None:
        return run_dir
    if resume_path is None:
        return None
    return Path(resume_path).resolve().parent.parent


def _save_checkpoint(
    path: Path,
    config: dict,
    epoch: int,
    student: torch.nn.Module,
    teacher: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    prototype_bank: PrototypeBank | None,
    metrics: dict[str, float],
    history: list[dict] | None = None,
    best_metric: float | None = None,
) -> None:
    torch.save(
        {
            "config": config,
            "epoch": epoch,
            "student_model": student.state_dict(),
            "teacher_model": teacher.state_dict() if teacher is not None else None,
            "optimizer": optimizer.state_dict(),
            "prototype_bank": prototype_bank.state_dict() if prototype_bank is not None else None,
            "metrics": metrics,
            "history": history or [],
            "best_metric": float("-inf") if best_metric is None else float(best_metric),
        },
        path,
    )


def _load_checkpoint(
    path: str | Path,
    student: torch.nn.Module,
    teacher: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    prototype_bank: PrototypeBank | None,
    device: torch.device,
) -> dict:
    state = torch.load(path, map_location=device)
    if "student_model" in state:
        student.load_state_dict(state["student_model"])
    elif "model" in state:
        student.load_state_dict(state["model"])
    else:
        raise KeyError("Checkpoint does not contain a student model state.")
    if teacher is not None and state.get("teacher_model") is not None:
        teacher.load_state_dict(state["teacher_model"])
    if state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if prototype_bank is not None and state.get("prototype_bank") is not None:
        prototype_bank.load_state_dict(state["prototype_bank"])
    return state


def run_training(
    config: dict,
    run_dir: str | Path | None = None,
    resume_path: str | Path | None = None,
) -> dict:
    set_seed(config["seed"])
    device = resolve_device(config)
    resume_path = Path(resume_path).resolve() if resume_path is not None else None
    run_dir = _make_run_dir(config, _resolve_run_dir(run_dir, resume_path))
    dump_config(config, run_dir / "config_resolved.yaml")
    logger = _build_logger(run_dir / "train.log", logger_name=f"udpc_mt_train_{timestamp()}")
    experiment_logger = build_experiment_logger(config, run_dir, logger)

    loaders, split_items = create_dataloaders(config)
    json_dump(
        {split_name: [item["case_id"] for item in items] for split_name, items in split_items.items()},
        run_dir / "splits.json",
    )

    student = build_model(config).to(device)
    teacher = None
    if config["mode"] != "baseline":
        teacher = copy.deepcopy(student).to(device)
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        teacher.eval()

    prototype_bank = None
    if config["mode"] == "udpc_mt" and config["prototype"]["enabled"]:
        prototype_bank = PrototypeBank(
            feature_dim=config["model"]["prototype_dim"],
            num_classes=config["prototype"]["num_classes"],
            momentum=config["prototype"]["momentum"],
            min_pixels=config["prototype"]["min_pixels"],
            normalize=config["prototype"]["normalize"],
            init_strategy=config["prototype"]["init_strategy"],
        ).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    history = []
    best_metric = float("-inf")
    start_epoch = 1
    metric_names = resolve_metric_names(config)
    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    last_checkpoint = run_dir / "checkpoints" / "last.pt"

    if resume_path is not None:
        state = _load_checkpoint(
            resume_path,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            prototype_bank=prototype_bank,
            device=device,
        )
        history = list(state.get("history", []))
        best_metric = float(
            state.get(
                "best_metric",
                state.get("metrics", {}).get(config["train"]["save_best_metric"], float("-inf")),
            )
        )
        start_epoch = int(state["epoch"]) + 1
        logger.info(
            "resumed checkpoint=%s start_epoch=%s best_metric=%s",
            resume_path,
            start_epoch,
            best_metric,
        )

    try:
        for epoch in range(start_epoch, config["train"]["epochs"] + 1):
            student.train()
            tracker = ScalarTracker()
            uncertainty_cfg = config.get("uncertainty", {"enabled": False, "mc_passes": 1, "threshold": 0.35})
            uncertainty_enabled = bool(uncertainty_cfg["enabled"])

            if config["mode"] == "baseline":
                labeled_loader = loaders["labeled_train"]
                max_steps = config["train"]["max_steps_per_epoch"] or len(labeled_loader)
                progress = tqdm(disable=False, total=max_steps, desc=f"Epoch {epoch:03d}")
                for step, batch in enumerate(labeled_loader, start=1):
                    if step > max_steps:
                        break
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device)
                    outputs = student(images)
                    loss_sup, sup_terms = supervised_segmentation_loss(outputs["logits"], masks, config)
                    optimizer.zero_grad(set_to_none=True)
                    loss_sup.backward()
                    optimizer.step()
                    metrics = compute_segmentation_metrics(
                        outputs["logits"].detach(),
                        masks,
                        metric_names=metric_names,
                    )
                    tracker.update({"loss_total": loss_sup.item(), **sup_terms, **metrics}, weight=images.size(0))
                    progress.update()
            else:
                labeled_iter = cycle_loader(loaders["labeled_train"])
                unlabeled_iter = cycle_loader(loaders["unlabeled_train"])
                max_steps = config["train"]["max_steps_per_epoch"] or max(
                    len(loaders["labeled_train"]), len(loaders["unlabeled_train"])
                )
                progress = tqdm(disable=False, total=max_steps, desc=f"Epoch {epoch:03d}")
                for step in range(1, max_steps + 1):
                    labeled_batch = next(labeled_iter)
                    unlabeled_batch = next(unlabeled_iter)

                    labeled_images = labeled_batch["image"].to(device)
                    labeled_masks = labeled_batch["mask"].to(device)
                    teacher_images = unlabeled_batch["teacher_image"].to(device)
                    student_images = unlabeled_batch["student_image"].to(device)

                    optimizer.zero_grad(set_to_none=True)

                    student_labeled = student(labeled_images)
                    loss_sup, sup_terms = supervised_segmentation_loss(
                        student_labeled["logits"], labeled_masks, config
                    )

                    student_unlabeled = student(student_images)

                    with torch.no_grad():
                        uncertainty_estimate = estimate_teacher_uncertainty(
                            teacher,
                            teacher_images,
                            stochastic_forward=teacher_forward,
                            mc_passes=uncertainty_cfg["mc_passes"] if uncertainty_enabled else 1,
                            threshold=uncertainty_cfg["threshold"],
                        )
                        teacher_probs = uncertainty_estimate.mean_probabilities
                        teacher_unlabeled_features = uncertainty_estimate.mean_features
                        teacher_labeled_outputs = teacher(labeled_images)

                    consistency_weights = (
                        uncertainty_estimate.low_uncertainty_mask if uncertainty_enabled else None
                    )
                    prototype_contrastive_mask = (
                        uncertainty_estimate.high_uncertainty_mask
                        if uncertainty_enabled
                        else torch.ones_like(teacher_probs[:, :1])
                    )
                    prototype_update_mask = (
                        uncertainty_estimate.low_uncertainty_mask
                        if uncertainty_enabled
                        else torch.ones_like(teacher_probs[:, :1])
                    )

                    ramp = sigmoid_rampup(
                        (epoch - 1) + step / max(max_steps, 1),
                        config["train"]["rampup_epochs"],
                    )
                    loss_pixel_consistency, consistency_metrics = decoupled_masked_mse_consistency_loss(
                        student_unlabeled["logits"],
                        teacher_probs,
                        low_uncertainty_mask=consistency_weights,
                        high_uncertainty_mask=uncertainty_estimate.high_uncertainty_mask if uncertainty_enabled else None,
                    )

                    loss_proto = student_labeled["logits"].new_tensor(0.0)
                    prototype_loss_metrics: dict[str, float] = {}
                    if prototype_bank is not None:
                        labeled_label_maps = targets_to_channel_label_map(labeled_masks)
                        pseudo_labels = hard_labels_from_probabilities(teacher_probs)
                        loss_proto, prototype_loss_metrics = pixel_to_prototype_contrastive_loss(
                            student_unlabeled["embeddings"],
                            pseudo_labels,
                            prototypes=prototype_bank.prototypes,
                            initialized=prototype_bank.initialized,
                            pixel_mask=prototype_contrastive_mask,
                            temperature=config["prototype"]["temperature"],
                            normalize=config["prototype"]["normalize"],
                        )
                        teacher_features = torch.cat(
                            [teacher_labeled_outputs["embeddings"], teacher_unlabeled_features], dim=0
                        )
                        prototype_labels = torch.cat(
                            [labeled_label_maps, pseudo_labels],
                            dim=0,
                        )
                        prototype_weights = torch.cat(
                            [
                                torch.ones_like(labeled_label_maps, dtype=prototype_update_mask.dtype),
                                prototype_update_mask,
                            ],
                            dim=0,
                        )
                    total_loss, total_loss_metrics = compose_total_loss(
                        supervised_loss=loss_sup,
                        pixel_consistency_loss=loss_pixel_consistency,
                        pixel_consistency_weight=config["loss"]["consistency_weight"] * ramp,
                        extra_unsupervised_losses=[
                            ("prototype", loss_proto, config["loss"]["prototype_weight"] * ramp)
                        ],
                    )

                    total_loss.backward()
                    optimizer.step()
                    update_ema(teacher, student, config["train"]["ema_decay"])

                    if prototype_bank is not None:
                        with torch.no_grad():
                            prototype_metrics = prototype_bank.update(
                                teacher_features,
                                prototype_labels,
                                prototype_weights,
                            )
                            tracker.update(
                                {**prototype_metrics, **prototype_loss_metrics},
                                weight=labeled_images.size(0),
                            )

                    metrics = compute_segmentation_metrics(
                        student_labeled["logits"].detach(),
                        labeled_masks,
                        metric_names=metric_names,
                    )
                    tracker.update(
                        {
                            **total_loss_metrics,
                            "uncertainty_mean": uncertainty_estimate.uncertainty_map.mean().item(),
                            **consistency_metrics,
                            **sup_terms,
                            **metrics,
                        },
                        weight=labeled_images.size(0),
                    )
                    progress.update()

            progress.close()
            train_metrics = tracker.average()
            val_metrics = evaluate(student, loaders["val"], device, config)
            epoch_record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(epoch_record)
            json_dump(history, run_dir / "history.json")
            logger.info("epoch=%s train=%s val=%s", epoch, train_metrics, val_metrics)

            tracked_metric = val_metrics[config["train"]["save_best_metric"]]
            if tracked_metric >= best_metric:
                best_metric = tracked_metric
                _save_checkpoint(
                    best_checkpoint,
                    config,
                    epoch,
                    student,
                    teacher,
                    optimizer,
                    prototype_bank,
                    val_metrics,
                    history=history,
                    best_metric=best_metric,
                )
            _save_checkpoint(
                last_checkpoint,
                config,
                epoch,
                student,
                teacher,
                optimizer,
                prototype_bank,
                val_metrics,
                history=history,
                best_metric=best_metric,
            )
            experiment_logger.log_epoch(epoch, train_metrics, val_metrics, best_metric)

        best_state = torch.load(best_checkpoint, map_location=device)
        student.load_state_dict(best_state["student_model"])
        test_metrics = evaluate(student, loaders["test"], device, config)
        summary = {
            "run_dir": str(run_dir),
            "best_val_metric": best_metric,
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "test_metrics": test_metrics,
        }
        if experiment_logger.url:
            summary["wandb_url"] = experiment_logger.url
        json_dump(summary, run_dir / "summary.json")
        experiment_logger.log_summary(summary)
        logger.info("summary=%s", summary)
        return summary
    finally:
        experiment_logger.finish()
