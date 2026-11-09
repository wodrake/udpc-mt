from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from datasets import create_dataloaders
from losses.consistency import decoupled_masked_mse_consistency_loss
from losses.supervised import probabilities_from_logits, supervised_segmentation_loss
from losses.total_loss import compose_total_loss
from models.ema_teacher import build_ema_teacher, update_ema_teacher
from models.unet import build_unet
from udpc_mt.utils.config import dump_config
from udpc_mt.utils.experiment_logger import build_experiment_logger
from udpc_mt.utils.misc import ScalarTracker, cycle_loader, ensure_dir, json_dump, set_seed, timestamp
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_segmentation_metrics, resolve_metric_names
from utils.uncertainty import estimate_teacher_uncertainty


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "cpu"))
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def make_run_dir(config: dict[str, Any], run_dir: str | Path | None) -> Path:
    if run_dir is None:
        run_dir = Path(config["output_root"]) / f"{config['project_name']}_{timestamp()}"
    run_path = Path(run_dir)
    ensure_dir(run_path / "checkpoints")
    return run_path


def resolve_run_dir(run_dir: str | Path | None, resume_path: str | Path | None) -> str | Path | None:
    if run_dir is not None:
        return run_dir
    if resume_path is None:
        return None
    return Path(resume_path).resolve().parent.parent


def build_logger(log_path: Path, logger_name: str) -> logging.Logger:
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


def get_uncertainty_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get(
        "uncertainty",
        {
            "enabled": False,
            "mc_passes": 1,
            "threshold": 0.35,
        },
    )


class SupervisedTrainer:
    def __init__(
        self,
        config: dict[str, Any],
        run_dir: str | Path | None = None,
        resume_path: str | Path | None = None,
        logger_name: str = "supervised_trainer",
    ) -> None:
        self.config = config
        self.resume_path = Path(resume_path).resolve() if resume_path is not None else None
        set_seed(config["seed"])
        self.device = resolve_device(config)
        self.run_dir = make_run_dir(config, resolve_run_dir(run_dir, self.resume_path))
        dump_config(config, self.run_dir / "config_resolved.yaml")
        self.logger = build_logger(self.run_dir / "train.log", logger_name=logger_name)
        self.experiment_logger = build_experiment_logger(config, self.run_dir, self.logger)

        self.loaders, self.split_items = create_dataloaders(config)
        json_dump(
            {split: [item["case_id"] for item in items] for split, items in self.split_items.items()},
            self.run_dir / "splits.json",
        )

        self.model = build_unet(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"],
        )
        self.history: list[dict[str, Any]] = []
        self.best_metric = float("-inf")
        self.start_epoch = 1
        self.metric_names = resolve_metric_names(config)

        if self.resume_path is not None:
            self._resume_from_checkpoint()

    def _compute_loss(self, logits: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        return supervised_segmentation_loss(
            logits,
            masks,
            bce_weight=self.config["loss"]["bce_weight"],
            dice_weight=self.config["loss"]["dice_weight"],
        )

    def _resume_from_checkpoint(self) -> None:
        state = load_checkpoint(
            self.resume_path,
            self.model,
            optimizer=self.optimizer,
            map_location=self.device,
        )
        self.history = list(state.get("history", []))
        self.best_metric = float(
            state.get(
                "best_metric",
                state.get("metrics", {}).get(self.config["train"]["save_best_metric"], float("-inf")),
            )
        )
        self.start_epoch = int(state["epoch"]) + 1
        self.logger.info(
            "resumed checkpoint=%s start_epoch=%s best_metric=%s",
            self.resume_path,
            self.start_epoch,
            self.best_metric,
        )

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        tracker = ScalarTracker()
        loader = self.loaders["labeled_train"]
        max_steps = self.config["train"]["max_steps_per_epoch"] or len(loader)
        progress = tqdm(total=max_steps, desc=f"Epoch {epoch:03d}")

        for step, batch in enumerate(loader, start=1):
            if step > max_steps:
                break
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            logits = self.model(images)
            loss, loss_parts = self._compute_loss(logits, masks)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            metrics = compute_segmentation_metrics(logits.detach(), masks, metric_names=self.metric_names)
            tracker.update({"loss_total": loss.item(), **loss_parts, **metrics}, weight=images.size(0))
            progress.update()
        progress.close()
        return tracker.average()

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> dict[str, float]:
        self.model.eval()
        tracker = ScalarTracker()
        loader = self.loaders[split]
        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            logits = self.model(images)
            loss, _ = self._compute_loss(logits, masks)
            metrics = compute_segmentation_metrics(logits, masks, metric_names=self.metric_names)
            tracker.update({"loss_total": loss.item(), **metrics}, weight=images.size(0))

        metrics = tracker.average()
        if metrics:
            return metrics
        return {
            "loss_total": 0.0,
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
        }

    def fit(self) -> dict[str, Any]:
        best_checkpoint = self.run_dir / "checkpoints" / "best.pt"
        last_checkpoint = self.run_dir / "checkpoints" / "last.pt"
        try:
            for epoch in range(self.start_epoch, self.config["train"]["epochs"] + 1):
                train_metrics = self.train_one_epoch(epoch)
                val_metrics = self.evaluate(split="val")
                record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
                self.history.append(record)
                json_dump(self.history, self.run_dir / "history.json")
                self.logger.info("epoch=%s train=%s val=%s", epoch, train_metrics, val_metrics)

                tracked_metric = val_metrics[self.config["train"]["save_best_metric"]]
                if tracked_metric >= self.best_metric:
                    self.best_metric = tracked_metric
                    save_checkpoint(
                        best_checkpoint,
                        config=self.config,
                        epoch=epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        metrics=val_metrics,
                        extra_state={"best_metric": self.best_metric, "history": self.history},
                    )
                save_checkpoint(
                    last_checkpoint,
                    config=self.config,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    metrics=val_metrics,
                    extra_state={"best_metric": self.best_metric, "history": self.history},
                )
                self.experiment_logger.log_epoch(epoch, train_metrics, val_metrics, self.best_metric)

            load_checkpoint(best_checkpoint, self.model, map_location=self.device)
            test_metrics = self.evaluate(split="test")
            summary = {
                "run_dir": str(self.run_dir),
                "best_checkpoint": str(best_checkpoint),
                "last_checkpoint": str(last_checkpoint),
                "best_val_metric": self.best_metric,
                "test_metrics": test_metrics,
            }
            if self.experiment_logger.url:
                summary["wandb_url"] = self.experiment_logger.url
            json_dump(summary, self.run_dir / "summary.json")
            self.experiment_logger.log_summary(summary)
            self.logger.info("summary=%s", summary)
            return summary
        finally:
            self.experiment_logger.finish()


class MeanTeacherTrainer:
    def __init__(
        self,
        config: dict[str, Any],
        run_dir: str | Path | None = None,
        resume_path: str | Path | None = None,
        logger_name: str = "mean_teacher_trainer",
    ) -> None:
        self.config = config
        self.resume_path = Path(resume_path).resolve() if resume_path is not None else None
        set_seed(config["seed"])
        self.device = resolve_device(config)
        self.run_dir = make_run_dir(config, resolve_run_dir(run_dir, self.resume_path))
        dump_config(config, self.run_dir / "config_resolved.yaml")
        self.logger = build_logger(self.run_dir / "train.log", logger_name=logger_name)
        self.experiment_logger = build_experiment_logger(config, self.run_dir, self.logger)

        self.loaders, self.split_items = create_dataloaders(config)
        if self.loaders["unlabeled_train"] is None:
            raise ValueError("Mean Teacher requires an unlabeled training loader.")
        json_dump(
            {split: [item["case_id"] for item in items] for split, items in self.split_items.items()},
            self.run_dir / "splits.json",
        )

        self.student = build_unet(config).to(self.device)
        self.teacher = build_ema_teacher(self.student).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"],
        )
        self.history: list[dict[str, Any]] = []
        self.best_metric = float("-inf")
        self.start_epoch = 1
        self.metric_names = resolve_metric_names(config)

        if self.resume_path is not None:
            self._resume_from_checkpoint()

    def _compute_supervised_loss(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return supervised_segmentation_loss(
            logits,
            masks,
            bce_weight=self.config["loss"]["bce_weight"],
            dice_weight=self.config["loss"]["dice_weight"],
        )

    def _resume_from_checkpoint(self) -> None:
        state = load_checkpoint(
            self.resume_path,
            self.student,
            optimizer=self.optimizer,
            teacher=self.teacher,
            map_location=self.device,
        )
        self.history = list(state.get("history", []))
        self.best_metric = float(
            state.get(
                "best_metric",
                state.get("metrics", {}).get(self.config["train"]["save_best_metric"], float("-inf")),
            )
        )
        self.start_epoch = int(state["epoch"]) + 1
        self.logger.info(
            "resumed checkpoint=%s start_epoch=%s best_metric=%s",
            self.resume_path,
            self.start_epoch,
            self.best_metric,
        )

    @staticmethod
    def _teacher_forward(
        teacher: torch.nn.Module,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        logits = teacher(images)
        return probabilities_from_logits(logits), None

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.student.train()
        self.teacher.eval()
        tracker = ScalarTracker()
        uncertainty_cfg = get_uncertainty_config(self.config)
        uncertainty_enabled = bool(uncertainty_cfg["enabled"])
        labeled_iter = cycle_loader(self.loaders["labeled_train"])
        unlabeled_iter = cycle_loader(self.loaders["unlabeled_train"])
        max_steps = self.config["train"]["max_steps_per_epoch"] or max(
            len(self.loaders["labeled_train"]),
            len(self.loaders["unlabeled_train"]),
        )
        progress = tqdm(total=max_steps, desc=f"Epoch {epoch:03d}")

        for _ in range(max_steps):
            labeled_batch = next(labeled_iter)
            unlabeled_batch = next(unlabeled_iter)

            labeled_images = labeled_batch["image"].to(self.device)
            labeled_masks = labeled_batch["mask"].to(self.device)
            teacher_images = unlabeled_batch["teacher_image"].to(self.device)
            student_images = unlabeled_batch["student_image"].to(self.device)

            labeled_logits = self.student(labeled_images)
            loss_sup, sup_terms = self._compute_supervised_loss(labeled_logits, labeled_masks)

            student_unlabeled_logits = self.student(student_images)
            with torch.no_grad():
                uncertainty_estimate = estimate_teacher_uncertainty(
                    self.teacher,
                    teacher_images,
                    stochastic_forward=self._teacher_forward,
                    mc_passes=uncertainty_cfg["mc_passes"] if uncertainty_enabled else 1,
                    threshold=uncertainty_cfg["threshold"],
                    normalize_feature_mean=False,
                )
                teacher_probabilities = uncertainty_estimate.mean_probabilities
            low_uncertainty_mask = (
                uncertainty_estimate.low_uncertainty_mask if uncertainty_enabled else None
            )
            loss_pixel_consistency, consistency_metrics = decoupled_masked_mse_consistency_loss(
                student_unlabeled_logits,
                teacher_probabilities,
                low_uncertainty_mask=low_uncertainty_mask,
                high_uncertainty_mask=uncertainty_estimate.high_uncertainty_mask if uncertainty_enabled else None,
            )
            total_loss, total_loss_metrics = compose_total_loss(
                supervised_loss=loss_sup,
                pixel_consistency_loss=loss_pixel_consistency,
                pixel_consistency_weight=self.config["train"]["unsupervised_weight"],
            )

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            self.optimizer.step()
            update_ema_teacher(self.teacher, self.student, self.config["train"]["ema_decay"])

            metrics = compute_segmentation_metrics(
                labeled_logits.detach(),
                labeled_masks,
                metric_names=self.metric_names,
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
        return tracker.average()

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> dict[str, float]:
        self.student.eval()
        tracker = ScalarTracker()
        loader = self.loaders[split]
        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            logits = self.student(images)
            loss, _ = self._compute_supervised_loss(logits, masks)
            metrics = compute_segmentation_metrics(logits, masks, metric_names=self.metric_names)
            tracker.update({"loss_total": loss.item(), **metrics}, weight=images.size(0))

        metrics = tracker.average()
        if metrics:
            return metrics
        return {
            "loss_total": 0.0,
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
        }

    def fit(self) -> dict[str, Any]:
        best_checkpoint = self.run_dir / "checkpoints" / "best.pt"
        last_checkpoint = self.run_dir / "checkpoints" / "last.pt"
        try:
            for epoch in range(self.start_epoch, self.config["train"]["epochs"] + 1):
                train_metrics = self.train_one_epoch(epoch)
                val_metrics = self.evaluate(split="val")
                record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
                self.history.append(record)
                json_dump(self.history, self.run_dir / "history.json")
                self.logger.info("epoch=%s train=%s val=%s", epoch, train_metrics, val_metrics)

                tracked_metric = val_metrics[self.config["train"]["save_best_metric"]]
                if tracked_metric >= self.best_metric:
                    self.best_metric = tracked_metric
                    save_checkpoint(
                        best_checkpoint,
                        config=self.config,
                        epoch=epoch,
                        model=self.student,
                        teacher=self.teacher,
                        optimizer=self.optimizer,
                        metrics=val_metrics,
                        extra_state={"best_metric": self.best_metric, "history": self.history},
                    )
                save_checkpoint(
                    last_checkpoint,
                    config=self.config,
                    epoch=epoch,
                    model=self.student,
                    teacher=self.teacher,
                    optimizer=self.optimizer,
                    metrics=val_metrics,
                    extra_state={"best_metric": self.best_metric, "history": self.history},
                )
                self.experiment_logger.log_epoch(epoch, train_metrics, val_metrics, self.best_metric)

            load_checkpoint(best_checkpoint, self.student, teacher=self.teacher, map_location=self.device)
            test_metrics = self.evaluate(split="test")
            summary = {
                "run_dir": str(self.run_dir),
                "best_checkpoint": str(best_checkpoint),
                "last_checkpoint": str(last_checkpoint),
                "best_val_metric": self.best_metric,
                "test_metrics": test_metrics,
            }
            if self.experiment_logger.url:
                summary["wandb_url"] = self.experiment_logger.url
            json_dump(summary, self.run_dir / "summary.json")
            self.experiment_logger.log_summary(summary)
            self.logger.info("summary=%s", summary)
            return summary
        finally:
            self.experiment_logger.finish()


def run_supervised_training(
    config: dict[str, Any],
    run_dir: str | Path | None = None,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    trainer = SupervisedTrainer(
        config,
        run_dir=run_dir,
        resume_path=resume_path,
        logger_name=f"supervised_train_{timestamp()}",
    )
    return trainer.fit()


def run_mean_teacher_training(
    config: dict[str, Any],
    run_dir: str | Path | None = None,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    trainer = MeanTeacherTrainer(
        config,
        run_dir=run_dir,
        resume_path=resume_path,
        logger_name=f"mean_teacher_train_{timestamp()}",
    )
    return trainer.fit()


def run_training(
    config: dict[str, Any],
    run_dir: str | Path | None = None,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    mode = config["mode"]
    if mode == "baseline":
        return run_supervised_training(config, run_dir=run_dir, resume_path=resume_path)
    if mode == "mean_teacher":
        return run_mean_teacher_training(config, run_dir=run_dir, resume_path=resume_path)
    raise ValueError(f"Unsupported root-level training mode: {mode}")


def validate_checkpoint(
    checkpoint_path: str | Path,
    split: str = "test",
    config_override: dict[str, Any] | None = None,
) -> dict[str, float]:
    state = torch.load(checkpoint_path, map_location="cpu")
    config = config_override or state["config"]
    if config["mode"] == "baseline":
        trainer = SupervisedTrainer(
            config,
            run_dir=Path(checkpoint_path).resolve().parent.parent.parent / "val_tmp",
            logger_name=f"supervised_val_{timestamp()}",
        )
        load_checkpoint(checkpoint_path, trainer.model, map_location=trainer.device)
        return trainer.evaluate(split=split)
    if config["mode"] == "mean_teacher":
        trainer = MeanTeacherTrainer(
            config,
            run_dir=Path(checkpoint_path).resolve().parent.parent.parent / "val_tmp",
            logger_name=f"mean_teacher_val_{timestamp()}",
        )
        load_checkpoint(checkpoint_path, trainer.student, teacher=trainer.teacher, map_location=trainer.device)
        return trainer.evaluate(split=split)
    raise ValueError(f"Unsupported validation mode: {config['mode']}")
