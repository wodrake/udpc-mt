from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def _flatten_metrics(prefix: str, metrics: dict[str, Any] | None) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, bool):
            flattened[f"{prefix}/{key}"] = float(value)
        elif isinstance(value, (int, float)):
            flattened[f"{prefix}/{key}"] = float(value)
    return flattened


class ExperimentLogger:
    enabled = False
    url: str | None = None

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any],
        best_metric: float,
    ) -> None:
        return None

    def log_summary(self, summary: dict[str, Any]) -> None:
        return None

    def finish(self) -> None:
        return None


class NullExperimentLogger(ExperimentLogger):
    pass


class WandbExperimentLogger(ExperimentLogger):
    enabled = True

    def __init__(self, run: Any, logger: logging.Logger, tracked_metric: str) -> None:
        self.run = run
        self.logger = logger
        self.tracked_metric = tracked_metric
        self.url = getattr(run, "url", None)
        if self.url:
            self.logger.info("wandb_url=%s", self.url)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any],
        best_metric: float,
    ) -> None:
        payload = {
            "epoch": epoch,
            f"val/best_{self.tracked_metric}": best_metric,
            **_flatten_metrics("train", train_metrics),
            **_flatten_metrics("val", val_metrics),
        }
        self.run.log(payload, step=epoch)

    def log_summary(self, summary: dict[str, Any]) -> None:
        self.run.summary.update(
            {
                "run_dir": summary.get("run_dir"),
                "best_checkpoint": summary.get("best_checkpoint"),
                "last_checkpoint": summary.get("last_checkpoint"),
                "best_val_metric": summary.get("best_val_metric"),
                **_flatten_metrics("test", summary.get("test_metrics")),
            }
        )

    def finish(self) -> None:
        self.run.finish()


def build_experiment_logger(
    config: dict[str, Any],
    run_dir: str | Path,
    logger: logging.Logger,
) -> ExperimentLogger:
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return NullExperimentLogger()

    try:
        import wandb  # type: ignore
    except ImportError:
        logger.warning("wandb is enabled in config but not installed; continuing without wandb logging.")
        return NullExperimentLogger()

    init_kwargs = {
        "project": wandb_config.get("project") or config.get("project_name", "udpc-mt"),
        "name": wandb_config.get("run_name") or Path(run_dir).name,
        "dir": str(run_dir),
        "config": config,
        "tags": wandb_config.get("tags") or [],
        "reinit": True,
    }
    for optional_key in ("entity", "group", "notes", "mode"):
        optional_value = wandb_config.get(optional_key)
        if optional_value:
            init_kwargs[optional_key] = optional_value

    try:
        wandb.login(timeout=10)
        run = wandb.init(**init_kwargs)
    except Exception as exc:  # pragma: no cover - depends on external wandb state.
        logger.warning("wandb initialization failed (%s); continuing with local logs only.", exc)
        return NullExperimentLogger()
    return WandbExperimentLogger(
        run=run,
        logger=logger,
        tracked_metric=str(config.get("train", {}).get("save_best_metric", "dice")),
    )
