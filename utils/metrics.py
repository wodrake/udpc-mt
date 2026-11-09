from __future__ import annotations

import torch

from losses.supervised import is_binary_segmentation, probabilities_from_logits, targets_to_label_map


DEFAULT_METRIC_NAMES = (
    "dice",
    "iou",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
)


def _compute_binary_metrics_from_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, float]:
    dims = (1, 2, 3)
    true_positive = (predictions * targets).sum(dim=dims)
    false_positive = (predictions * (1.0 - targets)).sum(dim=dims)
    false_negative = ((1.0 - predictions) * targets).sum(dim=dims)
    true_negative = ((1.0 - predictions) * (1.0 - targets)).sum(dim=dims)

    prediction_sum = predictions.sum(dim=dims)
    target_sum = targets.sum(dim=dims)
    both_empty = (prediction_sum == 0) & (target_sum == 0)

    dice = (2.0 * true_positive + eps) / (2.0 * true_positive + false_positive + false_negative + eps)
    iou = (true_positive + eps) / (true_positive + false_positive + false_negative + eps)
    precision = torch.where(
        true_positive + false_positive > 0,
        (true_positive + eps) / (true_positive + false_positive + eps),
        torch.ones_like(true_positive),
    )
    recall = torch.where(
        true_positive + false_negative > 0,
        (true_positive + eps) / (true_positive + false_negative + eps),
        torch.ones_like(true_positive),
    )
    specificity = torch.where(
        true_negative + false_positive > 0,
        (true_negative + eps) / (true_negative + false_positive + eps),
        torch.ones_like(true_positive),
    )

    dice = torch.where(both_empty, torch.ones_like(dice), dice)
    iou = torch.where(both_empty, torch.ones_like(iou), iou)
    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "sensitivity": recall.mean().item(),
        "specificity": specificity.mean().item(),
    }


def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    include_background: bool = False,
    metric_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    probabilities = probabilities_from_logits(logits)
    if is_binary_segmentation(logits):
        predictions = (probabilities >= threshold).float()
        targets = (targets >= 0.5).float()
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        metrics = _compute_binary_metrics_from_predictions(predictions, targets, eps=eps)
        return filter_metrics(metrics, metric_names)

    predictions = probabilities.argmax(dim=1)
    labels = targets_to_label_map(targets)
    class_range = range(0 if include_background else 1, logits.shape[1])
    metric_list = []
    for class_index in class_range:
        class_predictions = (predictions == class_index).float().unsqueeze(1)
        class_targets = (labels == class_index).float().unsqueeze(1)
        metric_list.append(
            _compute_binary_metrics_from_predictions(class_predictions, class_targets, eps=eps)
        )

    if not metric_list:
        metrics = {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
        }
        return filter_metrics(metrics, metric_names)

    keys = metric_list[0].keys()
    metrics = {key: sum(metric[key] for metric in metric_list) / len(metric_list) for key in keys}
    return filter_metrics(metrics, metric_names)


def filter_metrics(
    metrics: dict[str, float],
    metric_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    if metric_names is None:
        return metrics
    allowed = {str(name) for name in metric_names}
    return {key: value for key, value in metrics.items() if key in allowed}


def resolve_metric_names(config: dict | None = None) -> list[str]:
    if config is None:
        return list(DEFAULT_METRIC_NAMES)
    evaluation_cfg = config.get("evaluation", {})
    metric_names = evaluation_cfg.get("metrics")
    if not metric_names:
        return list(DEFAULT_METRIC_NAMES)
    return [str(name) for name in metric_names]


def compute_binary_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    metric_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    return compute_segmentation_metrics(
        logits,
        targets,
        threshold=threshold,
        eps=eps,
        include_background=False,
        metric_names=metric_names,
    )
