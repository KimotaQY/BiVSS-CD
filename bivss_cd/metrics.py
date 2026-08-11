from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    iou: float
    f1: float
    precision: float
    recall: float
    oa: float

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


def binary_metrics(prediction: Any, target: Any) -> BinaryMetrics:
    pred = np.asarray(prediction) > 0
    truth = np.asarray(target) > 0
    if pred.shape != truth.shape:
        raise ValueError(f"prediction shape {pred.shape} differs from target shape {truth.shape}")
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    safe = lambda numerator, denominator, empty=0.0: numerator / denominator if denominator else empty
    return BinaryMetrics(
        iou=safe(tp, tp + fp + fn, 1.0),
        f1=safe(2 * tp, 2 * tp + fp + fn, 1.0),
        precision=safe(tp, tp + fp),
        recall=safe(tp, tp + fn),
        oa=safe(tp + tn, tp + fp + fn + tn, 1.0),
    )


def multiclass_metrics(prediction: Any, target: Any, num_classes: int, ignore_index: int | None = 255) -> dict[str, Any]:
    pred, truth = np.asarray(prediction), np.asarray(target)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction shape {pred.shape} differs from target shape {truth.shape}")
    valid = np.ones(truth.shape, dtype=bool) if ignore_index is None else truth != ignore_index
    if np.any((pred[valid] < 0) | (pred[valid] >= num_classes)):
        raise ValueError("prediction contains class IDs outside the configured range")
    if np.any((truth[valid] < 0) | (truth[valid] >= num_classes)):
        raise ValueError("target contains class IDs outside the configured range")
    confusion = np.bincount(
        num_classes * truth[valid].astype(int) + pred[valid].astype(int),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    tp = np.diag(confusion).astype(float)
    fp, fn = confusion.sum(0) - tp, confusion.sum(1) - tp
    iou = np.divide(tp, tp + fp + fn, out=np.full(num_classes, np.nan), where=(tp + fp + fn) > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.full(num_classes, np.nan), where=(2 * tp + fp + fn) > 0)
    return {
        "mean_iou": float(np.nanmean(iou)),
        "mean_f1": float(np.nanmean(f1)),
        "per_class_iou": iou.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": confusion.tolist(),
    }
