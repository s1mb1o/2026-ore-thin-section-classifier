from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_confusion_counts(target: np.ndarray, pred: np.ndarray, valid: np.ndarray | None = None) -> dict[str, int]:
    target_bool = target.astype(bool)
    pred_bool = pred.astype(bool)
    valid_bool = np.ones(target_bool.shape, dtype=bool) if valid is None else valid.astype(bool)

    tp = int((target_bool & pred_bool & valid_bool).sum())
    fp = int((~target_bool & pred_bool & valid_bool).sum())
    fn = int((target_bool & ~pred_bool & valid_bool).sum())
    tn = int((~target_bool & ~pred_bool & valid_bool).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def binary_scores_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    total = tp + fp + fn + tn
    return {
        "iou_sulfide": safe_div(tp, tp + fp + fn),
        "iou_bg": safe_div(tn, tn + fp + fn),
        "precision_sulfide": safe_div(tp, tp + fp),
        "recall_sulfide": safe_div(tp, tp + fn),
        "f1_sulfide": safe_div(2 * tp, 2 * tp + fp + fn),
        "pixel_acc": safe_div(tp + tn, total),
    }


def auc_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray) -> float | None:
    pos_total = int(pos_hist.sum())
    neg_total = int(neg_hist.sum())
    if pos_total == 0 or neg_total == 0:
        return None

    auc_rank_sum = 0.0
    neg_less = 0.0
    for pos_count, neg_count in zip(pos_hist, neg_hist, strict=True):
        auc_rank_sum += float(pos_count) * (neg_less + 0.5 * float(neg_count))
        neg_less += float(neg_count)
    return auc_rank_sum / float(pos_total * neg_total)


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8)
    if int(mask_uint8.sum()) == 0:
        return np.zeros(mask_uint8.shape, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask_uint8, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    boundary = mask_uint8.astype(bool) & ~eroded.astype(bool)
    return boundary if bool(boundary.any()) else mask_uint8.astype(bool)


def hausdorff_and_hd95_px(
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[float, float]:
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    if valid is not None:
        valid_bool = valid.astype(bool)
        pred_bool = pred_bool & valid_bool
        target_bool = target_bool & valid_bool

    if not pred_bool.any() and not target_bool.any():
        return 0.0, 0.0

    max_dist = float(math.hypot(*pred_bool.shape))
    if pred_bool.any() != target_bool.any():
        return max_dist, max_dist

    pred_boundary = boundary_mask(pred_bool)
    target_boundary = boundary_mask(target_bool)
    distances = np.concatenate(
        [
            _boundary_distances(source=pred_boundary, reference=target_boundary),
            _boundary_distances(source=target_boundary, reference=pred_boundary),
        ]
    )
    if distances.size == 0:
        return 0.0, 0.0
    return float(distances.max()), float(np.percentile(distances, 95))


def _boundary_distances(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if not source.any():
        return np.array([], dtype=np.float32)
    if not reference.any():
        return np.full(int(source.sum()), math.hypot(*source.shape), dtype=np.float32)

    distance_input = np.where(reference, 0, 1).astype(np.uint8)
    distances = cv2.distanceTransform(distance_input, cv2.DIST_L2, 3)
    return distances[source]


@dataclass
class BinarySegmentationSummary:
    tp: int
    fp: int
    fn: int
    tn: int
    auc_sulfide: float | None
    hausdorff_items: int
    hausdorff_px_mean: float | None
    hausdorff_px_p95: float | None
    hd95_px_mean: float | None
    hd95_px_p95: float | None

    def to_dict(self) -> dict[str, Any]:
        scores = binary_scores_from_counts(self.tp, self.fp, self.fn, self.tn)
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            **scores,
            "auc_sulfide": self.auc_sulfide,
            "hausdorff_items": self.hausdorff_items,
            "hausdorff_px_mean": self.hausdorff_px_mean,
            "hausdorff_px_p95": self.hausdorff_px_p95,
            "hd95_px_mean": self.hd95_px_mean,
            "hd95_px_p95": self.hd95_px_p95,
        }


class BinarySegmentationAccumulator:
    def __init__(self, auc_bins: int = 256) -> None:
        if auc_bins < 2:
            raise ValueError("auc_bins must be >= 2")
        self.auc_bins = auc_bins
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.pos_hist = np.zeros(auc_bins, dtype=np.int64)
        self.neg_hist = np.zeros(auc_bins, dtype=np.int64)
        self.hausdorff_values: list[float] = []
        self.hd95_values: list[float] = []

    def update_confusion(
        self,
        target: np.ndarray,
        pred: np.ndarray,
        valid: np.ndarray | None = None,
        prob_sulfide: np.ndarray | None = None,
    ) -> None:
        counts = binary_confusion_counts(target, pred, valid)
        self.tp += counts["tp"]
        self.fp += counts["fp"]
        self.fn += counts["fn"]
        self.tn += counts["tn"]
        if prob_sulfide is not None:
            self._update_auc(target, prob_sulfide, valid)

    def update_hausdorff(self, target: np.ndarray, pred: np.ndarray, valid: np.ndarray | None = None) -> None:
        hausdorff_px, hd95_px = hausdorff_and_hd95_px(pred=pred, target=target, valid=valid)
        self.hausdorff_values.append(hausdorff_px)
        self.hd95_values.append(hd95_px)

    def _update_auc(self, target: np.ndarray, prob_sulfide: np.ndarray, valid: np.ndarray | None) -> None:
        target_bool = target.astype(bool)
        valid_bool = np.ones(target_bool.shape, dtype=bool) if valid is None else valid.astype(bool)
        probs = np.clip(prob_sulfide[valid_bool], 0.0, 1.0)
        labels = target_bool[valid_bool]
        if probs.size == 0:
            return
        bin_ids = np.minimum((probs * self.auc_bins).astype(np.int64), self.auc_bins - 1)
        self.pos_hist += np.bincount(bin_ids[labels], minlength=self.auc_bins)
        self.neg_hist += np.bincount(bin_ids[~labels], minlength=self.auc_bins)

    def summary(self) -> BinarySegmentationSummary:
        hausdorff = np.asarray(self.hausdorff_values, dtype=np.float64)
        hd95 = np.asarray(self.hd95_values, dtype=np.float64)
        return BinarySegmentationSummary(
            tp=self.tp,
            fp=self.fp,
            fn=self.fn,
            tn=self.tn,
            auc_sulfide=auc_from_hist(self.pos_hist, self.neg_hist),
            hausdorff_items=int(hausdorff.size),
            hausdorff_px_mean=float(hausdorff.mean()) if hausdorff.size else None,
            hausdorff_px_p95=float(np.percentile(hausdorff, 95)) if hausdorff.size else None,
            hd95_px_mean=float(hd95.mean()) if hd95.size else None,
            hd95_px_p95=float(np.percentile(hd95, 95)) if hd95.size else None,
        )
