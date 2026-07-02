from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class NearDuplicatePair:
    left_id: str
    right_id: str
    distance: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationLabelIssueSummary:
    issue_fraction: float
    mean_issue_confidence: float
    issue_mask: np.ndarray
    predicted_label: np.ndarray


def image_feature_vector(image: np.ndarray, histogram_bins: int = 8) -> np.ndarray:
    if histogram_bins < 2:
        raise ValueError("histogram_bins must be >= 2")
    array = image.astype(np.float32)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError("image must be HxW or HxWxC")
    if array.max() > 1.0:
        array = array / 255.0

    means = array.reshape(-1, array.shape[2]).mean(axis=0)
    stds = array.reshape(-1, array.shape[2]).std(axis=0)
    gray = array.mean(axis=2)
    hist, _ = np.histogram(gray, bins=histogram_bins, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float32) / max(int(hist.sum()), 1)
    return np.concatenate([means, stds, hist]).astype(np.float32)


def pairwise_distances(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("features must be a 2D array")
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt((diff * diff).sum(axis=2)).astype(np.float32)


def uniqueness_scores(features: np.ndarray, k: int = 1) -> np.ndarray:
    distances = pairwise_distances(features)
    n = distances.shape[0]
    if n == 0:
        return np.array([], dtype=np.float32)
    if n == 1:
        return np.ones(1, dtype=np.float32)
    k_eff = min(max(1, int(k)), n - 1)
    distances_without_self = distances.copy()
    np.fill_diagonal(distances_without_self, np.inf)
    sorted_distances = np.sort(distances_without_self, axis=1)
    nearest = sorted_distances[:, :k_eff].mean(axis=1)
    max_value = float(nearest.max())
    if max_value <= 0:
        return np.zeros(n, dtype=np.float32)
    return (nearest / max_value).astype(np.float32)


def near_duplicate_pairs(ids: list[str], features: np.ndarray, distance_threshold: float) -> list[NearDuplicatePair]:
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be non-negative")
    distances = pairwise_distances(features)
    if len(ids) != distances.shape[0]:
        raise ValueError("ids length must match number of feature rows")
    pairs: list[NearDuplicatePair] = []
    for i, j in combinations(range(len(ids)), 2):
        distance = float(distances[i, j])
        if distance <= distance_threshold:
            pairs.append(NearDuplicatePair(ids[i], ids[j], distance))
    pairs.sort(key=lambda item: item.distance)
    return pairs


def hardness_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.ndim < 2:
        raise ValueError("probabilities must include a class axis")
    max_prob = np.clip(probs.max(axis=0), 0.0, 1.0)
    return (1.0 - max_prob).astype(np.float32)


def segmentation_label_issue_summary(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    valid_mask: np.ndarray | None = None,
    confidence_threshold: float = 0.8,
) -> SegmentationLabelIssueSummary:
    if pred_probs.ndim != 3:
        raise ValueError("pred_probs must have shape KxHxW")
    if labels.shape != pred_probs.shape[1:]:
        raise ValueError("labels must match pred_probs spatial shape")
    valid = np.ones(labels.shape, dtype=bool) if valid_mask is None else valid_mask.astype(bool)
    if valid.shape != labels.shape:
        raise ValueError("valid_mask must match labels shape")

    predicted = pred_probs.argmax(axis=0).astype(labels.dtype)
    confidence = np.clip(pred_probs.max(axis=0), 0.0, 1.0)
    issues = (predicted != labels) & (confidence >= confidence_threshold) & valid
    valid_count = int(valid.sum())
    mean_conf = float(confidence[issues].mean()) if bool(issues.any()) else 0.0
    return SegmentationLabelIssueSummary(
        issue_fraction=float(issues.sum() / max(valid_count, 1)),
        mean_issue_confidence=mean_conf,
        issue_mask=issues.astype(np.uint8),
        predicted_label=predicted,
    )
