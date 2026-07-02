from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class MaskSource:
    name: str
    mask: np.ndarray
    weight: float = 1.0


@dataclass(frozen=True)
class FusedMask:
    probability: np.ndarray
    mask: np.ndarray
    disagreement: np.ndarray
    source_count: int
    positive_vote_count: np.ndarray

    def summary(self) -> dict[str, float | int]:
        return {
            "source_count": self.source_count,
            "positive_fraction_mean": float(self.probability.mean()),
            "disagreement_mean": float(self.disagreement.mean()),
            "fused_positive_fraction": float(self.mask.astype(bool).mean()),
        }


def fuse_source_masks(
    sources: list[MaskSource],
    threshold: float = 0.5,
    valid_mask: np.ndarray | None = None,
) -> FusedMask:
    names, masks, weights = _stack_sources(sources)
    del names
    valid = _valid_mask(valid_mask, masks.shape[1:])
    weighted_positive = (masks * weights[:, None, None]).sum(axis=0)
    total_weight = float(weights.sum())
    probability = weighted_positive / max(total_weight, 1e-12)
    probability = np.where(valid, probability, 0.0).astype(np.float32)
    positive_vote_count = masks.sum(axis=0).astype(np.uint8)
    disagreement = disagreement_from_probability(probability)
    disagreement = np.where(valid, disagreement, 0.0).astype(np.float32)
    return FusedMask(
        probability=probability,
        mask=((probability >= threshold) & valid).astype(np.uint8),
        disagreement=disagreement,
        source_count=int(masks.shape[0]),
        positive_vote_count=positive_vote_count,
    )


def disagreement_from_probability(probability: np.ndarray) -> np.ndarray:
    prob = np.clip(probability.astype(np.float32), 0.0, 1.0)
    return (4.0 * prob * (1.0 - prob)).astype(np.float32)


def source_agreement_summary(sources: list[MaskSource], valid_mask: np.ndarray | None = None) -> dict[str, object]:
    names, masks, _ = _stack_sources(sources)
    valid = _valid_mask(valid_mask, masks.shape[1:])
    valid_count = int(valid.sum())
    if valid_count == 0:
        raise ValueError("valid_mask excludes all pixels")

    all_positive = (masks.astype(bool).all(axis=0) & valid)
    all_negative = ((~masks.astype(bool)).all(axis=0) & valid)
    any_positive = (masks.astype(bool).any(axis=0) & valid)
    conflicting = any_positive & ~(all_positive | all_negative)

    pairwise_iou: dict[str, float] = {}
    for i, j in combinations(range(len(names)), 2):
        a = masks[i].astype(bool) & valid
        b = masks[j].astype(bool) & valid
        union = int((a | b).sum())
        pairwise_iou[f"{names[i]}__{names[j]}"] = float((a & b).sum() / union) if union else 1.0

    return {
        "source_count": int(masks.shape[0]),
        "valid_pixels": valid_count,
        "all_positive_fraction": float(all_positive.sum() / valid_count),
        "all_negative_fraction": float(all_negative.sum() / valid_count),
        "conflict_fraction": float(conflicting.sum() / valid_count),
        "any_positive_fraction": float(any_positive.sum() / valid_count),
        "pairwise_iou": pairwise_iou,
    }


def source_vote_records(sources: list[MaskSource]) -> list[dict[str, object]]:
    return [{"name": source.name, "weight": float(source.weight)} for source in sources]


def fused_summary_record(sources: list[MaskSource], fused: FusedMask) -> dict[str, object]:
    return {
        "sources": source_vote_records(sources),
        "fusion": fused.summary(),
    }


def _stack_sources(sources: list[MaskSource]) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not sources:
        raise ValueError("at least one source mask is required")
    shape = sources[0].mask.shape
    names: list[str] = []
    masks: list[np.ndarray] = []
    weights: list[float] = []
    for source in sources:
        if source.mask.shape != shape:
            raise ValueError("all source masks must have the same shape")
        if source.weight <= 0:
            raise ValueError("source weights must be positive")
        names.append(source.name)
        masks.append((source.mask > 0).astype(np.float32))
        weights.append(float(source.weight))
    return names, np.stack(masks, axis=0), np.asarray(weights, dtype=np.float32)


def _valid_mask(valid_mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if valid_mask is None:
        return np.ones(shape, dtype=bool)
    if valid_mask.shape != shape:
        raise ValueError("valid_mask must match source mask shape")
    return valid_mask.astype(bool)
