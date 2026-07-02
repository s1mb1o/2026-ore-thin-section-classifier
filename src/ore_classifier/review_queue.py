from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class ReviewCandidate:
    candidate_id: int
    x: int
    y: int
    width: int
    height: int
    area_px: int
    centroid_x: float
    centroid_y: float
    score: float
    uncertainty: float
    decision_impact: float
    novelty: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_review_queue(
    uncertainty_map: np.ndarray,
    decision_impact_map: np.ndarray | None = None,
    novelty_map: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    threshold: float = 0.25,
    min_area_px: int = 4,
    padding_px: int = 0,
    top_k: int | None = None,
) -> list[ReviewCandidate]:
    uncertainty = _score_map(uncertainty_map, "uncertainty_map")
    impact = np.ones_like(uncertainty) if decision_impact_map is None else _score_map(decision_impact_map, "decision_impact_map")
    novelty = np.ones_like(uncertainty) if novelty_map is None else _score_map(novelty_map, "novelty_map")
    valid = np.ones(uncertainty.shape, dtype=bool) if valid_mask is None else valid_mask.astype(bool)
    if valid.shape != uncertainty.shape:
        raise ValueError("valid_mask must match uncertainty_map shape")

    candidate_pixels = (uncertainty >= threshold) & valid
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_pixels.astype(np.uint8), connectivity=8)
    candidates: list[ReviewCandidate] = []
    for label_id in range(1, labels_count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        region = labels == label_id
        u = float(uncertainty[region].mean())
        d = float(impact[region].mean())
        n = float(novelty[region].mean())
        score = u * d * n
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        x, y, w, h = _pad_box(x, y, w, h, uncertainty.shape, padding_px)
        candidates.append(
            ReviewCandidate(
                candidate_id=int(label_id),
                x=x,
                y=y,
                width=w,
                height=h,
                area_px=area,
                centroid_x=float(centroids[label_id][0]),
                centroid_y=float(centroids[label_id][1]),
                score=float(score),
                uncertainty=u,
                decision_impact=d,
                novelty=n,
                reason=_reason(u, d, n),
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:top_k] if top_k is not None else candidates


def decision_impact_from_threshold_margin(
    values: np.ndarray,
    threshold: float,
    max_margin: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    if max_margin <= 0:
        raise ValueError("max_margin must be positive")
    impact = 1.0 - np.clip(np.abs(values.astype(np.float32) - float(threshold)) / float(max_margin), 0.0, 1.0)
    if valid_mask is not None:
        if valid_mask.shape != values.shape:
            raise ValueError("valid_mask must match values shape")
        impact = np.where(valid_mask.astype(bool), impact, 0.0)
    return impact.astype(np.float32)


def candidates_to_records(candidates: Iterable[ReviewCandidate]) -> list[dict[str, object]]:
    return [candidate.to_dict() for candidate in candidates]


def expert_questions_from_candidates(
    candidates: list[ReviewCandidate],
    image_id: str,
    question_limit: int = 20,
) -> list[dict[str, object]]:
    questions: list[dict[str, object]] = []
    for candidate in candidates[:question_limit]:
        questions.append(
            {
                "image_id": image_id,
                "candidate_id": candidate.candidate_id,
                "bbox_xywh": [candidate.x, candidate.y, candidate.width, candidate.height],
                "score": candidate.score,
                "why_it_matters": candidate.reason,
                "question_ru": _question_ru(candidate),
            }
        )
    return questions


def _score_map(values: np.ndarray, name: str) -> np.ndarray:
    array = values.astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return np.clip(array, 0.0, 1.0)


def _pad_box(x: int, y: int, w: int, h: int, shape: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    if padding <= 0:
        return x, y, w, h
    height, width = shape
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + w + padding)
    y1 = min(height, y + h + padding)
    return x0, y0, x1 - x0, y1 - y0


def _reason(uncertainty: float, decision_impact: float, novelty: float) -> str:
    parts: list[str] = []
    if uncertainty >= 0.75:
        parts.append("high uncertainty")
    if decision_impact >= 0.75:
        parts.append("near decision threshold")
    if novelty >= 0.75:
        parts.append("novel visual pattern")
    return ", ".join(parts) if parts else "moderate review value"


def _question_ru(candidate: ReviewCandidate) -> str:
    if candidate.decision_impact >= 0.75:
        return "Проверьте эту область: она может изменить итоговый класс руды или пороговое решение."
    if candidate.uncertainty >= 0.75:
        return "Проверьте границы и класс этой области: источники маски дают неуверенный результат."
    return "Проверьте эту область как полезный пример для доразметки и улучшения модели."
