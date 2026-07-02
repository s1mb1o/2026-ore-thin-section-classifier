from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ore_classifier.component_analysis import ComponentRuleConfig, OreSummary


@dataclass(frozen=True)
class AssociationContact:
    label_a: int
    label_b: int
    name_a: str
    name_b: str
    contact_px: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ComponentLiberationProxy:
    component_id: int
    area_px: int
    matrix_contact_px: int
    talc_contact_px: int
    other_sulfide_contact_px: int
    liberation_score: float
    touches_talc: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def association_contacts(
    class_mask: np.ndarray,
    label_names: dict[int, str] | None = None,
    valid_mask: np.ndarray | None = None,
) -> list[AssociationContact]:
    labels = class_mask.astype(np.int32)
    valid = np.ones(labels.shape, dtype=bool) if valid_mask is None else valid_mask.astype(bool)
    if valid.shape != labels.shape:
        raise ValueError("valid_mask must match class_mask shape")
    names = label_names or {}
    counts: dict[tuple[int, int], int] = {}
    for dy, dx in ((0, 1), (1, 0)):
        a = labels[: labels.shape[0] - dy or None, : labels.shape[1] - dx or None]
        b = labels[dy:, dx:]
        va = valid[: labels.shape[0] - dy or None, : labels.shape[1] - dx or None]
        vb = valid[dy:, dx:]
        changed = (a != b) & va & vb
        for left, right in zip(a[changed].tolist(), b[changed].tolist(), strict=True):
            key = (int(left), int(right)) if int(left) <= int(right) else (int(right), int(left))
            counts[key] = counts.get(key, 0) + 1
    return [
        AssociationContact(
            label_a=a,
            label_b=b,
            name_a=names.get(a, str(a)),
            name_b=names.get(b, str(b)),
            contact_px=count,
        )
        for (a, b), count in sorted(counts.items())
    ]


def component_liberation_proxies(
    sulfide_mask: np.ndarray,
    talc_mask: np.ndarray | None = None,
    min_area_px: int = 1,
) -> list[ComponentLiberationProxy]:
    sulfide = (sulfide_mask > 0).astype(np.uint8)
    talc = np.zeros_like(sulfide, dtype=np.uint8) if talc_mask is None else (talc_mask > 0).astype(np.uint8)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(sulfide, connectivity=8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    rows: list[ComponentLiberationProxy] = []
    for component_id in range(1, labels_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        component = (labels == component_id).astype(np.uint8)
        ring = (cv2.dilate(component, kernel, iterations=1) > 0) & (component == 0)
        matrix_contact = int((ring & (sulfide == 0) & (talc == 0)).sum())
        talc_contact = int((ring & (talc > 0)).sum())
        other_sulfide = int((ring & (sulfide > 0)).sum())
        total_contact = matrix_contact + talc_contact + other_sulfide
        liberation_score = float(matrix_contact / max(total_contact, 1))
        rows.append(
            ComponentLiberationProxy(
                component_id=int(component_id),
                area_px=area,
                matrix_contact_px=matrix_contact,
                talc_contact_px=talc_contact,
                other_sulfide_contact_px=other_sulfide,
                liberation_score=liberation_score,
                touches_talc=talc_contact > 0,
            )
        )
    return rows


def ore_decision_margins(
    summary: OreSummary,
    config: ComponentRuleConfig | None = None,
    talc_review_margin: float = 0.02,
    intergrowth_review_margin: float = 0.10,
) -> dict[str, object]:
    cfg = config or ComponentRuleConfig()
    talc_margin = float(summary.talc_fraction - cfg.talc_fraction_threshold)
    ordinary_minus_fine = float(summary.ordinary_sulfide_fraction - summary.fine_sulfide_fraction)
    needs_talc_review = abs(talc_margin) <= talc_review_margin
    needs_intergrowth_review = abs(ordinary_minus_fine) <= intergrowth_review_margin
    return {
        "ore_class": summary.ore_class,
        "talc_fraction": summary.talc_fraction,
        "talc_threshold": cfg.talc_fraction_threshold,
        "talc_margin": talc_margin,
        "ordinary_sulfide_fraction": summary.ordinary_sulfide_fraction,
        "fine_sulfide_fraction": summary.fine_sulfide_fraction,
        "ordinary_minus_fine_margin": ordinary_minus_fine,
        "needs_expert_review": bool(needs_talc_review or needs_intergrowth_review),
        "review_reasons": [
            reason
            for reason, enabled in (
                ("talc fraction near threshold", needs_talc_review),
                ("ordinary/fine split near threshold", needs_intergrowth_review),
            )
            if enabled
        ],
    }
