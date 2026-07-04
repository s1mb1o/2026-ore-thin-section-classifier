"""Model-free heuristic talcose-vs-not-talcose classifier via talc zones.

Pipeline (see docs/notes/2026-07-04-heuristic-talcose-classifier.md):

    RGB image
    -> opaque/ore mask (sulfides + gray magnetite; bright/opaque phases)
    -> matrix = analyzed area minus ore
    -> dark threshold  = K * median(luma of matrix)          [talc is darkest]
    -> flakes          = dark components, minus large SOLID blobs (pores/shadows/
                         resin) identified by low internal porosity
    -> zones           = proximity aggregation of flakes (morphological close,
                         DBSCAN-like) keeping clusters with enough flake pixels
    -> talc_fraction   = zone area / matrix area
    -> talcose if talc_fraction > classify_threshold

Rationale for each stage is documented per method. All spatial radii are defined
at a fixed processing width (`proc_width`) so results are resolution-independent.

The opaque/ore mask can be supplied directly (e.g. from the trained sulfide
segmentation model, which is preferred in production); `opaque_phase_mask` is a
self-contained brightness fallback.

Approved production parameters live in `TalcZoneConfig` defaults (2026-07-04).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np

LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def to_luma(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) @ LUMA_WEIGHTS


def _disk(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


@dataclass
class OpaqueMaskConfig:
    """Brightness heuristic for the opaque/ore phase (sulfides + gray magnetite)."""

    analyzed_min_luma: int = 18          # exclude near-black borders from analysis
    abs_luma_floor: float = 120.0        # v2: absolute bright floor
    abs_p995_frac: float = 0.75          # v2: or 0.75 * p99.5
    rel_p995_frac: float = 0.72          # v3: relative bright for dark-warm images
    warm_min: float = 25.0               # v3: R-B warmth for a bright pixel to count
    otsu_band_floor_offset: int = 35     # gray-phase floor above the matrix mode
    open_radius: int = 1                 # speckle cleanup
    min_component_frac: float = 0.00008  # drop components smaller than this * area


@dataclass
class TalcZoneConfig:
    """Approved production parameters for talc-zone detection (2026-07-04)."""

    proc_width: int = 800                # spatial radii are defined at this width
    k_threshold: float = 0.85            # dark threshold = k * median(matrix luma)
    blob_area_frac: float = 0.01         # components larger than this are "big"
    blob_porosity_max: float = 0.15      # big + porosity<this  => solid pore => drop
    footprint_close_radius: int = 4      # closing radius when measuring porosity
    proximity_radius: int = 25           # bridge flakes within this radius into a zone
    zone_min_flake_frac: float = 0.0003  # a zone needs at least this much flake area
    zone_min_flake_fill: float = 0.05    # and flakes must be >=5% of the bridged region
    classify_threshold: float = 0.50     # talcose if zone_fraction > this


@dataclass
class TalcZoneResult:
    talc_fraction: float
    is_talcose: bool
    zone_mask: np.ndarray                # bool, original resolution
    flake_mask: np.ndarray               # bool, original resolution
    matrix_median_luma: float
    dark_threshold: float
    matrix_area_px: int
    zone_area_px: int
    flake_area_px: int


def opaque_phase_mask(rgb: np.ndarray, config: OpaqueMaskConfig | None = None) -> np.ndarray:
    """Brightness heuristic for opaque/ore phases (sulfides + gray magnetite).

    Combines three cues so it works across the greenish, dark, and bright
    acquisition domains present in the official set:
      - v2 absolute brightness (normal/bright images);
      - v3 relative brightness gated by warm tone (dark-warm images);
      - a two-stage Otsu that first splits bright sulfide, then recovers the gray
        magnetite band above the silicate mode (all domains).
    Returns a boolean mask at the input resolution.

    In production the ore mask preferably comes from the trained sulfide model;
    this is a dependency-light fallback.
    """
    cfg = config or OpaqueMaskConfig()
    rgb = np.ascontiguousarray(rgb[..., :3])
    luma = to_luma(rgb)
    analyzed = luma >= cfg.analyzed_min_luma
    if analyzed.sum() < 16:
        return np.zeros(luma.shape, dtype=bool)

    vals = luma[analyzed]
    p995 = float(np.percentile(vals, 99.5))
    hist = np.bincount(vals.astype(np.int64).clip(0, 255), minlength=256).astype(np.float64)
    smooth = np.convolve(hist, np.ones(11) / 11, mode="same")
    cap = max(60, int(0.6 * p995))
    mode = int(np.argmax(smooth[:cap]))

    warm = rgb[..., 0].astype(np.int16) - rgb[..., 2].astype(np.int16)

    mask = (luma >= max(cfg.abs_luma_floor, cfg.abs_p995_frac * p995)) & analyzed          # v2
    mask |= (luma >= cfg.rel_p995_frac * p995) & (warm >= cfg.warm_min) & analyzed         # v3

    upper = luma[(luma >= mode + 20) & analyzed].astype(np.uint8)
    if upper.size > 1000:
        o1, _ = cv2.threshold(upper.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        band = luma[(luma >= mode + 20) & (luma < o1) & analyzed].astype(np.uint8)
        if band.size > 1000:
            o2, _ = cv2.threshold(band.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            o2 = max(float(o2), mode + cfg.otsu_band_floor_offset)
            mask |= (luma >= o2) & analyzed

    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _disk(cfg.open_radius))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    min_area = cfg.min_component_frac * luma.size
    keep = np.zeros_like(m, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = True
    return keep


def _extract_flakes(dark: np.ndarray, matrix: np.ndarray, cfg: TalcZoneConfig) -> np.ndarray:
    """Dark pixels minus large SOLID blobs (pores/shadows/resin, not talc).

    A component is dropped only if it is large AND non-porous: talc — even a dense
    connected mass — is lacy (white gaps between flakes), while a pore is a smooth
    filled region. Porosity = white gaps inside the component's filled footprint.
    """
    from scipy import ndimage

    num, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    cap = cfg.blob_area_frac * dark.size
    remove = np.zeros(dark.shape, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] <= cap:
            continue
        comp = labels == i
        footprint = ndimage.binary_fill_holes(
            cv2.morphologyEx(comp.astype(np.uint8), cv2.MORPH_CLOSE, _disk(cfg.footprint_close_radius)) > 0
        )
        internal_white = footprint & ~comp & matrix
        porosity = internal_white.sum() / max(footprint.sum(), 1)
        if porosity < cfg.blob_porosity_max:
            remove |= comp
    return (dark > 0) & ~remove


def _aggregate_zones(flakes: np.ndarray, matrix: np.ndarray, cfg: TalcZoneConfig) -> np.ndarray:
    """Proximity aggregation (DBSCAN-like): bridge nearby flakes into zones.

    Morphological closing connects flakes within `proximity_radius`; a bridged
    region is kept as a talc zone only if it actually contains enough flake pixels
    (rejects large hollow fills and single isolated flakes).
    """
    bridged = (cv2.morphologyEx(flakes.astype(np.uint8), cv2.MORPH_CLOSE, _disk(cfg.proximity_radius)) > 0) & matrix
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bridged.astype(np.uint8), 8)
    keep = np.zeros(bridged.shape, dtype=bool)
    min_flake_area = cfg.zone_min_flake_frac * flakes.size
    for i in range(1, num):
        comp = labels == i
        flake_area = (flakes & comp).sum()
        if flake_area >= min_flake_area and flake_area / max(comp.sum(), 1) >= cfg.zone_min_flake_fill:
            keep |= comp
    return keep


def detect_talc_zones(
    rgb: np.ndarray,
    ore_mask: np.ndarray | None = None,
    config: TalcZoneConfig | None = None,
    opaque_config: OpaqueMaskConfig | None = None,
) -> TalcZoneResult:
    """Full model-free talc-zone detection and talcose classification.

    `ore_mask` (bool, input resolution) is the opaque/ore phase to exclude; if
    None it is computed with `opaque_phase_mask`. Spatial steps run at
    `config.proc_width`, then the zone mask is upscaled to the input resolution.
    """
    cfg = config or TalcZoneConfig()
    rgb = np.ascontiguousarray(rgb[..., :3])
    h0, w0 = rgb.shape[:2]
    if ore_mask is None:
        ore_mask = opaque_phase_mask(rgb, opaque_config)

    # matrix = whole frame minus ore (no extra analyzed gate); the dark threshold
    # uses the full-resolution matrix median so it is independent of proc_width.
    matrix_full = ~ore_mask
    if matrix_full.sum() == 0:
        empty = np.zeros((h0, w0), dtype=bool)
        return TalcZoneResult(0.0, False, empty, empty, 0.0, 0.0, 0, 0, 0)
    median = float(np.median(to_luma(rgb)[matrix_full]))
    thr = cfg.k_threshold * median

    # Spatial stages run at a fixed processing width; use PIL BILINEAR/NEAREST to
    # match the reference experiment exactly.
    from PIL import Image as _Image

    scale = cfg.proc_width / float(w0)
    pw, ph = cfg.proc_width, max(1, int(round(h0 * scale)))
    luma = np.asarray(
        _Image.fromarray(rgb).convert("L").resize((pw, ph), _Image.Resampling.BILINEAR), dtype=np.float32
    )
    ore_small = np.asarray(
        _Image.fromarray(ore_mask.astype(np.uint8) * 255).resize((pw, ph), _Image.Resampling.NEAREST)
    ) > 0
    matrix = ~ore_small

    dark = (matrix & (luma <= thr)).astype(np.uint8)
    flakes = _extract_flakes(dark, matrix, cfg)
    zones = _aggregate_zones(flakes, matrix, cfg)

    matrix_area = int(matrix.sum())
    talc_fraction = float(zones.sum() / max(matrix_area, 1))

    zone_full = cv2.resize(zones.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST) > 0
    flake_full = cv2.resize(flakes.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST) > 0
    return TalcZoneResult(
        talc_fraction=talc_fraction,
        is_talcose=bool(talc_fraction > cfg.classify_threshold),
        zone_mask=zone_full,
        flake_mask=flake_full,
        matrix_median_luma=median,
        dark_threshold=thr,
        matrix_area_px=matrix_area,
        zone_area_px=int(zones.sum()),
        flake_area_px=int(flakes.sum()),
    )


def result_to_dict(result: TalcZoneResult, config: TalcZoneConfig) -> dict[str, Any]:
    return {
        "schema_version": "talc-zone-heuristic-v1",
        "talc_fraction": round(result.talc_fraction, 4),
        "ore_class": "talcose_ore" if result.is_talcose else "not_talcose",
        "is_talcose": result.is_talcose,
        "matrix_median_luma": round(result.matrix_median_luma, 2),
        "dark_threshold": round(result.dark_threshold, 2),
        "matrix_area_px": result.matrix_area_px,
        "zone_area_px": result.zone_area_px,
        "flake_area_px": result.flake_area_px,
        "config": asdict(config),
    }
