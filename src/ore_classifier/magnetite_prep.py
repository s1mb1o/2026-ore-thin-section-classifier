"""Adaptive magnetite-darkening preprocessing for sulfide segmentation (two-pass).

Problem: the sulfide segmenter merges gray magnetite hosts with the bright
sulfide grains they contain into one giant component, so a "magnetite slab with
sparse ore specks" reads as one massive ordinary chunk downstream.

Recipe (validated in the 2026-07-05 lab on 8 frames, see
``docs/notes/2026-07-05-magnetite-prep.md``):

1. pass 1: segmenter on the original image -> mask1.
2. gate: only frames where mask1's largest component is a real slab
   (>= ``min_giant_share`` of ore area AND >= ``min_giant_px``).
3. adaptive threshold T inside mask1 luma: two-step Otsu
   (t1 cuts the dark mode, t2 = magnetite | ore split), guarded by
   darkened-share and mode-separation checks.
4. darken pixels with luma <= T by ``darken`` -> pass 2: segmenter on the
   darkened image -> mask2.
5. giant-only post-filter: inside mask2 components that are still slabs
   (>= max(``postfilter_min_px``, ``postfilter_min_frac`` * mask area)), drop
   pixels whose ORIGINAL luma <= T (darkened magnetite the model re-claimed).
   Small components are never touched (dim ore grains survive).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class MagnetitePrepConfig:
    min_giant_share: float = 40.0      # % of ore area (pass-1 gate)
    min_giant_px: int = 200_000        # absolute size of the pass-1 giant
    min_darkened_share: float = 0.15   # of mask pixels below T
    max_darkened_share: float = 0.99
    min_separation: float = 35.0       # luma gap between the two clusters
    max_giant_bright_share: float = 0.25  # skip if the giant's >T content exceeds this
                                          # (solid two-phase ore, not sparse specks in magnetite)
    darken: float = 0.5                # hard darkening for definite magnetite (luma <= t1)
    darken_soft: float = 0.75          # soft darkening for the ambiguous band (t1 < luma <= T):
                                       # the model keeps textured dim ore, drops flat magnetite
    darken_dilate_px: int = 24         # darkening is applied only around slab components
    postfilter_min_px: int = 50_000
    postfilter_min_frac: float = 0.20  # of mask2 area


@dataclass
class MagnetitePrepDecision:
    applied: bool
    reason: str
    threshold: float | None = None
    t1: float | None = None
    darkened_share: float | None = None
    separation: float | None = None
    giant_share_pct: float | None = None
    giant_px: int | None = None
    giant_bright_share: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def luma_of(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _otsu(vals: np.ndarray) -> float | None:
    hist, _ = np.histogram(vals, bins=256, range=(0, 255))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return None
    csum = np.cumsum(hist)
    cmean = np.cumsum(hist * np.arange(256))
    w0 = csum / total
    mu0 = np.divide(cmean, csum, out=np.zeros_like(cmean), where=csum > 0)
    mu1 = np.divide(cmean[-1] - cmean, total - csum, out=np.zeros_like(cmean), where=(total - csum) > 0)
    sigma = w0 * (1 - w0) * (mu0 - mu1) ** 2
    return float(np.argmax(sigma))


def _valley_threshold(above: np.ndarray) -> float | None:
    """Antimode between the host (magnetite) and ore luma modes.

    Otsu bisects by variance and lands INSIDE a wide ore mode (cuts dim ore);
    the histogram valley separates the modes without biting into either."""
    hist, _ = np.histogram(above, bins=256, range=(0, 255))
    k = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
    h = np.convolve(hist.astype(np.float64), k / k.sum(), mode="same")
    peaks = [i for i in range(1, 255) if h[i] > h[i - 1] and h[i] >= h[i + 1] and h[i] > h.max() * 0.15]
    if len(peaks) < 2:
        return None
    # the two TALLEST peaks are the host and ore modes (tail bumps are ignored)
    top2 = sorted(sorted(peaks, key=lambda p: h[p], reverse=True)[:2])
    p_host, p_ore = top2
    if p_ore - p_host < 8:
        return None
    valley = p_host + int(np.argmin(h[p_host:p_ore + 1]))
    if h[valley] > 0.8 * min(h[p_host], h[p_ore]):
        # modes overlap without a real dip: split midway between the peaks
        return float((p_host + p_ore) / 2.0)
    return float(valley)


def _giant(mask: np.ndarray) -> tuple[float, int]:
    """(share % of ore area, px) of the largest connected component."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return 0.0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = int(areas.max())
    return big / max(int(areas.sum()), 1) * 100.0, big


def decide(rgb: np.ndarray, mask: np.ndarray, config: MagnetitePrepConfig | None = None) -> MagnetitePrepDecision:
    """Decide whether pass-2 darkening should run, and at which threshold."""
    cfg = config or MagnetitePrepConfig()
    mask = mask > 0
    share, big_px = _giant(mask)
    if share < cfg.min_giant_share or big_px < cfg.min_giant_px:
        return MagnetitePrepDecision(False, f"no_giant({share:.0f}%/{big_px}px)",
                                     giant_share_pct=share, giant_px=big_px)
    vals = luma_of(rgb)[mask]
    t1 = _otsu(vals)
    if t1 is None:
        return MagnetitePrepDecision(False, "empty_mask", giant_share_pct=share, giant_px=big_px)
    above = vals[vals > t1]
    if above.size < 200:
        return MagnetitePrepDecision(False, "unimodal", t1=t1, giant_share_pct=share, giant_px=big_px)
    t2 = _otsu(above)
    if t2 is None:
        return MagnetitePrepDecision(False, "unimodal", t1=t1, giant_share_pct=share, giant_px=big_px)
    valley = _valley_threshold(above)
    if valley is not None:
        t2 = valley  # antimode between host and ore; Otsu kept as fallback
    below_share = float((vals <= t2).mean())
    lo = vals[vals <= t2]
    hi = above[above > t2]
    separation = float(hi.mean() - lo.mean()) if lo.size and hi.size else 0.0
    if not (cfg.min_darkened_share <= below_share <= cfg.max_darkened_share):
        return MagnetitePrepDecision(False, f"darkened_share={below_share:.2f}", threshold=float(t2), t1=t1,
                                     darkened_share=below_share, separation=separation,
                                     giant_share_pct=share, giant_px=big_px)
    if separation < cfg.min_separation:
        return MagnetitePrepDecision(False, f"weak_separation={separation:.0f}", threshold=float(t2), t1=t1,
                                     darkened_share=below_share, separation=separation,
                                     giant_share_pct=share, giant_px=big_px)
    # sparsity gate: magnetite slabs hold only sparse bright specks; a giant whose
    # above-threshold content is a solid bright body is two-phase MASSIVE ORE - skip.
    lum = luma_of(rgb)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask).astype(np.uint8), connectivity=8)
    gid = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    giant = labels == gid
    bright_share = float((giant & (lum > t2)).sum() / max(giant.sum(), 1))
    if bright_share > cfg.max_giant_bright_share:
        return MagnetitePrepDecision(False, f"massive_ore(bright={bright_share:.0%})", threshold=float(t2), t1=t1,
                                     darkened_share=below_share, separation=separation,
                                     giant_share_pct=share, giant_px=big_px, giant_bright_share=bright_share)
    return MagnetitePrepDecision(True, "apply", threshold=float(t2), t1=t1,
                                 darkened_share=below_share, separation=separation,
                                 giant_share_pct=share, giant_px=big_px, giant_bright_share=bright_share)


def slabs_region(mask: np.ndarray, config: MagnetitePrepConfig | None = None) -> np.ndarray:
    """Union of slab-sized components of the pass-1 mask, dilated — the only
    area where darkening is allowed (small grains elsewhere stay untouched)."""
    cfg = config or MagnetitePrepConfig()
    m = mask > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    region = np.zeros_like(m)
    for cid in range(1, n):
        if int(stats[cid, cv2.CC_STAT_AREA]) >= cfg.min_giant_px:
            region |= labels == cid
    if cfg.darken_dilate_px > 0 and region.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.darken_dilate_px * 2 + 1,) * 2)
        region = cv2.dilate(region.astype(np.uint8), k).astype(bool)
    return region


def darken(rgb: np.ndarray, threshold: float, region: np.ndarray | None = None,
           config: MagnetitePrepConfig | None = None, t1: float | None = None) -> np.ndarray:
    """Two-level darkening inside ``region``: hard (x``darken``) for definite
    magnetite (luma <= t1), soft (x``darken_soft``) for the ambiguous band
    (t1 < luma <= threshold) where dim ore and light magnetite overlap —
    the segmenter separates them by texture at reduced brightness."""
    cfg = config or MagnetitePrepConfig()
    out = rgb.astype(np.float32).copy()
    lum = luma_of(rgb)
    inside = np.ones(lum.shape, dtype=bool) if region is None else region > 0
    hard_top = threshold if t1 is None else min(t1, threshold)
    out[inside & (lum <= hard_top)] *= cfg.darken
    if t1 is not None and threshold > t1:
        out[inside & (lum > t1) & (lum <= threshold)] *= cfg.darken_soft
    return out.clip(0, 255).astype(np.uint8)


def giant_only_postfilter(mask2: np.ndarray, original_luma: np.ndarray, threshold: float,
                          config: MagnetitePrepConfig | None = None) -> np.ndarray:
    """Scrub sub-threshold pixels ONLY inside still-slab components of mask2.

    Call with the DEFINITE-magnetite threshold (t1), not the darkening T:
    the ambiguous band belongs to the model's judgement, not the scrub."""
    cfg = config or MagnetitePrepConfig()
    m = (mask2 > 0).copy()
    min_px = max(cfg.postfilter_min_px, int(cfg.postfilter_min_frac * m.sum()))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    for cid in range(1, n):
        if int(stats[cid, cv2.CC_STAT_AREA]) >= min_px:
            comp = labels == cid
            m[comp & (original_luma <= threshold)] = False
    return m
