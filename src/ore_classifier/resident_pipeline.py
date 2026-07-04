"""Resident (single-load) ore pipeline.

Loads the binary-sulfide segmentation model once and runs the full per-image
pipeline in-process (tiled sulfide inference -> analyzed area -> optional auto
talc candidate -> deterministic ore analysis), producing the same artifact
layout and ``pipeline_summary.json`` schema as ``scripts/run_ore_pipeline.py``.

This removes the per-image cost the subprocess batch path pays: a Python
interpreter start plus a full checkpoint reload (``torch.load`` + model build +
``load_state_dict`` + move-to-device) for every image. The inference inner loop
and a few save helpers are intentionally duplicated from
``scripts/infer_binary_sulfide.py`` so that no currently-used per-image script is
modified (see ``docs/specs/resident-batch-inference.md``); a later cleanup can
dedupe.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from ore_classifier.analyzed_area import build_analyzed_mask
from ore_classifier.component_analysis import (
    ComponentRuleConfig,
    analyze_components,
    save_component_outputs,
)
from ore_classifier.datasets import IMAGENET_MEAN, IMAGENET_STD
from ore_classifier.model_io import (
    forward_logits,
    load_binary_segmentation_checkpoint,
    resolve_device,
)
from ore_classifier.pseudo_labels import brightness_sulfide_pseudo_mask
from ore_classifier.rule_config_io import default_rule_config
from ore_classifier.talc_candidate import (
    TalcCandidateConfig,
    estimate_talc_candidate_mask,
    save_talc_candidate_outputs,
)
from ore_classifier.tiling import Tile, iter_tiles, save_gray

Image.MAX_IMAGE_PIXELS = None


class ResidentSulfidePipeline:
    """Holds one loaded model and runs the full pipeline per image in-process."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "auto",
        tile_size: int = 1024,
        stride: int = 768,
        batch_size: int = 4,
        threshold: float = 0.5,
        talc_checkpoint: str | Path | None = None,
        talc_threshold: float = 0.5,
        preview_max_side: int = 1800,
    ) -> None:
        if stride > tile_size:
            raise ValueError("stride must be <= tile_size")
        self.device = resolve_device(device)
        self.model, self.checkpoint_meta = load_binary_segmentation_checkpoint(Path(checkpoint), self.device)
        self.model.eval()
        self.talc_model = None
        self.talc_checkpoint: str | None = None
        self.talc_checkpoint_meta: dict[str, Any] | None = None
        if talc_checkpoint is not None:
            self.talc_model, self.talc_checkpoint_meta = load_binary_segmentation_checkpoint(Path(talc_checkpoint), self.device)
            self.talc_model.eval()
            self.talc_checkpoint = str(talc_checkpoint)
        self.checkpoint = str(checkpoint)
        self.tile_size = tile_size
        self.stride = stride
        self.batch_size = batch_size
        self.threshold = threshold
        self.talc_threshold = talc_threshold
        self.preview_max_side = preview_max_side
        self._weight = _tile_weight(tile_size)

    # -- sulfide inference (mirrors scripts/infer_binary_sulfide.py main()) --
    def infer_sulfide(self, image: Image.Image, out_dir: Path, *, image_path: str) -> dict[str, Any]:
        started = time.time()
        out_dir.mkdir(parents=True, exist_ok=True)
        width, height = image.size
        tiles = iter_tiles(width=width, height=height, tile_size=self.tile_size, stride=self.stride)

        # Tiled probability accumulation with graceful degradation (plan 39 F2/F3):
        # OOM -> shrink batch and retry (memmap accumulators keep this off-RAM); a hard
        # model failure -> brightness-heuristic fallback. Every degradation is recorded
        # on this run so the result is never silently presented as nominal.
        degradations: list[dict[str, Any]] = []

        def _model_forward(batch_tiles: list[Tile]) -> np.ndarray:
            tensor = torch.stack([_preprocess_tile(image, tile) for tile in batch_tiles]).to(self.device)
            logits = forward_logits(self.model, tensor, (self.tile_size, self.tile_size))
            return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)

        try:
            prob, processed, effective_batch, oom_degradations = _accumulate_prob_map(
                forward_fn=_model_forward,
                tiles=tiles,
                weight=self._weight,
                width=width,
                height=height,
                batch_size=self.batch_size,
                out_dir=out_dir,
            )
            degradations.extend(oom_degradations)
            backend = "model_oom_shrunk" if oom_degradations else "model"
        except Exception as exc:  # noqa: BLE001 - degrade to a heuristic rather than losing the run
            _release_device_cache()
            prob = brightness_sulfide_pseudo_mask(np.asarray(image, dtype=np.uint8)).mask.astype(np.float32)
            processed = 0
            effective_batch = 0
            backend = "heuristic_fallback"
            degradations.append(
                {
                    "code": "model_fallback_heuristic",
                    "detail": f"sulfide model inference failed ({type(exc).__name__}: {exc}); used brightness heuristic",
                    "severity": "error",
                }
            )
        confidence = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        mask = (prob >= self.threshold).astype(np.uint8) * 255
        analyzed_mask = build_analyzed_mask(np.asarray(image, dtype=np.uint8))

        mask_path = out_dir / "sulfide_mask.png"
        confidence_path = out_dir / "confidence.png"
        analyzed_path = out_dir / "analyzed_mask.png"
        overlay_preview_path = out_dir / "overlay_preview.jpg"
        save_gray(mask_path, mask)
        save_gray(confidence_path, confidence)
        save_gray(analyzed_path, analyzed_mask.astype(np.uint8) * 255)
        _save_overlay(image=image, mask=mask, confidence=confidence, path=overlay_preview_path, max_side=self.preview_max_side)

        image_area = int(mask.size)
        analyzed_area = int((analyzed_mask > 0).sum())
        sulfide_area = int(((mask > 0) & (analyzed_mask > 0)).sum())
        summary = {
            "schema_version": "binary-sulfide-inference-v0.2",
            "image": image_path,
            "checkpoint": self.checkpoint,
            "checkpoint_meta": self.checkpoint_meta,
            "width": width,
            "height": height,
            "tile_size": self.tile_size,
            "stride": self.stride,
            "tiles": len(tiles),
            "tiles_processed": processed,
            "backend": backend,
            "batch_size_effective": effective_batch,
            "result_quality": "degraded" if degradations else "nominal",
            "degradations": degradations,
            "threshold": self.threshold,
            "device": str(self.device),
            "seconds": round(time.time() - started, 3),
            "image_area_px": image_area,
            "analyzed_area_px": analyzed_area,
            "analyzed_fraction": analyzed_area / max(image_area, 1),
            "sulfide_area_px": sulfide_area,
            "sulfide_fraction": sulfide_area / max(analyzed_area, 1),
            "sulfide_fraction_image": sulfide_area / max(image_area, 1),
            "paths": {
                "sulfide_mask": str(mask_path),
                "confidence": str(confidence_path),
                "analyzed_mask": str(analyzed_path),
                "overlay_preview": str(overlay_preview_path),
                "overlay_full": None,
            },
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return summary

    def infer_talc(
        self,
        image: Image.Image,
        sulfide_mask: np.ndarray,
        out_dir: Path,
        *,
        image_path: str,
    ) -> dict[str, Any]:
        if self.talc_model is None or self.talc_checkpoint is None:
            raise RuntimeError("talc checkpoint was not loaded")
        started = time.time()
        out_dir.mkdir(parents=True, exist_ok=True)
        width, height = image.size
        tiles = iter_tiles(width=width, height=height, tile_size=self.tile_size, stride=self.stride)

        with tempfile.TemporaryDirectory(prefix="talc_infer_", dir=str(out_dir)) as tmp:
            prob_sum = np.memmap(Path(tmp) / "prob_sum.dat", mode="w+", dtype=np.float32, shape=(height, width))
            weight_sum = np.memmap(Path(tmp) / "weight_sum.dat", mode="w+", dtype=np.float32, shape=(height, width))
            processed = 0
            with torch.no_grad():
                for batch_tiles in _batched(tiles, self.batch_size):
                    tensor = torch.stack([_preprocess_tile(image, tile) for tile in batch_tiles]).to(self.device)
                    logits = forward_logits(self.talc_model, tensor, (self.tile_size, self.tile_size))
                    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)
                    for tile, prob in zip(batch_tiles, probs, strict=True):
                        valid_h = min(tile.height, height - tile.y)
                        valid_w = min(tile.width, width - tile.x)
                        tile_weight_valid = self._weight[:valid_h, :valid_w]
                        y_slice = slice(tile.y, tile.y + valid_h)
                        x_slice = slice(tile.x, tile.x + valid_w)
                        prob_sum[y_slice, x_slice] += prob[:valid_h, :valid_w] * tile_weight_valid
                        weight_sum[y_slice, x_slice] += tile_weight_valid
                        processed += 1

            prob = np.asarray(prob_sum / np.maximum(weight_sum, 1e-6), dtype=np.float32)

        rgb = np.asarray(image, dtype=np.uint8)
        analyzed_mask = build_analyzed_mask(rgb).astype(bool)
        sulfide_bool = _as_bool_mask(sulfide_mask, (height, width))
        non_sulfide_mask = analyzed_mask & ~sulfide_bool
        talc_bool = (prob >= self.talc_threshold) & non_sulfide_mask

        confidence = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        confidence_non_sulfide = confidence.copy()
        confidence_non_sulfide[~non_sulfide_mask] = 0
        talc_mask = talc_bool.astype(np.uint8) * 255

        talc_path = out_dir / "talc_mask.png"
        confidence_path = out_dir / "confidence.png"
        confidence_non_sulfide_path = out_dir / "confidence_non_sulfide.png"
        analyzed_path = out_dir / "analyzed_mask.png"
        non_sulfide_path = out_dir / "non_sulfide_mask.png"
        sulfide_path = out_dir / "sulfide_mask_aligned.png"
        overlay_preview_path = out_dir / "overlay_preview.jpg"
        save_gray(talc_path, talc_mask)
        save_gray(confidence_path, confidence)
        save_gray(confidence_non_sulfide_path, confidence_non_sulfide)
        save_gray(analyzed_path, analyzed_mask.astype(np.uint8) * 255)
        save_gray(non_sulfide_path, non_sulfide_mask.astype(np.uint8) * 255)
        save_gray(sulfide_path, sulfide_bool.astype(np.uint8) * 255)
        _save_overlay(
            image=image,
            mask=talc_mask,
            confidence=confidence_non_sulfide,
            path=overlay_preview_path,
            max_side=self.preview_max_side,
            color_rgb=(255.0, 196.0, 0.0),
        )

        image_area = width * height
        analyzed_area = int(analyzed_mask.sum())
        non_sulfide_area = int(non_sulfide_mask.sum())
        talc_area = int(talc_bool.sum())
        summary = {
            "schema_version": "binary-talc-inference-v0.1",
            "image": image_path,
            "checkpoint": self.talc_checkpoint,
            "checkpoint_meta": self.talc_checkpoint_meta,
            "sulfide_mask": str(sulfide_path),
            "width": width,
            "height": height,
            "tile_size": self.tile_size,
            "stride": self.stride,
            "tiles": len(tiles),
            "tiles_processed": processed,
            "threshold": self.talc_threshold,
            "device": str(self.device),
            "seconds": round(time.time() - started, 3),
            "image_area_px": image_area,
            "analyzed_area_px": analyzed_area,
            "non_sulfide_area_px": non_sulfide_area,
            "talc_area_px": talc_area,
            "talc_fraction_non_sulfide": talc_area / max(non_sulfide_area, 1),
            "talc_fraction_analyzed": talc_area / max(analyzed_area, 1),
            "talc_fraction_image": talc_area / max(image_area, 1),
            "paths": {
                "talc_mask": str(talc_path),
                "confidence": str(confidence_path),
                "confidence_non_sulfide": str(confidence_non_sulfide_path),
                "analyzed_mask": str(analyzed_path),
                "non_sulfide_mask": str(non_sulfide_path),
                "sulfide_mask_aligned": str(sulfide_path),
                "overlay_preview": str(overlay_preview_path),
                "overlay_full": None,
            },
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return summary

    # -- full per-image pipeline (mirrors scripts/run_ore_pipeline.py) --
    def run_image(
        self,
        image_path: str | Path,
        out_dir: Path,
        *,
        rule_config: dict[str, Any] | None = None,
        min_component_area_px: int = 128,
        close_kernel_px: int = 21,
        talc_min_area_px: int = 320,
        auto_talc_candidate: bool = True,
    ) -> dict[str, Any]:
        image_path = str(image_path)
        out_dir = Path(out_dir)
        rule_config = rule_config or default_rule_config()
        inference_dir = out_dir / "binary_sulfide"
        analysis_dir = out_dir / "ore_analysis"
        talc_dir = out_dir / "talc_candidate"
        talc_model_dir = out_dir / "talc_model"

        image = Image.open(image_path).convert("RGB")
        image_arr = np.asarray(image, dtype=np.uint8)
        sulfide_summary = self.infer_sulfide(image, inference_dir, image_path=image_path)
        run_degradations: list[dict[str, Any]] = list(sulfide_summary.get("degradations", []))

        sulfide_arr = np.asarray(Image.open(inference_dir / "sulfide_mask.png").convert("L"))
        talc_mask_path: Path | None = None
        talc_paths: dict[str, str] = {}
        talc_summary: dict[str, Any] = {}
        talc_source = "none"
        talc_model_failed = False
        if self.talc_model is not None:
            try:
                talc_summary = self.infer_talc(
                    image=image,
                    sulfide_mask=sulfide_arr,
                    out_dir=talc_model_dir,
                    image_path=image_path,
                )
                talc_summary_paths = talc_summary.get("paths") if isinstance(talc_summary.get("paths"), dict) else {}
                talc_mask_path = Path(str(talc_summary_paths.get("talc_mask") or talc_model_dir / "talc_mask.png"))
                talc_paths = {
                    "talc_model_summary": str(talc_model_dir / "summary.json"),
                    "talc_model_overlay_preview": str(talc_summary_paths.get("overlay_preview") or talc_model_dir / "overlay_preview.jpg"),
                    "talc_model_confidence": str(talc_summary_paths.get("confidence") or talc_model_dir / "confidence.png"),
                    "talc_model_confidence_non_sulfide": str(
                        talc_summary_paths.get("confidence_non_sulfide") or talc_model_dir / "confidence_non_sulfide.png"
                    ),
                }
                talc_source = "ml_model"
            except Exception as exc:  # noqa: BLE001 - degrade to heuristic candidate, don't lose the run
                _release_device_cache()
                talc_model_failed = True
                run_degradations.append(
                    {
                        "code": "model_fallback_heuristic",
                        "detail": f"talc model inference failed ({type(exc).__name__}: {exc}); used heuristic talc candidate",
                        "severity": "error",
                    }
                )
        if talc_source == "none" and (auto_talc_candidate or talc_model_failed):
            cfg = TalcCandidateConfig(min_area_px=talc_min_area_px)
            talc_arr = estimate_talc_candidate_mask(image_arr, sulfide_mask=sulfide_arr, config=cfg)
            talc_paths = save_talc_candidate_outputs(
                out_dir=talc_dir,
                rgb=image_arr,
                talc_mask=talc_arr,
                sulfide_mask=sulfide_arr,
                config=cfg,
                preview_max_side=self.preview_max_side,
            )
            talc_mask_path = Path(talc_paths["talc_candidate_mask"])
            talc_source = "auto_candidate"

        talc_mask = None if talc_mask_path is None else np.asarray(Image.open(talc_mask_path).convert("L"))
        analyzed_mask = np.asarray(Image.open(inference_dir / "analyzed_mask.png").convert("L"))
        component_cfg = ComponentRuleConfig(
            min_component_area_px=min_component_area_px,
            close_kernel_px=close_kernel_px,
            fine_dark_inside_ratio=rule_config["fine_dark_inside_ratio"],
            fine_solidity_max=rule_config["fine_solidity_max"],
            fine_compactness_max=rule_config["fine_compactness_max"],
            talc_fraction_threshold=rule_config["talc_fraction_threshold"],
        )
        summary, components, classified = analyze_components(
            sulfide_mask=sulfide_arr,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            config=component_cfg,
        )
        save_component_outputs(
            out_dir=analysis_dir,
            summary=summary,
            components=components,
            classified_mask=classified,
            original_image=image_arr,
            talc_mask=talc_mask,
            analyzed_mask=analyzed_mask,
            preview_max_side=self.preview_max_side,
        )

        pipeline_summary = {
            "schema_version": "ore-pipeline-run-v0.2",
            "image": image_path,
            "result_quality": "degraded" if run_degradations else "nominal",
            "degradations": run_degradations,
            "checkpoint": self.checkpoint,
            "talc_source": talc_source,
            "talc_checkpoint": self.talc_checkpoint,
            "talc_threshold": self.talc_threshold if self.talc_checkpoint is not None else None,
            "talc_checkpoint_meta": talc_summary.get("checkpoint_meta") if talc_summary else None,
            "rule_config": rule_config,
            "paths": {
                "binary_sulfide_summary": str(inference_dir / "summary.json"),
                "sulfide_mask": str(inference_dir / "sulfide_mask.png"),
                "confidence": str(inference_dir / "confidence.png"),
                "analyzed_mask": str(inference_dir / "analyzed_mask.png"),
                "sulfide_overlay_preview": str(inference_dir / "overlay_preview.jpg"),
                "talc_mask": str(talc_mask_path) if talc_mask_path is not None else None,
                "talc_candidate_summary": talc_paths.get("talc_candidate_summary"),
                "talc_candidate_overlay_preview": talc_paths.get("talc_candidate_overlay_preview"),
                "talc_model_summary": talc_paths.get("talc_model_summary"),
                "talc_model_overlay_preview": talc_paths.get("talc_model_overlay_preview"),
                "talc_model_confidence": talc_paths.get("talc_model_confidence"),
                "talc_model_confidence_non_sulfide": talc_paths.get("talc_model_confidence_non_sulfide"),
                "ore_summary": str(analysis_dir / "ore_summary.json"),
                "component_features": str(analysis_dir / "component_features.csv"),
                "analysis_analyzed_mask": str(analysis_dir / "analyzed_mask.png"),
                "intergrowth_overlay_preview": str(analysis_dir / "intergrowth_overlay_preview.jpg"),
            },
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pipeline_summary.json").write_text(
            json.dumps(pipeline_summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return pipeline_summary


def _is_oom_error(exc: BaseException) -> bool:
    """True for GPU/host out-of-memory errors recoverable by shrinking the batch."""
    if isinstance(exc, MemoryError):
        return True
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(exc, cuda_oom):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _release_device_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cache release is best-effort
        pass


def _accumulate_prob_map(
    *,
    forward_fn,
    tiles: list[Tile],
    weight: np.ndarray,
    width: int,
    height: int,
    batch_size: int,
    out_dir: Path,
    oom_max_retries: int = 2,
) -> tuple[np.ndarray, int, int, list[dict[str, Any]]]:
    """Weighted tiled probability accumulation on disk-backed memmaps.

    ``forward_fn(batch_tiles)`` returns a ``[B, tile, tile]`` float32 array of
    positive-class probabilities. On out-of-memory the batch size is halved (after
    releasing the device cache) and the whole map is recomputed, recording an
    ``oom_batch_shrunk`` degradation each time. Returns ``(prob_map, tiles_processed,
    effective_batch_size, degradations)``. Non-OOM errors, and OOM that survives
    ``batch_size == 1``, propagate to the caller (which decides on a fallback).
    """
    degradations: list[dict[str, Any]] = []
    attempt_batch = max(1, int(batch_size))
    while True:
        try:
            with tempfile.TemporaryDirectory(prefix="resident_infer_", dir=str(out_dir)) as tmp:
                prob_sum = np.memmap(Path(tmp) / "prob_sum.dat", mode="w+", dtype=np.float32, shape=(height, width))
                weight_sum = np.memmap(Path(tmp) / "weight_sum.dat", mode="w+", dtype=np.float32, shape=(height, width))
                processed = 0
                with torch.no_grad():
                    for batch_tiles in _batched(tiles, attempt_batch):
                        probs = forward_fn(batch_tiles)
                        for tile, prob in zip(batch_tiles, probs, strict=True):
                            valid_h = min(tile.height, height - tile.y)
                            valid_w = min(tile.width, width - tile.x)
                            tile_weight_valid = weight[:valid_h, :valid_w]
                            y_slice = slice(tile.y, tile.y + valid_h)
                            x_slice = slice(tile.x, tile.x + valid_w)
                            prob_sum[y_slice, x_slice] += prob[:valid_h, :valid_w] * tile_weight_valid
                            weight_sum[y_slice, x_slice] += tile_weight_valid
                            processed += 1
                prob = np.asarray(prob_sum / np.maximum(weight_sum, 1e-6), dtype=np.float32)
            return prob, processed, attempt_batch, degradations
        except Exception as exc:  # noqa: BLE001 - classify OOM (retry) vs fatal (propagate)
            if _is_oom_error(exc) and attempt_batch > 1 and len(degradations) < oom_max_retries:
                _release_device_cache()
                new_batch = max(1, attempt_batch // 2)
                degradations.append(
                    {
                        "code": "oom_batch_shrunk",
                        "detail": f"out-of-memory at batch_size={attempt_batch}; retrying at {new_batch}",
                        "severity": "warning",
                    }
                )
                attempt_batch = new_batch
                continue
            raise


def _preprocess_tile(image: Image.Image, tile: Tile) -> torch.Tensor:
    crop = image.crop((tile.x, tile.y, tile.x + tile.width, tile.y + tile.height))
    if crop.size != (tile.width, tile.height):
        padded = Image.new("RGB", (tile.width, tile.height), (0, 0, 0))
        padded.paste(crop, (0, 0))
        crop = padded
    tensor = TF.to_tensor(crop)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def _tile_weight(tile_size: int) -> np.ndarray:
    if tile_size <= 2:
        return np.ones((tile_size, tile_size), dtype=np.float32)
    one_d = np.hanning(tile_size).astype(np.float32)
    one_d = np.maximum(one_d, 0.05)
    weight = np.outer(one_d, one_d)
    return (weight / weight.max()).astype(np.float32)


def _save_overlay(
    image: Image.Image,
    mask: np.ndarray,
    confidence: np.ndarray,
    path: Path,
    max_side: int,
    *,
    color_rgb: tuple[float, float, float] = (255.0, 216.0, 0.0),
) -> None:
    base = image.copy()
    mask_img = Image.fromarray(mask, mode="L")
    conf_img = Image.fromarray(confidence, mode="L")
    if max_side and max(base.size) > max_side:
        scale = max_side / float(max(base.size))
        new_size = (max(1, int(base.size[0] * scale)), max(1, int(base.size[1] * scale)))
        base = base.resize(new_size, Image.Resampling.BILINEAR)
        mask_img = mask_img.resize(new_size, Image.Resampling.NEAREST)
        conf_img = conf_img.resize(new_size, Image.Resampling.BILINEAR)
    base_arr = np.asarray(base).astype(np.float32)
    mask_arr = np.asarray(mask_img) > 0
    conf_arr = np.asarray(conf_img).astype(np.float32) / 255.0
    color = np.zeros_like(base_arr)
    color[..., 0] = color_rgb[0]
    color[..., 1] = color_rgb[1]
    color[..., 2] = color_rgb[2]
    alpha = np.where(mask_arr, 0.25 + 0.45 * conf_arr, 0.0)[..., None]
    overlay = base_arr * (1.0 - alpha) + color * alpha
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(path, quality=92, optimize=True)


def _as_bool_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] != target_hw:
        image = Image.fromarray(mask.astype(np.uint8), mode="L")
        image = image.resize((target_hw[1], target_hw[0]), Image.Resampling.NEAREST)
        mask = np.asarray(image, dtype=np.uint8)
    return mask > 0


def _batched(items: list[Tile], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]
