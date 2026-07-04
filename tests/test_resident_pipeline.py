"""Tests for the resident (single-load) ore pipeline.

Covers the pure helpers and a real end-to-end ``run_image`` on a tiny ResUNet
checkpoint (CPU), for both talc paths: the auto talc candidate and the loaded
talc model (``talc_source == "ml_model"``).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.models import create_resunet  # noqa: E402
from ore_classifier.resident_pipeline import (  # noqa: E402
    ResidentSulfidePipeline,
    _as_bool_mask,
    _batched,
    _preprocess_tile,
    _save_overlay,
    _tile_weight,
)
from ore_classifier.tiling import Tile  # noqa: E402


def write_tiny_checkpoint(path: Path, *, base_channels: int = 4) -> None:
    model = create_resunet(base_channels=base_channels)
    torch.save(
        {
            "model": model.state_dict(),
            "args": {"model": "resunet", "base_channels": base_channels},
            "epoch": 1,
            "best_iou_sulfide": 0.5,
        },
        path,
    )


def sample_image_array(width: int = 100, height: int = 80) -> np.ndarray:
    rng = np.random.default_rng(11)
    rgb = rng.integers(60, 200, size=(height, width, 3), dtype=np.uint8)
    rgb[20:60, 30:70] = (230, 228, 215)  # bright sulfide-like blob
    return rgb


class TileWeightTest(unittest.TestCase):
    def test_weight_shape_peak_and_floor(self) -> None:
        weight = _tile_weight(64)
        self.assertEqual(weight.shape, (64, 64))
        self.assertEqual(weight.dtype, np.float32)
        self.assertAlmostEqual(float(weight.max()), 1.0, places=6)
        # hanning is clamped at 0.05 before normalization so edges never zero out
        self.assertGreater(float(weight.min()), 0.0)

    def test_degenerate_tile_size_returns_ones(self) -> None:
        for size in (1, 2):
            weight = _tile_weight(size)
            self.assertEqual(weight.shape, (size, size))
            self.assertTrue(np.all(weight == 1.0))


class AsBoolMaskTest(unittest.TestCase):
    def test_matching_shape_thresholds_over_zero(self) -> None:
        mask = np.array([[0, 128], [255, 0]], dtype=np.uint8)
        result = _as_bool_mask(mask, (2, 2))
        self.assertEqual(result.dtype, np.bool_)
        self.assertEqual(result.tolist(), [[False, True], [True, False]])

    def test_mismatched_shape_is_resized_nearest(self) -> None:
        mask = np.zeros((2, 2), dtype=np.uint8)
        mask[0, 0] = 255
        result = _as_bool_mask(mask, (4, 4))
        self.assertEqual(result.shape, (4, 4))
        self.assertTrue(result[0, 0])
        self.assertFalse(result[3, 3])


class BatchedTest(unittest.TestCase):
    def test_chunks_preserve_order_and_tail(self) -> None:
        tiles = [Tile(x=i, y=0, width=8, height=8) for i in range(5)]
        batches = list(_batched(tiles, 2))
        self.assertEqual([len(b) for b in batches], [2, 2, 1])
        self.assertEqual([t.x for b in batches for t in b], [0, 1, 2, 3, 4])

    def test_empty_input_yields_nothing(self) -> None:
        self.assertEqual(list(_batched([], 3)), [])


class PreprocessTileTest(unittest.TestCase):
    def test_interior_tile_shape_and_normalization(self) -> None:
        image = Image.fromarray(sample_image_array(), mode="RGB")
        tensor = _preprocess_tile(image, Tile(x=0, y=0, width=32, height=32))
        self.assertEqual(tuple(tensor.shape), (3, 32, 32))
        # ImageNet normalization moves values out of [0, 1]
        self.assertLess(float(tensor.min()), 0.0)

    def test_edge_tile_is_padded_to_full_tile_size(self) -> None:
        image = Image.fromarray(sample_image_array(width=40, height=40), mode="RGB")
        # crop extends 24 px past the right/bottom edge -> black padding
        tensor = _preprocess_tile(image, Tile(x=16, y=16, width=48, height=48))
        self.assertEqual(tuple(tensor.shape), (3, 48, 48))


class SaveOverlayTest(unittest.TestCase):
    def test_writes_downscaled_rgb_jpeg(self) -> None:
        rgb = sample_image_array(width=120, height=90)
        mask = np.zeros((90, 120), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        confidence = np.full((90, 120), 200, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "overlay.jpg"
            _save_overlay(
                image=Image.fromarray(rgb, mode="RGB"),
                mask=mask,
                confidence=confidence,
                path=path,
                max_side=60,
            )
            self.assertTrue(path.exists())
            with Image.open(path) as overlay:
                self.assertEqual(overlay.mode, "RGB")
                self.assertEqual(max(overlay.size), 60)

    def test_no_downscale_when_within_max_side(self) -> None:
        rgb = sample_image_array(width=50, height=40)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "overlay.jpg"
            _save_overlay(
                image=Image.fromarray(rgb, mode="RGB"),
                mask=np.zeros((40, 50), dtype=np.uint8),
                confidence=np.zeros((40, 50), dtype=np.uint8),
                path=path,
                max_side=1800,
            )
            with Image.open(path) as overlay:
                self.assertEqual(overlay.size, (50, 40))


class ResidentPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp_path = Path(cls._tmp.name)
        cls.checkpoint = cls.tmp_path / "tiny_resunet.pt"
        write_tiny_checkpoint(cls.checkpoint)
        cls.image_path = cls.tmp_path / "sample.png"
        Image.fromarray(sample_image_array(), mode="RGB").save(cls.image_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def make_pipeline(self, **overrides) -> ResidentSulfidePipeline:
        kwargs = {
            "device": "cpu",
            "tile_size": 64,
            "stride": 48,
            "batch_size": 2,
            "preview_max_side": 64,
        }
        kwargs.update(overrides)
        return ResidentSulfidePipeline(self.checkpoint, **kwargs)

    def test_stride_larger_than_tile_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.make_pipeline(tile_size=32, stride=48)

    def test_infer_talc_without_talc_model_raises(self) -> None:
        pipeline = self.make_pipeline()
        with self.assertRaises(RuntimeError):
            pipeline.infer_talc(
                Image.open(self.image_path).convert("RGB"),
                np.zeros((80, 100), dtype=np.uint8),
                self.tmp_path / "talc_out",
                image_path=str(self.image_path),
            )

    def test_run_image_with_auto_talc_candidate(self) -> None:
        pipeline = self.make_pipeline()
        out_dir = self.tmp_path / "run_auto_talc"
        summary = pipeline.run_image(self.image_path, out_dir)

        self.assertEqual(summary["schema_version"], "ore-pipeline-run-v0.2")
        self.assertEqual(summary["talc_source"], "auto_candidate")
        self.assertIsNone(summary["talc_checkpoint"])
        self.assertTrue((out_dir / "pipeline_summary.json").exists())
        self.assertTrue((out_dir / "binary_sulfide/sulfide_mask.png").exists())
        self.assertTrue((out_dir / "binary_sulfide/confidence.png").exists())
        self.assertTrue((out_dir / "binary_sulfide/analyzed_mask.png").exists())
        self.assertTrue((out_dir / "binary_sulfide/overlay_preview.jpg").exists())
        self.assertTrue((out_dir / "ore_analysis/ore_summary.json").exists())
        self.assertTrue((out_dir / "ore_analysis/component_features.csv").exists())

        sulfide_summary = json.loads((out_dir / "binary_sulfide/summary.json").read_text(encoding="utf-8"))
        self.assertEqual(sulfide_summary["schema_version"], "binary-sulfide-inference-v0.2")
        self.assertEqual(sulfide_summary["width"], 100)
        self.assertEqual(sulfide_summary["height"], 80)
        self.assertEqual(sulfide_summary["tiles"], sulfide_summary["tiles_processed"])
        self.assertGreaterEqual(sulfide_summary["sulfide_fraction"], 0.0)
        self.assertLessEqual(sulfide_summary["sulfide_fraction"], 1.0)
        # the sulfide mask has the source geometry
        with Image.open(out_dir / "binary_sulfide/sulfide_mask.png") as mask:
            self.assertEqual(mask.size, (100, 80))

    def test_run_image_with_talc_model(self) -> None:
        # reuse the same tiny binary checkpoint as the talc segmenter
        pipeline = self.make_pipeline(talc_checkpoint=self.checkpoint, talc_threshold=0.4)
        out_dir = self.tmp_path / "run_ml_talc"
        summary = pipeline.run_image(self.image_path, out_dir)

        self.assertEqual(summary["talc_source"], "ml_model")
        self.assertEqual(summary["talc_checkpoint"], str(self.checkpoint))
        self.assertEqual(summary["talc_threshold"], 0.4)
        talc_summary = json.loads((out_dir / "talc_model/summary.json").read_text(encoding="utf-8"))
        self.assertEqual(talc_summary["schema_version"], "binary-talc-inference-v0.1")
        for key in (
            "talc_mask",
            "confidence",
            "confidence_non_sulfide",
            "analyzed_mask",
            "non_sulfide_mask",
            "sulfide_mask_aligned",
            "overlay_preview",
        ):
            self.assertTrue(Path(talc_summary["paths"][key]).exists(), key)
        # talc pixels never overlap sulfide pixels: talc is clipped to analyzed & ~sulfide
        talc = np.asarray(Image.open(out_dir / "talc_model/talc_mask.png").convert("L")) > 0
        sulfide = np.asarray(Image.open(out_dir / "talc_model/sulfide_mask_aligned.png").convert("L")) > 0
        self.assertEqual(int((talc & sulfide).sum()), 0)
        self.assertLessEqual(talc_summary["talc_fraction_non_sulfide"], 1.0)


if __name__ == "__main__":
    unittest.main()
