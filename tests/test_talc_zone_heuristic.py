"""Regression tests for the model-free heuristic talcose classifier.

Synthetic tests pin the dependency-light talc-zone pipeline and exercise the
standalone CLI artifact contract.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.talc_zone_heuristic import (  # noqa: E402
    TalcZoneConfig,
    detect_talc_zones,
    opaque_phase_mask,
    result_to_dict,
)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _scattered_flakes_image(h=400, w=600, matrix_luma=150, flake_density=0.0, seed=0):
    """Gray matrix with optional scattered small dark flakes, no ore."""
    rng = _rng(seed)
    img = np.full((h, w, 3), matrix_luma, dtype=np.uint8)
    img = (img.astype(np.int16) + rng.integers(-8, 9, img.shape)).clip(0, 255).astype(np.uint8)
    if flake_density > 0:
        n = int(flake_density * h * w / 9)
        ys = rng.integers(2, h - 3, n)
        xs = rng.integers(2, w - 3, n)
        for y, x in zip(ys, xs):
            img[y - 1:y + 2, x - 1:x + 2] = rng.integers(20, 60)  # dark 3x3 flake
    return img


class TalcZoneHeuristicTest(unittest.TestCase):
    def test_dense_scattered_flakes_are_talcose(self):
        img = _scattered_flakes_image(flake_density=0.5, seed=1)
        ore = np.zeros(img.shape[:2], dtype=bool)
        res = detect_talc_zones(img, ore_mask=ore, config=TalcZoneConfig())
        self.assertGreater(res.talc_fraction, 0.5)
        self.assertTrue(res.is_talcose)

    def test_plain_matrix_is_not_talcose(self):
        img = _scattered_flakes_image(flake_density=0.0, seed=2)
        ore = np.zeros(img.shape[:2], dtype=bool)
        res = detect_talc_zones(img, ore_mask=ore, config=TalcZoneConfig())
        self.assertLess(res.talc_fraction, 0.1)
        self.assertFalse(res.is_talcose)

    def test_large_solid_dark_blob_is_not_talc(self):
        # a big solid dark rectangle (pore/shadow) must be dropped, not counted as talc
        img = _scattered_flakes_image(flake_density=0.0, seed=3)
        img[80:320, 120:480] = 25  # solid dark block ~36% of frame
        ore = np.zeros(img.shape[:2], dtype=bool)
        res = detect_talc_zones(img, ore_mask=ore, config=TalcZoneConfig())
        self.assertLess(res.talc_fraction, 0.1)
        self.assertFalse(res.is_talcose)

    def test_ore_is_excluded_from_matrix(self):
        img = _scattered_flakes_image(flake_density=0.0, seed=4)
        img[:, :300] = 220  # bright ore-like half
        ore = np.zeros(img.shape[:2], dtype=bool)
        ore[:, :300] = True
        cfg = TalcZoneConfig()
        res = detect_talc_zones(img, ore_mask=ore, config=cfg)
        # matrix (measured at proc_width) is ~ the right (non-ore) half
        proc_area = cfg.proc_width * round(img.shape[0] * cfg.proc_width / img.shape[1])
        self.assertLess(abs(res.matrix_area_px / proc_area - 0.5), 0.05)

    def test_result_dict_shape(self):
        img = _scattered_flakes_image(flake_density=0.3, seed=5)
        cfg = TalcZoneConfig()
        res = detect_talc_zones(img, ore_mask=np.zeros(img.shape[:2], bool), config=cfg)
        d = result_to_dict(res, cfg)
        self.assertEqual(d["schema_version"], "talc-zone-heuristic-v1")
        self.assertIn(d["ore_class"], ("talcose_ore", "not_talcose"))
        self.assertIn("talc_fraction", d)
        self.assertIn("config", d)

    def test_opaque_phase_mask_selects_bright(self):
        img = _scattered_flakes_image(matrix_luma=90, flake_density=0.0, seed=6)
        img[:, :200] = 235  # bright opaque region
        mask = opaque_phase_mask(img)
        self.assertGreater(mask[:, :200].mean(), 0.8)   # bright region captured
        self.assertLess(mask[:, 400:].mean(), 0.2)      # dark matrix not captured

    def test_cli_single_image_writes_result_masks_and_overlay(self):
        outputs_root = ROOT / "outputs"
        outputs_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test_talc_zone_cli_", dir=outputs_root) as tmp:
            work = Path(tmp)
            image_path = work / "sample.png"
            ore_mask_path = work / "ore_mask.png"
            out_dir = work / "out"
            img = _scattered_flakes_image(h=180, w=240, flake_density=0.35, seed=7)
            ore_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            ore_mask[:, :20] = 255
            Image.fromarray(img, mode="RGB").save(image_path)
            Image.fromarray(ore_mask, mode="L").save(ore_mask_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/classify_talcose_heuristic.py"),
                    "--image",
                    str(image_path),
                    "--ore-mask",
                    str(ore_mask_path),
                    "--out-dir",
                    str(out_dir),
                    "--classify-threshold",
                    "0.01",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertIn("[1/1] sample.png:", completed.stdout)
            self.assertIn("done: 1 images", completed.stdout)
            sample_dir = out_dir / "sample"
            result_path = sample_dir / "talcose_result.json"
            self.assertTrue(result_path.exists())
            self.assertTrue((sample_dir / "talc_zone_mask.png").exists())
            self.assertTrue((sample_dir / "talc_flake_mask.png").exists())
            self.assertTrue((sample_dir / "overlay.jpg").exists())
            record = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(record["schema_version"], "talc-zone-heuristic-v1")
            self.assertEqual(record["image"], str(image_path))
            self.assertEqual(record["ore_mask_source"], str(ore_mask_path))
            self.assertIn(record["ore_class"], ("talcose_ore", "not_talcose"))
            self.assertIn("paths", record)


if __name__ == "__main__":
    unittest.main()
