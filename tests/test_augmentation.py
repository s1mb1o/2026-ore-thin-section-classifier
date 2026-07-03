from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.augmentation import apply_augmentation, normalize_augmentation_settings  # noqa: E402


class RuntimeAugmentationTest(unittest.TestCase):
    def test_surface_artifacts_are_deterministic_and_geometry_preserving(self) -> None:
        source = Image.fromarray(np.full((96, 128, 3), 128, dtype=np.uint8), mode="RGB")
        settings = normalize_augmentation_settings(
            {
                "enabled": True,
                "color": {
                    "brightness_pct": 0,
                    "contrast_pct": 0,
                    "saturation_pct": 0,
                    "hue_degrees": 0,
                    "gamma": 1,
                },
                "acquisition": {"blur_radius": 0, "gaussian_noise_std": 0},
                "surface_artifacts": {
                    "scratch_count": 12,
                    "scratch_intensity_pct": 35,
                    "polishing_haze_pct": 18,
                    "pit_count": 40,
                    "pit_intensity_pct": 30,
                },
                "runtime": {"random_seed": 123},
            }
        )

        first = np.asarray(apply_augmentation(source, settings))
        second = np.asarray(apply_augmentation(source, settings))

        self.assertEqual(first.shape, (96, 128, 3))
        self.assertFalse(np.array_equal(first, np.asarray(source)))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(settings["runtime"]["coordinate_mode"], "original")
        self.assertTrue(settings["runtime"]["geometry_preserving"])


if __name__ == "__main__":
    unittest.main()
