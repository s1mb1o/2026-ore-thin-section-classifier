from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.scribble_classifier import (  # noqa: E402
    extract_pixel_features,
    fit_scribble_pixel_classifier,
)


class ScribbleClassifierTest(unittest.TestCase):
    def test_feature_extraction_shape(self) -> None:
        image = np.zeros((10, 12, 3), dtype=np.uint8)
        features, names = extract_pixel_features(image, scales=(3,))
        self.assertEqual(features.shape[:2], (10, 12))
        self.assertEqual(features.shape[2], len(names))
        self.assertIn("gray_grad_3", names)

    def test_scribble_classifier_predicts_bright_square(self) -> None:
        image = np.full((32, 32, 3), 30, dtype=np.uint8)
        image[10:22, 10:22] = 230
        labels = np.zeros((32, 32), dtype=np.uint8)
        labels[0:4, 0:4] = 1
        labels[14:18, 14:18] = 2
        classifier = fit_scribble_pixel_classifier(image, labels, scales=(3,))
        pred = classifier.predict_mask(image)
        self.assertEqual(int(pred[16, 16]), 2)
        self.assertEqual(int(pred[2, 2]), 1)
        probs = classifier.predict_proba(image)
        self.assertEqual(probs.shape, (2, 32, 32))


if __name__ == "__main__":
    unittest.main()
