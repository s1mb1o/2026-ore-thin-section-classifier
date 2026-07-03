from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.datasets import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    BinarySulfideTileDataset,
)


def _write_png(path: Path, array: np.ndarray, mode: str) -> None:
    Image.fromarray(array, mode=mode).save(path)


def _build_manifest(base_dir: Path) -> Path:
    """Create a tiny two-item manifest (train + val) with matching tiles."""
    # train item: a 4x4 tile with a foreground square and one ignore pixel.
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[1:3, 1:3] = (200, 180, 120)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    ignore = np.zeros((4, 4), dtype=np.uint8)
    ignore[0, 0] = 255

    _write_png(base_dir / "train_img.png", image, "RGB")
    _write_png(base_dir / "train_mask.png", mask, "L")
    _write_png(base_dir / "train_ignore.png", ignore, "L")

    # val item: all-background tile with no ignore pixels.
    _write_png(base_dir / "val_img.png", np.zeros((4, 4, 3), dtype=np.uint8), "RGB")
    _write_png(base_dir / "val_mask.png", np.zeros((4, 4), dtype=np.uint8), "L")
    _write_png(base_dir / "val_ignore.png", np.zeros((4, 4), dtype=np.uint8), "L")

    manifest = {
        "items": [
            {
                "split": "train",
                "image": "train_img.png",
                "mask": "train_mask.png",
                "ignore": "train_ignore.png",
                "id": "t0",
            },
            {
                "split": "val",
                "image": "val_img.png",
                "mask": "val_mask.png",
                "ignore": "val_ignore.png",
                "id": "v0",
            },
        ]
    }
    manifest_path = base_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class BinarySulfideTileDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.manifest = _build_manifest(self.base)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_split_filtering_and_length(self) -> None:
        train = BinarySulfideTileDataset(self.manifest, split="train")
        val = BinarySulfideTileDataset(self.manifest, split="val")
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)

    def test_missing_split_raises(self) -> None:
        with self.assertRaises(ValueError):
            BinarySulfideTileDataset(self.manifest, split="test")

    def test_target_encoding_maps_ignore_mask_background(self) -> None:
        dataset = BinarySulfideTileDataset(self.manifest, split="train", normalize=False)
        sample = dataset[0]
        target = sample["mask"].numpy()
        self.assertEqual(target.dtype, np.int64)
        # ignore pixel wins over everything else.
        self.assertEqual(int(target[0, 0]), 255)
        # foreground square encodes to 1.
        self.assertEqual(int(target[1, 1]), 1)
        # a plain background pixel encodes to 0.
        self.assertEqual(int(target[3, 3]), 0)

    def test_meta_is_roundtrippable_json(self) -> None:
        dataset = BinarySulfideTileDataset(self.manifest, split="train")
        meta = json.loads(dataset[0]["meta"])
        self.assertEqual(meta["id"], "t0")
        self.assertEqual(meta["split"], "train")

    def test_image_tensor_shape_and_dtype(self) -> None:
        dataset = BinarySulfideTileDataset(self.manifest, split="train", normalize=False)
        image = dataset[0]["image"]
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(tuple(image.shape), (3, 4, 4))
        # without normalization pixel values stay in [0, 1].
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_normalization_shifts_values(self) -> None:
        plain = BinarySulfideTileDataset(self.manifest, split="train", normalize=False)[0]["image"]
        normed = BinarySulfideTileDataset(self.manifest, split="train", normalize=True)[0]["image"]
        # background (zero) pixels become -mean/std under ImageNet normalization.
        expected_bg = torch.tensor(
            [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
            dtype=normed.dtype,
        )
        self.assertTrue(torch.allclose(normed[:, 3, 3], expected_bg, atol=1e-5))
        self.assertFalse(torch.allclose(plain[:, 3, 3], normed[:, 3, 3]))

    def test_augment_preserves_shapes_and_labels(self) -> None:
        random.seed(1234)
        dataset = BinarySulfideTileDataset(self.manifest, split="train", augment=True, normalize=False)
        sample = dataset[0]
        target = sample["mask"].numpy()
        self.assertEqual(tuple(sample["image"].shape), (3, 4, 4))
        self.assertEqual(target.shape, (4, 4))
        # geometry may flip/rotate but the set of label values is preserved.
        self.assertEqual(set(np.unique(target).tolist()), {0, 1, 255})


if __name__ == "__main__":
    unittest.main()
