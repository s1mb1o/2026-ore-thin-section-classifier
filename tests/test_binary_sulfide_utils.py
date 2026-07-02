from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ore_classifier.pseudo_labels import (  # noqa: E402
    brightness_sulfide_pseudo_mask,
    lumenstone_binary_mask,
    parse_class_ids,
)
from ore_classifier.tiling import crop_array_with_pad, iter_tiles  # noqa: E402


class BinarySulfideUtilsTest(unittest.TestCase):
    def test_lumenstone_sulfide_mapping_excludes_magnetite(self) -> None:
        mask = np.array(
            [
                [[0, 0, 0], [1, 1, 1], [3, 3, 3]],
                [[5, 5, 5], [7, 7, 7], [10, 10, 10]],
            ],
            dtype=np.uint8,
        )
        pseudo = lumenstone_binary_mask(mask)
        self.assertEqual(int(pseudo.mask[0, 0]), 0)
        self.assertEqual(int(pseudo.mask[0, 1]), 1)
        self.assertEqual(int(pseudo.mask[0, 2]), 0)
        self.assertEqual(int(pseudo.mask[1, 0]), 1)
        self.assertEqual(int(pseudo.mask[1, 1]), 1)
        self.assertEqual(int(pseudo.mask[1, 2]), 0)
        self.assertEqual(int(pseudo.ignore.max()), 0)

    def test_parse_class_ids_accepts_comma_separated_override(self) -> None:
        self.assertEqual(parse_class_ids("1, 5,7"), (1, 5, 7))

    def test_brightness_pseudo_mask_finds_bright_metallic_region(self) -> None:
        image = np.full((64, 64, 3), (40, 55, 35), dtype=np.uint8)
        image[18:46, 20:48] = (235, 230, 165)
        pseudo = brightness_sulfide_pseudo_mask(image, min_area=12, uncertainty_margin=6)
        self.assertGreater(int(pseudo.mask[32, 32]), 0)
        self.assertEqual(int(pseudo.mask[5, 5]), 0)
        self.assertEqual(pseudo.mask.shape, (64, 64))
        self.assertEqual(pseudo.ignore.shape, (64, 64))
        self.assertIsNotNone(pseudo.threshold)

    def test_iter_tiles_covers_edges_and_padding(self) -> None:
        tiles = iter_tiles(width=10, height=7, tile_size=4, stride=3)
        starts = {(tile.x, tile.y) for tile in tiles}
        self.assertIn((0, 0), starts)
        self.assertIn((6, 3), starts)

        array = np.arange(3 * 5, dtype=np.uint8).reshape(3, 5)
        padded = crop_array_with_pad(array, tiles[-1], fill_value=255)
        self.assertEqual(padded.shape, (4, 4))
        self.assertEqual(int(padded[-1, -1]), 255)


if __name__ == "__main__":
    unittest.main()
