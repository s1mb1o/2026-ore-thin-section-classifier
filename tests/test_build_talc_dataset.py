from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_talc_dataset import assign_sample_splits, build_dataset, sample_group  # noqa: E402


def write_rgb(path: Path, height: int, width: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(60, 200, size=(height, width, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def write_mask(path: Path, height: int, width: int, box: tuple[int, int, int, int] | None) -> None:
    mask = np.zeros((height, width), dtype=np.uint8)
    if box is not None:
        y0, y1, x0, x1 = box
        mask[y0:y1, x0:x1] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path)


def make_sample(conversion_dir: Path, clean_dir: Path, sample_id: str, *, seed: int, talc_box, ignore_box=None) -> None:
    height = width = 128
    sample_dir = conversion_dir / "samples" / sample_id
    write_rgb(clean_dir / f"{sample_id}.JPG", height, width, seed)
    write_mask(sample_dir / "reviewed" / "reviewed_talc_mask.png", height, width, talc_box)
    write_mask(sample_dir / "reviewed" / "reviewed_ignore_mask.png", height, width, ignore_box)


def run_builder(root: Path, **overrides):
    kwargs = dict(
        conversion_dir=root / "conversion",
        clean_image_dir=root / "clean",
        negative_dirs=[],
        max_negative_images=0,
        out_dir=root / "out",
        tile_size=64,
        stride=48,
        val_fraction=0.5,
        val_samples=set(),
        seed=7,
        max_tiles_per_source=6,
        min_positive_fraction=0.002,
        min_valid_fraction=0.30,
        negative_keep_fraction=0.5,
        analyzed_min_value=8,
        downscale_max_side=0,
        overwrite=True,
    )
    kwargs.update(overrides)
    return build_dataset(**kwargs)


class BuildTalcDatasetTest(unittest.TestCase):
    def test_builds_tiles_with_per_image_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conversion = root / "conversion"
            clean = root / "clean"
            make_sample(conversion, clean, "DSCN0001", seed=1, talc_box=(10, 60, 10, 60))
            make_sample(conversion, clean, "DSCN0002", seed=2, talc_box=(40, 100, 40, 100), ignore_box=(0, 16, 0, 128))

            manifest = run_builder(root)
            out_dir = root / "out"

            self.assertTrue((out_dir / "manifest.json").exists())
            items = manifest["items"]
            self.assertGreater(len(items), 0)

            splits_by_sample = defaultdict(set)
            for item in items:
                self.assertTrue((out_dir / item["image"]).exists())
                self.assertTrue((out_dir / item["mask"]).exists())
                self.assertTrue((out_dir / item["ignore"]).exists())
                splits_by_sample[item["sample_id"]].add(item["split"])
            for sample_id, splits in splits_by_sample.items():
                self.assertEqual(len(splits), 1, f"{sample_id} tiles span multiple splits: {splits}")
            self.assertTrue(any(item["positive_fraction"] > 0 for item in items))

            with (out_dir / "manifest.json").open(encoding="utf-8") as f:
                stored = json.load(f)
            self.assertEqual(stored["task"], "binary_talc")
            self.assertEqual(len(stored["items"]), len(items))

    def test_reviewed_ignore_pixels_are_marked_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_sample(
                root / "conversion",
                root / "clean",
                "DSCN0003",
                seed=3,
                talc_box=(64, 120, 64, 120),
                ignore_box=(0, 128, 0, 32),
            )
            manifest = run_builder(root, val_fraction=0.0)
            out_dir = root / "out"

            left_edge_tiles = [item for item in manifest["items"] if item["x"] == 0]
            self.assertTrue(left_edge_tiles)
            found_ignore = False
            for item in left_edge_tiles:
                ignore = np.asarray(Image.open(out_dir / item["ignore"]).convert("L"))
                talc = np.asarray(Image.open(out_dir / item["mask"]).convert("L"))
                self.assertFalse(bool(((ignore > 0) & (talc > 0)).any()))
                if (ignore[:, :16] > 0).mean() > 0.9:
                    found_ignore = True
            self.assertTrue(found_ignore, "reviewed ignore stripe not reflected in ignore tiles")

    def test_talc_mask_overrides_analyzed_border_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conversion = root / "conversion"
            clean = root / "clean"
            sample_id = "DSCN0004"
            height = width = 128
            rng = np.random.default_rng(4)
            rgb = rng.integers(60, 200, size=(height, width, 3), dtype=np.uint8)
            rgb[32:96, 32:96] = 2  # near-black talc region below analyzed floor
            image_path = clean / f"{sample_id}.JPG"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb, mode="RGB").save(image_path)
            sample_dir = conversion / "samples" / sample_id
            write_mask(sample_dir / "reviewed" / "reviewed_talc_mask.png", height, width, (32, 96, 32, 96))

            manifest = run_builder(root, val_fraction=0.0)
            out_dir = root / "out"
            positive_total = 0
            for item in manifest["items"]:
                talc = np.asarray(Image.open(out_dir / item["mask"]).convert("L"))
                ignore = np.asarray(Image.open(out_dir / item["ignore"]).convert("L"))
                self.assertFalse(bool(((ignore > 0) & (talc > 0)).any()))
                positive_total += int((talc > 0).sum())
            self.assertGreater(positive_total, 0, "dark reviewed talc must stay positive, not ignored")

    def test_samples_without_reviewed_mask_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            conversion = root / "conversion"
            clean = root / "clean"
            make_sample(conversion, clean, "DSCN0005", seed=5, talc_box=(10, 60, 10, 60))
            unreviewed_dir = conversion / "samples" / "DSCN0006"
            unreviewed_dir.mkdir(parents=True)
            write_rgb(clean / "DSCN0006.JPG", 128, 128, seed=6)

            manifest = run_builder(root)

            self.assertEqual(manifest["stats"]["samples_without_reviewed_mask"], 1)
            self.assertEqual(manifest["stats"]["reviewed_samples"], 1)
            self.assertTrue(all(item["sample_id"] == "DSCN0005" for item in manifest["items"]))

    def test_negative_dir_adds_pure_negative_tiles_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_sample(root / "conversion", root / "clean", "DSCN0007", seed=7, talc_box=(10, 60, 10, 60))
            negative_dir = root / "negatives"
            write_rgb(negative_dir / "neg_a.JPG", 128, 128, seed=8)
            duplicate = negative_dir / "neg_b.JPG"
            duplicate.parent.mkdir(parents=True, exist_ok=True)
            duplicate.write_bytes((negative_dir / "neg_a.JPG").read_bytes())

            manifest = run_builder(
                root,
                negative_dirs=[negative_dir],
                max_negative_images=5,
            )

            negative_items = [item for item in manifest["items"] if item["source_type"] == "negative_official"]
            self.assertTrue(negative_items)
            self.assertTrue(all(item["positive_fraction"] == 0 for item in negative_items))
            self.assertEqual(manifest["stats"]["negative_duplicates_skipped"], 1)
            out_dir = root / "out"
            for item in negative_items:
                mask = np.asarray(Image.open(out_dir / item["mask"]).convert("L"))
                self.assertEqual(int((mask > 0).sum()), 0)

    def test_split_assignment_is_stratified_and_respects_forced_val(self) -> None:
        samples = [
            {"sample_id": f"DSCN000{i}", "group": "dscn_na"} for i in range(4)
        ] + [
            {"sample_id": f"255038{i}-1 10x", "group": "scan_10x"} for i in range(4)
        ]
        splits = assign_sample_splits(
            samples,
            val_fraction=0.25,
            forced_val={"DSCN0000"},
            rng=random.Random(3),
        )
        self.assertEqual(splits["DSCN0000"], "val")
        for group in ("dscn_na", "scan_10x"):
            group_ids = [s["sample_id"] for s in samples if s["group"] == group]
            self.assertEqual(sum(splits[sid] == "val" for sid in group_ids), 1)

    def test_sample_group_parsing(self) -> None:
        self.assertEqual(sample_group("DSCN3042"), "dscn_na")
        self.assertEqual(sample_group("2550381-1 10x"), "scan_10x")
        self.assertEqual(sample_group("2550375-3 5х"), "scan_5x")  # cyrillic х


if __name__ == "__main__":
    unittest.main()
