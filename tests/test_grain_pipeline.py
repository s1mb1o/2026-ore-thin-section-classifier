from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "apps"))

from ore_classifier.specimen import specimen_group  # noqa: E402
from ore_classifier.grain_features import (  # noqa: E402
    FEATURE_NAMES,
    GRAIN_CLASS_ORDER,
    build_grain_feature_matrix,
    grain_feature_vector,
    resolve_grain_label,
)
import aggregate_grade_from_grains as agg  # noqa: E402
import build_grain_dataset as bgd  # noqa: E402
import train_grain_classifier as tgc  # noqa: E402

import numpy as np  # noqa: E402


class SpecimenGroupTest(unittest.TestCase):
    def test_specimen_number_groups_photos_of_one_section(self):
        a = specimen_group("Фото руд по сортам. ч1/Труднообогатимые руды/2652976 10x 2.JPG")
        b = specimen_group("Фото руд по сортам. ч1/Труднообогатимые руды/2652976 5x 1.JPG")
        self.assertEqual(a, b)
        self.assertIn("spec:2652976", a)

    def test_ch2_sequential_counters_do_not_collide_across_grade_folders(self):
        # Cause B from the audit: '150_.jpg' (оталькованные) and '150.JPG' (рядовые)
        # are UNRELATED images and must not merge into one group.
        a = specimen_group("Фото руд по сортам. ч2/оталькованные/150_.jpg")
        b = specimen_group("Фото руд по сортам. ч2/рядовые/150.JPG")
        self.assertNotEqual(a, b)

    def test_dscn_names_fall_back_to_per_file_singletons(self):
        a = specimen_group("Фото руд по сортам. ч1/Оталькованные руды/DSCN4719.JPG")
        b = specimen_group("Фото руд по сортам. ч1/Оталькованные руды/DSCN4720.JPG")
        self.assertNotEqual(a, b)
        self.assertIn("file:DSCN4719", a)


class GrainFeatureTest(unittest.TestCase):
    def _row(self, **over):
        row = {
            "area_px": 1000, "footprint_area_px": 1250, "dark_inside_area_px": 250,
            "dark_inside_ratio": 0.2, "solidity": 0.8, "compactness": 0.3,
            "boundary_complexity": 5.0, "bbox_w": 50, "bbox_h": 25,
        }
        row.update(over)
        return row

    def test_feature_vector_length_matches_names(self):
        self.assertEqual(len(grain_feature_vector(self._row())), len(FEATURE_NAMES))

    def test_engineered_features_are_correct(self):
        vec = dict(zip(FEATURE_NAMES, grain_feature_vector(self._row())))
        self.assertAlmostEqual(vec["aspect_ratio"], 50 / 25)
        self.assertAlmostEqual(vec["extent"], 1000 / (50 * 25))
        self.assertAlmostEqual(vec["footprint_fill"], 1000 / 1250)
        self.assertAlmostEqual(vec["dark_inside_area_frac"], 250 / 1250)

    def test_matrix_handles_empty_and_bad_values(self):
        self.assertEqual(build_grain_feature_matrix([]).shape, (0, len(FEATURE_NAMES)))
        row = self._row(area_px="", bbox_h=0)  # bad/zero must not raise or produce nan/inf
        matrix = build_grain_feature_matrix([row])
        self.assertEqual(matrix.shape, (1, len(FEATURE_NAMES)))
        self.assertTrue((matrix == matrix).all())  # no NaN

    def test_resolve_label_human_wins_uncertain_skipped_bootstrap_fallback(self):
        row = {"grain_uid": "g1", "heuristic_label": "ordinary_intergrowth"}
        anns = {"g1": {"label": "fine_intergrowth"}}
        self.assertEqual(resolve_grain_label(row, anns, require_human=False), "fine_intergrowth")
        anns_unc = {"g1": {"label": "uncertain"}}
        self.assertIsNone(resolve_grain_label(row, anns_unc, require_human=False))
        # no annotation -> heuristic bootstrap, unless require_human
        self.assertEqual(resolve_grain_label(row, None, require_human=False), "ordinary_intergrowth")
        self.assertIsNone(resolve_grain_label(row, None, require_human=True))


class GradeRuleTest(unittest.TestCase):
    def test_predict_grade_priority(self):
        # talc dominates when above its threshold
        self.assertEqual(agg.predict_grade(0.9, 0.5, 0.4, 0.1), "talcose_ore")
        # fine fraction above tau_fine and talc below -> hard_to_process
        self.assertEqual(agg.predict_grade(0.6, 0.0, 0.4, 0.1), "hard_to_process_ore")
        # neither -> row_ore
        self.assertEqual(agg.predict_grade(0.1, 0.0, 0.4, 0.1), "row_ore")

    def test_macro_f1_perfect_and_zero(self):
        truth = ["row_ore", "hard_to_process_ore", "talcose_ore"]
        self.assertAlmostEqual(agg.macro_f1(truth, truth), 1.0)
        flipped = ["talcose_ore", "row_ore", "hard_to_process_ore"]
        self.assertLess(agg.macro_f1(truth, flipped), 1.0)


class CropGrainTest(unittest.TestCase):
    def test_crop_within_bounds_and_downscaled(self):
        img = Image.new("RGB", (400, 300), (10, 20, 30))
        grain = {"bbox_x": 380, "bbox_y": 280, "bbox_w": 40, "bbox_h": 40}  # bbox exceeds bounds
        crop = bgd.crop_grain(img, grain, pad=10, max_side=32)
        self.assertLessEqual(max(crop.size), 32)
        self.assertGreater(crop.size[0], 0)
        self.assertGreater(crop.size[1], 0)

    def test_select_grains_filters_and_sorts(self):
        import csv
        import tempfile

        rows = [
            {"component_id": "1", "area_px": "100"},
            {"component_id": "2", "area_px": "5000"},
            {"component_id": "3", "area_px": "800"},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["component_id", "area_px"])
            writer.writeheader()
            writer.writerows(rows)
            path = Path(f.name)
        selected = bgd.read_and_select_grains(path, min_area=300, max_grains=10)
        self.assertEqual([g["component_id"] for g in selected], ["2", "3"])  # 100 dropped, sorted desc
        path.unlink()


class GroupedSplitGuardTest(unittest.TestCase):
    def test_raises_when_class_confined_to_one_group(self):
        # A grade/grain class confined to a single specimen group cannot be
        # grouped-CV'd; both entrypoints must fail loudly, never floor at 2.
        y = np.array([0] * 5 + [1] * 5)
        groups = np.array(["g"] * 10)
        for fn in (tgc.grouped_n_splits, agg.grouped_n_splits):
            with self.assertRaises(SystemExit):
                fn(y, groups, 5)

    def test_healthy_case_returns_min_of_folds_and_groups(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        groups = np.array(["a", "b", "c", "d", "e", "f"])
        self.assertEqual(tgc.grouped_n_splits(y, groups, 5), 3)
        self.assertEqual(agg.grouped_n_splits(y, groups, 2), 2)


class CropSandboxTest(unittest.TestCase):
    def _make_store(self):
        import csv
        import tempfile

        from grain_review_web import GrainReviewStore

        d = Path(tempfile.mkdtemp())
        ds = d / "ds"
        (ds / "crops").mkdir(parents=True)
        with (ds / "grains_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["grain_uid", "crop_path", "grade_label", "heuristic_label"])
            w.writeheader()
            w.writerow({"grain_uid": "g1", "crop_path": "crops/x/g1.png", "grade_label": "talcose", "heuristic_label": "fine_intergrowth"})
        return GrainReviewStore(ds), d

    def test_sibling_prefix_dir_cannot_escape_sandbox(self):
        from grain_review_web import ApiError

        store, d = self._make_store()
        sib = d / "ds" / "crops_backup"
        sib.mkdir()
        (sib / "leak.png").write_bytes(b"SECRET")
        with self.assertRaises(ApiError):
            store.crop_file("../crops_backup/leak.png")


class GrainPayloadTest(unittest.TestCase):
    """The labeling app must surface the full feature report + heuristic reasons
    so the annotator can decide ordinary vs fine (matches the v2 pipeline report)."""

    def _store_with_features(self, **feature_over):
        import csv
        import tempfile

        from grain_review_web import FEATURE_FIELDS, GrainReviewStore

        d = Path(tempfile.mkdtemp())
        ds = d / "ds"
        (ds / "crops").mkdir(parents=True)
        cols = ["grain_uid", "crop_path", "grade_label", "heuristic_label", *FEATURE_FIELDS]
        row = {c: "" for c in cols}
        row.update({"grain_uid": "g1", "crop_path": "crops/x/g1.png", "grade_label": "fine_intergrowth", "heuristic_label": "fine_intergrowth"})
        # A clearly-fine grain: high replacement, low solidity, low compactness.
        row.update({"area_px": "1000", "footprint_area_px": "1400", "dark_inside_area_px": "420",
                    "dark_inside_ratio": "0.30", "solidity": "0.53", "compactness": "0.035",
                    "boundary_complexity": "6.1", "bbox_w": "40", "bbox_h": "25"})
        row.update(feature_over)
        with (ds / "grains_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerow(row)
        return GrainReviewStore(ds)

    def test_payload_carries_features_signals_and_reasons(self):
        store = self._store_with_features()
        item = store._item_payload(0)
        for key in ("features", "fine_signals", "fine_reasons"):
            self.assertIn(key, item)
        # all three decisive metrics trip -> fine, with a human-readable reason each
        self.assertTrue(all(item["fine_signals"].values()))
        self.assertEqual(len(item["fine_reasons"]), 3)
        self.assertAlmostEqual(item["features"]["dark_inside_ratio"], 0.30)

    def test_ordinary_grain_has_no_fine_signals(self):
        store = self._store_with_features(dark_inside_ratio="0.05", solidity="0.90", compactness="0.40", heuristic_label="ordinary_intergrowth")
        item = store._item_payload(0)
        self.assertFalse(any(item["fine_signals"].values()))
        self.assertEqual(item["fine_reasons"], [])

    def test_bad_numeric_cell_yields_none_not_crash(self):
        store = self._store_with_features(solidity="")  # missing value must not raise
        item = store._item_payload(0)
        self.assertIsNone(item["features"]["solidity"])
        self.assertFalse(item["fine_signals"]["solidity"])  # None never trips a threshold


if __name__ == "__main__":
    unittest.main()
