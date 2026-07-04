from __future__ import annotations

import json
import shutil
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from apps.ore_pipeline_web import OrePipelineStore  # noqa: E402
from ore_classifier.gis_export import (  # noqa: E402
    COORDINATE_SPACE,
    GIS_SCHEMA_VERSION,
    GisClassSpec,
    build_geojson_feature_collection,
    write_geojson_export,
    write_shapefile_zip_export,
)


CLASS_SPECS = [
    GisClassSpec(1, "ordinary", "ordinary intergrowth"),
    GisClassSpec(2, "fine", "fine intergrowth"),
    GisClassSpec(3, "talc", "talc"),
]


def calibrated_scale(area_um2_per_pixel: float = 0.25) -> dict[str, object]:
    return {
        "schema_version": "ore-pipeline-scale-v0.1",
        "available": True,
        "source_field": "microns_per_pixel",
        "microns_per_source_pixel": area_um2_per_pixel**0.5,
        "microns_per_analysis_pixel_x": area_um2_per_pixel**0.5,
        "microns_per_analysis_pixel_y": area_um2_per_pixel**0.5,
        "effective_microns_per_analysis_pixel": area_um2_per_pixel**0.5,
        "area_um2_per_analysis_pixel": area_um2_per_pixel,
        "scale_source": "calibration_slide",
        "scale_confidence": "calibrated",
        "source_width": 30,
        "source_height": 20,
        "analysis_width": 30,
        "analysis_height": 20,
    }


class GisGeojsonExportTest(unittest.TestCase):
    def test_geojson_features_include_classes_local_coordinates_and_area(self) -> None:
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[2:8, 3:13] = 1
        mask[10:18, 4:12] = 2
        mask[5:11, 20:27] = 3

        collection = build_geojson_feature_collection(
            mask,
            class_specs=CLASS_SPECS,
            run_id="run_geojson",
            source_mask="masks/final_mask.png",
            scale=calibrated_scale(0.25),
            simplify_tolerance_px=0,
        )

        self.assertEqual(collection["type"], "FeatureCollection")
        self.assertEqual(collection["metadata"]["schema_version"], GIS_SCHEMA_VERSION)
        self.assertEqual(collection["metadata"]["coordinate_space"], COORDINATE_SPACE)
        self.assertEqual(collection["metadata"]["feature_count"], 3)
        features_by_key = {feature["properties"]["class_key"]: feature for feature in collection["features"]}
        self.assertEqual(set(features_by_key), {"ordinary", "fine", "talc"})
        self.assertEqual(features_by_key["ordinary"]["properties"]["area_px"], 60)
        self.assertEqual(features_by_key["fine"]["properties"]["area_px"], 64)
        self.assertEqual(features_by_key["talc"]["properties"]["area_px"], 42)
        self.assertAlmostEqual(features_by_key["ordinary"]["properties"]["area_um2"], 15.0)
        self.assertEqual(features_by_key["ordinary"]["properties"]["coordinate_space"], COORDINATE_SPACE)
        for feature in collection["features"]:
            ring = feature["geometry"]["coordinates"][0]
            self.assertEqual(ring[0], ring[-1])
            self.assertEqual(feature["geometry"]["type"], "Polygon")

    def test_geojson_preserves_polygon_holes(self) -> None:
        mask = np.zeros((14, 14), dtype=np.uint8)
        mask[2:10, 2:10] = 1
        mask[4:6, 4:6] = 0

        collection = build_geojson_feature_collection(
            mask,
            class_specs=[CLASS_SPECS[0]],
            run_id="run_hole",
            source_mask="masks/final_mask.png",
            simplify_tolerance_px=0,
        )

        self.assertEqual(len(collection["features"]), 1)
        feature = collection["features"][0]
        self.assertEqual(feature["properties"]["area_px"], 60)
        self.assertEqual(len(feature["geometry"]["coordinates"]), 2)
        self.assertNotIn("area_um2", feature["properties"])

    def test_write_geojson_export_persists_utf8_collection(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="test_gis_geojson_"))
        try:
            mask = np.zeros((10, 10), dtype=np.uint8)
            mask[1:5, 1:5] = 1
            mask_path = root / "final_mask.png"
            output_path = root / "final_classes.geojson"
            Image.fromarray(mask, mode="L").save(mask_path)

            written = write_geojson_export(
                mask_path,
                output_path,
                class_specs=[GisClassSpec(1, "ordinary", "Обычные срастания")],
                run_id="run_write",
                source_mask="masks/final_mask.png",
            )

            loaded = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["metadata"]["schema_version"], GIS_SCHEMA_VERSION)
            self.assertEqual(loaded["features"][0]["properties"]["class_label"], "Обычные срастания")
            self.assertEqual(loaded["metadata"]["feature_count"], written["metadata"]["feature_count"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_write_shapefile_zip_exports_polygon_package(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="test_gis_shapefile_"))
        try:
            mask = np.zeros((20, 30), dtype=np.uint8)
            mask[2:8, 3:13] = 1
            mask[10:18, 4:12] = 2
            collection = build_geojson_feature_collection(
                mask,
                class_specs=CLASS_SPECS,
                run_id="run_shp",
                source_mask="masks/final_mask.png",
                scale=calibrated_scale(1.0),
                simplify_tolerance_px=0,
            )
            zip_path = root / "final_classes_shapefile.zip"

            result = write_shapefile_zip_export(collection, zip_path, layer_name="final_classes")

            self.assertEqual(result["feature_count"], 2)
            with zipfile.ZipFile(zip_path) as archive:
                names = set(archive.namelist())
                shp = archive.read("final_classes.shp")
                shx = archive.read("final_classes.shx")
                dbf = archive.read("final_classes.dbf")
                cpg = archive.read("final_classes.cpg")
            self.assertEqual(names, {"final_classes.shp", "final_classes.shx", "final_classes.dbf", "final_classes.cpg"})
            self.assertEqual(cpg, b"UTF-8\n")
            self.assertEqual(struct.unpack(">i", shp[:4])[0], 9994)
            self.assertEqual(struct.unpack("<i", shp[32:36])[0], 5)
            self.assertEqual(struct.unpack(">i", shx[:4])[0], 9994)
            self.assertEqual(struct.unpack("<L", dbf[4:8])[0], 2)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class OrePipelineGisFinalizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="test_ore_pipeline_gis_"))
        self.store = OrePipelineStore(
            workspace_dir=self.root / "workspace",
            backend="heuristic",
            checkpoint=None,
            processing_max_side=128,
            panorama_max_side=128,
            preview_max_sides=(64, 128),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_synthetic_completed_run_inputs(self) -> tuple[Path, dict[str, object]]:
        run_dir = self.store.runs_dir / "run_gis"
        (run_dir / "input/original_source").mkdir(parents=True)
        (run_dir / "reports").mkdir(parents=True)
        (run_dir / "display").mkdir(parents=True)
        (run_dir / "masks").mkdir(parents=True)
        Image.new("RGB", (12, 10), "white").save(run_dir / "input/preprocessed.png")
        Image.new("RGB", (12, 10), "white").save(run_dir / "input/original_source/sample.png")
        final_mask = np.zeros((10, 12), dtype=np.uint8)
        final_mask[1:4, 1:5] = 1
        final_mask[5:8, 1:7] = 2
        final_mask[2:6, 8:11] = 3
        Image.fromarray(final_mask, mode="L").save(run_dir / "masks/final_mask.png")
        Image.fromarray(((final_mask > 0).astype(np.uint8) * 255), mode="L").save(run_dir / "masks/sulfide_mask.png")
        Image.fromarray(((final_mask == 3).astype(np.uint8) * 255), mode="L").save(run_dir / "masks/talc_mask.png")
        Image.fromarray(np.full((10, 12), 255, dtype=np.uint8), mode="L").save(run_dir / "masks/analyzed_mask.png")
        Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8), mode="RGB").save(run_dir / "masks/sulfide_component_labels_rgb.png")
        summary = {
            "ore_class": "row_ore",
            "ore_class_ru": "рядовая руда",
            "sulfide_fraction": 30 / 120,
            "sulfide_fraction_image": 30 / 120,
            "ordinary_sulfide_fraction": 12 / 30,
            "fine_sulfide_fraction": 18 / 30,
            "talc_fraction": 12 / 120,
            "talc_fraction_image": 12 / 120,
            "sulfide_area_px": 30,
            "ordinary_sulfide_area_px": 12,
            "fine_sulfide_area_px": 18,
            "talc_area_px": 12,
            "image_area_px": 120,
            "analysis_area_px": 120,
            "analyzed_fraction": 1.0,
            "component_count": 2,
            "warnings": [],
        }
        (run_dir / "reports/ore_summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
        (run_dir / "reports/component_features.csv").write_text("", encoding="utf-8")
        (run_dir / "display/display.json").write_text(json.dumps({"layers": {}}), encoding="utf-8")
        metadata = {
            "run_id": "run_gis",
            "status": "complete",
            "backend": "heuristic",
            "input": {
                "upload_id": "upload_gis",
                "original_artifact_path": str(run_dir / "input/original_source/sample.png"),
                "curated_metadata": {
                    "domain": {
                        "microns_per_pixel": 2.0,
                        "scale_source": "calibration_slide",
                        "scale_confidence": "calibrated",
                    }
                },
            },
            "preprocess": {"enabled": True, "preset": {}},
            "tiling": {
                "source_width": 12,
                "source_height": 10,
                "analysis_width": 12,
                "analysis_height": 10,
            },
        }
        return run_dir, metadata

    def test_run_finalization_writes_geojson_report_metadata(self) -> None:
        run_dir, metadata = self._write_synthetic_completed_run_inputs()

        self.store._finalize_run_metadata(metadata, run_dir)

        geojson_path = run_dir / "reports/final_classes.geojson"
        shapefile_path = run_dir / "reports/final_classes_shapefile.zip"
        self.assertTrue(geojson_path.exists())
        self.assertTrue(shapefile_path.exists())
        self.assertEqual(metadata["reports"]["final_classes_geojson"], str(geojson_path))
        self.assertEqual(metadata["reports"]["final_classes_shapefile_zip"], str(shapefile_path))
        self.assertEqual(metadata["gis_exports"]["geojson"], str(geojson_path))
        self.assertEqual(metadata["gis_exports"]["shapefile_zip"], str(shapefile_path))
        payload = json.loads(geojson_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["metadata"]["feature_count"], 3)
        self.assertEqual(payload["metadata"]["coordinate_space"], COORDINATE_SPACE)
        ordinary = next(feature for feature in payload["features"] if feature["properties"]["class_key"] == "ordinary")
        self.assertEqual(ordinary["properties"]["area_px"], 12)
        self.assertAlmostEqual(ordinary["properties"]["area_um2"], 48.0)
        with zipfile.ZipFile(shapefile_path) as archive:
            self.assertIn("final_classes.shp", archive.namelist())

    def test_run_files_payload_and_artifact_zip_include_gis_exports(self) -> None:
        run_dir, metadata = self._write_synthetic_completed_run_inputs()
        self.store._finalize_run_metadata(metadata, run_dir)
        self.store._write_json(run_dir / "run.json", metadata)

        files_payload = self.store.run_files_payload("run_gis")
        files_by_path = {item["path"]: item for item in files_payload["files"]}

        self.assertIn("reports/final_classes.geojson", files_by_path)
        self.assertIn("reports/final_classes_shapefile.zip", files_by_path)
        self.assertFalse(files_by_path["reports/final_classes.geojson"]["is_image"])
        self.assertFalse(files_by_path["reports/final_classes_shapefile.zip"]["is_image"])

        run_zip = self.store.run_zip_path("run_gis")
        with zipfile.ZipFile(run_zip) as archive:
            names = set(archive.namelist())
        self.assertIn("reports/final_classes.geojson", names)
        self.assertIn("reports/final_classes_shapefile.zip", names)
        self.assertNotIn("reports/run_artifacts.zip", names)


if __name__ == "__main__":
    unittest.main()
