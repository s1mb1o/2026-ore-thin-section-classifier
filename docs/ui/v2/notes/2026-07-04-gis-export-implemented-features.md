# GIS Export Implemented Features

Date: 2026-07-04

Scope:

```text
apps/ore_pipeline_web.py
src/ore_classifier/gis_export.py
tests/test_gis_export.py
```

Related contract:

```text
docs/ui/v2/specs/ore-pipeline-gis-export-v0.1.md
docs/ui/v2/plans/41_ore-pipeline-gis-export.md
```

## Implemented

1. GeoJSON run export.
   - Completed runs write `reports/final_classes.geojson`.
   - The export is generated from `masks/final_mask.png`.
   - The file is a standard GeoJSON `FeatureCollection`.
   - The top-level metadata records schema version, run id, source mask,
     image dimensions, feature count, local coordinate contract, simplification
     tolerance, and scale provenance.

2. Shapefile run export.
   - Completed runs write `reports/final_classes_shapefile.zip`.
   - The package contains:
     - `final_classes.shp`
     - `final_classes.shx`
     - `final_classes.dbf`
     - `final_classes.cpg`
   - The writer is implemented locally and does not require GDAL, Fiona,
     GeoPandas, Rasterio, or pyshp.
   - DBF output uses compact fields for stable GIS inspection:
     `FID`, `RUN_ID`, `CLASS_ID`, `CLASS_KEY`, `LABEL`, `AREA_PX`,
     `AREA_UM2`, `AREA_MM2`.

3. Final-class polygon layers.
   - Exported classes match the final mask semantics:
     - `1`: `ordinary`
     - `2`: `fine`
     - `3`: `talc`
   - Polygon features are extracted with OpenCV contours.
   - Polygon holes are preserved in GeoJSON and Shapefile records.
   - Small contours below the configured minimum area are skipped to avoid
     single-pixel GIS noise.

4. Coordinate contract.
   - Coordinates are local analysis-image pixels.
   - Origin is the top-left of the analysis mask grid.
   - `x` is pixel column and `y` is pixel row.
   - The coordinate space is recorded as
     `local_image_pixel_top_left`.
   - No longitude/latitude, projected CRS, or stage/world georeferencing is
     claimed.

5. Physical area fields.
   - Pixel area is always exported as `area_px`.
   - `area_um2` and `area_mm2` are exported only when the existing calibrated
     scale metadata contract is satisfied.
   - The scale rule is shared with `ore-pipeline-scale-metrics-v0.1.md`.
   - Uncalibrated runs keep GIS polygons but omit physical-area properties.

6. Run metadata integration.
   - Completed `run.json` payloads include:
     - `reports.final_classes_geojson`
     - `reports.final_classes_shapefile_zip`
     - `gis_exports.geojson`
     - `gis_exports.shapefile_zip`
     - `gis_exports.feature_count`
     - `gis_exports.shapefile_feature_count`
   - Export generation is hooked into `_finalize_run_metadata`, so normal runs
     and edit-derived completed runs share the same export path.

7. Artifact listing and bundle integration.
   - `GET /api/runs/{run_id}/files` lists the GeoJSON and Shapefile ZIP as
     regular run files.
   - `GET /api/runs/{run_id}/artifacts.zip` includes:
     - `reports/final_classes.geojson`
     - `reports/final_classes_shapefile.zip`
   - The generated `reports/run_artifacts.zip` still excludes itself.

8. Content validation tests.
   - `tests/test_gis_export.py` validates:
     - GeoJSON class features, local coordinate metadata, closed rings, and
       calibrated physical areas.
     - Polygon holes.
     - UTF-8 GeoJSON labels.
     - Shapefile ZIP members.
     - Shapefile headers, polygon shape type, record count, part count, point
       count, and record bounding box.
     - DBF rows and expected class/area attributes.
     - Run finalization metadata.
     - Run file listing and full artifact ZIP contents, including nested
       Shapefile ZIP members.

## Verification Commands

```bash
python3 -m py_compile tests/test_gis_export.py src/ore_classifier/gis_export.py apps/ore_pipeline_web.py
python3 -m unittest discover -s tests -p 'test_gis_export.py' -v
git diff --check -- tests/test_gis_export.py src/ore_classifier/gis_export.py apps/ore_pipeline_web.py docs/ui/v2/notes/2026-07-04-gis-export-implemented-features.md
```

## Commits

```text
6c805fd docs: specify ore pipeline GIS exports
feca7c3 feat: export ore classes as GeoJSON
83c01da feat: package ore GIS exports as Shapefile
8311877 test: cover ore GIS artifact exports
d3cc2a4 test: validate GIS export contents
```

## Not Implemented

- True CRS/georeferencing.
- `.prj` export.
- Longitude/latitude or projected coordinates.
- Component-level GIS layers.
- Edit-history vector layers.
- Direct GIS server integration.
