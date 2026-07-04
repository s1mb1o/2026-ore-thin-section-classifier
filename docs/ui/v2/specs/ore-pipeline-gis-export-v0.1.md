# Ore Pipeline GIS Export v0.1

Date: 2026-07-04

## Scope

This spec applies to the v2 local ore pipeline service:

```text
apps/ore_pipeline_web.py
```

It covers the official wish-list requirement:

```text
Интеграция с ГИС: экспорт результатов в форматы, совместимые с
геологическими информационными системами (Shapefile, GeoJSON).
```

## Goals

- Export mask-derived final classification polygons in a GIS-readable format.
- Support GeoJSON first, then Shapefile as a packaged `.zip`.
- Keep the output deterministic and local-only.
- Preserve the existing run artifact model: GIS exports are immutable files in
  the run directory and therefore appear in file listings and artifact ZIPs.
- Avoid false georeferencing claims when the input has no stage/world transform.

## Coordinate Contract

The current official image inputs do not provide world coordinates, stage
coordinates, CRS, or a calibrated affine transform. GIS exports therefore use a
local image pixel coordinate space:

```text
coordinate_space = local_image_pixel_top_left
x = analysis pixel column
y = analysis pixel row
origin = top-left of the analysis mask grid
```

The coordinate values are directly comparable to the generated masks and
overlays. They are not longitude/latitude and not projected map coordinates.

When calibrated scale metadata is available through the existing v2 metadata
contract, polygon properties include physical area fields:

```json
{
  "domain": {
    "microns_per_pixel": 0.5,
    "scale_source": "calibration_slide",
    "scale_confidence": "calibrated"
  }
}
```

Scale handling follows `ore-pipeline-scale-metrics-v0.1.md`: physical areas are
emitted only when the source is calibrated and the pixel size is positive.

## Exported Layers

Phase 1 exports the final three-class result mask:

```text
reports/final_classes.geojson
```

Features are generated from `masks/final_mask.png` with these class values:

| Value | Key | Label |
| --- | --- | --- |
| 1 | `ordinary` | `Обычные срастания` |
| 2 | `fine` | `Тонкие срастания` |
| 3 | `talc` | `Тальк` |

Phase 2 adds the Shapefile package:

```text
reports/final_classes_shapefile.zip
```

The package contains:

- `final_classes.shp`
- `final_classes.shx`
- `final_classes.dbf`
- `final_classes.cpg`

## GeoJSON Contract

The GeoJSON file is a `FeatureCollection` with project metadata at top level:

```json
{
  "type": "FeatureCollection",
  "metadata": {
    "schema_version": "ore-pipeline-gis-export-v0.1",
    "coordinate_space": "local_image_pixel_top_left",
    "source_mask": "masks/final_mask.png"
  },
  "features": []
}
```

Each feature is a polygon or multi-part polygon encoded as standard GeoJSON
geometry. Feature properties include:

- `feature_id`
- `run_id`
- `class_id`
- `class_key`
- `class_label`
- `source_mask`
- `area_px`
- `bbox_px`
- `coordinate_space`
- `area_um2` and `area_mm2` when calibrated scale is available

Small contours below the configured minimum polygon area are skipped to avoid
noisy single-pixel GIS artifacts.

## Shapefile Contract

The Shapefile export mirrors the GeoJSON features in ESRI Polygon format. It
uses the same local pixel coordinate space and stores compact DBF attributes:

| DBF field | Meaning |
| --- | --- |
| `FID` | Stable feature index |
| `RUN_ID` | Run identifier, truncated if needed |
| `CLASS_ID` | Numeric final-mask class |
| `CLASS_KEY` | Stable ASCII class key |
| `LABEL` | Human-readable class label |
| `AREA_PX` | Polygon area in analysis pixels |
| `AREA_UM2` | Physical area, blank when unavailable |
| `AREA_MM2` | Physical area, blank when unavailable |

The package includes `final_classes.cpg` with `UTF-8`. No `.prj` is emitted
until the system has a real CRS or local engineering coordinate definition.

## Non-Goals

- Do not infer CRS, stage coordinates, or georeferencing from EXIF/DPI data.
- Do not export latitude/longitude.
- Do not require GDAL/Fiona/GeoPandas for the baseline local build.
- Do not change mask semantics or classification thresholds.
- Do not make component-level GIS layers part of v0.1; they can be added after
  final-class exports are stable.

## Acceptance Criteria

- Every completed run with `masks/final_mask.png` writes
  `reports/final_classes.geojson`.
- GeoJSON features contain the official final classes and local image-pixel
  coordinates.
- Calibrated runs include physical polygon areas; uncalibrated runs omit them.
- The run metadata points to the generated GIS exports.
- The run file listing and artifact ZIP include GIS exports automatically.
- Phase 2 writes a valid Shapefile ZIP without adding heavyweight GIS runtime
  dependencies.

## Verification

```bash
python3 -m py_compile src/ore_classifier/gis_export.py apps/ore_pipeline_web.py tests/test_gis_export.py
python3 -m unittest discover -s tests -p 'test_gis_export.py' -v
git diff --check -- \
  src/ore_classifier/gis_export.py \
  apps/ore_pipeline_web.py \
  tests/test_gis_export.py \
  docs/ui/v2/specs/ore-pipeline-gis-export-v0.1.md \
  docs/ui/v2/plans/41_ore-pipeline-gis-export.md
```
