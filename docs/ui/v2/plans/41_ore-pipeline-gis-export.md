# Ore Pipeline GIS Export Implementation Plan

Date: 2026-07-04

Spec:

```text
docs/ui/v2/specs/ore-pipeline-gis-export-v0.1.md
```

Implementation targets:

```text
src/ore_classifier/gis_export.py
apps/ore_pipeline_web.py
tests/test_gis_export.py
```

## Decision

Implement GIS exports as deterministic run artifacts created from
`masks/final_mask.png`. The output uses local analysis-pixel coordinates because
the current official image package does not provide stage/world georeferencing.
This keeps the feature useful in QGIS/ArcGIS for inspection and overlay while
remaining honest about coordinate provenance.

## Phase 0: Spec And Plan

Status: done.

1. Add the v0.1 GIS export contract under `docs/ui/v2/specs/`.
2. Reserve this plan as `docs/ui/v2/plans/41_ore-pipeline-gis-export.md`.
3. Document GeoJSON as the first implementation phase and Shapefile as the
   second phase.
4. Commit only the spec and plan.

## Phase 1: GeoJSON

Status: done.

1. Add `src/ore_classifier/gis_export.py`.
2. Convert final-mask class regions into polygon features with OpenCV contours.
3. Store top-level GeoJSON metadata with coordinate-space and scale provenance.
4. Add calibrated physical area properties when scale metadata is valid.
5. Hook export generation into completed run finalization in
   `apps/ore_pipeline_web.py`.
6. Add focused unit tests for polygon generation and run finalization.
7. Commit the GeoJSON implementation separately.

## Phase 2: Shapefile

Status: pending.

1. Add a lightweight Shapefile writer for the GeoJSON feature set.
2. Package `.shp`, `.shx`, `.dbf`, and `.cpg` as
   `reports/final_classes_shapefile.zip`.
3. Hook Shapefile generation into the same run finalization path.
4. Add tests that validate package contents, Shapefile headers, and DBF record
   count.
5. Commit the Shapefile implementation separately.

## Done Criteria

- `reports/final_classes.geojson` is produced for completed runs.
- `reports/final_classes_shapefile.zip` is produced for completed runs.
- Both artifacts use the same feature semantics and local pixel coordinate
  contract.
- `run.json` exposes generated GIS export paths.
- Focused tests pass.
- `git diff --check` passes for touched files.

## Out Of Scope

- Full GIS server integration.
- True map CRS/georeferencing before calibrated stage/world transforms exist.
- Component-level layers and edit-history vector layers.
- Adding GDAL/Fiona/GeoPandas as required runtime dependencies.
