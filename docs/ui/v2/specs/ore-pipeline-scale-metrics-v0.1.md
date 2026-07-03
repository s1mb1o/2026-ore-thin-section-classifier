# Ore Pipeline Scale Metrics v0.1

Date: 2026-07-03

## Scope

This spec applies to the v2 local ore pipeline UI:

```text
apps/ore_pipeline_web.py
```

It covers the official requirement:

```text
Расчет площадей и процентных долей с учетом масштаба изображения.
Таблица с метриками в интерфейсе и возможность экспорта в CSV.
```

## Goals

- Keep percentage fractions visible for every run.
- Always expose pixel areas for mask-backed metric rows.
- Convert pixel areas to physical areas only when calibrated scale metadata is
  explicitly supplied.
- Preserve conservative scale handling: DPI, EXIF focal length, digital zoom,
  and filename `5x` / `10x` hints are not calibrated specimen scale.
- Make the same metric fields available in the result UI table and
  per-run `metrics.csv`.

## Scale Input Contract

Scale comes from curated run metadata:

```json
{
  "domain": {
    "microns_per_pixel": 0.5,
    "scale_source": "calibration_slide",
    "scale_confidence": "calibrated"
  }
}
```

The existing UI field name remains supported:

```json
{
  "domain": {
    "pixel_size_um": 0.5,
    "scale_source": "calibration_slide",
    "scale_confidence": "calibrated"
  }
}
```

Physical areas are emitted only when:

- `microns_per_pixel` or `pixel_size_um` parses as a positive number;
- `scale_confidence == "calibrated"`;
- `scale_source` is not empty, `none`, or `unavailable`.

## Scaling Rule

The UI may run inference on a downscaled analysis image for large inputs. The
entered scale is treated as micrometers per source-image pixel. The run converts
it to the analysis mask grid before calculating area:

```text
microns_per_analysis_pixel_x = microns_per_source_pixel * source_width / analysis_width
microns_per_analysis_pixel_y = microns_per_source_pixel * source_height / analysis_height
area_um2 = area_px * microns_per_analysis_pixel_x * microns_per_analysis_pixel_y
area_mm2 = area_um2 / 1_000_000
```

If no scaling metadata is available, the analysis grid is assumed to match the
source grid.

## Metric Rows

Each completed run stores `metrics[]` rows in `run.json`.

Rows with mask-backed areas:

- `sulfide_fraction`
- `ordinary_sulfide_fraction`
- `fine_sulfide_fraction`
- `talc_fraction`
- `analyzed_fraction`

Each area row contains:

```json
{
  "key": "talc_fraction",
  "value": 0.12,
  "percent": 12.0,
  "area_px": 12345,
  "area_um2": 3086.25,
  "area_mm2": 0.00308625
}
```

When scale is unavailable, `area_px` remains present and `area_um2` /
`area_mm2` are omitted.

The component-count row remains count-only.

## UI And CSV

The result UI metrics table shows:

- metric label;
- value / percent;
- area in pixels;
- physical area when calibrated scale is available.

The `metrics.csv` export includes:

- `metric`
- `key`
- `value`
- `percent`
- `area_px`
- `area_um2`
- `area_mm2`
- `microns_per_pixel`
- `effective_microns_per_analysis_pixel`
- `scale_source`
- `scale_confidence`

## Non-Goals

- Do not infer scale from DPI/JFIF density/EXIF fields.
- Do not add stage/world georeferencing.
- Do not change the current analysis-resolution mask editing model.
- Do not claim physical areas when scale confidence is weak or absent.

## Acceptance Criteria

- A run without calibrated scale shows percentages and pixel areas, but no
  physical areas.
- A run with `domain.microns_per_pixel`, calibrated confidence, and a calibrated
  source shows correct `area_um2` and `area_mm2` in `run.json`.
- `metrics.csv` contains the same pixel and physical area values as `run.json`.
- The browser result table has columns for metric, value, pixel area, and
  physical area.
- Edit-derived runs inherit parent scale metadata and recompute physical areas
  from the edited mask areas.

## Verification

```bash
python3 -m py_compile apps/ore_pipeline_web.py tests/test_ore_pipeline_web.py
python3 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v
git diff --check -- apps/ore_pipeline_web.py tests/test_ore_pipeline_web.py docs/ui/v2/specs/ore-pipeline-scale-metrics-v0.1.md docs/ui/v2/plans/33_ore-pipeline-scale-metrics.md
```
