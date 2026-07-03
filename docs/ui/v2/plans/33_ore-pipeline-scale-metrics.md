# Ore Pipeline Scale Metrics Implementation Plan

Date: 2026-07-03

Spec:

```text
docs/ui/v2/specs/ore-pipeline-scale-metrics-v0.1.md
```

Implementation target:

```text
apps/ore_pipeline_web.py
tests/test_ore_pipeline_web.py
```

## Decision

Implement this inside the existing v2 ore pipeline metrics path. The current
run already writes `summary`, `metrics`, `reports/metrics.csv`, and result-page
HTML from one source of truth, so adding physical-area fields there keeps UI and
CSV consistent without adding another reporting module.

## Plan

1. Add scale parsing.
   - Read `domain.microns_per_pixel` first, then existing
     `domain.pixel_size_um`.
   - Require `scale_confidence=calibrated` and a non-empty calibrated source.
   - Keep uncalibrated values as metadata only.

2. Adjust for analysis-image scaling.
   - Use the run tiling manifest's `source_width`, `source_height`,
     `analysis_width`, and `analysis_height`.
   - Compute `area_um2_per_analysis_pixel`.
   - Store a compact `run.json.scale` block for auditability.

3. Enrich metric rows.
   - Add `area_px` for class/fraction rows and analyzed area.
   - Add `area_um2` and `area_mm2` when scale is available.
   - Leave component count as count-only.

4. Update exports.
   - Expand `reports/metrics.csv` with pixel area, physical area, and scale
     provenance columns.
   - Keep existing `value` and `percent` columns stable.

5. Update UI.
   - Add result-table columns for pixel area and physical area.
   - Localize column labels in Russian and English.
   - Leave physical-area cells empty when no calibrated scale is present.

6. Test.
   - Add a focused regression that starts a run with
     `domain.microns_per_pixel=0.5`, `scale_source=calibration_slide`, and
     `scale_confidence=calibrated`.
   - Assert `run.json.metrics` and `metrics.csv` match the expected
     `area_px * 0.25` physical area.
   - Assert the HTML contains the new metrics table contract.

## Done Criteria

- Focused `test_ore_pipeline_web.py` passes.
- `run.json` and `metrics.csv` agree for calibrated physical areas.
- The result table contract is documented and covered by an HTML smoke test.
- `ChangeLog.md`, `SMOKE_TESTS.md`, and `docs/session-sync.md` mention the
  scale-aware metrics behavior.
