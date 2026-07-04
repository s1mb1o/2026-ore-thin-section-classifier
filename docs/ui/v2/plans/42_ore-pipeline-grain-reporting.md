# Plan 42: Ore Pipeline Grain Reporting

Date: 2026-07-04

Status: Implemented 2026-07-04.

Spec: `../specs/ore-pipeline-grain-reporting-v0.1.md`

## Goal

Upgrade the existing sulfide-grain table into a practical OM-derived ore-grain report surface: area, equivalent diameter, perimeter, sulfide area share, contact/liberation proxies, and locked/composite proxy flag.

Non-goals: no new imaging modalities, chemistry basis, class ontology, or new upload metadata for this task.

## Implementation Steps

1. Extend component features.
   - Add `perimeter_px` to `ComponentFeatures`.
   - Keep CSV compatibility by deriving perimeter in the UI payload if older CSVs do not contain the column.

2. Enrich the run API payload.
   - Reuse `ore_classifier.component_reports.component_liberation_proxies`.
   - Load the run sulfide and talc masks in `_sulfide_grains_payload`, compute contact proxies, and merge them into `_read_sulfide_grain_rows` by `component_id`.
   - Add `equivalent_diameter_px`, `perimeter_px`, `sulfide_area_share`, `liberation_proxy`, `locked_composite_proxy`, `contacts`, and `association_percentages`.
   - Expose the helper's residual non-matrix contact bucket as `other_contact_px` in the API/UI copy; do not label it as a true mineral association.
   - Use explicit formulas from the spec and keep numeric fields stable for older completed runs.

3. Update the result UI table.
   - Keep the existing checkbox outline behavior.
   - Add columns for the reportable grain fields.
   - Localize RU/EN headers and clarify these are OM-mask proxies, not chemistry-grade liberation analysis.
   - Keep the table compact enough for repeated review: show IDs/type/outline controls first, then area/diameter/perimeter/share/proxies.

4. Tests and docs.
   - Extend the full upload/run test to assert enriched grain fields.
   - Extend the static UI test to assert headers/functions/labels.
   - Update `SMOKE_TESTS.md`, `docs/ui/v2/README.md`, `ChangeLog.md`, and `docs/session-sync.md`.

## Verification

Run from the v2 root:

```bash
python3 -m py_compile src/ore_classifier/component_analysis.py apps/ore_pipeline_web.py
python3 -m unittest discover -s tests -p 'test_ore_pipeline_grain_reporting.py' -v
python3 -m unittest discover -s tests -p 'test_component_analysis.py' -v
python3 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v
.venv/bin/python -m pytest tests/browser/test_ore_pipeline_ui_browser.py -q
```
