# Ore Pipeline Grain Reporting v0.1

Date: 2026-07-04

Status: Implemented 2026-07-04.

## Scope

Improve the v2 ore pipeline result report for ore/sulfide grains using only artifacts already produced by the current OM pipeline:

- area;
- equivalent diameter;
- perimeter;
- sulfide area share;
- liberation/contact proxy;
- locked/composite proxy flag;
- contact lengths against matrix, talc, and an `other contact` proxy bucket from the current OM masks.

This is a reporting feature over the current OM segmentation output. It does not add new modalities and it does not claim chemistry-grade mineral classification from RGB OM images.

## Existing Inputs

The current pipeline already writes `reports/component_features.csv` with one row per connected sulfide component:

- `component_id`;
- `label` (`ordinary_intergrowth` or `fine_intergrowth`);
- `area_px`;
- `footprint_area_px`;
- `dark_inside_area_px`;
- `dark_inside_ratio`;
- `solidity`;
- `compactness`;
- `boundary_complexity`;
- `bbox_x/y/w/h`;
- `centroid_x/y`.

The code also already has `ore_classifier.component_reports.component_liberation_proxies`, which can derive contact-count proxies from sulfide and talc masks. The current UI exposes a sulfide-grain table and a component label map for click-to-outline.

## Data Contract

New runs should write `perimeter_px` in `component_features.csv`. Existing runs without that column remain supported by deriving perimeter from `boundary_complexity * sqrt(area_px)` in the UI payload.

Derived values:

- `equivalent_diameter_px = sqrt(4 * area_px / pi)`;
- `perimeter_px = row.perimeter_px` or derived fallback;
- `sulfide_area_share = area_px / sulfide_area_px`;
- `liberation_proxy = matrix_contact_px / max(total_contact_px, 1)`;
- `locked_composite_proxy = true` when the component has material non-matrix contact or a low liberation proxy.

`GET /api/runs/{run_id}` keeps `sulfide_grains.schema_version = ore-pipeline-sulfide-grains-v0.1`, but enriches every item:

```json
{
  "component_id": 12,
  "type": "ordinary_intergrowth",
  "area_px": 1234,
  "equivalent_diameter_px": 39.64,
  "perimeter_px": 151.2,
  "sulfide_area_share": 0.045,
  "share_percent": 4.5,
  "liberation_proxy": 0.82,
  "locked_composite_proxy": false,
  "contacts": {
    "matrix_px": 83,
    "talc_px": 0,
    "other_contact_px": 18,
    "total_px": 101
  },
  "association_percentages": {
    "matrix": 82.18,
    "talc": 0.0,
    "other_contact": 17.82
  }
}
```

Terminology:

- `sulfide_area_share` is the component area divided by total sulfide area.
- `liberation_proxy` is a contact-based proxy: matrix contact divided by all one-pixel boundary contacts.
- `locked_composite_proxy` is a conservative proxy flag when the component has material non-matrix contact or a low liberation proxy.
- `other_contact_px` is not a mineral class. It is the residual non-matrix/non-talc contact bucket available from the current OM masks.
- Contact lengths are pixel-boundary counts from OM masks, not calibrated microns unless a later export multiplies by calibrated scale.

## UI Contract

The result `Sulfide grains` table becomes the ore-grain report table. It keeps row checkboxes for image outlines and adds columns:

- area, px;
- equivalent diameter, px;
- perimeter, px;
- sulfide area share;
- liberation proxy;
- contacts, px;
- locked/composite proxy.

The note above the table must explicitly say that liberation/contact/locked-composite values are OM mask proxies and are not chemistry-based mineral liberation analysis.

No new upload fields, modality checklist, chemistry metadata panel, or class table is added for this task.

## Acceptance Criteria

- New component CSVs include `perimeter_px`.
- Existing component CSVs without `perimeter_px` still render with derived perimeter.
- `GET /api/runs/{run_id}` returns enriched grain rows with equivalent diameter, perimeter, sulfide area share, liberation/contact proxies, contact lengths, association percentages, and locked/composite proxy flag.
- The static UI renders the new table columns and localized RU/EN labels.
- The UI copy makes the proxy limitation visible near the table, without implying chemistry-grade mineral liberation analysis.
- Focused tests cover the enriched API payload and required UI hooks.

## Implementation Notes

- `ComponentFeatures` now writes `perimeter_px` to new `reports/component_features.csv` files.
- `GET /api/runs/{run_id}` enriches `sulfide_grains.items` from current component CSVs plus persisted sulfide/talc masks.
- The static UI renders the added proxy columns and keeps row checkbox outlines unchanged.
