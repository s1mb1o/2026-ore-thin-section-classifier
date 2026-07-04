# Ore Pipeline Talc Cluster Areas v0.1

## Purpose

Port the talc annotation tool's "talc cluster areas" concept into the v2 ore pipeline UI so dense local talc zones are visible, configurable, reproducible, and reported with the same immutable run semantics as segmentation results.

## Scope

- Add talc cluster area visibility controls to both top-left and top-right segmentation class legends.
- Display talc cluster areas only in `final` image mode and in side-by-side panels whose selected layer is `final`.
- Add system-wide default talc clusterization parameters on the Settings page.
- Add a workspace run Configuration dialog under Metadata with a `Talc clusterization` section.
- Persist the effective talc clusterization parameters with each run.
- Compute a talc cluster mask from the final talc mask and analyzed area.
- Add talc cluster area to the UI metrics table, CSV export, PDF report data path, and run summary.

## Non-Goals

- No new top-level view mode is introduced; cluster areas are part of the final segmentation overlay.
- The feature does not replace talc detection; it derives local dense zones from the talc mask produced by the selected runtime.
- The feature does not add hand-editing of talc clusters directly. Editing still happens through final segmentation or artefact masking.

## User Experience

### Workspace

Under the existing Metadata button, show a `Configuration...` button after an image is loaded. It opens a modal with the `Talc clusterization` section:

- `Radius, px`
- `Min local talc, %`
- `Opacity, %`

The modal is prefilled from Settings defaults for new uploads. If a run is loaded from History, the modal is prefilled from that run's saved configuration. The values are submitted with `Start`.

### Settings

Settings gains a `Talc clusterization defaults` card with the same three fields. Saving settings persists defaults server-side and in the browser's normalized settings state.

### Viewer

For `final` view:

- Existing final class legend remains visible.
- Add a checkable row `talc cluster areas` with a magenta/pink swatch.
- The row shows the cluster area percentage, calculated from analyzed area.
- Toggling the row controls only the cluster overlay, not the base talc class overlay.

For side-by-side:

- The right legend shows the same talc cluster row when the side-by-side layer is `final`.
- The left and right legends use the same visibility state as the other class checkboxes.

## Run Semantics

Runs are immutable. Each run stores the exact talc clusterization parameters used:

```json
{
  "talc_clusterization": {
    "schema_version": "ore-pipeline-talc-clusterization-v0.1",
    "radius_px": 64,
    "min_local_talc_percent": 4.0,
    "opacity_percent": 45.0
  }
}
```

When `Start` is pressed:

- If a prepared run already exists and the configuration has changed, the prepared run is treated as stale and recreated before start.
- If a completed run is loaded and parameters are changed, the next execution creates a new run through the existing immutable run flow.
- Batch items inherit the shared talc clusterization settings from the batch settings payload.

## Cluster Algorithm

Input:

- binary final talc mask
- analyzed mask after artefact exclusion
- `radius_px`
- `min_local_talc_percent`

For each pixel, calculate local talc density inside the clipped square window:

```text
window = [x-radius, x+radius] x [y-radius, y+radius]
density = talc_pixels_in_window / analyzed_pixels_in_window
cluster = analyzed && density >= min_local_talc_percent / 100
```

Implementation may use OpenCV `boxFilter` on the talc mask and analyzed mask to avoid a per-pixel Python loop on large images.

Output:

- `masks/talc_cluster_mask.png`
- `display.talc_cluster_overlay` preview pyramid
- `summary.talc_cluster_area_px`
- `summary.talc_cluster_fraction`
- `summary.talc_clusterization`

## Metrics

The metrics table adds a child row under `Доля талька`:

- RU label: `Площадь кластеров талька`
- EN label: `Talc cluster areas`
- Denominator: analyzed area
- Value: percent
- Area columns: pixel area and physical area when calibrated scale is available

The denominator note must mention that talc cluster areas are derived from the talc mask and use analyzed area as the denominator.

## Validation

- Unit test normalization of talc clusterization settings.
- Unit test cluster mask calculation on a small deterministic mask.
- End-to-end run test confirms summary fields, mask file, display layer, metrics row, and run configuration.
- Static UI test confirms Settings fields, Configuration dialog, legend controls, draw path, and translations.

