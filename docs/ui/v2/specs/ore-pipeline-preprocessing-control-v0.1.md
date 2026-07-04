# Ore Pipeline Preprocessing Control v0.1

Date: 2026-07-03

## Scope

This spec applies only to the v2 local ore pipeline UI:

```text
apps/ore_pipeline_web.py
```

It replaces the always-visible preprocessing checklist in the left sidebar with a compact preprocessing gate and a settings popup.

## Goals

- Show preprocessing as one compact control:

```text
[ ] Preprocessing   [Edit...] [Apply]
```

- Keep preprocessing disabled by default and remember the user's choice.
- Keep individual preprocessing filter options selected by default as the
  ready-to-use preset for when the main `Preprocessing` checkbox is enabled.
- Move these settings into an `Edit...` popup:
  - illumination normalization `(?)`;
  - noise reduction `(?)`;
  - contrast correction `(?)`;
  - panorama image scaling `(?)`.
- Show a small hover/focus hint on each `(?)` explaining what the setting does.
- Make panorama image scaling explicit: when enabled, the user chooses either a longest-side pixel bound or a scale factor.
- Make `Apply` run the current preprocessing settings for preview and force the main preprocessing checkbox to checked.
- Make the main preprocessing checkbox control whether preprocessing is applied when `Start` is pressed.
- If preprocessing is skipped on `Start`, do not expose a `preprocessed` display layer for that run; the preprocessed view and side-by-side option must stay disabled/greyed.
- Preserve the existing immutable run model.

## Non-Goals

- Do not add new preprocessing algorithms in this version.
- Do not change the segmentation/edit/recalculate algorithms.
- Do not send full-resolution large images to the browser.
- Do not add a separate augmentation stage here.

## User Flow

1. User selects an image.
2. The sidebar shows one preprocessing row:
   - checkbox `Preprocessing`;
   - button `Edit...`;
   - button `Apply`.
3. User can open `Edit...` and change the four preprocessing settings.
   Each setting label includes a compact `(?)` hint:
   - illumination normalization `(?)`: balances uneven lighting before segmentation;
   - noise reduction `(?)`: suppresses small image noise while keeping larger ore structures;
   - contrast correction `(?)`: gently increases tonal separation for visual inspection;
   - panorama image scaling `(?)`: explicitly downsamples panoramas either to a configured longest-side bound or by a configured factor.
   The panorama setting exposes:
   - mode `longest side bound` with a numeric pixel value;
   - mode `scale factor` with a numeric multiplier.
4. Pressing `Apply`:
   - checks `Preprocessing`;
   - runs `/api/uploads/{upload_id}/preprocess`;
   - updates the preprocessed preview layer.
5. If the `Preprocessing` checkbox is unchecked and the user presses `Start`:
   - preprocessing is skipped;
   - the run records `preprocess.enabled = false`;
   - the pipeline uses an analysis-scale copy of the original image internally;
   - the `preprocessed` view is unavailable.
6. If the checkbox is checked and the user presses `Start`:
   - preprocessing is applied using the saved popup settings;
   - the run records the preset and exposes the `preprocessed` display layer.

## Data Contract

The preprocessing preset sent by the browser includes:

```json
{
  "preprocessing_enabled": false,
  "illumination_normalization": true,
  "denoise": true,
  "contrast_correction": true,
  "panorama_scaling": true,
  "panorama_scaling_mode": "max_side",
  "panorama_max_side_px": 1800,
  "panorama_scale_factor": 0.5
}
```

Compatibility aliases may still be accepted:

- `enabled`
- `illumination`
- `noise_reduction`
- `contrast`
- `panoramaScaling`
- `panoramaScalingMode`
- `panoramaMaxSidePx`
- `panoramaScaleFactor`

Panorama scaling semantics:

- `panorama_scaling=false`: no special panorama downscaling is applied; the image falls back to the normal processing max side.
- `panorama_scaling=true` and `panorama_scaling_mode="max_side"`: downscale to `panorama_max_side_px` on the longest side when the source is larger.
- `panorama_scaling=true` and `panorama_scaling_mode="scale_factor"`: compute the processing longest side as `source_longest_side * panorama_scale_factor`.
- Tiling is independent from this setting. Turning panorama scaling off does not disable tiling; the tiling manifest is still created from the analysis image and is enabled when the source was scaled or the analysis image needs multiple tiles.

Run metadata records:

```json
{
  "preprocess": {
    "enabled": true,
    "preset": {},
    "target_max_side": 1800,
    "panorama_scaling": {
      "enabled": true,
      "mode": "max_side",
      "target_max_side": 1800,
      "source_longest_side": 27025,
      "max_side_px": 1800,
      "scale_factor": 0.5
    }
  }
}
```

The internal analysis input can still be stored at `input/preprocessed.png` for compatibility, but when preprocessing is disabled it is an analysis-scale original copy and must not be exposed as the user-facing preprocessed layer.

## Acceptance Criteria

- The sidebar no longer shows the four preprocessing items directly.
- `Edit...` opens a popup containing the four settings.
- Each popup setting shows a small `(?)` help control with a localized hover/focus description.
- Panorama scaling shows explicit mode/value controls and the summary reports the active rule, such as `панорама до 1800 px` or `панорама 0.5x`.
- `Apply` checks the main preprocessing checkbox and updates the preprocessed preview.
- `Start` with the main checkbox unchecked creates a complete run without a preprocessed display layer.
- `preprocessed` and side-by-side `preprocessed` controls are disabled when no preprocessed display layer exists or preprocessing is currently unchecked for tuning.
- The default main preprocessing checkbox is unchecked, while the default
  filter/panorama options are checked.
- The user's preprocessing enabled/settings choice persists in local storage.
- Focused unit tests cover the UI controls and disabled-preprocessed run behavior.
