# Talc Mask Review Web App v0.1

Date: 2026-07-03

Status: Implemented in `apps/talc_review_web.py`.

## Purpose

Replace the unreliable Streamlit mask-editing experience with a narrow local
browser application for talc mask QA. This app is talc-only because the source
annotations are hand-drawn MS Paint pen marks over talc regions. Those marks are
draft annotations, not masks: many regions are open, rough, and not reliable
closed shapes.

The app matches each annotated image from `Области оталькования` to its original
image in the parent `Оталькованные руды` folder by the same filename, derives an
initial talc mask from the pen annotations, and lets the user correct that talc
mask directly. It is not the training loop, final inference engine, sulfide QA
tool, or old broad OM/SEM/XRD platform.

## Proposed Product Decision

Build a lightweight local web app:

- Python backend with `http.server.ThreadingHTTPServer` or equivalent minimal
  stdlib server.
- Generated HTML/CSS plus vanilla JavaScript.
- Browser `canvas` for image display, mask overlay, fill, Similar
  intensity assist, polygon/rectangle editing, brush left-draw/right-erase
  editing, pan/zoom, and SAM2 prompts.
- File-based outputs compatible with the current converter and training
  manifests.

This satisfies the optional web-interface/dashboard requirement without
depending on Streamlit/Dash/Gradio templates. It keeps the computation in the
existing Python stack: OpenCV, scikit-image, torch, SAM2, and project code under
`src/ore_classifier/`.

## Input Pairing and Preparation

The review app works directly from the official blue-line annotation folder and
the original talcose-ore image folder.

Source folders:

- annotated images:
  `dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования`
- original images:
  `dataset/Фото руд по сортам. ч1/Оталькованные руды`

The app pairs files by exact filename. For example:

```text
dataset/.../Оталькованные руды/Области оталькования/DSCN3042.JPG
dataset/.../Оталькованные руды/DSCN3042.JPG
```

The annotated image is used to detect draft talc regions from hand-drawn pen
marks. The original image is used as the clean review background and final
overlay source.

Primary launch input:

```bash
python3 apps/talc_review_web.py \
  --annotated-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --original-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды" \
  --workspace-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port <free-local-port>
```

`--original-dir` defaults to the parent of `--annotated-dir`, so the shorter
command is also valid for the official folder layout:

```bash
python3 apps/talc_review_web.py \
  --annotated-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --workspace-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port <free-local-port>
```

On startup, the app checks whether the workspace contains a current manifest and
per-sample working masks. If the workspace is missing or `--reconvert` is
requested, it runs the same conversion logic as:

```bash
python3 scripts/convert_talc_blue_lines.py \
  --input "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --output-dir outputs/talc_blue_line_conversion \
  --summary-json outputs/talc_blue_line_conversion_summary.json
```

The conversion copies or references each blue-line annotation image, extracts
blue strokes, closes/fills candidate regions, subtracts or marks sulfide
overlap, and writes per-sample mask artifacts plus
`outputs/talc_blue_line_conversion/manifest.json`.

The new app must additionally record the matched clean original image path for
each sample. If an annotated file has no original file with the same filename,
the sample remains visible in the queue with `missing_original` status and no
editing canvas until the match is fixed.

Optional display-only evidence masks can be supplied by image stem:

```bash
python3 apps/talc_review_web.py \
  --annotated-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --original-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды" \
  --workspace-dir outputs/talc_blue_line_conversion \
  --sulfide-mask-dir path/to/binary_sulfide_masks \
  --silicate-mask-dir path/to/silicate_support_masks
```

Prepared-workspace mode is still supported for reproducibility/debugging:

```bash
python3 apps/talc_review_web.py \
  --conversion-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port <free-local-port>
```

If neither a raw input folder nor a valid conversion manifest is available, the
app should show a clear setup screen with the exact direct-input command.

## Scope

### In Scope for v0.1

- Accept the official `Области оталькования` folder directly through
  `--annotated-dir`.
- Match each annotated image to the clean original image by the same filename in
  `--original-dir`.
- Create or reuse a talc conversion/review workspace under
  `outputs/talc_blue_line_conversion`.
- Auto-save the autodetected mask as the current working talc mask when a sample
  is first opened.
- Load the generated talc-only review manifest from that workspace.
- Show a review queue with status, reason, source image name, and existing
  candidate statistics.
- Display the clean original photo, the annotated MS Paint draft, the
  autodetected talc region, the current talc mask, and optional display-only
  evidence layers such as sulfide overlap or silicate support.
- Edit the talc mask directly, not the pen strokes.
- Protect talc-mask edits from sulfide overlap by default and provide an
  explicit manual action to subtract sulfide pixels from the current mask.
- Save reviewed masks under each sample's `reviewed/` directory.
- Preserve an auditable edit patch with vector geometry and/or mask deltas.
- Keep SAM2 optional and lazy-loaded.
- Provide exact-coordinate fallback forms behind an advanced panel.
- Run locally without internet after dependencies and model cache are present.

### Out of Scope

- Sulfide QA or ordinary/fine intergrowth QA.
- Old broad OM/SEM/XRD upload portal.
- XRD, SEM registration, defect/product platform UI.
- Editing ignore masks as a primary user task.
- Multi-user authentication or public deployment.
- Live model training from the UI.
- Downloading model weights into the repository.
- Treating non-expert edits as geological ground truth without review status.

## Users

- Reviewer: corrects talc masks from draft MS Paint annotations.
- ML engineer: uses reviewed talc masks and patches for training data.
- Presenter: demonstrates original annotation, converted mask, manual
  correction, SAM2 assist, and final reviewed overlay.

## Primary Workflow

1. Launch the app with the official `Области оталькования` annotated folder.
2. App matches each annotated file to the original file with the same filename.
3. App creates or reuses `outputs/talc_blue_line_conversion`.
4. App opens the generated review queue.
5. Select a sample from the queue.
6. On first open, app saves the autodetected talc mask as the current working
   talc mask.
7. Inspect the clean original, annotated MS Paint draft, autodetected mask, and
   current talc mask.
8. Edit the current talc mask with brush left-draw/right-erase, fill, Similar
   Talc intensity assist, direct editable polygon regions, direct editable
   rectangle regions, or SAM2 assist.
9. Use undo when an edit is wrong.
10. Press top-right `Save` to persist the current sample.
11. Press top-right `Save & Next` to persist the current sample and move to the next
    sample in the queue, or press transparent `Next` to move without saving.
12. Press top-right `Download` to export the current sample image with the enabled
    background, class masks, display layers, comparison layer, and talc-cluster layer
    as a full-resolution PNG.
13. Copy the `/sample/<slug>` URL when a reviewer needs to reopen or share the
    current image state.

## UX Requirements

### Layout

The first screen is the working review interface, not a landing page.

- Left column: sample queue with filters.
- Center: canvas viewer and editing controls.
- Right column: sample details, layer toggles, current metrics, save status.
- Bottom or collapsible side panel: edit history and advanced exact-coordinate
  tools.
- Theme selector: `System`, `Light`, and `Dark`, persisted in browser storage.

The reviewer must always see that the edited object is the `Talc mask`.

### Queue Filters

- All samples.
- Needs review.
- Sulfide overlap.
- Candidate OK.
- Reviewed.
- Missing original.
- Search by sample id/stem.

### Viewer Layers

Base image modes:

- Original photo.
- MS Paint annotated photo.
- QA overlay.
- Sulfide mask (sulfide/non-sulfide mask segmentation).
- Current mask view.

The viewer includes a display-only dark-pixel brightness threshold slider for
photo backgrounds. It computes perceptual luma per pixel as
`0.299*R + 0.587*G + 0.114*B`; pixels with luma less than or equal to the
slider value stay visible as the original RGB pixel, while pixels brighter than
the slider value are painted white. `255` is the off state, `0` paints the
background white, and `90` is exposed as a quick practical talc-candidate
starting point for reflected-light images. The control also reports the current
visible-pixel share/count (`luma <= threshold`) for the active photo
background, and reports inactive when the background mode is mask-only or
sulfide-mask. The filter must not alter the talc mask; threshold and
visible-pixel statistics are stored only as review/view metadata in working
state and reviewed patch JSON.

The right panel also includes a display-only talc cluster overlay. It highlights
areas where the selected talc source is locally dense, without modifying any
mask. The reviewer can tune:

- source: `Talc class` or `Positive bag + Talc`;
- radius in pixels for the local-density window;
- minimum local talc percentage required to highlight a region;
- overlay opacity.

The cluster overlay uses the current working masks, updates after edits, and is
stored only as review/view metadata in working state and reviewed patch JSON.

Segmentation controls are a compact top-left overlay widget on the image
viewer. Editable class rows have a visibility checkbox, live percentage, and
edit-target radio button:

- Positive bag.
- Talc.
- Not Talc.

A separate top-right `Display layers` overlay widget contains the display-only
`Background` and `Talc cluster areas` controls. `Background` is checked by
default to preserve the current base-image view; unchecking it hides only the
base background image while keeping the editable masks, derived overlays, and
tool previews visible. It must not warn about a missing selected background
while the background checkbox is off.

Both top overlay widgets must stay anchored to the visible viewer viewport, not
to the scrolled canvas content. After sample load, fit-to-view, or free-pan
origin changes, `Segmentation classes` remains top-left and `Display layers`
remains top-right inside the visible viewer.

The top-right `Talc cluster areas` row has a visibility checkbox and percentage.
It mirrors the right-panel cluster overlay toggle and reports highlighted
non-sulfide cluster pixels as a percentage of image pixels. Cluster areas must
never be painted over sulfide-mask pixels and must not use sulfide pixels as
local-density source evidence. It has no edit-target radio because cluster areas
are derived display evidence, not a saved segmentation class.

The edit-target radio controls Brush, Fill, Rectangle, and Polygon. Selecting
`Positive bag` writes those tools into `current_positive_bag_mask`; selecting
`Talc` writes those tools into `current_talc_node_mask`; selecting `Not Talc`
writes those tools into `current_not_talc_mask`. `Not Talc` is a hard-negative
class for dark non-talc objects such as pores, scratches, dark matrix, and other
false positives. It is saved separately, never participates in
`current_talc_mask` / `reviewed_talc_mask`, and must exclude confirmed `Talc`
pixels at save time. `Positive bag` may overlap `Not Talc`, because the bag is a
weak container rather than a pixel-accurate talc class. Similar always writes
Talc, excludes `Not Talc`, and SAM2 remains a Positive bag assist unless
explicitly changed later.

Because the source folder `Оталькованные руды/Области оталькования` is treated
as a talcose source where confirmed talc should be at least `10%` of visible
image pixels, the widget also shows live visible-pixel percentages for
`Positive bag` and `Talc`, plus a compact target status. The top-right display
layers widget shows `Background`, `Blank White`, `Original blue lines`, `Talc
cluster areas`, and `Sulfides`, including the `Talc cluster areas` percentage.
`Blank White` is display-only and fills the canvas white only when `Background`
is unchecked. The target status is based on confirmed `Talc` pixels, not the
rough `Positive bag`: below `10%` it shows how many percentage points are
missing; at or above `10%` it reports the target as met.

### Comparison Modes

The right panel includes a `Comparison mode` selector with six modes:

- `Current`: shows only the normal editable classes (`Positive bag`, `Talc`,
  `Not Talc`) plus display-only controls in the separate top-right
  `Display layers` widget.
- `Heuristic`: overlays only the non-neural talc-zone mask.
- `Neural Model`: overlays only the trained neural talc-model prediction.
- `Current vs Heuristic`: overlays the current `Talc` class mask against the
  non-neural talc-zone mask.
- `Current vs Neural Model`: overlays the current `Talc` class mask against the
  trained neural talc-model prediction.
- `Heuristic vs Neural Model`: overlays the non-neural talc-zone mask against
  the trained neural talc-model prediction.

The heuristic mode can generate its own comparison source. Pressing
`Run non-neural classifier` calls
`POST /api/samples/{sample_id}/talcose-heuristic`, runs the standalone
talc-zone heuristic from `src/ore_classifier/talc_zone_heuristic.py`, and
writes the following per-sample artifacts under `qa/non_neural_talcose/`:

- `talc_zone_mask.png`;
- `talc_flake_mask.png`;
- `overlay.jpg`;
- `talcose_result.json`.

When the converter produced a sulfide mask, the app passes that mask as the
heuristic ore-exclusion mask. If no sulfide mask is available, the heuristic
uses its brightness fallback and records that fallback in `talcose_result.json`.
The plain `Heuristic` mode shows the heuristic layer alone. `Current vs
Heuristic` highlights `agreement`, `heuristic only`, `current only`, and
`sulfide conflict`; the stats row reports the ore class, talc-zone fraction,
ore-mask source, comparison percentages when applicable, and a link to the
overlay artifact. Heuristic-only pixels use the distinct fuchsia color
`#ec4899`, not the orange/red/yellow colors already used by other classes or
QA states.

The neural mode is read-only and must not change masks. The app may load an
optional trained talc-model prediction mask from `--talc-model-mask-dir`, from
well-known files in the sample directory such as `model_talc_mask.png` /
`predicted_talc_mask.png`, or from manifest paths when present. The plain
`Neural Model` mode shows the neural layer alone. `Current vs Neural Model`
highlights `agreement`, `neural only`, `current only`, and `sulfide conflict`.
If no model mask is available, either neural mode shows an unavailable status
instead of silently dropping the overlay.

The `Neural Model` panel exposes an editable `ML talc probability threshold`
numeric field. It defaults to the server-side `--talc-threshold` value (`0.50`
unless overridden at launch), accepts `0.00..1.00`, and is passed to
`POST /api/samples/{sample_id}/neural-model` so `Run model` writes the
sample-level `model_talc_mask.png` using the same threshold shown in the UI.

Additional display-only layer toggles:

- Autodetected talc region.
- Sulfide overlap, display-only.
- Raw blue strokes.
- Silicate support, display-only when available.
- SAM2 preview mask, when available.

Layer opacity must be adjustable without rerunning Python.

The right panel must include a default-on `Protect sulfides while drawing`
control. When enabled, additive tools clip newly applied talc pixels against
the loaded sulfide mask without silently changing pre-existing overlap. A
separate `Subtract sulfides from mask` command removes any existing talc pixels
on the sulfide mask and autosaves the working mask. `Current on sulfide px`
shows whether any overlap still exists.

### Canvas Interaction

Required controls:

- A single top toolbar ordered icon-only Brush, Fill, Similar, Rectangle,
  Polygon, then SAM2, Undo, followed by active-tool parameters. The icon-only
  tool buttons expose the tool name through hover title and accessible labels.
- Top-right Save, Save & Next, transparent Next, and Download actions.
- Toolbar controls may wrap at narrower viewer widths, but wrapped rows must
  increase the topbar height so controls are never hidden behind the viewer.
- Zoom by mouse wheel/trackpad over the canvas, plus a bottom-left vertical
  viewer widget with Fit, Actual size, Zoom In, current zoom percentage, and
  Zoom Out. The top toolbar must not duplicate these zoom controls.
- Pan by pressing and dragging the mouse wheel / middle button over the canvas,
  without entering edit-tool drawing or native browser middle-click scrolling.
  Pan is not clamped to the image top-left: the canvas can move past the viewer
  edges so image coordinates can be negative inside the visible viewport.
- A below-viewer hint row states `Mouse wheel - zoom in / out` and
  `Mouse wheel press - pan`.
- Brush add with left mouse into the selected edit class.
- Brush erase with right mouse from the selected edit class.
- Fill bounded area.
- Similar intensity assist.
- Filled polygon.
- Filled rectangle.
- SAM2 assist.
- Undo.
- Keyboard shortcuts: `B` selects Brush, `F` selects Fill, and text inputs keep
  normal typing behavior.
Brush width appears in the toolbar only when Brush is active, supports `2-240 px`,
and applies only to Brush. In Brush mode, left mouse adds the selected edit class and right mouse
erases the selected edit class without opening the browser context menu. While
hovering over the image in Brush mode, the canvas shows a circle matching the
current draw/erase area and selected class color. Fill, Similar,
rectangle, polygon, and SAM2 operate on filled regions. Similar
parameters appear in the toolbar only when Similar is active. Polygon and
rectangle drafts are cancelled with right mouse, not with separate `Apply` or
`Cancel` controls. SAM2 parameters appear in the same toolbar only when SAM2 is
active.

Edits update the current working talc mask. The app should auto-save this
working mask after each applied edit so leaving and reopening a sample does not
lose work. `Save` marks the current working mask as reviewed and writes the
review patch; `Save & Next` does the same and navigates to the next queue row.
`Next` navigates to the next visible queue row without saving and has no filled
button background.
`Download` writes a PNG of the current sample at image resolution, composited
from the enabled persistent classes and display/comparison/cluster layers, with
no UI widgets, brush cursor, draft handles, or transient tool previews.
Live polygon/rectangle regions stay editable only while the current image is
open; saving flattens them into the reviewed mask PNG.

### Fill Tool

Fill click adds talc to the connected non-boundary region under the cursor.
Boundaries are raw/closed blue annotation strokes, sulfide pixels, existing
current talc-mask regions, and image edges. Fill is additive, undoable,
autosaved, and respects the default sulfide-protection guard for newly added
pixels. Fill writes the selected edit class.

### Similar Tool

Similar helps turn a known dark talc flake/grain into additional candidates
based on intensity, color, and local texture, not object shape. The reviewer can
add positive and negative seeds. Positive seeds mean "this is talc"; negative
seeds mean "this dark object is not talc". The app samples the original photo
around each seed as anchors. `positive_bag` means a rough region that may
contain talc segments, not confirmed talc pixels. If a positive seed is already
inside the current `positive_bag`, nearby bag pixels may refine the positive
calibration only after they pass luma/color/texture similarity checks against
the seed patch. Broad matrix-heavy positive bags must not be averaged
wholesale, because that makes the preview too permissive. The app then previews
luma/color/texture-similar pixels across the image while excluding sulfide-mask
pixels, existing talc-node pixels, existing `Not Talc` pixels, pixels close to
negative seeds, and isolated single pixel noise. Preview and applied talc nodes
may overlap positive-bag pixels.

Behavior:

- The Similar toolbar exposes `+ seed` and `- seed` modes.
- In `+ seed` mode, left-click adds a positive seed and creates or refreshes a
  non-destructive yellow preview.
- In `- seed` mode, left-click adds a negative seed and refreshes the preview;
  negative seeds are also useful when the reviewer has not yet drawn a
  permanent `Not Talc` region.
- Right-click clears the preview.
- `Strictness` controls how narrowly luma/color similarity must match the seed;
  higher values produce smaller candidate masks.
- At `Strictness=100`, matching should stay tight enough that a click inside a
  positive bag still follows the clicked grain, not the average bag color.
- `Apply Similar` merges the preview into the `talc_node` class, records an
  auditable `similar_talc_add` edit with `target_class: "talc_node"`, autosaves,
  and remains undoable.
- `Save` and `Save & Next` must also merge an active Similar preview before
  writing reviewed outputs, so a visible Similar result is not silently
  lost if the reviewer saves without pressing `Apply Similar`.
- Similar may mark pixels inside an existing `positive_bag` as talc nodes;
  the positive bag remains as the rough containing region.
- The preview is rejected if it would cover an implausibly large portion of the
  image, preventing a whole-screen assist result from being merged.
- `Not Talc` regions and negative seeds are stored in the audit trail so they
  can be exported later as hard negatives for model training.

### Polygon Tool

Polygon editing must support:

- Add point by clicking empty canvas.
- Close/finalize the polygon by clicking its first point after at least three
  points exist.
- Right-click a polygon point to remove it.
- Right-click away from polygon points to cancel the current draft polygon.
- Fill the talc region immediately after the polygon is closed.
- Keep completed polygon regions editable until another image is opened or the
  sample is saved.
- Insert point by clicking an existing completed polygon edge.
- Drag existing completed polygon points.
- Move the completed polygon as a filled region.
- Update the displayed talc mask live while the completed polygon is edited.
- Delete/Backspace removes the selected completed polygon.

### Rectangle Tool

Rectangle editing must support:

- Draw by drag or by clicking one corner and then the opposite corner.
- Right-click to cancel the current draft rectangle.
- Fill the talc region immediately after the rectangle is drawn.
- Keep completed rectangle regions editable until another image is opened or the
  sample is saved.
- Drag corners.
- Drag edges.
- Move the completed rectangle as a filled region.
- Update the displayed talc mask live while the completed rectangle is edited.
- Delete/Backspace removes the selected completed rectangle.

### SAM2 Tool

SAM2 must be a canvas tool, not a separate coordinate-only form.

Prompt modes:

- Rectangle, default because it bounds SAM2 proposals.
- Positive point.
- Polygon/rectangle refinement if practical in v0.1; otherwise keep as v0.2.

SAM2 behavior:

- Show model/device/load status.
- Load lazily only after explicit reviewer action.
- Use default Hugging Face/model cache, not project-local model downloads.
- Show a dashed canvas hover preview for the proposed SAM2 box/point prompt
  area before the reviewer clicks.
- In point mode, if the reviewer hovers without moving for about two seconds,
  request SAM2 and show the returned mask as a non-destructive orange preview
  overlay.
- Display returned mask as a preview layer.
- Reviewer must explicitly apply the SAM2 point preview to the talc mask with
  the `Apply SAM2` button. If no preview is ready, the same button may run and
  apply the current hover point.
- Clip box/polygon SAM2 results to the reviewer-drawn prompt bounds.
- Reject implausibly large SAM2 results, currently any returned mask covering
  more than half the image, instead of merging a whole-screen mask into the talc
  mask.
- If SAM2 is unavailable, show a local dependency message and keep manual tools
  usable.

## Data Model

### Sample Record

Each sample shown by the app is resolved from `manifest.json` and per-sample
`conversion_summary.json`.

Required fields:

```json
{
  "sample_id": "DSCN3042",
  "sample_dir": "outputs/talc_blue_line_conversion/DSCN3042",
  "annotated_image": "dataset/.../Области оталькования/DSCN3042.JPG",
  "original_image": "dataset/.../Оталькованные руды/DSCN3042.JPG",
  "raw_blue_stroke": "raw_blue_stroke.png",
  "autodetected_talc_mask": "final_talc_mask.png",
  "current_talc_mask": "current_talc_mask.png",
  "sulfide_overlap_mask": "sulfide_overlap_mask.png",
  "qa_overlay": "qa_overlay.png",
  "status": "needs_manual_review",
  "match_status": "matched",
  "review_status": "unreviewed"
}
```

Optional fields:

- `silicate_support_mask`
- `silicate_supported_talc_mask`
- `silicate_unsupported_talc_mask`
- `talc_positive_core_mask`
- `silicate_hard_negative_mask`
- `reviewed_talc_mask`
- `reviewed_positive_bag_mask`
- `reviewed_talc_node_mask`
- `reviewed_not_talc_mask`
- `model_talc_mask`
- `human_review_masks`
- `non_neural_talcose_qa`
- `review_patch`

### Mask Semantics

- `autodetected_talc_mask`: initial mask produced from MS Paint pen annotations.
- `positive_bag`: original blue-line-derived region that can contain talc
  segments, plus manual Brush, Fill, Rectangle, Polygon, and SAM2 edits.
- `talc_node`: talc pixels created by the Similar intensity assist.
- `not_talc`: explicit hard-negative pixels. These are visually dark or
  otherwise talc-like false positives that the reviewer says are not talc.
- `current_positive_bag_mask`: editable working `positive_bag` class mask.
- `current_talc_node_mask`: editable working `talc_node` class mask.
- `current_not_talc_mask`: editable working `not_talc` hard-negative class.
- `current_talc_mask`: compatibility union of `positive_bag | talc_node`.
  `talc_node` may overlap `positive_bag`; `not_talc` is excluded from
  `talc_node` and not included in this union.
- `reviewed_positive_bag_mask`: final saved `positive_bag` class.
- `reviewed_talc_node_mask`: final saved `talc_node` class.
- `reviewed_not_talc_mask`: final saved hard-negative class.
- `reviewed_talc_mask`: final saved union mask for existing training/review
  export code.
- `model_talc_mask`: optional read-only trained model prediction for QA.
- `human_review_masks`: optional read-only masks from teammate review
  workspaces for agreement/disagreement QA.
- `non_neural_talcose_qa`: optional read-only talcose/not-talcose classifier
  QA result produced by the standalone non-neural talc-zone heuristic.
- `sulfide_overlap_mask`, `silicate_support_mask`, and related masks:
  display-only evidence layers for review; they are not edited by this app.
- Unlabeled background outside the reviewed talc/not-talc masks remains unknown
  for training unless downstream exporters explicitly treat it as negative.

## Patch Format

Every save writes both raster masks and an auditable patch.

```json
{
  "schema_version": "talc-mask-review-patch-v0.2",
  "sample_id": "DSCN3042",
  "reviewer": "",
  "review_status": "reviewed",
  "source": {
    "conversion_summary": "conversion_summary.json",
    "annotated_image": "dataset/.../Области оталькования/DSCN3042.JPG",
    "original_image": "dataset/.../Оталькованные руды/DSCN3042.JPG",
    "autodetected_talc_mask": "final_talc_mask.png",
    "base_working_talc_mask": "current_talc_mask.png",
    "working_positive_bag_mask": "current_positive_bag_mask.png",
    "working_talc_node_mask": "current_talc_node_mask.png",
    "working_not_talc_mask": "current_not_talc_mask.png",
    "model_talc_mask": "model_talc_mask.png",
    "human_review_masks": ["teammate/reviewed_talc_node_mask.png"]
  },
  "class_definitions": {
    "positive_bag": "blue-line/manual/SAM2 talc bag",
    "talc_node": "confirmed talc pixels",
    "not_talc": "explicit dark non-talc hard-negative pixels"
  },
  "edits": [
    {
      "edit_id": "edit_0001",
      "timestamp_utc": "2026-07-03T00:00:00Z",
      "tool": "filled_polygon",
      "operation": "add",
      "geometry": {
        "type": "polygon_xy",
        "points": [[10, 10], [100, 20], [80, 90]]
      },
      "notes": ""
    }
  ],
  "outputs": {
    "reviewed_talc_mask": "reviewed/reviewed_talc_mask.png",
    "reviewed_positive_bag_mask": "reviewed/reviewed_positive_bag_mask.png",
    "reviewed_talc_node_mask": "reviewed/reviewed_talc_node_mask.png",
    "reviewed_not_talc_mask": "reviewed/reviewed_not_talc_mask.png",
    "reviewed_overlay": "reviewed/reviewed_overlay.png",
    "review_summary": "reviewed/review_summary.json"
  }
}
```

Operations:

- `add`
- `remove`
- `replace`

Tools:

- `brush`
- `filled_polygon`
- `filled_rectangle`
- `sam2_point`
- `sam2_rectangle`
- `uploaded_mask`
- `advanced_coordinates`

## Backend API

The app can be implemented with manual routes. API responses are JSON unless
serving artifact bytes.

### Pages

- `GET /`: main app shell.
- `GET /samples/{sample_id}`: app shell preselected to one sample.

### Artifacts

- `GET /artifacts/{relative_path}`: serve only allowlisted files under the
  configured conversion/review directories.

### JSON Endpoints

- `GET /api/conversion/status`
- `POST /api/conversion/reconvert`
- `GET /api/manifest`
- `GET /api/samples/{sample_id}`
- `GET /api/samples/{sample_id}/mask-state`
- `POST /api/samples/{sample_id}/autosave-working-mask`
- `POST /api/samples/{sample_id}/draft`
- `POST /api/samples/{sample_id}/sam2-assist`
- `POST /api/samples/{sample_id}/save-review`
- `POST /api/samples/{sample_id}/reload-base`
- `POST /api/samples/{sample_id}/set-review-status`
- `GET /api/health`

### Save Review Request

```json
{
  "review_status": "reviewed",
  "reviewer": "",
  "notes": "",
  "edits": [],
  "view_settings": {
    "brightness_threshold_luma": 90,
    "brightness_threshold_formula": "luma = 0.299*R + 0.587*G + 0.114*B; luma <= threshold keeps the pixel, luma > threshold paints it white",
    "brightness_visible_pixels": 12345,
    "brightness_visible_total_pixels": 100000,
    "brightness_visible_fraction": 0.12345,
    "talc_cluster_overlay": {
      "enabled": true,
      "source": "talc_node",
      "radius_px": 64,
      "min_density_percent": 4,
      "opacity_percent": 45
    },
    "background_mode": "original"
  },
  "current_talc_mask_png_base64": "",
  "navigate": "stay"
}
```

The backend validates dimensions, writes masks, writes the patch JSON, and
returns updated metrics. `navigate` may be `stay` for `Save` or `next` for
`Save & Next`.

## File Layout Proposal

Implemented files:

```text
apps/talc_review_web.py
tests/test_talc_review_web.py
```

The application remains narrow and local: generated HTML/CSS/JavaScript live in
the Python app, while tests cover pairing, first-open autosave, reviewed saves,
theme controls, and HTTP endpoints without requiring a browser.

## Launch Command

Primary direct-input command:

```bash
python3 apps/talc_review_web.py \
  --annotated-dir "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --workspace-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port <free-local-port>
```

Prepared-workspace command:

```bash
python3 apps/talc_review_web.py \
  --conversion-dir outputs/talc_blue_line_conversion \
  --host 127.0.0.1 \
  --port <free-local-port>
```

No fixed default port is assigned in this spec. Before implementation chooses a
default port, check `/Users/ashmelev/Projects/10_admin/mbp2023/PORTS_USED.md`
and update it if a new persistent default is reserved.

## Validation

### Unit Tests

- Patch schema validates required fields.
- Direct-input startup creates a conversion workspace when no manifest exists.
- Direct-input startup matches annotated/original files by exact filename and
  reports `missing_original` samples.
- Prepared-workspace startup loads an existing manifest without reconverting.
- Opening a sample for the first time copies `autodetected_talc_mask` to
  `current_positive_bag_mask`, initializes `current_talc_node_mask` empty, and
  writes union `current_talc_mask`.
- Fill, polygon, rectangle, brush, and SAM2 mask previews convert to expected
  `positive_bag` rasters; Similar converts to expected `talc_node`
  rasters on synthetic inputs.
- Brush, Fill, Polygon, and Rectangle can write `not_talc`, and saved review
  outputs include `reviewed_not_talc_mask` without adding it to
  `reviewed_talc_mask`.
- Comparison mode defaults to `Current` and adds no extra QA layer.
- Plain `Heuristic` and `Neural Model` modes show only their corresponding
  prediction layer.
- The separate top-right `Display layers` widget has `Background` checked by
  default; unchecking it hides only the base image and still renders selected
  mask/QA/cluster overlays. Checking `Blank White` while `Background` is
  unchecked fills the otherwise empty canvas with white.
- Heuristic comparison reports agreement, heuristic-only, current-only, and
  sulfide-conflict counts when a heuristic talc-zone mask is available.
- Neural comparison reports agreement, neural-only, current-only, and
  sulfide-conflict counts when a model mask is available.
- Similar positive/negative seeds and `not_talc` masks constrain preview
  candidates and are recorded in edit metadata.
- Save-review rejects wrong-size masks.
- Artifact serving rejects paths outside the configured conversion directory.

### Browser/UX Smoke

Use Playwright or equivalent browser automation.

Required checks:

- App opens and loads the manifest.
- App launched with `--annotated-dir` shows conversion progress or a completed
  conversion status before opening the queue.
- Canvas renders a nonblank source image.
- Layer toggles and the over-image segmentation class widget update without a
  Python rerun.
- Brightness-threshold slider `0..255` updates the photo background without
  changing mask pixels: `255` shows the original image, `90` keeps dark
  talc-candidate pixels visible while whitening brighter matrix/sulfides, and
  `0` paints the background white.
- The bottom-left zoom widget and mouse-wheel zoom work and do not change mask
  geometry.
- The below-viewer mouse hint row is visible without opening a help panel.
- Polygon points can be added, closed by clicking the first point, edited, and
  flattened on save.
- Rectangles can be drawn by drag or two corner clicks; corners/edges can be
  dragged after drawing and flattened on save.
- Selected completed polygon/rectangle regions can be removed with
  Delete/Backspace without affecting focused text fields.
- Brush left/right mouse strokes modify only the selected edit class.
- Fill adds a bounded region without crossing blue strokes, sulfide pixels,
  current selected-class regions, or image edges, and writes the selected edit
  class.
- Similar previews intensity/color/texture-similar non-sulfide pixels from
  positive talc seeds, can be constrained with negative seeds / `Not Talc`
  hard negatives, can be tightened with Strictness, and is non-destructive until
  `Apply Similar`, `Save`, or `Save & Next` is pressed.
- Heuristic comparison overlay highlights heuristic-only in fuchsia
  (`#ec4899`), current-only, agreement, and sulfide conflict without mutating
  masks.
- Neural comparison overlay highlights neural-only, current-only, agreement,
  and sulfide conflict without mutating masks.
- Undo restores prior mask state.
- Completing or editing a shape updates/autosaves `current_talc_mask`.
- With sulfide protection enabled, brush/polygon/rectangle/SAM2 additions over
  sulfide pixels are clipped and the status line reports newly protected pixels
  without silently cleaning older overlap.
- SAM2 point hover preview is non-destructive until the reviewer presses
  `Apply SAM2`.
- `Subtract sulfides from mask` removes existing overlap, autosaves, and leaves
  `Current on sulfide px` at zero.
- `Save` writes reviewed masks and patch JSON.
- `Save & Next` writes reviewed masks and patch JSON, then selects the next
  sample in the queue.
- `Next` selects the next sample in the queue without writing reviewed outputs.
- `Download` exports `<sample>_enabled_layers.png` at the sample image
  resolution with enabled persistent classes and layers, and without viewer UI
  chrome or transient edit previews.
- SAM2 unavailable state does not block manual editing.

### Manual Acceptance

- A reviewer can correct one `needs_manual_review` talc sample in under two
  minutes after launch.
- The app makes it visually obvious that the reviewer is editing the talc mask.
- Original blue lines remain visible as reference but are never the edited
  object.
- Saved reviewed overlay matches the current canvas state.

## Migration Plan

1. Keep `apps/deprecated/streamlit/talc_review_streamlit.py` available as fallback.
2. Implement direct-input startup that creates or reuses the same conversion
   output directory.
3. Keep prepared-workspace mode for debugging/reproducibility.
4. Verify on at least one `candidate_ok`, one `needs_manual_review`, and one
   `sulfide_overlap_review_required` sample.
5. Update `COMMANDS.md` with the new launch command.
6. Mark Streamlit as deprecated only after the new app passes unit and browser
   smoke tests.

## Review Questions

1. Should the save output remain exactly under each sample's `reviewed/`
   directory, or should reviewed masks also be collected into a global training
   manifest immediately?
2. Should SAM2 polygon refinement be required for v0.1, or is point/rectangle
   enough for the first reliable replacement?
3. Should the app keep a reviewer name field, or is anonymous local review
   sufficient?
4. Should Streamlit remain as a documented fallback after the new app is
   accepted, or should it be removed to avoid two QA paths?

## Proposed v0.1 Acceptance Criteria

- The app has no Streamlit runtime dependency.
- The app can start directly from the official `Области оталькования` folder.
- The app matches each annotated file to the clean original by identical
  filename.
- The app can review all `42` talc samples after creating or reusing the
  conversion workspace.
- Opening or selecting a sample updates the browser URL to `/sample/<slug>`,
  and loading that URL directly selects the same sample.
- The app edits class masks, not blue annotation strokes: Brush, Fill,
  Rectangle, and Polygon edit whichever class is selected in the over-image
  Segmentation classes widget; SAM2 edits `positive_bag`; Similar edits
  `talc_node`.
- Toolbar controls are ordered icon-only Brush, Fill, Similar, Rectangle,
  Polygon, then SAM2, Undo, with active-tool parameters at the end; toolbar
  zoom controls are not shown.
- Mouse wheel zooms over the canvas without changing mask geometry, and the
  bottom-left zoom widget exposes one vertical stack: Fit, Actual size,
  Zoom In, the live percent, and Zoom Out controls.
- Pressing and dragging the mouse wheel / middle button over the canvas pans
  the view before any edit-tool handling and suppresses native browser
  middle-click scrolling. The pan range includes gutter space around the canvas,
  so the image can move partly outside the visible viewport like the v2 ore UI.
- The viewer shows the same mouse hints as the v2 ore UI below the main view:
  `Mouse wheel - zoom in / out` and `Mouse wheel press - pan`.
- Brightness threshold preview is available, reports the visible-pixel
  percentage/count for the active photo background, and does not change mask
  geometry or saved mask pixels.
- Talc cluster overlay is available as a separated row in the Segmentation
  classes widget, tunable by source/radius/density/opacity in the right panel,
  reports highlighted-area percentage, excludes sulfide pixels from source and
  highlight regions, and does not change mask geometry or saved mask pixels.
- Brush left mouse draws the selected edit class; Brush right mouse erases it.
- `B` selects Brush and `F` selects Fill without hijacking focused text inputs.
- Fill, polygon, and rectangle are direct filled-area tools for drawing the
  selected edit class; SAM2 is a direct filled-area tool for positive bag;
  Similar is a direct filled-area tool for adding talc nodes.
- `Not Talc` is an editable hard-negative class with its own visible overlay,
  live percentage, save output, and audit metadata; it does not enter the
  compatibility `reviewed_talc_mask`.
- Fill can add a bounded region without crossing blue strokes, sulfide pixels,
  current talc mask regions, or image edges.
- Similar can preview and apply luma/color/texture-similar non-sulfide talc-node
  candidates from positive seeds, while excluding negative seeds and `Not Talc`,
  without changing the mask before Apply.
- Optional heuristic comparison mode highlights `heuristic only`,
  `current only`, `agreement`, and `sulfide conflict`.
- Optional neural comparison mode highlights `neural only`, `current only`,
  `agreement`, and `sulfide conflict`.
- Selected completed polygon and rectangle regions can be deleted with
  Delete/Backspace.
- Sulfide protection is enabled by default; additive tools cannot add new talc
  pixels on the sulfide mask unless the reviewer disables protection.
- Manual sulfide subtraction is available and autosaves the current working
  mask.
- Undo is available for editing mistakes.
- Top-right `Save`, `Save & Next`, `Next`, and `Download` are all available.
- SAM2 is available as a canvas tool and optional dependency.
- SAM2 point mode supports idle hover preview plus explicit `Apply SAM2`.
- Reviewed outputs are raster masks plus a machine-readable patch JSON.
- UI work is local and file-based, so model training and inference remain
  runnable from CLI without the app.
