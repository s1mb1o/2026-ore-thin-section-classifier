# Talc Blue-Line Conversion

Date: 2026-07-03

This note records the v2 talc annotation conversion path for
`Области оталькования`.

## Purpose

The official talc annotations are not binary masks. They are microscope photos
with blue hand-drawn boundary strokes. Some strokes are open, and some marked
regions overlap bright sulfide grains. The v2 converter produces conservative
candidate talc masks plus QA artifacts for manual review. It can also consume an
optional silicon/silicate support mask as evidence: supported pixels inside the
talc candidate become stronger positives, unsupported candidate pixels become
uncertain, and supported pixels outside the candidate become hard negatives.

## Code

- `src/ore_classifier/talc_blue_line_converter.py`: conversion core and review-mask helpers.
- `scripts/convert_talc_blue_lines.py`: CLI.
- `apps/deprecated/streamlit/talc_review_streamlit.py`: Streamlit QA app.
- `src/ore_classifier/sam2_region_assist.py`: optional SAM2 point/box assist, active only when `torch` and `sam2` are installed.
- `tests/test_talc_blue_line_converter.py`: focused synthetic unit tests.

## Command

```bash
python3 scripts/convert_talc_blue_lines.py \
  --input "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --output-dir outputs/talc_blue_line_conversion \
  --summary-json outputs/talc_blue_line_conversion_summary.json
```

Optional external sulfide masks can be supplied by image stem:

```bash
python3 scripts/convert_talc_blue_lines.py \
  --input "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --output-dir outputs/talc_blue_line_conversion \
  --sulfide-mask-dir path/to/binary_sulfide_masks
```

Optional silicon/silicate support masks can also be supplied by image stem:

```bash
python3 scripts/convert_talc_blue_lines.py \
  --input "dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования" \
  --output-dir outputs/talc_blue_line_conversion \
  --sulfide-mask-dir path/to/binary_sulfide_masks \
  --silicate-mask-dir path/to/silicate_support_masks
```

## Outputs

Each sample directory contains:

- `raw_blue_stroke.png`
- `closed_blue_stroke.png`
- `filled_talc_region.png`
- `candidate_talc_mask.png`
- `sulfide_mask.png`
- `sulfide_overlap_mask.png`
- `silicate_support_mask.png`
- `silicate_supported_talc_mask.png`
- `silicate_unsupported_talc_mask.png`
- `talc_positive_core_mask.png`
- `silicate_hard_negative_mask.png`
- `ignore_mask.png`
- `final_talc_mask.png`
- `qa_overlay.png`
- `conversion_summary.json`

For training, use `talc_positive_core_mask.png` as conservative positive talc,
`silicate_hard_negative_mask.png` as `not_talc` hard negatives, and
`ignore_mask.png` for uncertain/markup/sulfide-overlap pixels. Do not treat the
whole silicate support mask as talc.

The current v2 full run below was generated without an external silicate support
mask:

```text
outputs/talc_blue_line_conversion/manifest.json
sample_count: 42
candidate_ok: 31
needs_manual_review: 9
sulfide_overlap_review_required: 2
```

Samples requiring review:

```text
2550376-1 5x
2550377-1 5x
2550381-2 10x
2550382-1 10x
DSCN3042
DSCN3056
DSCN3057
DSCN4714
DSCN4715
DSCN4755
DSCN5180
```

## Review UI

```bash
streamlit run apps/deprecated/streamlit/talc_review_streamlit.py -- \
  --conversion-dir outputs/talc_blue_line_conversion
```

The app displays original blue annotation lines next to the QA overlay by
default. Editing is mask-first: canvas defaults to `Current mask`, and the
original blue-line image remains available as a reference background.

The edit area uses a stateful `Workspace` segmented control so the selected
workspace survives Streamlit reruns after applying edits. Each editing
workspace shows local `Current talc px`, `Current ignore px`, and `Unsaved
edits` counters.

Main `Review canvas` tools:

- `Brush` and `Erase` apply stroke-width edits to the current mask.
- `Filled polygon` and `Filled box` apply editable filled areas; polygon
  vertices can be dragged, inserted, and deleted, and boxes support corner/edge
  drag before applying.
- `SAM2 assist` uses the same canvas background and action controls, with
  draggable point or box prompts plus `Load/check SAM2` and `Run SAM2`.

`Advanced` keeps exact-coordinate fallbacks out of the normal review path:
polygon table, rectangle form, and coordinate SAM2 prompt. SAM2 remains
optional and uses explicit model/device controls plus a `Load/check SAM2`
button before running point or box prompts. The model cache should stay in the
normal Hugging Face cache, not under the project tree.

Reviewed outputs are saved under each sample directory:

```text
reviewed/
  reviewed_talc_mask.png
  reviewed_ignore_mask.png
  reviewed_overlay.png
  review_patch.json
  review_summary.json
```

`Reload base masks` resets the current session masks and clears unsaved edit
history/canvas objects.

## Current Boundary

Convex-hull fallback is disabled by default because it overfills open strokes.
Use `--fallback-hull` only for exploratory QA, not final training masks without
manual review.
