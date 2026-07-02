# Session Sync

Date: 2026-07-03

This is the shared handoff file for the clean v2 Nornickel hackathon workspace.

## Required Read Order

1. `~/.claude/CLAUDE.md`
2. Project `AGENTS.md` or `CLAUDE.md`
3. This file: `docs/session-sync.md`
4. `ChangeLog.md`
5. `ResearchLog.md`
6. `SMOKE_TESTS.md`
7. The focused official docs and plans linked below

## Current Objective

Build the official P0 optical-microscopy classifier for `Скажи мне, кто твой шлиф` without carrying over the old broad OM/SEM/XRD application surface.

Current target pipeline:

```text
official panorama image
-> binary sulfide segmentation
-> sulfide connected components
-> ordinary_intergrowth / fine_intergrowth classification
-> talc detection and talc fraction
-> deterministic ore class rule
-> mask, overlay, confidence heatmap, metrics, report
```

## Scope Boundaries

- In scope: official OM images, sulfide/non-sulfide segmentation, talc detection, high-resolution overlapping tiling, Streamlit QA, training manifests, model evaluation, final artifacts.
- Out of scope by default: SEM, XRD, defect/product platform UI, old generic QC dashboard, broad mineral ontology.
- Pseudo-labels are weak supervision, not expert geological ground truth.
- Non-expert QA may fix visually obvious mask errors, mark uncertain/excluded areas, and produce training patches.

## Dataset

The local `dataset` entry is a symlink:

```text
dataset -> ../2026_Nornikel_Hackaton/dataset
```

Source dataset facts from the original repository handoff:

- official download complete;
- `dataset/_download_manifest.json` verified `1236/1236` files;
- total verified bytes: `3,018,194,503`;
- panoramas: `14` JPG images, largest around `27025 x 21227` px;
- image-level class folders include ordinary/row, fine/hard-to-process, and talcose ore examples;
- `Области оталькования` contains blue-line talc annotations and must be inspected before training the talc detector.

## Focused Docs

- `presentation.md`: slide-by-slide explanation of the current official OM-only approach for the project presentation.
- `docs/official/Скажи мне кто твой шлиф.md`: saved official task page.
- `docs/official/Постановка задачи.docx`: official source document copy.
- `docs/specs/official-tz-solution-map.ru.md`: requirement-by-requirement solution mapping.
- `docs/plans/25_standalone-ore-classifier-project.md`: standalone implementation plan.
- `docs/plans/26_weak-supervision-sulfide-binary-model.md`: binary sulfide weak-supervision plan with Streamlit QA.
- `docs/notes/2026-07-03-gpu-training-status.md`: current binary sulfide dataset, gx10 ResUNet job, and zelda blocker.
- `docs/notes/talc-blue-line-conversion.md`: v2 talc blue-line converter/review note.
- `docs/notes/2026-07-02-domain-datasets-search.md`: official dataset inventory and external dataset context.
- `docs/notes/2026-07-02-targeted-om-datasets-models.md`: targeted OM sulfide/talc dataset/model review.
- `docs/notes/2026-07-02-telegram-shlif-captains-chat.md`: organizer confirmations.

## Immediate Next Steps

1. Monitor gx10 `tmux nornickel_v2_resunet` and pull `outputs/train_resunet_gx10_20260703_004425` after completion.
2. Monitor zelda `tmux nornickel_v2_segformer_b0` and pull `outputs/train_segformer_b0_zelda_20260702_220225` after completion.
3. Compare ResUNet vs SegFormer validation IoU and choose the binary sulfide model checkpoint for downstream component rules.
4. Implement `apps/sulfide_qa_streamlit.py` as a file-based QA app with overlays and JSON patches.
5. Add component-level ordinary/fine rules and talc fraction logic.
6. Review and accept/fix talc masks from `outputs/talc_blue_line_conversion` in `apps/talc_review_streamlit.py`.

## Known Risks

- No geologist is currently available; QA must be framed as non-expert cleanup and uncertainty marking.
- Official class folders are image-level labels, not pixel masks.
- Talc annotations are drawn as colored lines, so mask extraction needs pairing checks and visual QA.
- Very large panoramas require overlapping tiling and streamed stitching; full-image probability tensors can exceed practical memory.
- Zelda `root@161.104.48.181` initially booted without a visible NVIDIA GPU, then recovered after retry/reboot. Re-check `nvidia-smi` after any restart before assuming CUDA is available.

## Implemented Binary Sulfide Block

- Official manifest generation works with very large panoramas by disabling the PIL decompression limit; `outputs/official_manifest.json` contains `1236` images.
- Binary sulfide dataset builder writes tiled RGB images, masks, ignore masks, and a JSON manifest under `outputs/binary_sulfide_dataset_v0`.
- Training script supports `resunet`, `segformer_b0`, and `segformer_b1` with ignored pixels, AMP, checkpoints, CSV logs, and IoU metrics.
- Local smoke tests passed for ResUNet and SegFormer-B0 on `outputs/smoke_binary_sulfide_dataset`.
- gx10 ResUNet training is active in `tmux nornickel_v2_resunet`.
- zelda SegFormer-B0 training is active in `tmux nornickel_v2_segformer_b0`.

## Implemented Talc Block

- Code now lives in the v2 layout: `src/ore_classifier/talc_blue_line_converter.py`, `src/ore_classifier/sam2_region_assist.py`, `scripts/convert_talc_blue_lines.py`, `apps/talc_review_streamlit.py`, and `tests/test_talc_blue_line_converter.py`.
- Full conversion of `dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования` was regenerated under `outputs/talc_blue_line_conversion`.
- Manifest counts: `42` samples, `31` `candidate_ok`, `9` `needs_manual_review`, `2` `sulfide_overlap_review_required`.
- Streamlit talc review displays original blue annotation lines explicitly but edits the current talc/ignore mask, not the blue strokes. Canvas defaults to `Current mask`; pen/eraser use stroke width, polygon/box use filled areas, and `Move/resize` exposes Fabric object transforms for area objects.
- Full geometry editing is implemented through the local Streamlit component at `apps/components/mask_shape_editor/index.html`. The `Geometry` editor supports polygon vertex drag, polygon point insertion/deletion, and box corner/edge drag before applying the filled area as a mask. `Polygon table` and `Rectangle form` remain exact-coordinate fallbacks.
- UX check on 2026-07-03 replaced rerun-resetting `st.tabs` with a stateful `Editor` segmented control, added local `Current talc px` / `Current ignore px` / `Unsaved edits` metrics inside edit modes, and changed edit actions to rerun with flash messages so overlays/metrics update immediately. `Reload base masks` now also clears unsaved edit history and canvas objects.
- The SAM2 editor now has explicit model/device fields plus a `Load/check SAM2` button before running point or box prompts. SAM2 remains optional and requires local `torch` plus the official `facebookresearch/sam2` package.
