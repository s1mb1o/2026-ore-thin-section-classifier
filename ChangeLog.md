# ChangeLog

## 2026-07-03

- Added binary sulfide training utilities: official manifest generation, LumenStone/official pseudo-label tile dataset builder, ResUNet/SegFormer training script, dataset loader, model code, and smoke-tested both ResUNet and SegFormer-B0 locally.
- Built `outputs/binary_sulfide_dataset_v0` with `8536` tiles (`6948` train / `1588` val) from LumenStone masks and official heuristic pseudo masks.
- Synced v2 code and the binary sulfide tile dataset to gx10 and launched ResUNet training in `tmux` session `nornickel_v2_resunet`, output `outputs/train_resunet_gx10_20260703_004425`; first completed epochs reached sulfide IoU `0.912475` and `0.924633`.
- Prepared the new zelda host `root@161.104.48.181` with synced v2 code/dataset and a project venv. A later retry found the NVIDIA GPU attached, passed a SegFormer-B0 CUDA sanity run, and launched SegFormer-B0 training in `tmux` session `nornickel_v2_segformer_b0`, output `outputs/train_segformer_b0_zelda_20260702_220225`; observed epoch 9 val sulfide IoU `0.948772`.
- Added binary sulfide utility unit tests for LumenStone class-id mapping, brightness pseudo masks, class-id parsing, and tile edge padding; local `unittest` now covers `7` tests.
- Added `docs/notes/2026-07-03-gpu-training-status.md` with the current gx10/zelda training status and blocker.
- Updated `apps/talc_review_streamlit.py` for mask-first talc QA: canvas now defaults to current mask view, pen/eraser are the only stroke-width tools, polygon/box edits are filled areas, Fabric `Move/resize` is exposed for drawn area objects, a local `Geometry` component supports polygon vertex drag/add/delete and box corner/edge drag, table/form geometry fallbacks remain available, and SAM2 now has explicit model/device load-check controls plus point/box prompts.
- UX-tested the talc Streamlit review flow in browser and fixed interaction issues: editor selection now uses a stateful segmented control instead of rerun-resetting tabs, edit actions rerun with flash messages so overlays/metrics update immediately, edit modes show local current-mask counters, and `Reload base masks` clears unsaved edit history.
- Added `presentation.md` with a slide-by-slide explanation of the official OM-only ore-classifier approach.
- Ported the talc blue-line conversion/review block into the v2 layout: `src/ore_classifier/talc_blue_line_converter.py`, `src/ore_classifier/sam2_region_assist.py`, `scripts/convert_talc_blue_lines.py`, `apps/talc_review_streamlit.py`, tests, and `requirements.txt`. Regenerated all `42` `Области оталькования` samples under `outputs/talc_blue_line_conversion`; status counts are `31` `candidate_ok`, `9` `needs_manual_review`, and `2` `sulfide_overlap_review_required`.
- Created the clean v2 workspace for the official OM-only ore-classifier path.
- Added a relative `dataset` symlink to the verified dataset in `../2026_Nornikel_Hackaton/dataset`.
- Copied the selected official task docs, focused plans, specs, and research notes needed for the current sulfide/talc pipeline.
- Added v2-specific `AGENTS.md`, `README.md`, `docs/session-sync.md`, `ResearchLog.md`, `SMOKE_TESTS.md`, and initial placeholders for `apps/`, `scripts/`, `src/ore_classifier/`, `outputs/`, and `models/`.
