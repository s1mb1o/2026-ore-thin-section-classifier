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
- `docs/plans/27_talc-silicate-support-labeling.md`: optional silicate-support plan for conservative talc positives and hard negatives.
- `docs/benchmarks/01_binary_sulfide_model_benchmark.md`: binary sulfide model benchmark; SegFormer-B2 is the current default checkpoint, SegFormer-B1/B0 remain mirrored fallbacks.
- `docs/cards/binary-sulfide-model-card.md`: model provenance, metrics, limitations, and B0/B1 checkpoint status.
- `docs/cards/official-balanced-eval-dataset-card.md`: balanced image-level eval split and panorama caveats.
- `docs/cards/demo-run-fact-sheet.md`: B1 demo pipeline input, parameters, outputs, and deterministic result.
- `docs/notes/2026-07-03-official-metrics-and-panorama-split.md`: organizer clarification on IoU/Hausdorff, F1/AUC, and unlabelled panorama usage.
- `docs/notes/2026-07-03-research-mindstorm-improvements.ru.md`: research-backed Russian mindstorm for killer features, training upgrades, annotation strategy, and presentation framing.
- `docs/notes/2026-07-03-reusable-demo-libraries.md`: shared libraries for source fusion, active-review queues, dataset curation, component reports, report cards, and scribble classifiers.
- `docs/notes/2026-07-03-b1-visual-validation-pack.md`: six-image final-B1 visual sanity pack and calibration finding for ordinary/fine rules.
- `docs/notes/2026-07-03-heuristic-segmentation-subproject.md`: separate non-neural segmentation baseline, smoke result, limits, and intended use as a disagreement source.
- `docs/notes/2026-07-03-gpu-training-status.md`: current binary sulfide dataset, gx10 ResUNet job, and zelda blocker.
- `docs/notes/talc-blue-line-conversion.md`: v2 talc blue-line converter/review note.
- `docs/notes/2026-07-02-domain-datasets-search.md`: official dataset inventory and external dataset context.
- `docs/notes/2026-07-02-targeted-om-datasets-models.md`: targeted OM sulfide/talc dataset/model review.
- `docs/notes/2026-07-02-telegram-shlif-captains-chat.md`: organizer confirmations.

## Immediate Next Steps

1. Monitor gx10 `tmux nornickel_v2_resunet` and update `docs/benchmarks/01_binary_sulfide_model_benchmark.md` after epoch 30; latest observed ResUNet best is epoch 21 IoU `0.950462`.
2. Use the local SegFormer-B2 mirror at `models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/` as the default sulfide checkpoint.
3. Keep the local SegFormer-B1 mirror at `models/binary_sulfide/segformer_b1_dataset_v0_zelda_20260703_overnight_safetensors/` as the faster fallback checkpoint.
4. Use `outputs/official_balanced_eval_split.json` for balanced labelled-class evaluation; keep weak-label metrics separate from image-level class metrics.
5. Calibrate component ordinary/fine thresholds using the six-image visual pack plus the balanced split; current visual pack shows rule disagreements on 2 of 4 ordinary/fine examples.
6. Compare SegFormer-B1/B0 predictions against `heuristic_segmentation/` outputs and surface disagreement areas as the first sulfide QA queue.
7. Wire the reusable demo libraries into pipeline outputs: `source_fusion` -> `review_queue` -> `component_reports` -> `report_cards`, with `curation` for split/label QA and `scribble_classifier` as an optional reviewer-assist source.
8. Use the research mindstorm note to prioritize the remaining differentiating features: robustness certificate, illumination/flat-field artifact, high-loss pseudo-label cleanup, annotation-budget simulation, OIA-style report protocol, and MIL over sulfide components.
9. Use `apps/sulfide_qa_streamlit.py` as a file-based QA app with overlays, confidence heatmaps, disagreement layers, and JSON verdicts.
10. Review and accept/fix talc masks from `outputs/talc_blue_line_conversion` in `apps/talc_review_streamlit.py`.

## Known Risks

- No geologist is currently available; QA must be framed as non-expert cleanup and uncertainty marking.
- Official class folders are image-level labels, not pixel masks.
- Official panoramas are unannotated and unclassified in the provided dataset; use them for performance, visual QA, and stress testing unless a new annotation/classification pass is created.
- Organizer-recommended production metrics are IoU and Hausdorff distance for segmentation, F1 and AUC for classification; current weak-label IoU benchmarks are incomplete by that standard.
- Talc annotations are drawn as colored lines, so mask extraction needs pairing checks and visual QA.
- Very large panoramas require overlapping tiling and streamed stitching; full-image probability tensors can exceed practical memory.
- Zelda `root@161.104.48.181` initially booted without a visible NVIDIA GPU, then recovered after retry/reboot. Re-check `nvidia-smi` after any restart before assuming CUDA is available.

## Implemented Binary Sulfide Block

- Official manifest generation works with very large panoramas by disabling the PIL decompression limit; `outputs/official_manifest.json` contains `1236` images.
- Binary sulfide dataset builder writes tiled RGB images, masks, ignore masks, and a JSON manifest under `outputs/binary_sulfide_dataset_v0`.
- Training script supports `resunet`, `segformer_b0`, `segformer_b1`, and `segformer_b2` with ignored pixels, AMP, checkpoints, CSV logs, and IoU metrics.
- `scripts/evaluate_binary_sulfide.py` reports IoU, F1, AUC, Hausdorff, and HD95. SegFormer-B1 best eval: sulfide IoU `0.971548`, F1 `0.985569`, AUC `0.998522`, HD95 mean `26.25 px` on 512 sampled val tiles.
- `scripts/infer_binary_sulfide.py` runs overlapping tiled inference and writes `sulfide_mask.png`, `confidence.png`, `overlay_preview.jpg`, and `summary.json`.
- `scripts/analyze_ore_from_masks.py` computes connected-component ordinary/fine features, ore class rule output, `component_features.csv`, and intergrowth overlay.
- `scripts/run_ore_pipeline.py` runs image -> sulfide mask -> ore summary in one command.
- `scripts/build_official_balanced_eval_split.py` generated `outputs/official_balanced_eval_split.json` / `.csv` with `129` samples per ordinary/fine/talcose class; panoramas are listed separately as unlabelled.
- Final B2 demo output exists under `outputs/inference_demo/b2_final_row_2539589_1/`: final B2 inference on official row ore image, sulfide fraction `0.296259`, component summary, confidence map, and overlays.
- Local smoke tests passed for ResUNet and SegFormer-B0 on `outputs/smoke_binary_sulfide_dataset`; full local unit tests now cover `31` tests.
- gx10 ResUNet training is active in `tmux nornickel_v2_resunet`.
- zelda SegFormer-B2 training completed 30 epochs; best validation sulfide IoU is `0.974381` at epoch 20, with final epoch 30 IoU `0.969119`.
- zelda SegFormer-B1 training completed 30 epochs; best validation sulfide IoU is `0.971548` at epoch 16, with final epoch 30 IoU `0.964032`.
- gx10 ResUNet training completed 30 epochs; best validation sulfide IoU is `0.956436` at epoch 26, with final epoch 30 IoU `0.953216`.
- zelda SegFormer-B0 training completed 30 epochs; current best validation sulfide IoU is `0.953371` at epoch 13, with final epoch 30 IoU `0.951119`.
- zelda SegFormer-B2/B1/B0 and gx10 ResUNet `best.pt`, `last.pt`, `train_log.csv`, and `metrics.json` were mirrored locally under `models/binary_sulfide/`.
- Current binary sulfide benchmark is saved in `docs/benchmarks/01_binary_sulfide_model_benchmark.md`.

## Implemented Heuristic Block

- Separate subproject lives under `heuristic_segmentation/` and does not modify the neural training path.
- CLI: `python3 heuristic_segmentation/run_heuristic_segmentation.py --image ... --output-dir ...`.
- Outputs: `class_mask.png`, `sulfide_mask.png`, `talc_candidate_mask.png`, `analyzed_mask.png`, `overlay.png`, `components.csv`, `metrics.json`, `run_summary.json`, and `batch_summary.json`.
- Current method: analyzed-area mask, illumination-normalized brightness threshold, green/blue artifact suppression, morphology, connected components, ordinary/fine rules from area/solidity/compactness/replacement ratio, and conservative `talc_candidate`.
- Smoke on `dataset/Фото руд по сортам. ч1/Рядовые руды/DSCN2176.JPG` at `--max-side 900` passed; metrics were `sulfide_fraction 0.164864`, `talc_candidate_fraction 0.000708`, `component_count 70`.
- Unit tests: `python3 -m unittest discover -s heuristic_segmentation/tests -p 'test_*.py' -v`.

## Implemented Reusable Demo Libraries

- `src/ore_classifier/source_fusion.py` fuses heuristic/model/SAM/manual masks, writes weighted probabilities, positive vote counts, fused masks, disagreement maps, and source-agreement summaries.
- `src/ore_classifier/review_queue.py` turns uncertainty, decision impact, and novelty maps into ranked review candidates and Russian expert-question prompts.
- `src/ore_classifier/curation.py` provides lightweight image uniqueness, near-duplicate, hardness, and segmentation label-issue helpers without adding FiftyOne/cleanlab dependencies.
- `src/ore_classifier/component_reports.py` adds association contacts, sulfide liberation proxies, and deterministic ore-decision margin flags.
- `src/ore_classifier/report_cards.py` renders model cards, dataset cards, and run fact sheets for reproducibility/provenance outputs.
- `src/ore_classifier/scribble_classifier.py` provides an ilastik/Labkit-style nearest-centroid pixel classifier from sparse foreground/background scribbles.
- Full local unit tests now cover `31` tests with `python3 -m unittest discover -s tests -p 'test_*.py' -v`.

## Implemented Talc Block

- Code now lives in the v2 layout: `src/ore_classifier/talc_blue_line_converter.py`, `src/ore_classifier/sam2_region_assist.py`, `scripts/convert_talc_blue_lines.py`, `apps/talc_review_streamlit.py`, and `tests/test_talc_blue_line_converter.py`.
- Presentation framing now treats missing LumenStone/Petroscope silicate/matrix support on a talc candidate as a hard-negative or uncertainty signal, not as direct talc supervision.
- The talc converter now accepts optional `--silicate-mask-dir` masks by image stem. With support masks, `final_talc_mask` is the supported candidate, unsupported candidate pixels are added to `ignore_mask`, `talc_positive_core_mask` gives conservative positives, and `silicate_hard_negative_mask` gives `not_talc` hard negatives outside the annotation.
- Full conversion of `dataset/Фото руд по сортам. ч1/Оталькованные руды/Области оталькования` was regenerated under `outputs/talc_blue_line_conversion`.
- Manifest counts: `42` samples, `31` `candidate_ok`, `9` `needs_manual_review`, `2` `sulfide_overlap_review_required`.
- Streamlit talc review displays original blue annotation lines explicitly but edits the current talc/ignore mask, not the blue strokes. The main `Workspace` is `Review canvas`, which defaults to `Current mask` and has five clear tools: `Brush`, `Erase`, `Filled polygon`, `Filled box`, and `SAM2 assist`.
- Full filled-area editing is implemented through the local Streamlit component at `apps/components/mask_shape_editor/index.html`. In `Review canvas`, polygon vertices can be dragged, inserted, and deleted; boxes support corner/edge drag before applying the filled area as a mask.
- UX check on 2026-07-03 replaced rerun-resetting `st.tabs` with a stateful `Workspace` segmented control, added local `Current talc px` / `Current ignore px` / `Unsaved edits` metrics inside edit workspaces, and changed edit actions to rerun with flash messages so overlays/metrics update immediately. `Reload base masks` now also clears unsaved edit history and canvas/shape drafts.
- `Advanced` keeps exact-coordinate fallbacks out of the normal review path: polygon table, rectangle form, and coordinate SAM2 prompt. SAM2 remains optional, is available as `SAM2 assist` in `Review canvas`, and requires local `torch` plus the official `facebookresearch/sam2` package.
