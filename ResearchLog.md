# ResearchLog

## 2026-07-03

### Binary sulfide evaluation and ore pipeline

- Added official-metric evaluator `scripts/evaluate_binary_sulfide.py` with IoU, F1, AUC, Hausdorff, and HD95.
- Evaluated SegFormer-B1 best checkpoint on the full val split: sulfide IoU `0.971548`, F1 `0.985569`, AUC `0.998522`, HD95 mean `26.25 px` on 512 sampled val tiles.
- Evaluated SegFormer-B0 best checkpoint on the full val split: sulfide IoU `0.953371`, F1 `0.976129`, AUC `0.996154`, HD95 mean `33.92 px` on 512 sampled val tiles.
- Added tiled inference `scripts/infer_binary_sulfide.py`, component/ore analysis `scripts/analyze_ore_from_masks.py`, and one-command runner `scripts/run_ore_pipeline.py`.
- Final B2 demo on `2539589-1.JPG` produced `sulfide_fraction 0.296259` and a deterministic `hard_to_process_ore` summary with ordinary/fine component overlays under `outputs/inference_demo/b2_final_row_2539589_1/`.
- Ran a six-image final-B1 visual validation pack from the balanced split under `outputs/visual_validation_b1_final/`. Masks/overlays rendered, but deterministic ordinary/fine rules disagreed with folder labels on 2 of 4 ordinary/fine examples; saved the calibration finding in `docs/notes/2026-07-03-b1-visual-validation-pack.md`.
- Added balanced official image-level split generation: `outputs/official_balanced_eval_split.json` has `387` labelled images, `129` per ordinary/fine/talcose class, and keeps `14` panoramas separate as unlabelled stress images.
- Added model/data/run cards under `docs/cards/` for checkpoint provenance, balanced split caveats, and the B1 demo pipeline run.
- Added official label audit and deconflicted split generation. SHA-256 audit over official labelled folders found `1180` labelled images, `1124` unique hashes, `56` duplicate-content groups, and `24` conflicting-label duplicate groups. The preferred deconflicted balanced split is `outputs/official_balanced_eval_split_deconflicted.json` with `345` images (`115` per class).
- Added analyzed-area denominator support to the ML inference/analysis path: black/blue-markup excluded pixels no longer dilute `sulfide_fraction` and `talc_fraction`; full-image fractions remain as `*_fraction_image`, and `ore_summary.json` now exposes decision margins plus review warnings.
- Added `scripts/calibrate_ore_rules.py` so a completed B2 official batch can grid-search deterministic talc/ordinary/fine thresholds from `summary.csv` and per-run `component_features.csv`; output is an explicit calibration artifact because it uses image-level folder labels rather than pixel-level geological ground truth.

### Heuristic segmentation baseline

- Added a separate non-neural segmentation subproject under `heuristic_segmentation/`.
- The baseline uses illumination-normalized brightness, green/blue artifact suppression, morphology, connected components, and component-level ordinary/fine rules.
- Official-image smoke on `dataset/Фото руд по сортам. ч1/Рядовые руды/DSCN2176.JPG` wrote masks/overlay/components to `outputs/heuristic_segmentation_smoke`; result was `sulfide_fraction 0.164864`, `talc_candidate_fraction 0.000708`, and `70` sulfide components.
- Saved method, smoke result, and limits in `docs/notes/2026-07-03-heuristic-segmentation-subproject.md`.

### Official metrics and panorama split clarification

- Recorded organizer clarification: production metrics to track are IoU and Hausdorff distance for segmentation, F1 and AUC for classification.
- Panoramas can be used for testing/stress testing, but in the provided dataset they are unannotated and unclassified; evaluation should prefer a balanced sample from labelled classes.
- Saved implications in `docs/notes/2026-07-03-official-metrics-and-panorama-split.md` and updated the current binary sulfide benchmark caveats.

### Research mindstorm for project improvements

- Checked current segmentation, weak-supervision, active-learning, uncertainty, and geology-microscopy papers/sources for ideas that fit the v2 official OM-only path.
- Saved Russian idea map in `docs/notes/2026-07-03-research-mindstorm-improvements.ru.md`.
- Highest-value directions: source-disagreement maps, active-learning queue for the most valuable crops, SAM2-assisted QA, per-component ordinary/fine passports, decision margins near official thresholds, robustness certificates, semi-supervised/CPS training, SAM-enhanced WSSS from image-level folders, and MIL over sulfide components.
- Deeper search added stronger implementation candidates: OIA-style protocol/report framing, enhanced thresholding/hard negatives for talc, Snorkel/STAPLE-like source fusion, co-teaching/high-loss rejection for pseudo masks, CLAM-like MIL over sulfide components, FDA-style microscope color adaptation, exact tile/halo inference, Hann blending, and rare-class losses such as Focal Tversky/Lovasz.
- Reddit/practitioner search added implementation and demo ideas: avoid presenting a plain SAM wrapper, use SAM only in a review loop, add flat-field/illumination normalization artifacts, export high-loss tiles for pseudo-label cleanup, combine uncertainty with diversity and decision impact for active learning, keep local brush/magic-wand annotation UX simple, and make component passports resemble mineral-property observation cards.
- Broader tooling/workflow search added more project directions: ilastik/Labkit/Weka/QuPath-style scribble pixel classifiers, RootPainter/MONAI/AIDE-style corrective annotation, FiftyOne/cleanlab-style dataset curation and label issue detection, MicroNet/SSL microscopy pretraining, μSAM/MatSAM prompt planning, object-level component classifiers, mineral liberation-style association reports, model/data cards, expert-question generation, and annotation-budget simulation.
- Implemented the applicable demo-ready ideas as reusable libraries with synthetic tests: source fusion/disagreement maps, active-review queue and expert questions, lightweight dataset curation/label-issue helpers, component association/liberation proxies, model/dataset/run cards, and a dependency-light scribble classifier. Saved implementation note in `docs/notes/2026-07-03-reusable-demo-libraries.md`.
- No drop-in public polished-section OM talc segmentation dataset was found; official blue-line talc annotations remain the primary talc supervision source.

### Binary sulfide model benchmark

- Benchmarked the first binary `sulfide / not_sulfide` segmentation runs on `outputs/binary_sulfide_dataset_v0`.
- Current best checkpoint is SegFormer-B2 on zelda: best validation sulfide IoU `0.974381` at epoch 20; final epoch 30 sulfide IoU `0.969119`.
- ResUNet on gx10 completed 30 epochs: best validation sulfide IoU `0.956436` at epoch 26; final epoch 30 sulfide IoU `0.953216`.
- SegFormer-B2 extended eval reached F1 `0.987024`, AUC `0.998811`, and HD95 mean `23.57 px`, beating B1 on all tracked weak-label metrics.
- Mirrored SegFormer-B0/B1/B2 and ResUNet `best.pt`, `last.pt`, `train_log.csv`, and `metrics.json` locally under `models/binary_sulfide/`.
- Saved the benchmark details and weak-label caveats in `docs/benchmarks/01_binary_sulfide_model_benchmark.md`.

### Clean v2 extraction

- Created a focused v2 workspace from the original hackathon project.
- Carried forward only official task sources and sulfide/talc implementation plans.
- Kept the verified official dataset as a relative symlink instead of copying data.

Detailed current direction is in `docs/session-sync.md`.
