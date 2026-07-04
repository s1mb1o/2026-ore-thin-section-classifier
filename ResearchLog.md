# ResearchLog

## 2026-07-04

### Current-state idea review

- Reviewed the research mindstorm, pipeline proposals, and v2 UI backlog against
  the current project state after the latest UI/runtime/model work.
- Saved the updated priority review in
  `docs/notes/2026-07-04-current-state-ideas-review.md`.
- Key conclusion: stop treating the backlog as "invent more features"; prioritize
  integration of measured strengths: Path A ordinary/fine decision lane,
  trained talc branch with honest fraction caveats, source-disagreement map,
  review candidates, panorama compliance evidence, and complete runtime/evidence
  provenance.
- Added a current-state addendum to
  `docs/notes/2026-07-03-pipeline-improvement-proposals.md` so stale 2026-07-03
  claims are not mistaken for current facts.
- Refreshed `docs/ui/v2/TODO_CANDIDATES.md`: demoted already-implemented
  ZIP/status work, promoted decision-lane provenance, talc claim guard,
  source-disagreement layer, review candidates, panorama run card, and workspace
  demo health strip.

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
- Wired calibration artifacts back into the runnable pipeline: `--rule-config-json` accepts `ore_rule_calibration.json` in `scripts/analyze_ore_from_masks.py`, `scripts/run_ore_pipeline.py`, and `scripts/run_official_batch.py`, so a selected `best_config` can be applied to later batches or demos without manually copying four threshold flags.
- Added a reproducible merge step for sharded official batches (`scripts/merge_official_batch_shards.py`), replacing the previous one-off combine snippet with a tested command before evaluation/calibration.
- During the live zelda sharded B2 batch, fine-intergrowth CPU analysis was slower than GPU inference because component morphology ran on a full-frame mask for every component. The component feature path now crops each component to its padded bounding box before morphology.
- Completed the zelda B2 deconflicted balanced batch: `345` rows, `0` failures. The deterministic rule scored macro F1 `0.1849` and macro AUC OVR `0.4264`; grid-search calibration improved macro F1 only to `0.2743`. A cross-validated ExtraTrees classifier over the extracted image/component features reached macro F1 `0.7439` and macro AUC OVR `0.8802`, making the feature-classifier path the strongest current image-level F1/AUC artifact.
- Diagnosed the local ML UI failure as Transformers SegFormer namespace drift: zelda checkpoints use `segformer.stages.*`, while the local runtime creates `segformer.encoder.*`. Implemented a strict all-keys/all-shapes remap in `src/ore_classifier/model_io.py`, saved details in `docs/notes/2026-07-03-segformer-transformers-namespace-compatibility.md`, and added the consolidated work/verification handoff `docs/notes/2026-07-03-ml-runtime-fix-work-summary.md`.

### Heuristic segmentation baseline

- Added a separate non-neural segmentation subproject under `heuristic_segmentation/`.
- The baseline uses illumination-normalized brightness, green/blue artifact suppression, morphology, connected components, and component-level ordinary/fine rules.
- Official-image smoke on `dataset/Фото руд по сортам. ч1/Рядовые руды/DSCN2176.JPG` wrote masks/overlay/components to `outputs/heuristic_segmentation_smoke`; result was `sulfide_fraction 0.164864`, `talc_candidate_fraction 0.000708`, and `70` sulfide components.
- Saved method, smoke result, and limits in `docs/notes/2026-07-03-heuristic-segmentation-subproject.md`.

### Talc non-sulfide segmentation baseline

- Updated `scripts/build_talc_dataset.py` so `sulfide_mask.png` pixels are ignored by default for talc training; the model learns `talc` vs `not_talc` only on non-sulfide analyzed pixels.
- Added `scripts/train_talc_segmentation.py` and `scripts/infer_talc_segmentation.py` for talc-named training and non-sulfide-clipped tiled inference.
- Built `outputs/talc_non_sulfide_dataset_v0` from all 42 reviewed talc samples: `1510` tiles (`1150` train / `360` val), all 42 sulfide masks loaded, `25,036,407` sulfide pixels ignored.
- Ran a local MPS ResUNet baseline under `models/talc_segmentation/resunet_non_sulfide_20260703_local`; best validation talc IoU was `0.526502` at epoch 1.
- Inference smoke on held-out `DSCN4714` wrote `outputs/talc_segmentation_predictions/resunet_non_sulfide_20260703_local_DSCN4714`; final talc mask has `0` pixels on sulfides and non-sulfide IoU `0.693980` vs the reviewed mask.
- Added `scripts/run_talc_segformer_folds.py`, an image-level fold runner that rebuilds non-sulfide talc datasets, trains pretrained SegFormer checkpoints, and calibrates probability thresholds on validation tiles.
- Ran a local SegFormer-B0 fold smoke under `outputs/talc_segformer_folds/segformer_b0_smoke_20260703`: `nvidia/mit-b0` loaded, one capped fold trained for 10 steps, and best calibrated validation tile talc IoU was `0.373902` at threshold `0.40`. This verifies the path but is under-trained.
- Completed the full zelda SegFormer-B0 5-fold run under `outputs/talc_segformer_folds/segformer_b0_full_20260703`: mean calibrated talc IoU `0.644191`, mean F1 `0.782301`, with per-fold thresholds `0.50`, `0.50`, `0.40`, `0.35`, and `0.55`.
- Saved details in `docs/notes/2026-07-03-talc-non-sulfide-segmentation-training.md`.

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
