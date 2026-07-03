# Pipeline Improvement Proposals (Research Only)

Date: 2026-07-03 (state verified against the repo at ~09:40)

Method: multi-agent docs+code review — 5 parallel readers over all `docs/` clusters and the actual code, 6 improvement lenses, merge/dedup to 23 proposals, adversarial verification (2 skeptics per proposal). A rate limit stopped verification after the top proposals, so items below are marked **CONFIRMED** (verified by an adversarial checker against the repo, or re-verified by hand during synthesis) or **unverified** (agent claim, spot-check before acting). No code was written for this note.

Relationship to existing notes: this complements `docs/notes/2026-07-03-research-mindstorm-improvements.ru.md`. The mindstorm collects research-backed ideas and presentation framing; this note is a current-state-verified engineering gap list, prioritized against the submission deadline.

**Moving target caveat**: the repo advanced substantially while this review ran (SegFormer-B2, `run_ore_pipeline.py`, `run_official_batch.py`, `evaluate_ore_classification.py`, balanced eval split, talc web review app all landed in parallel). Every proposal below was re-checked against the state at ~09:40; statuses may lag further parallel work.

## Hard constraints driving priorities

- **Deadline: 2026-07-04 23:59** (official task page). The Telegram schedule note says upload by 07-05 08:00 — a discrepancy worth resolving with organizers, but plan for the earlier date.
- Official numeric bars: talc-fraction error ≤ ±3 pp vs expert annotation; intergrowth classification F1 ≥ 90%; one 10k×10k panorama in ≤ 5 minutes on a CPU/GPU workstation.
- Organizer metrics: IoU + Hausdorff (segmentation), F1 + AUC (classification). No automated scoring; jury review.
- One non-expert human, gx10 + zelda GPUs, no geologist.

## Current state (verified 2026-07-03 ~09:40)

What exists: SegFormer-B2 default checkpoint (val sulfide IoU 0.9744 / F1 0.9870 / AUC 0.9988 / HD95 23.57 px on the weak-label split); Hann-blended memmap tiled inference (`scripts/infer_binary_sulfide.py`); one-command `scripts/run_ore_pipeline.py` with `--talc-mask` / `--auto-talc-candidate`; `scripts/run_official_batch.py` + balanced 387-image split (129/129/129) + `scripts/evaluate_ore_classification.py` (image-level accuracy, P/R/F1, macro F1, one-vs-rest AUC, confusion); manual review pack generator; talc blue-line converter + Streamlit and new web canvas review apps; model/data/run cards.

The two loudest facts:

1. **No image-level metric has ever been produced.** The balanced split exists, the batch runner exists, the evaluator exists — but only one-image smoke runs have been made. The single recorded end-to-end demo classified an image from the *ordinary* folder as `hard_to_process_ore` with 80.1% fine share (`docs/cards/demo-run-fact-sheet.md`), i.e. the ordinary/fine thresholds are visibly miscalibrated. The 90% F1 bar is currently unmeasured, and likely far away.
2. **Headline segmentation numbers are partly self-confirming.** 700 of 1588 val tiles are scored against the same Otsu heuristic that generated their training labels; only the 888 LumenStone tiles have human masks. There is zero human-verified official-domain segmentation evidence.

---

## Tier 0 — correctness fixes that gate every number (do first, all cheap)

### 1. Dedupe + label-conflict quarantine + group split keys (P2) — **partially CONFIRMED, re-run audit first**

The finder agent hashed the official class folders and reports: 1180 labeled images, only 1124 unique contents; 56 duplicate-content groups, **24 with conflicting labels** (e.g. byte-identical `DSCN4695.JPG` filed as «Оталькованные» in ч1 and «рядовые» in ч2; `2550376-2 10x.JPG` is both talcose and fine *and* one of the 42 blue-line annotations). The balanced eval split reportedly contains ~18 conflict-file entries, and `binary_sulfide_dataset_v0` has identical content in train and val (literal leak). Additionally, splits are per image with no physical-sample grouping, while filename stems prove multi-view same-sample groups (`2544791-2 10x` vs `20x`).

Proposal: `scripts/audit_official_labels.py` (SHA-256 + perceptual hash) emitting `duplicates.json` + `label_conflicts.csv`; one entry per unique content; conflicting-label contents quarantined from training/calibration/eval (keep the list — they double as organizer-level borderline cases); split assignment by physical-sample group key (leading numeric stem; ч1/ч2 in the key), never per file. Rebuild the balanced split (v2) and dataset manifest (v1) under these rules.

Why first: with ~4.6% of the 387-item split unscoreable, every F1/AUC number carries a built-in error floor, and the train/val leak inflates segmentation val. Effort S. Risk: shrinks the talcose eval pool; document hashes and both labels rather than silently dropping. *The hash counts come from the agent's own audit run — re-run the audit as step one; the script is the deliverable anyway.*

### 2. Analyzed-area denominator fix (P3) — **CONFIRMED current** (`scripts/infer_binary_sulfide.py:97`, `src/ore_classifier/component_analysis.py:71`)

Both shipped fraction sites divide by the **full image** including black borders, glare, and blue annotation strokes, contradicting plan 25 §8 ("analyzed non-excluded pixels"). Because the ore rule is strict `talc_fraction > 0.10`, an inflated denominator systematically underestimates talc: a panorama with ~15% dark border turns true 11% talc into a reported ~9.4% and silently flips the class off talcose. A tested producer already exists (`heuristic_segmentation._build_analyzed_mask`, V≥18 + blue-stroke exclusion) but no ML-path consumer uses it.

Proposal: port the analyzed-mask producer into `src/ore_classifier`, emit `analyzed_mask.png`, report fractions both of-image and of-analyzed, and make all downstream fraction math (talc %, predominance) consume the analyzed denominator. Must land **before** threshold calibration, or calibrated thresholds refer to the wrong quantity. Effort S. Risk: over-aggressive glare exclusion can eat bright sulfides — require full saturation + low chroma and log excluded area.

### 3. Decision margins, `needs_expert_review`, zero-sulfide warning (P10) — **CONFIRMED current** (no margin/warning fields in `component_analysis.py`)

A zero-sulfide image currently yields `ordinary` silently (0 ≥ 0 predominance); nothing reports distance to the 10% talc or 50/50 intergrowth thresholds although the fixed conventions (tie → ordinary + warning) require it. Proposal: add `talc_margin`, `intergrowth_margin`, `needs_expert_review` (near-threshold, zero sulfide, low analyzed fraction), and a `warnings` list to `OreSummary`/`ore_summary.json`, plus the exact TZ-format RU verdict. The 24 organizer-side label conflicts are the jury story: "our review margin catches exactly the cases the organizers themselves labeled inconsistently." Effort S, additive JSON only. Endorsed from the mindstorm (idea 5) — now with a concrete calibration check attached.

---

## Tier 1 — close the measurement loop (the actual critical path to the official bars)

### 4. Run and calibrate the image-level ore-classification harness (P5) — **CONFIRMED gap; harness partially built since**

The F1 ≥ 90% bar is unmeasured. Since the review started, `run_official_batch.py` + `evaluate_ore_classification.py` landed — so the remaining work is: (a) actually batch-run the balanced split (after Tier 0 dedupe/denominator), caching per-image `component_features.csv` + `ore_summary.json`; (b) add `scripts/calibrate_ore_rule.py` that re-scores any threshold vector from **cached features** (CPU, seconds per iteration — no GPU re-runs), grid-searching `ComponentRuleConfig` on ordinary-vs-fine images only (talcose-folder images excluded from the intergrowth fit — their intergrowth composition is unlabeled), with group-aware CV and a ч1↔ч2 cross-batch generalization check; (c) define AUC scores as the rule's own continuous decision variables (`talc_fraction` for talcose-vs-rest, `fine_share_among_sulfides` for fine-vs-ordinary) so AUC is statistically valid and the 10% / 50-50 operating points sit on the ROC.

Two divergent hand-set threshold sets currently coexist unvalidated (**CONFIRMED**): `ComponentRuleConfig` (dark_inside_ratio ≥ 0.18, solidity ≤ 0.62, compactness ≤ 0.12) vs `HeuristicConfig` (0.22 / 0.78 / 0.24, area ≤ 450). Exclude the 42 blue-line images from any threshold calibration — they are the pixel-level ±3 pp holdout. Effort M (GPU batch ~hours on zelda + cheap CPU search). Risk: first holdout F1 may be far below 90% — that is signal, not a harness defect; its per-image errors become the review queue.

### 5. Talc-fraction evaluator over the 42 blue-line pairs (P4) — **CONFIRMED, refined by verifiers; partially unblocked since**

`run_ore_pipeline.py --auto-talc-candidate` now makes the talcose branch reachable (it was structurally dead when the review ran). Still missing, and required for the ±3 pp claim: `scripts/evaluate_talc_fraction.py` — for each of the 42 pairs, reference = reviewed mask if present, else converter candidate for `candidate_ok`; predict talc on the **clean parent image** (paired by filename stem); exclude the dilated blue-stroke band from numerator and denominator; report per-image error in pp, MAE, share within ±3 pp, ore-class flips at 10%, bootstrap CI, and reference provenance. Spend the single human review pass on the 11 flagged conversions (9 `needs_manual_review` + 2 `sulfide_overlap_review_required`) in the new web review app so all 42 references become usable.

Verifier-added leakage guard (**CONFIRMED**): all 42 clean parents sit inside the 129 talcose images of the balanced eval split — if any of the 42 calibrate the talc heuristic, report talcose F1 both including and excluding those stems, or regenerate the split. Effort M. Risk: the HSV candidate may have poor recall (only smoke evidence: fraction 0.0007 on an ordinary image) — the harness quantifies it; worst case the ±3 pp claim is scoped down honestly.

### 6. Source-stratified evaluation + honest checkpoint selection + Hausdorff fixes (P6) — **stands; unverified details**

Report metrics by `source_type` (LumenStone-GT vs "agreement with labeling heuristic") in both training-time eval and `evaluate_binary_sulfide.py`; select `best.pt` by LumenStone-GT IoU, not the pooled/polluted number; final checkpoint choice by image-level ore-class macro-F1 over a fixed balanced subset (the metric the jury cares about). Fix Hausdorff: random stratified subsample instead of the first-512-in-loader-order tiles (reportedly 100% LumenStone), and count empty-vs-nonempty tiles separately instead of substituting the ~724 px tile diagonal into the mean (86.2 px reported mean vs 33.9 HD95 suggests heavy inflation). No retraining needed. Effort S. The manifest already carries `source_type` per tile.

---

## Tier 2 — cheap accuracy levers (each measurable via Tier 1)

### 7. Retrain from ImageNet-pretrained init (P7) — **CONFIRMED by hand: B2 `args.json` has `pretrained_model: null`**

Every deployed checkpoint (B0, B1, B2) was trained from **random initialization** although `train_binary_sulfide.py` contains a working `from_pretrained('nvidia/mit-bX')` path. B2 takes ~78 s/epoch on zelda → a controlled 30-epoch A/B costs ~40 GPU-minutes. Pre-download HF weights on zelda and verify from the log that pretrained weights actually loaded (the coded fallback to random init would silently degenerate the A/B). Rank with the stratified eval (item 6), not pooled val IoU. Likely the best accuracy-per-hour available in the repo. Effort S. Risk: low.

### 8. Threshold + temperature calibration on GT-only val tiles (P12) — **stands**

The 0.5 operating threshold has never been tuned, and the pipeline's outputs are *fractions* — a small consistent per-pixel bias converts directly into talc/predominance fraction error. Sweep threshold on the 888 GT val tiles picking the value that zeroes signed sulfide-fraction bias subject to IoU within 0.5 pp of max; fit one temperature for the confidence map (uncalibrated softmax makes margins and review-queue ranking numerically meaningless); write `calibration.json` next to the checkpoint, consumed by the infer script by default. Effort S. Risk: calibrates to the LumenStone proxy domain — report official-domain bias separately.

### 9. Post-stitch mask cleanup + component table from the neural path (P9) — **stands (no morphology in `infer_binary_sulfide.py` as of 09:40)**

Training pseudo-labels were built with open(3)/close(5) morphology and <48 px component drop, but stitched inference output is used raw — speckle noise then enters component classification, where small fragments bias the fine/ordinary vote. Apply the same morphology post-stitch, emit `component_stats.csv` (incl. `touches_image_border`, `crosses_tile_boundary`), and log raw-vs-cleaned fractions for auditability. Effort S–M. Risk: the 48 px minimum was tuned at photo resolution; log both fractions.

### 10. Aggregate-level fragmentation grouping (P19) — **stands; unverified line-level details**

Two biases reportedly push fine ore toward "ordinary": components below the min-area cutoff are excluded from the predominance vote while their pixels still count in sulfide area, and the ML-path rule (unlike the heuristic subproject) has no small-area clause — isolated fragments of a replaced grain are compact/solid and classify as ordinary one by one. Proposal: dilate/close the sulfide mask with an adaptive radius, take connected components of that as *grain aggregates*, compute per-aggregate replacement ratio and fragment density, classify fine/ordinary at aggregate level, components inherit the label. This measures exactly the TZ definition ("sulfides significantly replaced by non-ore phase"). Feed the aggregate radius into the calibration grid (item 4). Effort M — do it if item 4's first F1 comes out far below the bar, because this is the most likely fix.

---

## Tier 3 — compliance and robustness evidence

### 11. Panorama 5-minute compliance benchmark + OOM defusing (P8) — **stands; no panorama timing exists anywhere**

The only recorded end-to-end timing is 3.5 s for a 2272×1704 photo. The TZ bar is 10k×10k ≤ 5 min; real panoramas reach 27025×21227. Work package on the existing script: fp16 autocast around the forward pass; thread-pool tile prefetch; band-wise finalize over the memmaps; bounded-memory overlay path (the full-overlay path reportedly allocates tens of GB — **unverified**, check before relying on it); `--consistency-check` second pass with the grid shifted by stride/2 emitting flip rate + fraction drift + would-the-class-change; run on the largest panorama and a 10k×10k crop, GPU and CPU-only, record in `docs/benchmarks/`. Effort M. This converts an unverified TZ requirement into citable evidence and removes the most likely live-demo crash.

### 12. Tile-size consistency check + magnification-aware augmentation (P21) — **stands; 512-train vs 1024-infer CONFIRMED in demo params**

Every training tile is 512 px but inference defaults to 1024 — an input-size shift never measured. The 30-minute experiment first: infer 5–10 official images at tile 512/768/1024 and compare sulfide-fraction drift and component counts; if drift is material (> 0.5 pp), pin inference to 512 or retrain. Separately, official folders mix 5x/10x/20x magnifications of the same sections (verified in filenames), while augmentation has no scale jitter — add scale jitter, blur/sharpen, JPEG re-encode, gamma to `_augment` for the next retrain. Effort S (check) + M (retrain).

### 13. Margin-gated TTA — measure before adopting (P23) — **stands**

Blanket 4-transform TTA quadruples inference cost and threatens the 5-minute bar. Measure first on GT val tiles: if TTA moves fractions < 0.2 pp, ship without TTA and cite the measurement. If it matters, apply TTA only when an image's decision margin falls inside the review band, restricted to low-confidence tiles. Effort S to measure.

---

## Tier 4 — data/training upgrades (stretch; post-deadline material unless time appears)

### 14. Non-expert gold eval set for official sulfide tiles (P13) — the highest-value use of one human's 2–3 hours

There is zero human-verified official-domain segmentation evidence. Rank ~40–60 official val tiles by model-vs-heuristic disagreement × decision impact, correct only visually obvious errors in the existing review tooling (the new web canvas app or the review-pack flow — do **not** build a new QA app), freeze as `official_sulfide_eval_v0` with hashes, and report IoU/F1/HD95 on it as a separate row next to LumenStone-GT and heuristic-agreement rows. The gap between heuristic-agreement IoU and gold IoU quantifies pseudo-label quality and is itself the honesty headline. Label it non-expert QA, never expert ground truth.

### 15. Talc model path (P14 + P15 + P16) — the ±3 pp bar realistically needs more than an HSV heuristic

- **Clean-parent diff masks (P14)**: all 42 annotated files pair 1:1 by stem with clean parents; the converter currently emits the *annotated* image as the training image, excluding strokes only by color detection + 4 px dilation — JPEG halo extends further, so a model can learn "blue tint = talc". Use the clean parent as the training image and `|annotated − parent|` diff as the authoritative stroke-ignore. Small, surgical, kills the annotation-leak channel.
- **Talc segmenter via the existing trainer (P15)**: build a talc tile dataset from converted+reviewed masks (positives = conservative core, ignore = stroke band + margin ring), group-split by stem, train the existing SegFormer path (pretrained init), and select checkpoints by **held-out full-image talc-fraction MAE**, not tile IoU — the TZ target is a fraction metric. Keep talc a separate binary model; it composes at inference (talc searched only in non-sulfide matrix).
- **Domain-native hard negatives (P16)**: sample dark-matrix tiles from ordinary/fine folder images as `not_talc` (folder label bounds true talc ≤ 10%), with a contamination-aware weight — attacks the "everything dark = talc" failure mode without building a silicate detector; plan 27's silicate-support path is currently dead code (no silicate mask source exists).

### 16. Pseudo-label quality round (P17 + P18)

- **Plausibility gate (P17)**: manifest stats reportedly show whole official images whose Otsu threshold pinned at the clip floor labeled most of the matrix as sulfide (per-tile positive-fraction p90 up to 0.885) — these poison train *and* val. Gate at dataset build: reject/downgrade sources with implausible image-level fractions or clip-floor thresholds, cross-check against a second cheap source, record `label_quality` per source. *Numbers unverified — recompute from the manifest before acting.*
- **One self-training round (P18)**: plan 26's "≥2 sources agree → label, else ignore" fusion was never implemented — official supervision is pure single-source Otsu. Generate dataset v1 by model∧heuristic consensus (label 1 where prob ≥ 0.8 AND Otsu = 1; label 0 where prob ≤ 0.2 AND Otsu = 0; else ignore), keep LumenStone-GT val fixed as the anti-circularity anchor (if LumenStone IoU drops while official agreement rises, roll back), one round only, ~1–2 h GPU.

### 17. Multi-checkpoint disagreement maps at panorama scale (P22)

B2/B1/ResUNet checkpoints exist and one loader handles all families — extend the infer script to accept multiple checkpoints, emit ensemble mean + bit-encoded `source_votes.png` + `disagreement_map.png` in the band-wise finalize. Offline evidence mode only; never the timed path. This is what makes the mindstorm's KF-1 disagreement story real on actual panoramas.

---

## Cross-cutting (fits anywhere)

### 18. License/provenance table + card generation (P11) — **partially done since** (hand-written cards + `EXTERNAL_DATASETS.md` exist)

Remaining: a `licenses.md` per-asset table (LumenStone usage terms, HF `nvidia/mit-bX`, SAM2, Streamlit components — with an explicit "unverified" flag where status is unclear) — plans 22/24 treat this as submission-blocking; and init-provenance (random vs pretrained) in the model card, which currently omits it and would have caught item 7 earlier. Auto-generation from checkpoint `args.json` is nice-to-have, not critical.

### 19. Frozen eval manifests + benchmark ledger + metric golden tests (P20)

Freeze each eval set as a content-hashed manifest; append one row per (model, checkpoint SHA, eval-set version) to `docs/benchmarks/benchmark_ledger.csv`; add golden-value unit tests for `segmentation_metrics.py` so metric-code changes (e.g. the Hausdorff fixes) fail loudly instead of drifting silently. Matters because Tier 0 will regenerate datasets — without pinned eval versions, before/after comparisons stop being comparable. Effort S.

---

## Additional gaps noted during synthesis (the completeness-critic agent was rate-limited; these are mine)

1. **Preprocessing per TZ**: the ML inference path applies no illumination normalization; the TZ names it explicitly. The heuristic subproject has one. At minimum, state in the report which preprocessing the ML path uses and why (trained-domain match); at best, fold illumination-robustness into the consistency-check evidence (item 11).
2. **µm scale**: `metrics.csv`/area outputs are pixel-fraction only; scale metadata is absent from the dataset. Keep the plan-25 behavior — report fractions and warn that absolute areas need scale — but make sure the warning actually appears in `ore_summary.json`.
3. **Submission packaging is on the critical path** and none of it is pipeline work: README run commands from a clean checkout, the demo video, and the presentation consume hours. Time-box Tier 1 GPU work so packaging starts on 07-04 morning at the latest.
4. **Deadline ambiguity** (07-04 23:59 vs 07-05 08:00) — ask in the organizer chat today.

## Suggested execution order for the remaining ~36 h

1. Tier 0 (items 1–3) — half a day total, unblocks trustworthy numbers.
2. Item 7 (pretrained-init retrain) fired on zelda in the background immediately — it needs no decisions.
3. Item 4 (batch + calibrate) → first honest image-level F1. In parallel: item 5 (talc evaluator + review of the 11 flagged conversions).
4. Item 6 + 8 (stratified eval, calibration) → pick the shipping checkpoint honestly.
5. Item 11 (panorama benchmark) once the shipping checkpoint is fixed; item 12's 30-minute tile-size check alongside.
6. If F1 is far under the bar: item 10 (aggregates) is the most likely fix; recalibrate via cached features (cheap).
7. Packaging (README, video, presentation, licenses table) from 07-04 morning; Tier 4 only if everything above is green.

## What not to do before the deadline (re-affirmed from mindstorm + plans, still valid)

- No new UI surfaces (plan 25's "do not build a third UI" — the web talc review app now exists; repoint it rather than adding another).
- No Mask2Former / foundation-model replacement of the P0 path.
- No blanket TTA without the measurement in item 13.
- Never present weak-label agreement (the 0.97 IoU) as production accuracy; the stratified table (item 6) is the honest replacement.

## Sources

- Full agent outputs (readers, 23 merged proposals with evidence, verification verdicts): workflow run `wf_0c479e4e-e8e`, session transcript dir `subagents/workflows/wf_0c479e4e-e8e/journal.jsonl`.
- First-hand verified during synthesis: `docs/official/Скажи мне кто твой шлиф.md` (deadline, numeric bars), `docs/plans/25/26`, `docs/notes/2026-07-03-official-metrics-and-panorama-split.md`, `docs/benchmarks/01_binary_sulfide_model_benchmark.md`, `docs/cards/*`, `scripts/README.md`, `ChangeLog.md`, current line-level checks in `scripts/infer_binary_sulfide.py`, `src/ore_classifier/component_analysis.py`, `scripts/run_ore_pipeline.py`, B2 `args.json`.
