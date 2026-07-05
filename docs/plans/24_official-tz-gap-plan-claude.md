# Official TZ Gap Plan (Claude Cross-Check)

Date: 2026-07-02

Independent second-opinion gap analysis of the official hackathon TZ against the current repository, written without reusing the reasoning of `docs/plans/23_official-tz-gap-plan.md`. Both plans agree on the core diagnosis; this one adds a full requirement-by-requirement traceability matrix, verified file-level facts, a supervision strategy derived from the official package metadata, submission logistics, and a day-by-day schedule to the deadline.

Revised the same day after the official page copy (`docs/official/Скажи мне кто твой шлиф.md`) and the Yandex case-data metadata (`docs/notes/2026-07-02-domain-datasets-search.md`) became available: §4 replaces the earlier A/B/C data contingencies with a concrete hybrid supervision strategy, and §11 records the cross-check of the updated plan 23.

## 0. Hard Timeline Context

- Submission deadline: 2026-07-04 23:59 — roughly two working days from now.
- Official case data is already published: `https://disk.yandex.ru/d/Fo5eIM984glHaA`, linked from the saved official page `docs/official/Скажи мне кто твой шлиф.md`; a local download is in progress and folder-level metadata is recorded in `docs/notes/2026-07-02-domain-datasets-search.md`.
- 2026-07-03 schedule (captains chat, `docs/notes/2026-07-02-telegram-shlif-captains-chat.md`): accesses from 08:00, opening/task presentation 17:00, Q&A 17:45 — use the Q&A to confirm edge-case semantics (fraction denominators, tie rule, video length limit), not to wait for data.
- Organizer confirmations from the same chat: the Yandex dataset is final/real/unaugmented and OM-only; no XRD will be provided or evaluated; the annotation format is blue lines drawn on the photos; each team gets one T4-class GPU and external infrastructure is allowed — so the inference path must fit a T4 memory budget.
- Consequence: data inventory starts immediately (2026-07-02 evening) in parallel with the data-independent plumbing; 2026-07-03 evening and 2026-07-04 are reserved for calibration, validation, and packaging — not core plumbing.

## 1. What the TZ Actually Requires (Compressed Contract)

Requirements source: `docs/official/Скажи мне кто твой шлиф.md` (converted 2026-07-02 from the official task-3 page); it matches the user-supplied TZ text.

Input: panoramic OM image of a polished section (TIFF/PNG/JPEG, up to ~10000 x 10000 px; the problem statement separately mentions panoramas up to several gigapixels).

Output per image:

1. Pixel mask with exactly three highlighted classes over the source image:
   - green = ordinary sulfide intergrowths (`обычные срастания`): large, isolated sulfides with minimal replacement by gray/dark phase;
   - red = fine sulfide intergrowths (`тонкие срастания`): sulfides significantly replaced by non-ore phase;
   - blue = talc (`тальк`): dark dispersed phase in the non-ore matrix (annotated with a colored line in training data).
2. Metrics table: total sulfide fraction, fraction per intergrowth type, talc fraction.
3. Deterministic text conclusion by the expert rule:
   - talc fraction `> 10%` -> `оталькованная руда`;
   - else ordinary predominates -> `рядовая руда`;
   - else fine predominates -> `труднообогатимая руда`.
   - Sample wording: `«Руда классифицирована как оталькованная: содержание талька — 14%, преобладание тонких срастаний — 62%»`.

Hard quality/performance bars: talc fraction error <= ±3% vs expert markup; intergrowth-type classification F1 >= 90%; 10000 x 10000 px in <= 5 minutes on a CPU/GPU workstation; robust to lighting/contrast variation and polishing artifacts; local deployment.

Submission by 2026-07-04 23:59: VCS link, cloud-disk archive of the code, video demo, presentation, optional deployed instance link.

## 2. Verified Current-State Facts

Checked directly in code on 2026-07-02 (not from memory). Status note on
2026-07-05: the bullets below describe the old broad QC codebase at the time of
the gap analysis; the v2 implementation now lives under `src/ore_classifier/`,
`scripts/`, and `apps/ore_pipeline_web.py`.

- At the time, the OM ontology was still the LumenStone S1 mineral label set (`bg`, `ccp`, `gl`, `br`, `py`, `sh`, `tnt`) and talc/intergrowth/ore-class concepts were not implemented in code. In v2, the official ore classes are represented by `apps/ore_pipeline_web.py` (`CLASS_COLORS`, `CLASS_LABELS_RU`, `REPORT_CLASS_SPECS`) and the component analysis modules.
- At the time, the old broad label set was spread across many pipeline/report/test files. In v2, the official task path is concentrated in `src/ore_classifier/component_analysis.py`, `src/ore_classifier/component_reports.py`, `src/ore_classifier/resident_pipeline.py`, and `apps/ore_pipeline_web.py`.
- Tiled inference with overlap is now represented by `scripts/infer_binary_sulfide.py`, `scripts/run_ore_pipeline.py`, and the resident path in `src/ore_classifier/resident_pipeline.py`; full-resolution/panorama behavior is benchmarked separately in `docs/benchmarks/`.
- The full-resolution `logits_sum` buffer is `num_classes x H x W float32`; at 10000 x 10000 x 7 that is ~2.8 GB on the inference device. Needs a memory strategy (band-wise accumulation, float16, or uint8 vote counting) before the 10k benchmark can pass.
- YOLO11 tiled evaluation with confidence stitching is proven (full/void mIoU `0.8741`/`0.8881` at 1280/320 on S1 proxy; `docs/benchmarks/46_yolo_om_segmentation_tiled_eval.md`) — a second deployable model family.
- The current deterministic ore classifier is `src/ore_classifier/component_analysis.py` (`classify_ore_type`, `ComponentRuleConfig`), using the strict talc `> 10%` rule and ordinary/fine predominance.
- The current fraction/report path is `src/ore_classifier/component_reports.py` plus `apps/ore_pipeline_web.py`, which writes `reports/metrics.csv`, `reports/ore_summary.json`, PDF, overlays, GIS exports, and artifact bundles.
- The vendored `experiments/petroscope/petroscope/analysis/geometry.py` already converts masks to polygons and saves GeoJSON — the GIS-export wish is mostly a wiring task.
- Preprocessing normalization is essentially absent as a pipeline stage: the only CLAHE usage is in `prepare_manual_mvp_sample.py` (sample prep), not in the inference/report path. The TZ lists illumination normalization/denoising/contrast correction as functional requirements.
- Confidence heatmap ("карта уверенности") is absent; logits are already computed in the tiled path, so per-pixel max-softmax export is cheap.
- The result-page interactive preview, per-class toggles, exclusion muting, zoom/pan, history, series, and API surface are now in `apps/ore_pipeline_web.py`.
- Physical scale: SEM footer OCR scale parsing and manual metadata fields exist, but the report table does not compute absolute areas (µm²); fractions only. TZ requires area computation "с учётом масштаба изображения".
- Batch mode (`scripts/run_official_batch.py`, `scripts/run_resident_batch.py`, UI Series page, `summary.csv`), PDF reports, CSV exports, run-parameter logging, Docker/local deploy, expert correction, exclusion masks, robustness/evaluation harness, RU/EN manuals, and the RU pitch deck cover most of the TZ "Визуализация и экспорт", "Пакетная обработка", "Интерфейс", "Безопасность", and expert-review wishes.
- Git remote exists: `git@github.com:s1mb1o/microstructure-qc-assistant.git` (single `main` branch). Repo visibility/jury access must be verified before submission.
- Organizer constraints from `docs/plans/22_official-qna-metric-license-update.md`: open-license pretrained models only, no automatic checking, recommended metrics IoU/Hausdorff (segmentation) + F1/AUC (classification). License provenance for every shipped checkpoint is a submission-blocking item (note: YOLO11/Ultralytics is AGPL-3.0 — open but copyleft; SegFormer/ResUNet/SAM2 checkpoint licenses must be verified and recorded individually rather than assumed).

## 3. Requirement Traceability Matrix

Status legend: `DONE` (exists, may need relabeling), `PART` (infrastructure exists, task-specific piece missing), `MISS` (absent), `UNVER` (cannot be verified until official data).

### Образ решения

| # | TZ requirement | Status | Evidence / gap |
| --- | --- | --- | --- |
| 1 | Segment + classify sulfide intergrowths (ordinary vs fine) | MISS | No model, no labels, no classes in code; S1 minerals are a proxy only. Two-stage fallback possible (see §4). |
| 2 | Detect + quantify talc share | MISS | No talc concept anywhere in code; training-data "colored line" annotation needs a conversion tool. |
| 3 | Expert rule talc>10% / predominance | DONE | Implemented in `src/ore_classifier/component_analysis.py` (`classify_ore_type`, strict `> 10%`). |
| 4a | Color mask green/red/blue over source | PART | Overlay/legend/interactive preview exist; need exact 3-class palette + legend wording. |
| 4b | Table with fractions | PART | `phase_fractions` table + CSV exist; need task-specific rows (total sulfides, per-type shares, talc). |
| 4c | Text conclusion | PART | Report text + optional LLM narrator exist; need the deterministic RU sentence exactly per TZ (never LLM-dependent). |

### Функциональные требования

| TZ requirement | Status | Evidence / gap |
| --- | --- | --- |
| TIFF/PNG/JPEG high-res input | DONE | `input_loaders.py`, upload UI, TIFF->PNG browser previews. |
| Preprocessing: illumination normalization, denoise, contrast, scaling | PART | Only ad-hoc CLAHE in sample prep; need an explicit, logged, optional preprocessing stage in the inference path. |
| Pixel segmentation preserving morphology | PART | Tiling+stitching exists but smokes use 768 px resize; full-res tiled mode must become the panorama default and be profiled. |
| Sulfide (bright) vs matrix (dark) separation | PART | Heuristic bright-phase and S1 mineral models exist as proxies; task-specific sulfide union needed. |
| Intergrowth classification by replacement degree | MISS | Core new work (§4). |
| Talc as dark dispersed phase in matrix | MISS | Core new work (§4). |
| Area/percent computation with image scale | PART | Fractions with exclusion-adjusted denominator: done. Absolute µm² areas from metadata scale: missing. |
| Mask overlay + interactive zoomed view | PART | Interactive preview with per-class toggles: done. Zoom/pan: missing (small JS task). |
| Metrics table in UI + CSV export | DONE | Result page table + `reports/metrics.csv` with official v2 rows. |
| Text conclusion + PDF report export | DONE | RU/EN PDF generation from result pages/API; needs task wording block. |
| Optional Streamlit/Gradio web UI | DONE | `apps/ore_pipeline_web.py` exceeds the ask (jobs, history, API/OpenAPI, status, RU/EN). |
| Batch processing without user | DONE | `scripts/run_official_batch.py`, `scripts/run_resident_batch.py`, UI Series page, `summary.csv`, `batch_summary.json`, `ore_class` columns. |
| Analysis-parameter logging for reproducibility | DONE | Run summaries, inference metadata, command transcripts, manifests. |

### Нефункциональные требования

| TZ requirement | Status | Evidence / gap |
| --- | --- | --- |
| 10000x10000 px <= 5 min on CPU/GPU workstation | UNVER | Tiling exists; no 10k benchmark exists; memory strategy for full-res logits needed. Must measure, not assume. |
| Robust to uneven lighting, scratches, contamination | PART | Strong story already: exclusion masks, defect candidates, robustness service with perturbation verdicts. Needs one ore-classifier-specific scorecard. |
| Talc share error <= ±3% | UNVER | Metric not implemented; buildable now, verifiable only with expert markup. |
| Intergrowth classification F1 >= 90% | UNVER | F1 metric not implemented (per plan 22 it is planned); target achievable only after seeing data. |
| Interface intuitive for geologists; manual mask correction | DONE | RU/EN UI; expert correction modal + SAM2 assist + correction-exclusion rerun — a genuine differentiator. |
| Local deployment for confidential data | DONE | Local run + Dockerfile + deploy smoke; no cloud dependency. |

### Дополнительные пожелания

| Wish | Status | Evidence / gap |
| --- | --- | --- |
| Expert-check mode feeding retraining set | PART | Correction patches -> training manifest + CVAT/Label Studio exports exist; needs 3-class alignment. Live fine-tuning stays roadmap. |
| Metadata (scale, acquisition, deposit type) | DONE | Manual metadata + sidecars + SEM footer OCR. |
| Confidence heatmap for disputed areas | MISS | Cheap from existing logits in tiled path. |
| GIS export (Shapefile, GeoJSON) | MISS | GeoJSON: reuse `petroscope/analysis/geometry.py`. Shapefile: skip unless a library is already vendored. |
| Documentation with typical/borderline case walkthroughs | PART | RU/EN manuals + quick starts exist; add a classification-cases section. |

### Формат сдачи (2026-07-04 23:59)

| Item | Status | Action |
| --- | --- | --- |
| VCS link | PART | GitHub repo exists; verify visibility (public or jury access) and clean README entry point. |
| Cloud-disk archive of source | MISS | Build zip, upload to Yandex/Google disk. |
| Video demo | MISS | Nothing recorded; needs a script + recording slot on 2026-07-04 (target ~3 min through the upload UI happy path). |
| Presentation | PART | `docs/presentations/hackathon-project-pitch.ru.pdf` exists but leads with the broad multimodal QC story; must be reframed to the OM ore classifier with the three-color mask and the expert rule as the lead. |
| Deployed solution (optional) | MISS | Options: ship Docker one-liner instructions only, or additionally deploy the upload UI to an owned VPS. |

## 4. Data Reality and Supervision Strategy (updated after package metadata)

Yandex Disk folder metadata (recorded without download in `docs/notes/2026-07-02-domain-datasets-search.md`) resolves most of the earlier format uncertainty — the TZ "colored line" wording and the Telegram "partial class labels without segmentation" answer turn out to be both true, for different parts of the package:

- `Панорамы`: 14 unlabeled JPG panoramas, ~1.31 GiB total (~96 MB per JPG — decoded size may far exceed 10000 x 10000 px) — the demo/inference targets.
- `Фото руд по сортам. ч1`: image-level class folders — 42 `Оталькованные руды`, 68 `Рядовые руды`, 68 `Труднообогатимые руды`.
- `Фото руд по сортам. ч1/Оталькованные руды/Области оталькования`: 42 JPGs — the talc annotations; organizers confirmed the annotation format is blue lines drawn on the photos (captains chat), so first inspection must classify whether the lines sit on the original pixels (then an extraction/cleanup step is required so models do not train on the markup) or on paired copies.
- `Фото руд по сортам. ч2`: 87 `оталькованные`, 497 `рядовые`, 418 `тонкие` — roughly a thousand more image-level labels.
- No pixel masks for intergrowth types are visible at metadata level.

This selects a hybrid supervision strategy (replaces the earlier A/B/C contingency branches):

- Talc — region-supervised: convert the 42 `Области оталькования` annotations into filled talc masks (color threshold -> contour close -> fill, visual QA overlay for non-closed lines); train/calibrate a talc segmenter and measure the ±3% talc-fraction error on held-out annotated pairs.
- Intergrowths — weakly supervised: per-connected-component classifier on top of binary sulfide segmentation (replacement-degree morphology: dark-inclusion ratio inside the grain footprint, grain size, boundary complexity), thresholds calibrated against the ~1050 ordinary/fine image-level labels; the measurable headline metric is image-level intergrowth-type / ore-class F1, which is also the most defensible reading of the TZ "классификация типа срастаний ... F1". Fully deterministic and explainable — directly answers the TZ "предметная точность" requirement.
- Optional upgrade (only if all P0 is green): self-training — pseudo-label the class-folder photos with the calibrated two-stage pipeline and fine-tune a dense 4-class model (SegFormer/ResUNet first, YOLO11-seg tiled as the packaging-friendly alternative); this is the only realistic mask-supervised path for intergrowths given the package contents.
- Split discipline: split by sample stem (and by panorama for panorama-derived crops), never randomly by file, to avoid near-duplicate leakage; ч1 vs ч2 look like different acquisition batches — keep one as a holdout candidate.

Shared work unchanged by data format (build first): ontology module, deterministic rule, report/UI/PDF/CSV rewiring, full-res tiled inference + 10k benchmark + confidence map, metrics module, batch `ore_class`, judge artifact command, submission logistics.

Disaster fallback (package unusable): the same two-stage pipeline on adapted LumenStone proxy fixtures with the calibration procedure documented — the old Branch C, now only insurance.

## 5. P0 Workstreams

### W1. Ore ontology module

Current v2 equivalent: the class registry `{background, ordinary_intergrowth,
fine_intergrowth, talc}` is encoded in `apps/ore_pipeline_web.py`
(`CLASS_COLORS`, `CLASS_LABELS_RU`, `REPORT_CLASS_SPECS`) and exported through
overlays, reports, GIS files, and UI legends.

Acceptance: report/overlay/CSV/PDF/UI render the three classes with TZ colors; unknown labels fail loudly; S1 mode still passes its smokes.

### W2. Deterministic ore classifier

Current v2 equivalent: `src/ore_classifier/component_analysis.py` consumes
sulfide/talc masks and component statistics, emits `total_sulfide_fraction`,
`ordinary_intergrowth_fraction`, `fine_intergrowth_fraction`, `talc_fraction`,
predominance shares among sulfides, `ore_class` in `{talcose_ore,
ordinary_ore, hard_to_process_ore}`, and the RU conclusion used by UI/PDF/CSV.

Semantics to fix in code and tests (and to confirm at the Q&A):

- talc fraction denominator = analyzed (non-excluded) pixels;
- predominance = ordinary vs fine share of sulfide area only;
- `> 10%` is strict; at exactly 10% the ore is not talcose;
- tie at 50/50 predominance defaults to `рядовая руда` plus an expert-review warning.

Acceptance: boundary unit tests (10% exact, tie, zero sulfides, talc-only); conclusion never depends on the LLM narrator; exposed in result JSON, UI, CSV, PDF.

### W3. Segmentation path per supervision strategy (§4)

Build the two-stage interpretable core first (guaranteed shippable): binary sulfide segmentation -> per-component intergrowth typing -> talc detector. Add the colored-line -> filled-mask converter for `Области оталькования` with a visual QA overlay, then the talc calibration/eval loop on held-out annotated pairs and the intergrowth threshold calibration against class-folder labels. Keep the 4-class SegFormer/YOLO fine-tune recipe ready for the optional pseudo-label upgrade, wired to gx10.

Acceptance: one command produces a 4-class mask from a raw OM image; talc-fraction error and intergrowth/ore-class F1 measured on leakage-safe splits of the official package.

### W4. Full-resolution panorama inference + 10k benchmark + confidence map

Current v2 path: `scripts/infer_binary_sulfide.py`, `scripts/run_ore_pipeline.py`,
and `src/ore_classifier/resident_pipeline.py` provide tiled inference and
resident model loading; panorama performance evidence is recorded in
`docs/benchmarks/07_panorama_performance_20260705.md` and
`docs/benchmarks/08_largest_panorama_16jpg_zelda_20260705.md`.

Acceptance: 10k smoke passes and prints measured runtime; failure on low memory is graceful with a documented downscale fallback; heatmap artifact appears in result page/report links.

### W5. Report / UI / PDF / batch rewiring

Task-specific result block: three-color legend, fraction table rows (total sulfides / ordinary / fine / talc), ore-class conclusion in RU (mirrored in EN report), confidence heatmap link; `summary.csv` and batch index gain `ore_class` + `talc_fraction` columns so a batch reads as a technological map input.

Acceptance: upload UI happy path from a raw OM panorama to a PDF containing mask, table, and the TZ-style sentence, with no S1 mineral wording anywhere in the ore-classifier mode.

### W6. Task metrics module

`evaluate_ore_classification.py`: talc-fraction absolute error, intergrowth F1 (per-pixel where masks exist, per-component and per-image ore-class F1 otherwise), per-class IoU, optional HD95 (plan 22 alignment); emits `metrics.json`/`.csv`/Markdown.

Acceptance: runs on a held-out split of whatever official data exists; on proxy fixtures produces the same artifact shape labeled `proxy`.

### W7. Final judge artifact + license manifest

One command: raw image folder -> per-image masks/overlays/heatmaps/JSON + batch CSV + localized PDF + metrics + model/data license manifest + README with exact commands. Refresh `submissions/runner_variants/` selection to the 3-class contract once the official format is known.

Acceptance: the artifact used in the presentation is produced by this exact command from raw inputs; every shipped checkpoint has recorded license/provenance; no closed-license dependency in the final package.

### W8. Submission logistics

Repo access check + README lead paragraph rewrite; source zip to cloud disk; ~3-minute video demo script (upload -> progress -> mask/table/conclusion -> correction -> batch CSV -> PDF) and recording; pitch deck reframe: lead slide becomes `Интерпретируемый OM-классификатор руды: срастания, тальк, автоматический тип руды`, with the existing multimodal QC platform demoted to one "platform readiness" backup slide; optional deployed instance decision.

Acceptance: all four submission links resolvable by an outsider before 2026-07-04 20:00 (self-imposed buffer).

## 6. P1 (Only After All P0 Green)

1. Zoom/pan on the result canvas (TZ asks for zoomed interactive viewing).
2. Explicit preprocessing presets (illumination normalization/CLAHE, denoise, contrast) as logged optional stages with before/after thumbnails in the report.
3. GeoJSON export of class polygons via `petroscope/analysis/geometry.py` (+ pixel->µm affine when scale metadata exists); Shapefile only if a ready library is acceptable.
4. Absolute µm² areas in the fraction table when scale metadata is present.
5. Ore-classifier-specific robustness scorecard via the existing robustness service (brightness/contrast/blur/scratch perturbations -> ore-class stability verdict).
6. Correction-editor phase list aligned to the three official classes + retraining manifest export.
7. Documentation section with typical/borderline case walkthroughs (talc ~10%, near-tie predominance, artifact-heavy panoramas).

## 7. Explicit Freezes Until Submission

- No SEM, XRD, synthetic-SEM, defect-model, Rietveld, or LLM-narrator work; they are backup slides, not build targets.
- No new model families or ablations beyond the branch decision in §4.
- No refactor of the 13 S1-hardcoded files beyond the injection points W1 needs.
- No live fine-tuning UI.

## 8. Day-by-Day Schedule

2026-07-02 evening:

- finish the Yandex package download; build the official-data manifest (folder class, stem, dimensions, hashes) and visually inspect `Области оталькования` to classify the talc annotation format — this decides the converter design;
- W1 ontology module + W2 deterministic classifier with boundary tests (pure code, no data dependency);
- start W4: `--no-resize` tiled mode + memory strategy; decode one real panorama and get a first runtime/memory number.

2026-07-03 day (before 17:30 Q&A):

- finish W4 benchmark + confidence heatmap on a real panorama;
- W5 report/UI/PDF/batch rewiring;
- W3 two-stage pipeline runnable end-to-end on official photos; colored-line -> mask converter for talc;
- W6 metrics module; first talc-error and intergrowth-F1 numbers on leakage-safe splits;
- Q&A questions: talc-fraction denominator, predominance definition/tie rule, whether the 14 panoramas are the judged inputs, video length limit, deployed-instance value.

2026-07-03 evening (after Q&A):

- calibration iteration on the full class folders; optionally launch the pseudo-label dense fine-tune on gx10 overnight;
- start pitch-deck reframe (W8) in parallel with training.

2026-07-04:

- morning: validate (W6 metrics on held-out split), fix worst failure modes, finalize judge artifact (W7);
- afternoon: record video, finish deck, build zip, upload everything, verify all links (W8);
- hard buffer: all links live by 20:00, submission before 23:59.

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| `Области оталькования` turns out to be crops/photo variants without extractable line geometry | Talc supervision weakens; ±3% claim unverifiable | Inspect first thing on 2026-07-02 evening; fallback: hand-trace a small validation subset in the existing correction editor to measure talc error honestly. |
| Panoramas decode far beyond 10000 x 10000 (14 JPGs averaging ~96 MB) | Decode/memory failures before inference even starts | Deliberately raise the Pillow `MAX_IMAGE_PIXELS` guard, decode once and process band-wise; if a full decode does not fit, switch to a streamed/banded decoding path and document the supported limit. |
| F1 >= 90% / ±3% talc not reachable in one day on real data | Misses stated bars | Report honest measured metrics + calibration procedure; per-component and per-image F1 are more attainable than per-pixel and still match the TZ wording ("классификация типа срастаний"). |
| 10k x 10k memory/runtime blows the 5-min bar on CPU | Non-functional requirement fails | Measure early (2026-07-02); band-wise accumulation, fp16, larger tiles on GPU/MPS; documented downscale fallback as last resort. |
| Ontology swap breaks existing smokes | Regression churn during the busiest days | Additive label-map injection; S1 mode untouched; run the existing test suite after W1/W5. |
| License provenance of a shipped checkpoint unclear (e.g., Petroscope ResUNet weights) | Submission compliance risk per organizer answer | License manifest in W7 is blocking; prefer checkpoints with verified open licenses; drop or retrain anything unclear. |
| Video/presentation squeezed out by coding | Weak jury-facing output despite strong code | W8 has reserved slots on 2026-07-04 afternoon; deck reframe starts 2026-07-03 evening regardless of model state. |
| Repo private / links broken at submission | Formal disqualification risk | Link check with an outside account is an explicit W8 acceptance item. |

## 10. Decisions Needed From the User

1. Hybrid supervision per §4 (region-supervised talc + weakly supervised per-component intergrowth typing, dense fine-tune only as pseudo-label upgrade) confirmed as the default — agree?
2. Deployed instance: Docker instructions only, or also a live VPS deployment of the upload UI?
3. Video demo language (RU assumed) and target length (~3 min assumed; TZ leaves the limit blank).
4. Should the S1 mineral mode remain visible in the UI as a "proxy demo" selector, or be hidden for the jury build?

## 11. Relation to Plan 23 (Codex) — Checked Against Its 2026-07-02 Update

Plan 23 was re-verified after Codex updated it to cite the saved official page and the case-data link. Checked line-by-line against `docs/official/Скажи мне кто твой шлиф.md`: its requirement extraction, deadline, quality bars, decision rule (tie resolved to ordinary ore via `>=` — same choice as W2 here), P0-0 data inventory, ontology adapter, deterministic classifier, tiled inference, judge artifact, submission-format coverage, and presentation reframe are all accurate and consistent with this plan. No factual errors found. Its earlier gaps (no data-inventory step, no submission-format row) are now closed by P0-0 and the expanded coverage table/P0-5.

Deltas this plan still adds on top of plan 23:

- deadline-anchored day-by-day schedule with reserved video/presentation slots on 2026-07-04;
- memory math for the full-resolution logits buffer (~2.8 GB at 10k x 10k x 7 float32) and the banded-accumulation requirement in W4;
- the gigapixel panorama decode risk (14 JPGs averaging ~96 MB) with its Pillow-guard/banded-decode mitigation;
- explicit classifier edge-case semantics (fraction denominators, strict `>10%`, tie warning) turned into unit tests and Q&A questions;
- zoom, µm² scale-aware areas, and preprocessing-visibility specifics from the TZ functional list;
- the hybrid supervision reading of the package metadata (§4): region-supervised talc via `Области оталькования`, weakly supervised intergrowth typing via ~1050 class-folder labels, image-level F1 as the headline classification metric.

Divergence that still stands: plan 23's P0-3 candidate-model order presumes mask-supervised fine-tuning as the primary intergrowth path. The package metadata shows image-level labels plus talc regions only, so this plan keeps the interpretable two-stage classifier as the default deliverable and treats dense fine-tuning as a pseudo-label upgrade. Plan 23's own P0-0 ("inspect annotation format first") hedges in the same direction, so the two plans converge in practice once the inventory confirms the metadata.
