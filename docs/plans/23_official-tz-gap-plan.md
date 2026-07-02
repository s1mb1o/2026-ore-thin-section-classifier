# Official TZ Gap Plan

Date: 2026-07-02

## Source

Local source of truth:

- `docs/official/Скажи мне кто твой шлиф.md`
- extracted from `docs/official/Скажи мне кто твой шлиф.html`
- official page: `https://nornickel-ai-hackathon.ru/task-3`
- conversion date recorded in the Markdown: 2026-07-02

Submission deadline in the official page: 2026-07-04 23:59.

Case data URL in the official page: `https://disk.yandex.ru/d/Fo5eIM984glHaA`.

Official contacts and documents are also preserved in the Markdown: `ask@pgenesis.ru`, Telegram chat, hackathon rules, and privacy policy links.

The task is narrower than the earlier broad SEM/OM/XRD wording. The primary product must now be an end-to-end system for automatic ore classification from panoramic OM images of polished sections.

## Task Core

Required output:

- pixel-level mask over the source OM image;
- green class: ordinary sulfide intergrowths;
- red class: fine sulfide intergrowths;
- blue class: talc;
- table with total sulfides, ordinary intergrowth fraction, fine intergrowth fraction, talc fraction;
- text conclusion using the explicit geological rule:
  - talc fraction `> 10%` means talcose ore;
  - otherwise, ordinary intergrowth predominance means ordinary ore;
  - otherwise, fine intergrowth predominance means hard-to-process ore.

Target quality and constraints:

- talc fraction error within `+-3%` against expert annotation;
- intergrowth-type classification at least `90%` F1-score;
- images up to `10000 x 10000` px in less than `5` minutes on CPU/GPU workstation;
- local deployment for confidential geology data;
- batch processing, reproducible logs, CSV/PDF exports;
- optional expert correction, metadata, confidence heatmap, and GIS export.

## Current Coverage

| TZ requirement | What is ready | Gap |
| --- | --- | --- |
| TIFF/PNG/JPEG input and local processing | Upload UI/API, CLI input loaders, Docker/local deployment, PDF/CSV/JSON artifacts are already implemented. | Need final demo path trimmed to OM-only ore-classification flow. |
| Pixel segmentation and overlay | OM segmentation infrastructure, overlays, interactive browser preview, correction editor, and strong proxy OM benchmarks exist. | Current ontology is phase/mineral-style S1 labels, not `ordinary_intergrowth`, `fine_intergrowth`, `talc`. |
| Quantitative areas/fractions | Phase-fraction report, artifact/exclusion denominator handling, CSV exports, and PDF reports are implemented. | Need task-specific fraction names and talc `>10%` decision logic. |
| Text conclusion | Reports and optional narration exist. | Need deterministic geological conclusion, not generic narrative. |
| Expert correction / active learning | Correction modal, region edits, exclusions, annotation exports, and rerun with correction exclusions exist. | Need correction actions aligned to the three official classes and training manifest for this dataset. |
| Batch processing and logging | E2E batch mode, summary CSV/JSON, command transcripts, smoke logs exist. | Need final batch command for official OM images and official-style output folder. |
| Robustness to lighting/artifacts | Robustness service, perturbation checks, artifact/exclusion masks, OM artifact proxy datasets exist. | Need a small judge-visible robustness scorecard specifically for this OM classifier. |
| High-res panorama handling | YOLO eval-time tiled inference and other model plumbing exist. | Need one final tiled inference path profiled on `10000 x 10000` or a realistic generated panorama. |
| Metrics | IoU/mIoU and pixel accuracy exist; F1/AUC/Hausdorff are planned in `docs/plans/22_official-qna-metric-license-update.md`. | Need task metrics: talc fraction error, intergrowth F1, per-class IoU/HD95 where masks exist. |
| Official case data | Official page points to Yandex Disk case data. | Need download/access check, manifest, annotation-format inspection, license/provenance note, and split strategy. |
| SEM/XRD/synthetic data | Broad SEM/XRD pipeline, synthetic SEM path, XRD support, references, and reports exist. | For this TZ they should be secondary evidence/backup, not the final P0 storyline unless official data requires them. |
| Submission format | VCS repo, docs, PDF deck, Docker/local portal, final demo package, and source archive tooling are mostly available. | Need final VCS access check, cloud archive upload, video demo, revised presentation, and optional deployed-solution link. |
| Presentation, docs, video-ready UI | Russian pitch deck, quick starts, manuals, final demo package, and upload UI are implemented. | Need revise pitch/video around `OM ore classifier`, the three-color mask, and the explicit ore-type rule. |

## Main Risk

The repository is rich enough, but it currently answers a broader QC-assistant problem. The winning risk for this TZ is ontology mismatch:

```text
Current core: phase/mineral segmentation and multimodal QC.
Required core: OM ore classification by ordinary intergrowth / fine intergrowth / talc.
```

If we do not adapt the label map, metrics, report wording, and demo, the solution will look impressive but not task-specific.

## P0 Plan

### P0-0. Inventory Official Source and Case Data

Use `docs/official/Скажи мне кто твой шлиф.md` as the requirements source and the Yandex Disk URL from that file as the case-data source.

Acceptance criteria:

- official data archive is downloaded or access-blocked status is documented;
- data manifest records file names, sizes, hashes, formats, and annotation types;
- annotation format is classified as pixel masks, colored-line markup, region labels, image/sample labels, or unknown;
- all requirements, data, and contact/document links are traceable back to the local official Markdown.

### P0-1. Add Official Ontology Adapter

Create a task-specific class registry:

| id | label | RU label | color |
| ---: | --- | --- | --- |
| 0 | background_matrix | Матрица / фон | transparent or gray |
| 1 | ordinary_intergrowth | Обычные срастания | green |
| 2 | fine_intergrowth | Тонкие срастания | red |
| 3 | talc | Тальк | blue |

Acceptance criteria:

- masks, overlays, legends, CSV, JSON, and PDF can render these labels;
- official masks, colored lines, or sample annotations can be converted into this internal ontology;
- unknown labels are rejected with a clear error.

### P0-2. Implement Deterministic Ore Classifier

Add a module that consumes predicted mask fractions and emits:

- `talc_fraction`;
- `ordinary_intergrowth_fraction`;
- `fine_intergrowth_fraction`;
- `total_sulfide_fraction`;
- `ore_class`: `talcose_ore`, `ordinary_ore`, or `hard_to_process_ore`;
- Russian conclusion string matching the task statement.

Decision rule:

```text
if talc_fraction > 0.10:
    ore_class = talcose_ore
elif ordinary_intergrowth_fraction >= fine_intergrowth_fraction:
    ore_class = ordinary_ore
else:
    ore_class = hard_to_process_ore
```

Acceptance criteria:

- unit tests cover boundary cases around `10%` talc and equal ordinary/fine fractions;
- report and API expose both fractions and final class;
- conclusion is deterministic and does not depend on the optional LLM narrator.

### P0-3. Create Official Training/Validation Path

When official data is available:

- inspect annotation format first;
- convert colored talc lines to a reproducible mask representation if that is how the dataset is supplied;
- split train/validation with leakage checks by sample/panorama;
- fine-tune the strongest deployable model path.

Candidate model order:

1. ResUNet/Petroscope-style dense segmentation if checkpoint/runtime fits.
2. Mask2Former if packaging and runtime are acceptable.
3. YOLO11s tiled segmentation if runner simplicity matters more.
4. Heuristic fallback only for UI/demo, not as final quality claim.

Acceptance criteria:

- one command trains or fine-tunes;
- one command validates;
- output includes per-class IoU, F1, talc fraction error, and optional Hausdorff/HD95;
- all used pretrained weights have open-license provenance recorded.

### P0-4. Build High-Resolution Tiled Inference

Unify tiled inference for official OM panoramas:

- tile large images without resizing away morphology;
- stitch probability or class masks deterministically;
- record tile size, overlap, device, runtime, and memory;
- preserve output at source resolution or a documented scale.

Acceptance criteria:

- smoke on a synthetic or assembled `10000 x 10000` image;
- runtime target measured against the `5` minute requirement;
- failure mode is graceful if memory is insufficient.

### P0-5. Produce Final Judge Artifact

One command should produce:

- predictions and overlays for the official/demo samples;
- `metrics.json`, `metrics.csv`, and brief metric Markdown;
- per-sample ore-classification JSON;
- CSV summary for a batch;
- localized PDF report;
- model/data/license manifest;
- demo README with exact run command;
- source archive contents list for cloud upload;
- video-demo script checklist;
- presentation artifact path;
- optional deployable upload UI link or local-start instruction.

Acceptance criteria:

- the command starts from raw images and produces the same artifacts used in the presentation;
- no closed-license model/data dependency is advertised as final;
- XRD/SEM are clearly labeled optional support, not required for the core task answer.

## P1 Plan

1. Add confidence heatmaps from class probabilities or tile-vote agreement.
2. Add task-specific preprocessing presets: illumination normalization, contrast correction, denoising, and artifact masking.
3. Add robustness scorecard for brightness, contrast, blur, scratches, and polishing artifacts.
4. Add active-learning export: corrected official-class regions to CVAT/Label Studio and training manifest.
5. Add batch dashboard showing ore-class distribution and talc-heavy samples first.

## P2 Plan

1. GeoJSON export for mask polygons or connected components.
2. Shapefile export only if a GIS library is already acceptable in the final environment.
3. SEM/XRD support as optional review evidence after the OM task is solid.
4. A polished Streamlit/Gradio wrapper only if the existing upload UI is not acceptable for jury review.

## Presentation Reframe

Replace the broad lead:

```text
Multimodal OM/SEM/XRD QC assistant
```

with the task-specific lead:

```text
Интерпретируемый OM-классификатор руды: обычные срастания, тонкие срастания, тальк и автоматический вывод типа руды.
```

Keep a short backup slide:

- existing QC platform also supports SEM, XRD, correction workflow, PDF/API/Docker, and robustness checks;
- these modules show integration readiness but are not required to understand the main answer.

## Immediate Work Order

1. Download or document access status for the official Yandex Disk case data from `docs/official/Скажи мне кто твой шлиф.md`.
2. Inspect data/annotation format and write a manifest before choosing the model branch.
3. Implement ontology adapter and deterministic ore classifier.
4. Wire the three-class mask/fraction/classification into report JSON, CSV, PDF, and UI labels.
5. Add task metrics: talc fraction error and intergrowth F1 first; Hausdorff/HD95 next.
6. Add tiled high-resolution inference smoke and record runtime.
7. Create final judge artifact command and README.
8. Update pitch deck and video script around the official three-color mask and rule-based ore conclusion.

## Bottom Line

Do not build more breadth before the official OM classifier path is coherent. The fastest route to a credible submission is to reuse the existing mature UI/report/batch/correction infrastructure, but swap the center of gravity to the official three-class OM mask and deterministic ore-classification rule.
