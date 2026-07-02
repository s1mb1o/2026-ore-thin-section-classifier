# Official Q&A Metric and License Update Plan

Date: 2026-07-02

## Trigger

User-supplied organizer clarification:

- "Предобученные модели разрешены, если распространяются и используются по открытой лицензии."
- "Автоматической проверки решений не будет."
- "Основные метрики, на которые стоит обратить внимание при обучении модели - IoU и расстояние Хаусдорфа для сегментации, F1 и AUC для классификации."

Source status:

- Treat this as a current working requirement from the user.
- Archive the original organizer message or updated Telegram export when available.
- Until archived, keep the wording explicitly marked as user-supplied clarification rather than independently verified raw-source evidence.

## Strategic Update

The project should shift from a leaderboard-first assumption to a judge-facing reproducibility and validation package.

Keep:

- one-command local runs;
- clean model/data provenance;
- validation metrics with saved JSON/CSV/HTML/PDF artifacts;
- upload UI and report flow for jury review;
- runner templates as backup and reproducibility scaffolding.

Reduce priority:

- strict official-runner bridge work that only matters for automatic evaluation;
- packaging variants whose only purpose is unknown server-side scoring.

Increase priority:

- open-license audit for every pretrained model and external data source;
- segmentation metrics beyond mIoU, especially Hausdorff distance;
- explicit classification path with F1 and AUC;
- final demo package that a judge can run and inspect without a hidden leaderboard.

## P0 Changes

| Area | Change | Acceptance criteria |
| --- | --- | --- |
| Source tracking | Add a focused note for the 2026-07-02 clarification after raw source is available. | `ResearchLog.md`, `docs/session-sync.md`, and a `docs/notes/` source note agree on what is confirmed, what is user-supplied, and what remains unknown. |
| Model license audit | Build a manifest of pretrained models, checkpoints, model cards, source URLs, license names, and usage status. | `model_license_manifest.json` or Markdown equivalent lists SAM2, YOLO, SegFormer, Mask2Former, ResUNet/Petroscope, DINO/CLIP-style candidates if used, and any other shipped weights. No closed or unknown-license model is presented as allowed. |
| Data license audit | Extend the external dataset/reference inventory with license and allowed-use status. | Open proxy datasets, COD/USGS/opXRD/RRUFF/AMCSD references, and any official samples have explicit license/provenance rows. ICDD/PDF remains production-only unless licensed. |
| Segmentation metrics | Add Hausdorff distance next to existing IoU/mIoU metrics. | Shared metric output includes per-class IoU, mean IoU, Hausdorff distance in pixels, and preferably robust HD95 when masks exist. Missing masks produce `not_available`, not fake scores. |
| Classification metrics | Make classification evaluation first-class. | Evaluation output includes macro/weighted F1, per-class F1, ROC-AUC/PR-AUC where labels and class counts make AUC valid, and an explicit `not_available` reason otherwise. |
| Classification module | Promote microstructure/class-label output from secondary report field to a real pipeline module. | E2E summaries contain `microstructure_classification` or `sample_classification` with labels, confidence, model/rule source, and metric status. |
| Final rehearsal | Reframe the final rehearsal around a judge-visible package. | One command produces: run artifacts, validation metrics, license manifest, result report/PDF, demo instructions, and optional runner zip if still useful. |
| Presentation | Align the pitch with "no automatic check." | Deck/report distinguish validation metrics used during training from jury/demo evaluation. Do not imply there is a hidden leaderboard score. |

## P1 Changes

1. Update postprocessing selection so it can optimize or report both IoU and Hausdorff when masks are available.
2. Add class-threshold calibration for F1/AUC when official class labels are available.
3. Add a compact validation dashboard section: segmentation metrics, classification metrics, license status, and known limitations.
4. Add a "model/data provenance" page to the final report or demo README.
5. Keep defect/artifact outputs honest: if labels are classification-only, report defect candidates as review evidence unless official defect labels exist.

## Implementation Order

1. Archive the source clarification in `docs/notes/` once the raw message/export is available.
2. Create the model/data license manifest and mark unknowns before changing code.
3. Add metric primitives for Hausdorff/HD95 and classification F1/AUC with unit tests.
4. Wire new metrics into existing validation summaries and benchmark documents.
5. Promote classification output in the E2E summary/report path.
6. Add final rehearsal output that bundles metrics, report, provenance, and demo instructions.
7. Update presentation and README wording to match "no automatic check."

## Non-Goals

- Do not remove runner templates; keep them as backup and reproducibility tooling.
- Do not ship or advertise a pretrained model with unknown, closed, or incompatible license.
- Do not report AUC when the label structure makes it statistically invalid.
- Do not present segmentation metrics as official scoring if official segmentation labels are absent.
- Do not expand UI breadth before metric, provenance, and final rehearsal artifacts are coherent.

## Open Questions

- Which exact class labels are provided, and at what level: image, region, sample, modality, or thin section?
- Are segmentation masks ever expected as submitted artifacts, or only as explainability evidence?
- Are defect/artifact labels classification labels, segmentation masks, bounding boxes, or not provided?
- Is use of external open datasets allowed in addition to open-license pretrained models?
- What hardware/runtime constraints apply to the judge-visible demo, if any?

## Bottom Line

The highest-value change is not another broad feature. The solution now needs a tight validation-and-provenance package: open-license models, IoU plus Hausdorff for segmentation, F1 plus AUC for classification, and a reproducible demo/report path that the jury can inspect directly.
