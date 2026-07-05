# Official Metrics and Panorama Split Clarification

Date: 2026-07-03

Source: user-provided organizer clarification in the project thread.

## Clarification

Organizer guidance:

- Automated scoring will not be used for the hackathon solution check.
- The target metrics to pay attention to while training models are:
  - segmentation: IoU and Hausdorff distance;
  - classification: F1 and AUC.
- These are production-solution target metrics, not a request to overfit the provided dataset.
- Teams should not try to reach `100%` on the provided data.
- Panoramas may be used as a test set, but the recommended evaluation set is a balanced sample from several classes.
- In the provided dataset, panoramas are unannotated and unclassified images.

## Implications for This Project

Current `binary_sulfide_dataset_v0` metrics are useful for model selection, but incomplete:

- current benchmark has IoU and pixel accuracy;
- next benchmark should add Hausdorff distance or HD95 for segmentation masks;
- downstream ore/image classification should report F1 and AUC;
- validation and test splits should be balanced by class where class labels exist;
- panoramas should be used for demo, performance, and optional unlabelled stress testing unless a separate annotation/classification pass is created.

Do not present the current SegFormer-B0 IoU result as final production accuracy. Present it as a weak-label baseline for choosing a sulfide segmentation checkpoint.

## Evaluation Plan Update

For the next measurable iteration:

1. Keep `binary_sulfide_dataset_v0` as weak-label baseline only.
2. Extend evaluation code to emit:
   - IoU;
   - Hausdorff distance or HD95;
   - F1 for binary segmentation thresholded at the chosen operating point;
   - AUC from probability maps where available.
3. Build a balanced image-level validation table from official class folders for:
   - ordinary / row ore;
   - fine / difficult ore;
   - talcose ore.
4. Use panoramas separately:
   - tiled inference performance;
   - visual QA overlays;
   - confidence heatmaps;
   - stress testing on large unannotated images.
5. If panoramas are later annotated or assigned class labels, record that as a new dataset version and do not mix it silently with the current unlabelled panorama set.

## QA session #4: no talc-fraction GT + jury accepts the team's own definition

Source (verbatim, with timecodes): `docs/official/2026-07-04-qna-session-4-transcript.md`
(video SHA-256 `a1df6a549d189bd1ecf6b30a7fda96144bc27be2996a7d7b49eb5945ab5066b7`).

### (a) No expert talc-fraction ground truth exists
- **[29:44] our question:** «Можно ли сообщить хотя бы выборочно, сколько доля талька
  у тех изображений, что размечено синими линиями?»
- **[30:46–31:15] expert answer:** «у меня нет такой экспертизы в геологии, чтобы
  посмотреть на картинку… сказать, да, тут точно 16%. Результат лаборатории у меня
  нету… Если ничего дополнительно не предоставили организаторы, то скорее всего уже
  ничего… всё, что есть, вам показали, дополнительно ничего не можем уточнить.»

→ There is **no expert talc-fraction reference at all** (participants, jury, or
organizers). The "talc-fraction error ≤ ±3%" criterion is therefore **not gradeable**;
"not met" is an absence of any reference, not a solution deficiency. Report leak-free
proxies instead: OOF fraction error vs the reviewed masks (median ~5.5 pp) and the
grade-relevant "talc > 10%" decision agreement (90% on the 42; talcose F1 0.851 in the
pipeline). See `docs/notes/2026-07-05-consolidated-metrics.md` §5.

### (b) Jury ruling — grade against the TEAM'S OWN documented definition
- **[28:04 / 43:57–44:05]:** «жюри при оценке будет учитывать мнение команды на тему
  того, что такое тальк, оталькованная руда… тальк — это не то, что геолог думает, а
  то, что команда прописала, как она поняла, что это такое» — **valid only if the team
  states its definition explicitly** in the README / solution description.
- **[45:07–47:34]:** the same applies to intergrowth types (ordinary/fine) — write the
  operational definition/methodology ("предпосылки") the team used.
- **[46:21–46:33 / 46:53]:** **keep the 10% talc threshold**; you may adjust the
  *definition* of talc, not the threshold.
- **[01:00:41]:** organizers acknowledge the annotation is weak («по разметке поняли…
  ещё месяц назад»).

**Action for submission:** state our operational definitions explicitly in the
README/presentation — talc (what pixels/regions we count, denominator, >10% rule),
ordinary vs fine intergrowth (morphology criteria), and the sulfide basis. Per the
ruling, we are then graded against *our* stated definition, which converts the
"no GT" problem into a documented, defensible choice.
