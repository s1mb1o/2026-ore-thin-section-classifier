# Telegram Captains Chat for Official Shlif Task

Date: 2026-07-02

## Source

- Export: `/Users/ashmelev/Downloads/Telegram Desktop/ChatExport_2026-07-02/result.json`
- Chat: `Трек «Скажи мне, кто твой шлиф» | Норникель 2026`
- Telegram export type: `private_supergroup`
- Exported messages inspected: `38`
- Message date range in export: 2026-07-01 to 2026-07-02

This note records task-relevant organizer answers and operational implications. It does not copy the raw Telegram JSON into the repository.

## Confirmed Organizer Information

### Task Publication and Schedule

- The detailed task description was announced as published on 2026-07-02 at 22:00 MSK.
- Official task page linked in chat: https://nornickel-ai-hackathon.ru/task-3
- Schedule for 2026-07-03 MSK:
  - from 08:00: accesses are provided;
  - 17:00: opening ceremony and task presentation in Zoom;
  - 17:45: Q&A session in Zoom.

### Compute Access

- The accesses are for fully allocated GPU servers with T4/T4i.
- Later clarification: one T4 GPU per team.
- External infrastructure is allowed; one organizer answer explicitly says they do not restrict use of external infrastructure.

Implication:

- The runnable path should be T4-compatible and not assume large VRAM.
- The official page's performance line, `up to 10,000 x 10,000 px panorama in <=5 minutes on CPU/GPU workstation`, should be treated as a practical design target.
- External deployment can be used for demo/jury access, but model and data license provenance still need to follow official rules and prior organizer/license clarifications.

### Dataset Scope

Organizer answers in the chat establish:

- the Yandex Disk dataset linked from the task page is the final version;
- the data are real;
- no augmentation was performed by organizers;
- the Yandex Disk content is all the data that will be provided;
- the final data are only optical microscopy;
- there will be no XRD profiles;
- XRD is not evaluated because it is not in the dataset.

Implication:

- P0 must be OM-only.
- Do not spend final-score effort on XRD for this task.
- SEM/XRD modules can remain as product/support material, but the judged task path should not depend on them.
- Sim-to-real framing should be revised: the official task data are real, so robustness should focus on magnification, tiling, staining/color/illumination, and panorama-scale handling rather than synthetic-to-real claims.

### Annotation and Target Output

Organizer answer:

- annotation in the provided data is the blue lines on the photos;
- teams should focus on segmentation and classification.

Implication:

- The official package likely provides partial region/line supervision rather than clean dense masks.
- First data-inspection step after download should determine whether blue-line annotations are:
  - overlaid directly on image pixels;
  - paired annotation images;
  - masks or pseudo-masks;
  - class-specific line colors or only one annotation color.
- If blue lines are overlaid on image pixels, build an extraction/cleanup step before using them as supervision.
- Treat classification and segmentation as both required visible outputs in the judge-facing path.

## Updated P0 Interpretation

The current best interpretation after this chat:

1. Input: panoramic optical microscopy images and class-folder microscopy examples from the official Yandex package.
2. Supervision: real images with partial blue-line annotations plus folder-level class labels.
3. Required product behavior:
   - segment/mark relevant ore regions or class regions;
   - classify ore/image/sample into ordinary intergrowth, fine/difficult intergrowth, or talcose logic from the official task;
   - produce a judge-visible result on optical microscopy only.
4. Non-goals for final scoring:
   - XRD profile interpretation;
   - SEM inference;
   - broad multimodal fusion as a dependency.

## Impact on Existing Plans

- `docs/plans/23_official-tz-gap-plan.md` remains directionally correct: the main gap is the official OM ore-classifier path.
- `docs/plans/24_official-tz-gap-plan-claude.md` should be read with the 2026-07-02 chat as the newer source for data availability:
  - no XRD data;
  - real, final official dataset;
  - blue-line annotations are the confirmed annotation format;
  - one T4 per team and external infrastructure allowed.
- Older 2026-06-29 Telegram note still matters for format/UI assumptions, but its unanswered data questions are now partly answered here.

## Immediate Next Actions

1. After the Yandex download finishes, create a manifest for all files and verify folder-level classes, image dimensions, extensions, and magnification tokens.
2. Visually inspect blue-line annotation examples and decide whether they can be converted into masks or only weak region prompts.
3. Build a small `blue_line_annotation_extraction` experiment:
   - detect line color;
   - remove or ignore line pixels from training images;
   - optionally dilate lines into weak region masks;
   - keep original image provenance.
4. Create a T4-compatible inference smoke:
   - tiled processing for up to `10,000 x 10,000` px;
   - memory cap suitable for T4;
   - target runtime under 5 minutes per panorama.
5. Move XRD from P0 scoring path to optional presentation/support appendix.
