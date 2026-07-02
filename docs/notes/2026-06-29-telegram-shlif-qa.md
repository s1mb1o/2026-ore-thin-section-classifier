# 2026-06-29 Telegram Q&A: "Скажи мне, кто твой шлиф"

Date checked: 2026-06-29

Raw source:
- `docs/raw/2026-06-29_telegram/result.json`

Scope:
- Telegram export for `Хакатон Норникель AI Science Hack`.
- Focus only on messages related to the track `Скажи мне, кто твой шлиф` plus general organizer answers that affect participation, schedule, or judging.
- Message IDs below refer to the Telegram export.

## Confirmed From Organizer Answers

### 1. When detailed task questions will be answered

Answer:
- The track topic was opened on 2026-06-01, but organizers pointed participants to the public site until the opening meetup.
- Task questions are expected after the opening on 2026-07-03.

Evidence:
- `id=1036`

Implication:
- Keep current assumptions as provisional until the 2026-07-03 opening/Q&A. The project should be ready to pivot on data format, labels, metric, and submission packaging.

### 2. Solution format: Colab/API/site/UI

Question:
- Is a Colab with upload/output enough, or is a separate website required?

Answer:
- The form of the solution is the team's choice.
- Organizers recommend at least a minimal UI so jury members can inspect the solution conveniently.

Evidence:
- question `id=1831`
- answer `id=1839`

Implication:
- The existing upload UI and report flow are aligned with organizer preference. It should stay demo-friendly, but the core runner/model path must remain runnable without the UI.

### 3. Team size

Question:
- Can a one-person team participate?

Answer:
- No. Teams must have 2 to 5 participants.

Evidence:
- question `id=1848`
- answer `id=1851`

### 4. Registration, confirmation, and track changes

Answer:
- Registration and team applications close on 2026-06-30 at 23:59 MSK.
- Confirmations and invitations are expected on 2026-07-01 to 2026-07-02.
- Before registration closes, a captain may withdraw the submitted team application, change team composition or track, and submit again.
- After registration closes, the track cannot be changed.

Evidence:
- `id=1824`
- `id=2085`

### 5. Hackathon schedule

Answer:
- 2026-07-03 17:30: opening and expert Q&A; recording will be available.
- 2026-07-04: team checkpoints.
- 2026-07-05 08:00: solution upload deadline.
- 2026-07-05 08:00-12:00: expert review.
- 2026-07-05 13:00-16:00: finalist defenses.
- 2026-07-05 16:00-18:00: winner selection.
- 2026-07-05 18:00: closing and awards.

Evidence:
- `id=1965`
- `id=2085`

Implication:
- The real implementation window after task details is short: evening 2026-07-03 through early morning 2026-07-05. Prioritize a ready skeleton, one-command packaging, and a short pitch/report path.

### 6. Labels and annotation format for the "шлиф" track

Question:
- Will the data have full, partial, or no labels? If labels exist, in what format?

Answer:
- Organizers stated: partial class labels, no segmentation.

Evidence:
- question `id=1975`
- answer `id=1977`

Implication:
- Do not assume official pixel masks for training or scoring.
- Current segmentation work remains valuable as a product/demo differentiator, but the official task may emphasize classification, weak supervision, or expert-assessed outputs unless clarified.
- The strongest strategy is to keep segmentation as an explainable intermediate artifact while supporting class-level labels and weak-label evaluation.

### 7. "Idea of solution" field in the application

Question:
- What should be written in the `идея решения` field, and is it mandatory?

Answer:
- General input about the solution is enough.
- The field is not mandatory, but organizers prefer that teams fill it so the idea and direction are understandable before solution upload.

Evidence:
- question `id=1975`
- answer `id=1977`

### 8. Team selection criteria

Question:
- Will all applicants pass, or will there be selection?

Answer:
- Selection is only by mandatory criteria: age, citizenship, and residence.
- Age threshold is 18+.
- A separate question about per-track participant limits was answered as: no such restriction is provided.

Evidence:
- `id=1980`
- `id=1991`
- `id=2124`
- `id=2126`

## Unanswered In This Export

These questions appeared in the chat or follow directly from organizer answers, but no organizer answer was present in the 2026-06-29 export.

### 1. What exactly are the partial class labels?

Ask:
- Какие именно классы будут размечены?
- Разметка будет на уровне изображения, области/патча, образца, отдельного шлифа или модальности?
- Будет ли привязка классов к OM, SEM, XRD или общему образцу?
- Будут ли метки дефектов/артефактов отдельно от фаз/типов микроструктуры?

Why it matters:
- This decides whether the official model should be classification, weakly supervised segmentation, MIL, detection, or a multimodal sample-level classifier.

### 2. If segmentation is not labeled, how will segmentation be evaluated?

Ask:
- Сегментация является обязательной частью результата или только способом объяснения решения?
- Если сегментация не размечена, будет ли она оцениваться экспертно, по косвенной метрике, или не будет входить в формальную оценку?

Evidence for need:
- Organizer answer says there will be no segmentation labels (`id=1977`).

### 3. Detection labels and defect task boundary

Chat follow-up:
- A participant asked whether there will also be no labels for detection.
- No organizer answer was found.

Evidence:
- `id=1982`

Ask:
- Для детекции дефектов/артефактов будут ли bounding boxes, masks, point labels, image-level labels, or no labels?
- Нужно ли дефекты находить как формальный target или достаточно показывать review-only candidate regions?

### 4. Sim-to-real data split

Chat follow-up:
- A participant asked whether sim-to-real means both simulated and experimental data will be in the dataset.
- No organizer answer was found.

Evidence:
- `id=1986`

Ask:
- Будут ли в выдаче отдельно simulated и experimental данные?
- Будет ли train/test split проверять перенос с синтетики на эксперимент?
- Нужно ли явно отчитываться о robust/sim-to-real проверках?

### 5. Scope prioritization across subtasks

Chat question:
- The task contains several large subtasks; should teams go deep on a couple of them or cover all evenly?
- No organizer answer was found.

Evidence:
- `id=2041`

Ask:
- Что важнее в оценке: равномерное покрытие OM/SEM/XRD/дефектов/отчета или качественная реализация ограниченного ядра?
- Есть ли обязательные и дополнительные части задачи?

### 6. Official input bundle

Ask:
- Какие модальности будут в обязательном входе: OM, SEM, XRD, metadata?
- Будут ли OM и SEM paired по одному физическому шлифу?
- Будет ли XRD sample-level или привязанный к ROI?
- Какие форматы файлов: TIFF/PNG/JPG для изображений; CSV/TXT/XY/XLSX для XRD; JSON/CSV/YAML для metadata?

Why it matters:
- The current project can load many formats, but final runner/package logic must match the official folder contract exactly.

### 7. Metric, leaderboard, and judging contract

Ask:
- Будет ли leaderboard?
- Какая метрика будет основной?
- Будет ли автоматическая проверка архива с кодом или только экспертная проверка решения?
- Если есть автоматическая проверка: какой entrypoint, Docker/runtime, CPU/GPU/RAM/time limits, internet policy, archive size?

Why it matters:
- The project currently has both a product UI and official-runner templates; the official scoring path must be narrowed immediately after clarification.

### 8. Compute provided by organizers

Chat question:
- A participant asked whether organizers will provide compute or teams use their own.
- No organizer answer was found.

Evidence:
- `id=1893`

Ask:
- Будут ли предоставлены вычислительные мощности для обучения или инференса?
- Если да, какие GPU/CPU/RAM/time limits?
- Если нет, можно ли использовать внешние ресурсы for training before final submission?

### 9. External data and pretrained models

Ask:
- Разрешены ли внешние открытые датасеты, self-supervised pretraining, foundation models, SAM/SAM2, CLIP/DINO, YOLO, SegFormer, LLM/API?
- Есть ли ограничения по иностранным моделям, лицензиям, коммерческому использованию, доступности в РФ?
- Нужно ли прикладывать список внешних источников и лицензий?

Why it matters:
- The project uses open proxy datasets, pretrained models, optional SAM2, and open XRD references. Licensing and provenance should be explicit in the final package.

## Ready Questions For 2026-07-03 Q&A

1. Правильно ли мы понимаем, что по треку "Скажи мне, кто твой шлиф" официальная разметка будет только частично по классам и без pixel-level segmentation masks? На каком уровне будут эти классы: изображение, область, образец, шлиф или модальность?
2. Если сегментационные маски не выдаются, ожидается ли от участников сегментация как обязательный результат, как explainability artifact, или она не входит в формальную оценку?
3. Какие именно targets оцениваются: тип/класс шлифа, фазы, микроструктура, дефекты/артефакты, фазовые доли, качество отчета, или комбинация?
4. Какие входные модальности будут обязательными и какие optional: OM, SEM, XRD, metadata? Будут ли они paired по одному физическому образцу?
5. Что означает sim-to-real в этом треке: будут ли simulated и experimental данные, отдельный test split на перенос, или это ожидаемый раздел в отчете?
6. Какой официальный формат сдачи: архив с кодом и entrypoint, Git repo, notebook/Colab, UI/demo, API, отчет/презентация? Что обязательно, а что дает дополнительные баллы?
7. Будет ли автоматическая проверка и leaderboard? Если да, какая метрика, runner image, hardware/time limits, archive size, internet policy?
8. Как жюри будет сравнивать команды: важнее глубоко закрыть один-два ключевых блока или показать широкий end-to-end workflow по OM/SEM/XRD/дефектам/отчету?
9. Можно ли использовать внешние данные и pretrained/foundation models? Какие требования к лицензиям, доступности в РФ и коммерческому использованию?
10. Дадут ли организаторы compute для обучения/инференса или команды полностью используют свои ресурсы?

## Project Direction Update

Keep:
- Minimal UI and report flow, because organizer answer recommends UI for jury review.
- Modular OM/SEM/XRD/defect architecture, because the public task remains broad.
- Official-runner packaging templates, because automatic evaluation is still unknown.

Adjust:
- Treat official segmentation labels as absent until contradicted by the 2026-07-03 Q&A.
- Add/keep a weak-label or class-label path in the final runner strategy.
- Phrase segmentation, defect regions, and XRD support as explainable evidence unless official labels/metrics make them primary targets.

Immediate practical focus:
- Prepare a Q&A checklist for 2026-07-03.
- Keep one-command demo/report and one-command runner rehearsal ready.
- Be ready to pivot from dense segmentation scoring to class-level or weak-supervision scoring.
