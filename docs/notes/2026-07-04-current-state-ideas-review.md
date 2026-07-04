# Current-State Ideas Review

Дата: 2026-07-04

Контекст: этот обзор пересматривает идеи из `docs/notes/2026-07-03-research-mindstorm-improvements.ru.md`,
`docs/notes/2026-07-03-pipeline-improvement-proposals.md` и
`docs/ui/v2/TODO_CANDIDATES.md` с учетом фактического состояния проекта на
2026-07-04. Главный вывод: проект уже прошел стадию "придумать много
возможностей"; теперь ценность дают не новые широкие ветки, а аккуратная
сборка уже реализованных сильных частей в честный judged path.

## Что изменилось с прошлого mindstorm

- UI уже не является прототипом одного экрана: есть Workspace, History, Series,
  Settings, Status, API page, runtime test, PDF/CSV, `View files`, `Download ZIP`,
  runtime provenance и Docker/gx10 путь.
- Данные и метрики стали сильнее: есть deconflicted split на 345 изображений,
  аудит дубликатов/конфликтов, analyzed-area denominator, warnings/margins,
  official evaluation harness и robustness ladder.
- Сегментация сульфидов остается нашим ядром: SegFormer-B2 сейчас основной
  checkpoint, а resident inference снимает большой runtime overhead.
- Тальк стал модельной веткой, а не только HSV-кандидатом: SegFormer-B0 дает
  talc IoU `0.6410` и F1 `0.7812`, но fraction MAE пока `8.551` п.п., поэтому
  нельзя заявлять точность `+/-3` п.п.
- Классификация сорта стала двухветочной:
  - segmentation-first features дают около `0.7467` macro-F1 на 345 split;
  - Path A whole-image `efficientnet_b3` дает `0.9303` macro-F1 на held-out
    ordinary/fine 230 изображениях, но без talcose;
  - Path B grain-level HIL реализован, но с bootstrap labels пока `0.1895`
    macro-F1 и требует реальных human grain labels.

## Пересмотр идеи "killer feature"

Самая сильная история для жюри теперь такая:

```text
шлиф / панорама
-> сильная сульфидная сегментация
-> отдельная модель талька с честным caveat по fraction error
-> параллельная grade-ветка: Path A для ordinary/fine + talc-seg branch
-> объяснимые компоненты и provenance
-> карта спора источников + review queue для слабых мест
-> воспроизводимый ZIP/PDF/API/Docker evidence bundle
```

То есть "killer feature" не должен быть еще одним большим экраном или новой
архитектурой. Он должен показать, что система умеет:

1. честно отделять измеренные факты от слабых/pseudo labels;
2. объяснять, почему итоговый сорт получился именно таким;
3. показывать спорные зоны вместо маски с ложной уверенностью;
4. быстро воспроизводить результат и все артефакты.

## P0: идеи, которые стоит доводить сейчас

### 1. Decision lane: Path A + talc branch + pipeline provenance

Path A уже доказал, что ordinary/fine лучше решать supervised CNN-веткой, а не
только hand rules. Практический P0 теперь не "придумать классификатор", а
собрать демонстрационную decision lane:

- ordinary/fine: Path A `efficientnet_b3`, явно помеченный как 2-class branch;
- talcose: trained talc segmentation branch, пока с caveat по fraction MAE;
- segmentation/rules: оставить как объяснимую evidence layer, не как единственный
  headline grade score;
- UI/report: показывать метод решения и source/provenance, чтобы не смешивать
  rule, feature-CV и Path A числа.

### 2. Калибровка тальковой доли и worst-error review

Talc IoU уже хороший для демонстрации масок, но fraction MAE `8.551` п.п.
слишком велик для официального `+/-3` п.п. Следующий полезный тест:

- threshold/fraction calibration по 42 reviewed masks;
- отдельный разбор worst cases из `docs/benchmarks/02_talc_model_benchmark.md`;
- report wording: "segmentation improved over blue-line baseline" вместо
  "fraction is solved";
- UI/report guard: не показывать `+/-3 pp` claim до прохождения метрики.

### 3. Карта спора источников вместо обычной heatmap

Идея из mindstorm остается сильной, но теперь MVP должен быть узким:

- источники: model sulfide mask/probability, heuristic baseline, artifact/analyzed
  mask;
- позже: teacher/Petroscope/LumenStone source, TTA/ensemble instability;
- выход: `agreement_class_map.png`, `disagreement_score.png`,
  `source_manifest.json`, `review_candidates.csv/json`;
- UI: слой `сомнения / disagreement` с адаптивной легендой;
- framing: это не ground truth, а карта согласия/конфликта источников.

План уже есть: `docs/ui/v2/plans/36_ore-pipeline-source-disagreement-map.md`.
Реализовывать стоит через существующие `source_fusion.py` и `review_queue.py`,
а не создавать новую параллельную систему.

### 4. Evidence bundle как воспроизводимый run package

`Download ZIP`, `reports/runtime.json`, PDF и file browser уже есть, поэтому
идея "artifact bundle" перестала быть P0 с нуля. Теперь P0-смысл другой:

- убедиться, что ZIP содержит runtime provenance, rule config, model checkpoints
  references, talc source, warnings, masks, overlays, metrics, PDF, CSV;
- добавить disagreement/review-candidate artifacts после реализации карты;
- в демо открывать один ZIP/PDF как доказательство воспроизводимости.

### 5. Real panorama compliance card

Большие изображения уже лучше готовятся в UI, resident inference ускоряет batch,
но у жюри останется вопрос "а панорама?". Нужен короткий evidence card:

- input size, analysis size, tile size/stride, tile count;
- elapsed time, backend/device, peak memory if available;
- whether live run or pre-generated run was used;
- explicit caveat: public Nornickel VM больше не использовать.

## P1: полезно после P0

### Review candidates rail

После disagreement map добавить компактный список спорных crop-кандидатов:
reason, bbox, score, expected impact on decision. Это даст human-in-loop историю
без строительства еще одного QA-приложения.

### Series / resident bridge

Series orchestration работает, но ML Series узкое место в per-item pipeline.
Следующий UX/infra шаг: использовать resident path или хотя бы показывать, что
медленно именно выполнение модели, а не Series UI.

### Grain-level Path B только после реальных labels

Path B важен как объяснимая стратегия, но bootstrap result не headline. Дальше
его стоит продвигать только через реальные human labels в `apps/grain_review_web.py`
и через trained talc model branch.

### Dataset curation and pseudo-label cleanup

`curation.py` уже есть. Использовать его лучше точечно: найти label conflicts,
hard/high-loss tiles, near-duplicates и кандидаты для ручной проверки. Не надо
запускать большой новый data-cleaning проект без прямой связи с метрикой.

## P2 / отложить

- Новые backbone-архитектуры для сульфидов: текущий B2 уже силен, а риск
  упаковки/демо выше потенциального выигрыша.
- Полноценный MIL/CLAM как отдельная модель: оставить как research direction,
  пока Path A уже закрывает ordinary/fine accuracy.
- Новые UI-страницы ради "полировки": сначала finish decision lane, provenance,
  disagreement и demo script.
- Public VM demo: не использовать и не redeploy из-за stop request.
- Любые SEM/XRD/product-QC расширения: не относятся к v2 official OM path.

## Обновленный приоритет для финального демо

1. Показать strongest measured numbers честно: sulfide segmentation, Path A
   ordinary/fine, talc segmentation, robustness.
2. Не смешивать 2-class Path A score с full 3-class score.
3. Не заявлять `+/-3 pp` по тальку до calibration/review.
4. Сделать source-disagreement MVP, если остается время на одну "умную" фичу.
5. Упаковать доказательства: prepared runs, PDF, ZIP, runtime provenance,
   demo script и presentation package.
