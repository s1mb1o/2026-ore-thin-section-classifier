# Component grade model: обученный классификатор компонентов ordinary/fine

Дата: 2026-07-04

## Что это

Обученная замена ручного OR-правила формы (`fine_dark_inside_ratio | fine_solidity_max |
fine_compactness_max`) в `component_analysis`: HistGradientBoosting по тем же
per-component признакам (+ log-area, заполнение bbox, аспект, увеличение из имени
файла). Метки слабые — по папке класса (Рядовые → ordinary, Труднообогатимые →
fine). Классифицируются **найденные маски** (компоненты сульфидной сегментации), а
не кадр целиком, поэтому процентное соотношение ordinary/fine сохраняется — меняется
только присвоение метки компоненту перед той же площадной агрегацией.

## Цифры (бенчмарк: 100 тёмных+зелёных кадров ч1, тальк off)

| классификатор компонентов | точность кадра | ordinary recall | fine recall |
|---|---|---|---|
| правило OR (прод до этого) | 43.0% | 22.5% | 56.7% |
| **модель (image-level GroupKFold CV)** | **73.0%** | 57.5% | 83.3% |

In-sample финального артефакта на тех же 100: 80% (ожидаемо выше CV; честная оценка — 73%).
Известный системный кейс: магнетитовая плита с редкими сульфидными вкраплениями —
правило тянет в «рядовую» из-за компактности крупного куска; по вводной это
«тонкая» (см. обсуждение в сессии; таргет для следующей итерации разметки).

## Артефакт

`models/component_grade/hgb_weak100_20260704/{model.joblib,meta.json}` (84 KB,
обычный git-файл, не LFS). meta.json содержит список признаков и CV-метрики.
Обучение: `scripts/train_component_grade_model.py --runs-dir <batch>/runs --out-dir <dir>`.

## Интеграция (все точки — один параметр `component_classifier` в `analyze_components`)

- `src/ore_classifier/component_grade_model.py` — загрузка, признаки, labeler.
- `scripts/analyze_ore_from_masks.py --component-model <model.joblib>`
- `scripts/run_ore_pipeline.py --component-model …` (passthrough + поле в summary)
- `scripts/run_resident_batch.py --component-model …` → `ResidentSulfidePipeline(component_model=…)`
- `apps/ore_pipeline_web.py --component-model …` — **по умолчанию ВКЛЮЧЕНО**, если
  артефакт существует (паттерн как у `--grade-checkpoint`); `--component-model none`
  возвращает правило. Конструктор `OrePipelineStore` по умолчанию None → тесты и
  программные вызовы не затронуты (52/52 тестов зелёные).

## Ограничения / дальше

- Обучено на слабых папочных метках 100 тёмных кадров ч1; ordinary recall 57.5% —
  главный резерв. Идёт ручное поклеточное ревью масок (`apps/grade_mask_review.py`,
  сид от этой же модели) — переобучение на честных метках должно поднять точность.
- Сегментация не отделяет магнетит от сульфидов; модель учится компенсировать это
  на уровне класса (магнетитовые плиты → fine), проценты минералогически неточны.
- Увеличение парсится из имени файла (`10x`/`5x`…); без него — «cam».
