# MODEL_CARD.md — модели решения

Провенанс, обучение, метрики, назначение и ограничения всех моделей конвейера
«Скажи мне, кто твой шлиф». Метрики продублированы из [EVALUATION.md](EVALUATION.md) с той же
разметкой по источнику (**weak-label / silver / proxy / folder-GT**). Числа — из
`docs/benchmarks/` и [`docs/cards/binary-sulfide-model-card.md`](docs/cards/binary-sulfide-model-card.md).

## Обзор

| Роль | Модель | Статус | Метрика (источник) |
| --- | --- | --- | --- |
| Сегментация сульфидов | **SegFormer-B2** | развёрнута (по умолчанию) | IoU 0.974 (weak-label) |
| Сегментация талька | **SegFormer-B0 (5-fold)** | развёрнута | IoU 0.644 / F1 0.782 (silver) |
| Тип срастаний | **EfficientNet-B3 (Grade-CNN)** | развёрнута (параллельная ветка) | macro-F1 0.930/0.939 (folder-GT) |
| Сегментация (fallback) | Эвристика (без torch) | развёрнута | baseline/объяснимая |
| Сегментация сульфидов | SegFormer-B0/B1/B3, ResUNet, Mask2Former | бенчмарк-варианты | см. EVALUATION §1 |
| Тип срастаний (по зёрнам) | Grain-классификатор (tabular) | экспериментальный (путь B) | proxy |

Общее: все нейросетевые backbone **предобучены на ImageNet** и дообучены на наших метках
(это факт из train-логов, не «с нуля»). Обучение — на GPU-серверах (`zelda` RTX 4090,
`gx10` GB10), не на Mac.

---

## 1. SegFormer-B2 — сегментация сульфидов (основная модель)

- **Назначение:** бинарная маска `сульфид / не-сульфид` — фундамент всего конвейера.
- **Архитектура:** SegFormer (MiT-B2), backbone `nvidia/mit-b2` (ImageNet), дообучен.
- **Данные:** `binary_sulfide_dataset_v0` — 8536 тайлов 512 px (2976 LumenStone S1/S2 с
  реальными пиксельными масками + 5560 официальных фото с псевдо-метками Otsu), шаг 384.
- **Обучение:** `scripts/train_binary_sulfide.py`, `CrossEntropyLoss(ignore_index)` (спорные
  пиксели и границы — в `ignore`, не в метку), аугментации: флипы, повороты 90°, мягкий
  color-jitter. 30 эпох на zelda.
- **Метрики (weak-label val):** IoU **0.9744**, F1 0.9870, AUC 0.9988, HD95 23.6 px. Лучший из
  6 бенчмарк-вариантов (см. EVALUATION §1).
- **Чекпойнт:** `models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt`
  — **в репозитории (Git LFS)**.
- **Ограничения:** метрика weak-label (частичная цикличность с Otsu-эвристикой) → завышена
  относительно геологической точности; это критерий выбора чекпойнта.

## 2. SegFormer-B0 — сегментация талька

- **Назначение:** детекция талька **только в не-сульфидной анализируемой области**
  (`область − сульфиды − артефакты`) → доля талька для правила `> 10%`.
- **Архитектура:** SegFormer (MiT-B0), ImageNet-претрен, non-sulfide-clipped выход.
- **Данные:** 42 изображения «Области оталькования» → авто-конверсия синих линий в кандидат-маску
  (штрих в `ignore`) → перенос на чистый парный оригинал → **ручное ревью всех 42**
  (`apps/talc_review_web.py`) → silver-GT. Обучение только в не-сульфидной матрице,
  `scripts/train_talc_segmentation.py` + `scripts/run_talc_segformer_folds.py` (5-fold,
  калибровка порога на валидации).
- **Метрики (silver, 5-fold):** talc IoU **0.644**, F1 **0.782**; побит oracle-luma baseline
  (0.502); старый HSV-кандидат давал IoU 0.000.
- **Порог по умолчанию:** talc probability `0.50`.
- **Чекпойнт:** `outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt`
  — **не в клоне** (под `outputs/`); доступен на развёрнутом демо и по ссылкам.
- **Ограничения:** silver-маски (не эксперт), ревью велось с подсказкой по яркости (частичная
  цикличность), все 42 фото делят условия съёмки. Самый хрупкий источник в проекте.

## 3. EfficientNet-B3 — Grade-CNN (тип срастаний)

- **Назначение:** метка `обычное ↔ тонкое` (`ordinary_intergrowth ↔ fine_intergrowth`) на уровне
  изображения — несёт критерий ТЗ «тип срастаний ≥ 90%». Работает **параллельной веткой**;
  во фьюзе решает не-оталькованные случаи.
- **Архитектура:** EfficientNet-B3 (torchvision, ImageNet), img 384, class-weighted CE,
  preproc-aware аугментации. `scripts/train_grade_classifier.py`.
- **Данные/сплит:** ~755 обучающих изображений (единый group-aware сплит train/val 85/15 по
  аншлифу, `scripts/train_grade_classifier.py:grouped_split`, без k-fold), leak-free held-out 230
  (фиксированный тестовый набор, не используется при подборе модели).
- **Метрики (folder-GT, held-out 230):** macro-F1 **0.930** (raw) / **0.939** (preproc-aware);
  устойчивость к сложным изображениям — EVALUATION §3.
- **Sweep архитектур:** не выполнялся — `convnext_tiny`/`resnet50` заскаффолжены в
  `BACKBONES` (`scripts/train_grade_classifier.py`), но не обучены, чекпойнтов/eval-артефактов
  нет (см. EVALUATION §2 «sweep архитектур — не выполнялся»); effb3 выбран за data-efficiency
  и отсутствие новой зависимости, не по результатам сравнения.
- **Чекпойнт:** `models/grade_classifier/effb3_ordfine_ppaug_20260704/best.pt`
  — в git (LFS), доступен после `git lfs pull` на свежем клоне. Есть также 4-классовый
  бенчмарк-вариант `effb3_4class_20260704` (val F1-macro 0.771, 4-классовая схема
  ordinary/thin/talc/refractory, не в git — только на развёрнутом демо).
- **Ограничения:** image-level (не покомпонентная); GT — метка папки на аншлиф, не пиксельная.

## 4. Эвристический бэкенд (fallback, без torch)

- **Назначение:** объяснимый baseline и fallback без GPU/torch (`heuristic_segmentation/`).
- **Метод:** нормализация освещения, подавление зелёных/синих артефактов, морфология, связные
  компоненты, покомпонентные правила `обычное/тонкое` (оптимизировано на padded-ROI).
- **Метрики:** QA-сигнал, не экспертная истина; см.
  [`docs/notes/2026-07-03-heuristic-segmentation-subproject.md`](docs/notes/2026-07-03-heuristic-segmentation-subproject.md).
- **Чекпойнты:** не требуются.

## 5. Бенчмарк-варианты (не развёрнуты)

SegFormer-B0/B1/B3, Mask2Former-Swin-T, ResUNet — обучены для сравнения; B2 остался дефолтным
(EVALUATION §1). Grain-классификатор (`models/grain_classifier/bootstrap_v0`) — экспериментальный
путь B (сорт по зёрнам, human-in-the-loop).

---

## Доступность чекпойнтов

| Чекпойнт | В git (LFS) | В свежем клоне | На развёрнутом демо |
| --- | :---: | :---: | :---: |
| Сульфид SegFormer-B2 (default) | ✅ | ✅ (`git lfs pull`) | ✅ |
| Сульфид SegFormer-B0/B1, ResUNet | ✅ | ✅ | ✅ |
| Тальк ResUNet (local) | ✅ | ✅ | — |
| **Тальк SegFormer-B0 (развёрнутый, default)** | ✅ (`fold_00` only) | ✅ (`git lfs pull`) | ✅ |
| **Grade-CNN EfficientNet-B3 (`ppaug`, default)** | ✅ | ✅ (`git lfs pull`) | ✅ |
| Grade-CNN, другие варианты (raw/4class/ppaug07+acq) | ❌ | ❌ | частично (только если явно передан чекпойнт) |
| Тальк SegFormer-B0, folds 1–4 + smoke (не дефолт) | ❌ | ❌ | — |
| Grain-классификатор | ❌ | ❌ | частично |

**Практический вывод:**

- После `git lfs pull` сульфидная модель B2, тальк-`SegFormer-B0` (fold_00) и Grade-CNN
  (`ppaug`) — все три дефолтных чекпойнта — работают сразу на свежем клоне; `DEFAULT_*_BACKEND`
  в `apps/ore_pipeline_web.py` автовыбирает `ml`, когда чекпойнт присутствует.
- Только вспомогательные варианты (другие Grade-CNN рецепты, талька folds 1-4/smoke)
  не входят в git — они нужны лишь для сравнения архитектур/рецептов в EVALUATION.md, не для
  дефолтного пути приложения.
- Все веса Hugging Face резолвятся из общего кэша `~/.cache/huggingface/`, а не из репозитория.

## Назначение и не-назначение

- **Назначение:** ассистент-классификатор руды по OM-шлифам для лаборатории; человек
  проверяет маску и заключение.
- **Не назначение:** автономная замена эксперта; SEM/XRD; количественная минералогия за
  рамками сульфид/тальк/матрица. Полные ограничения — [LIMITATIONS.md](LIMITATIONS.md).
