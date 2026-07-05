# Nornickel AI Science Hack 2026 — Скажи мне, кто твой шлиф

**Задача.** По панорамному OM-изображению полированного шлифа определить тип руды:
`рядовая` / `труднообогатимая` / `оталькованная` — с проверяемой маской, долями фаз
и текстовым заключением.

**Решение.** Интерпретируемый измерительный конвейер, а не «чёрный ящик»:
`панорама → маска талька и его доля → класс руды`. Каждый процент прослеживается до
маски, источника сигнала и параметров запуска.

**Основной проверяемый контур:** `изображение -> доля талька -> правило > 10% -> класс руды`.
Официальное правило простое и детерминированное: **`доля талька > 10%` → `оталькованная`**
(строго `>`; ровно 10% — ещё не оталькованная).

- **Развёрнутое решение (ничего ставить не нужно):** <https://nornickel-ai-hackathon.alola.ru/>
  — доступ на слайде «Ссылки» презентации (резерв: <https://nornickel-ai-hackathon.my.3simbio.ru/>).
- **Локально:** через venv и `python apps/ore_pipeline_web.py --host 127.0.0.1 --port 0`
  (или `docker compose up --build` -> `http://<host>:8080/workspace`).
- **С чего начать ревью кода:** [`apps/ore_pipeline_web.py`](apps/ore_pipeline_web.py),
  [`src/ore_classifier/resident_pipeline.py`](src/ore_classifier/resident_pipeline.py),
  [`scripts/run_ore_pipeline.py`](scripts/run_ore_pipeline.py) — карта в [CODE_REVIEW.md](CODE_REVIEW.md).
- **Артефакты запуска:** маска сульфидов, маска талька, `ore_summary.json`, `summary.csv`,
  таблицы признаков зёрен, PDF-отчёт, `run.json` + `reports/runtime.json`.

> Судейские документы: [SUBMISSION_README.md](SUBMISSION_README.md) · [QUICKSTART.md](QUICKSTART.md)
> · [EVALUATION.md](EVALUATION.md) · [MODEL_CARD.md](MODEL_CARD.md) · [DATA_CARD.md](DATA_CARD.md)
> · [LIMITATIONS.md](LIMITATIONS.md) · [CODE_REVIEW.md](CODE_REVIEW.md) ·
> [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## Что делает система

1. **Сегментация сульфидов** (SegFormer-B2): бинарная маска `сульфид / не-сульфид`.
2. **Детектор талька** (SegFormer-B0): тальк ищется **только в нерудной матрице**
   (`анализируемая область − сульфиды − артефакты`).
3. **Связные компоненты** сульфидов → отдельные зёрна с морфологией.
4. **Тип срастания** `обычное / тонкое` — по морфологии зерна (не по яркости пикселя);
   метку сорта даёт Grade-CNN (EfficientNet-B3), покомпонентное правило — объяснение.
5. **Детерминированное правило класса руды** (из ТЗ) + доли, `decision margin` и предупреждения.
6. **Маска, оверлеи, heatmap, метрики, отчёт** (PDF/CSV/ZIP), провенанс каждого запуска.

Четыре интерфейса — один движок: **WebUI**, **CLI**, **REST API + OpenAPI 3.1**,
**MCP-сервер** для AI-агентов.

## Официальное правило классификации

```text
если доля талька > 10%:                     -> оталькованная руда
иначе Grade-CNN решает рядовая ↔ труднообогатимая
    (фоллбэк-правило по морфологии: обычные >= тонкие -> рядовая, иначе труднообогатимая)
```

- `доля талька = площадь талька / площадь анализируемой области` (артефакты шлифовки
  исключены из знаменателя; полнокадровая доля сохраняется как `*_fraction_image`);
- строго `>`: ровно `10%` — **ещё не** оталькованная;
- ноль сульфидов и зоны артефактов исключаются из расчёта.

### Наши операционные определения

На QA-сессии №4 организаторы подтвердили: **экспертной «доли талька» не существует**, и
решения оцениваются **по определению, которое команда выписала явно** (порог 10% сохраняется).
Наши определения:

- **Сульфид** — светлая рудная фаза (бинарная сегментация SegFormer-B2). Всё «серое» —
  не-сульфид (нерудная матрица / пустая порода).
- **Тип срастания** — по **морфологии** восстановленного зерна: крупное, компактное,
  слабо замещённое нерудной фазой → **обычное** (рядовая); ажурное/фрагментированное,
  сильно замещённое (высокий replacement-ratio, низкие solidity/компактность) → **тонкое**
  (труднообогатимая).
- **Тальк** — нерудная фаза (SegFormer-B0) в пределах не-сульфидной анализируемой области.
- **Итог (фьюз):** тальк-ветка решает `оталькованная`; иначе Grade-CNN — `рядовая ↔ труднообогатимая`.

Полная методология и дословные цитаты QA#4 — в
[`docs/notes/2026-07-03-official-metrics-and-panorama-split.md`](docs/notes/2026-07-03-official-metrics-and-panorama-split.md)
и слайде 8 презентации.

## Попробовать

```bash
# 1. Развёрнутое решение (Selectel Alize, 1×L4 24 ГБ) — ничего ставить не нужно
#    https://nornickel-ai-hackathon.alola.ru/   (доступ — на слайде «Ссылки» презентации)

# 2. Docker (CPU/эвристика по умолчанию)
docker compose up --build          # -> http://<host>:8080/workspace
docker compose --profile gpu up --build   # ML-бэкенд на NVIDIA GPU (порт 8210)

# 3. Локально без Docker
python3 -m venv /tmp/nornikel_v2_ml_venv
source /tmp/nornikel_v2_ml_venv/bin/activate
python -m pip install -r requirements.txt
python apps/ore_pipeline_web.py --host 127.0.0.1 --port 0
```

Подробные инструкции для жюри — [QUICKSTART.md](QUICKSTART.md).

## Что на выходе (артефакты)

Каждый запуск — неизменяемая (immutable) папка со следующими файлами:

- `masks/sulfide_mask.png` — бинарная маска сульфидов (этап 1);
- `masks/talc_mask.png`, `masks/talc_cluster_mask.png` — тальк (клипован по не-сульфиду);
- confidence-heatmap и оверлеи сульфидов/срастаний для визуальной проверки;
- `component_features.csv` — покомпонентные признаки зёрен (площадь, replacement-ratio, solidity…);
- `ore_summary.json` — машиночитаемое решение, доли, `decision margin`, предупреждения;
- `run.json` + `reports/runtime.json` — провенанс (бэкенд, чекпойнт, устройство, тайлы, пороги);
- экспорт: PDF-отчёт, `metrics.csv`, ZIP, ГИС-экспорт (GeoJSON/Shapefile).

## Карта репозитория

```text
apps/                     браузерные приложения (stdlib http.server, без Streamlit)
  ore_pipeline_web.py     главный UI пайплайна (Рабочее место/Серии/История/Статус/API/Настройки)
  talc_review_web.py      QA-инструмент ревью масок талька (производит silver-разметку)
  grain_review_web.py     ревью зёрен (human-in-the-loop разметка обычное/тонкое)
  ore_mcp_server.py       MCP-сервер: пайплайн как инструмент AI-агента
src/ore_classifier/       ядро: resident_pipeline, model_io, component_analysis, tiling, ...
scripts/                  CLI-утилиты: run_ore_pipeline, run_official_batch, train_*, evaluate_*
heuristic_segmentation/   отдельный не-нейросетевой baseline (fallback без torch)
docs/                     official/ · specs/ · plans/ · notes/ · benchmarks/ · cards/ · ui/v2/
models/                   чекпойнты (Git LFS); HF-кэш живёт вне репозитория
dataset/                  локальная копия официального датасета (в git не коммитится)
outputs/                  генерируемые артефакты (в git не коммитится)
```

## Быстрый локальный запуск (CLI)

Один снимок через end-to-end путь:

```bash
python3 scripts/run_ore_pipeline.py \
  --image "dataset/Фото руд по сортам. ч1/Рядовые руды/2539589-1.JPG" \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --talc-checkpoint outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt \
  --out-dir outputs/demo_ore_pipeline \
  --tile-size 1024 --stride 768 --batch-size 4
```

Для больших панорам используйте путь по файлу (path-based), а не base64-загрузку.
Без чекпойнтов доступен объяснимый эвристический baseline
(`heuristic_segmentation/run_heuristic_segmentation.py`).

## API / пакетная обработка

- **REST API + OpenAPI 3.1:** машиночитаемая спецификация — `GET /api/openapi.json`
  (открыта даже при включённой парольной защите); интерактивная страница — `/api`.
- **Серии / batch:** пакетная обработка партии снимков в UI (`Серии`) и в CLI
  (`scripts/run_official_batch.py`).
- **MCP:** `apps/ore_mcp_server.py` — инструменты `classify_thin_section` и `get_config`,
  модель остаётся тёплой между вызовами.

## Провенанс моделей и данных

- **Модели:** SegFormer-B2 (сульфиды), SegFormer-B0 (тальк, 5-fold), EfficientNet-B3
  (Grade-CNN, тип срастаний), ResUNet и эвристика как baseline/fallback. Все backbone
  предобучены на ImageNet и дообучены на наших метках. Провенанс, метрики и статус
  чекпойнтов — в [MODEL_CARD.md](MODEL_CARD.md).
- **Данные:** официальный пакет Норникеля (1236 файлов, ~3.0 ГБ, SHA-256 сверен) с метками
  **уровня папки** + 42 примера «синих линий» талька; публичный LumenStone S1/S2 как
  proxy-претрен пиксельных масок. **Пиксельной экспертной GT нет** — вся тренировка это
  слабый надзор. Подробно — [DATA_CARD.md](DATA_CARD.md).

## Валидация и тесты

- **Метрики** (все размечены по источнику — weak-label / silver / proxy / folder-GT):
  сульфид IoU `0.974` (weak-label); тальк IoU `0.644` / F1 `0.782` (silver, 5-fold);
  Grade-CNN тип срастаний macro-F1 `0.930–0.939` (folder-GT, held-out 230) — критерий
  ТЗ «≥ 90%» закрыт; фьюз-вердикт 3 класса macro-F1 `0.861` (leak-free 345). Сводка —
  [EVALUATION.md](EVALUATION.md) и
  [`docs/notes/2026-07-05-consolidated-metrics.md`](docs/notes/2026-07-05-consolidated-metrics.md).
- **Производительность:** цель ТЗ ≤ 5 мин на 10000×10000 закрыта (панорама 126 Мп за
  99.6–186.9 с на трёх машинах); крупнейшая панорама 574 Мп — 7:08 end-to-end на RTX 4090.
- **Тесты:** 40 модулей unit/интеграционных тестов в `tests/` + браузерные Playwright-тесты
  + формальная валидация OpenAPI. `python3 -m pytest tests/`.

## Известные ограничения

Кратко: нет экспертной пиксельной GT (сегментационные метрики weak-label/silver, завышены
относительно геологической точности); критерий «ошибка доли талька ≤ ±3%» **не проверяем ни
для кого** — организаторы (QA#4) подтвердили, что не располагают эталонными долями талька;
ось `обычное↔тонкое` — слабое место детерминированного правила (её закрывает Grade-CNN);
детектор талька обучен на 42 фото с общими условиями съёмки (silver-маски). Полный список —
[LIMITATIONS.md](LIMITATIONS.md).

## Ссылки

- **GitHub (основной):** <https://github.com/s1mb1o/2026-ore-thin-section-classifier>
- **SourceCraft (зеркало, auto-sync):** <https://sourcecraft.dev/s1mb1o/2026-ore-thin-section-classifier>
- **Развёрнутое решение:** <https://nornickel-ai-hackathon.alola.ru/> (доступ — на слайде «Ссылки» презентации)
- **Резерв:** <https://nornickel-ai-hackathon.my.3simbio.ru/> · <https://nornickel-backup.my.3simbio.ru/workspace>
- **Презентация:** `presentation/` (RU-дек `presentation_ru.md`, рендер `presentation.html`)
- **Постановка задачи:** [`docs/official/Скажи мне кто твой шлиф.md`](docs/official/Скажи мне кто твой шлиф.md)
