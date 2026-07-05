# CODE_REVIEW.md — карта кода для ревьюера

Цель файла — за несколько минут показать, **где живёт логика**, **что читать первым** и
**какие инженерные решения приняты осознанно**. Все указатели — `путь:строка`.

## 1. Архитектура одним взглядом

```text
                 ┌───────────────────────────────────────────────┐
   изображение → │  apps/ore_pipeline_web.py   (UI + HTTP + REST) │
   (UI/CLI/API/  │  scripts/run_ore_pipeline.py (CLI, один снимок) │
    MCP)         │  apps/ore_mcp_server.py      (MCP-инструменты)  │
                 └───────────────────────┬───────────────────────┘
                                         │  все входы сходятся в один движок
                 ┌───────────────────────▼───────────────────────┐
                 │  src/ore_classifier/resident_pipeline.py       │
                 │   run_image(): тайлинг → сегментация →         │
                 │   сшивка → компоненты → правило руды           │
                 └───────────────────────┬───────────────────────┘
        ┌────────────────────┬───────────┴───────────┬───────────────────┐
        ▼                    ▼                        ▼                   ▼
  model_io.py          tiling.py            component_analysis.py    talc_candidate.py
  (загрузка            (перекрытие/         (морфология зерна,        + talc SegFormer-B0
   SegFormer/           Hann-сшивка)         правило «> 10%»)          (в матрице)
   ResUNet/effb3)
```

Ключевой принцип — **не** сквозной `изображение → класс`, а разложение на проверяемые этапы.
Это осознанное решение (см. презентацию, слайд 2): ошибка локализуется на своём этапе, где её
видно и можно исправить.

## 2. Что читать первым (4 файла)

| Порядок | Файл | Зачем |
| --- | --- | --- |
| 1 | [`src/ore_classifier/component_analysis.py`](src/ore_classifier/component_analysis.py) | **Ядро правила руды** — компактный, читается за 10 минут |
| 2 | [`src/ore_classifier/resident_pipeline.py`](src/ore_classifier/resident_pipeline.py) (582 стр.) | Оркестрация: `run_image()` на строке 299 |
| 3 | [`scripts/run_ore_pipeline.py`](scripts/run_ore_pipeline.py) | Тонкий CLI-враппер поверх движка |
| 4 | [`apps/ore_pipeline_web.py`](apps/ore_pipeline_web.py) (7341 стр.) | UI/HTTP/REST/безопасность — большой, но модульный |

## 3. Официальное правило — где именно оно в коде

Правило детерминированное и локализовано в одном месте
([`component_analysis.py`](src/ore_classifier/component_analysis.py)):

- `component_analysis.py:69` — порог `talc_fraction_threshold: float = 0.10`;
- `component_analysis.py:131` — `talc_fraction = talc_area / max(analysis_area, 1)`
  (знаменатель = анализируемая область, артефакты исключены);
- `component_analysis.py:134` — `if talc_fraction > cfg.talc_fraction_threshold:` → `talcose_ore`
  (строго `>`, ровно 10% — не оталькованная);
- `component_analysis.py:143` — `talc_margin` (decision margin для предупреждений);
- `component_analysis.py:300` — `rule_text_ru()` формирует человекочитаемое заключение;
- `component_analysis.py:312` — `summary_warnings()` помечает граничные случаи
  (`talc_fraction_near_threshold` и т. п.).

Тип срастания по морфологии зерна: `analyze_components()` (`:78`),
`reconstructed_footprint()` (`:266`, замыкание контура), `component_solidity()` (`:285`),
`component_features()` (`:201`, replacement-ratio/компактность). LLM в решении **нет**.

## 4. Два бэкенда (надёжность)

- **ML** — `model_io.py` загружает SegFormer/ResUNet/EfficientNet. Обратите внимание на
  строгий remap namespace-дрейфа SegFormer (`segformer.stages.*` ↔ `segformer.encoder.*`) —
  все ключи/формы проверяются; см.
  [`docs/notes/2026-07-03-segformer-transformers-namespace-compatibility.md`](docs/notes/2026-07-03-segformer-transformers-namespace-compatibility.md).
- **Эвристика** — отдельный подпроект [`heuristic_segmentation/`](heuristic_segmentation/):
  сегментация без torch, работает как fallback и объяснимый baseline. Компонентная
  классификация оптимизирована на padded-ROI (регрессионный тест против полнокадровой версии).

Переключение бэкенда — на лету в `Настройках` с неразрушающей проверкой чекпойнта `Test`.

## 5. Большие панорамы

`resident_pipeline.py`: `run_image()` (`:299`) режет панораму на перекрывающиеся тайлы,
инференс батчами, взвешенная сшивка Hann (`_tile_weight()`, `:532`) в memmap, затем
глобальный connected-component postprocess. Панорама **не ресайзится** — морфология зёрен
сохраняется; пик GPU-памяти не растёт с размером панорамы (тайлы + memmap). Замеры —
[`docs/benchmarks/07_...`](docs/benchmarks/07_panorama_performance_20260705.md),
[`docs/benchmarks/08_...`](docs/benchmarks/08_largest_panorama_16jpg_zelda_20260705.md).

## 6. Безопасность и устойчивость сервиса (сделано осознанно)

В [`apps/ore_pipeline_web.py`](apps/ore_pipeline_web.py):

- **Детектор декомпрессионных бомб** — `describe_decode_bomb()` (`:344`), пороги
  `DECODE_BOMB_MAX_MEGAPIXELS = 1000` / `..._EXPANSION_RATIO = 300` (`:100`), проверка **до**
  полного декодирования в `_register_upload_file()` (`:2188`); отказ `HTTP 413` с
  `code: "decode_bomb"`. Порог 1000 МП калиброван по всему датасету (0/1236 ложных; крупнейшая
  реальная панорама 574 МП проходит).
  - **Осознанный компромисс:** `Image.MAX_IMAGE_PIXELS = None` (`:78`) отключает встроенный
    guard PIL, потому что легитимные панорамы (574 МП) превышают дефолт; вместо него —
    собственная предекодная проверка выше. Это документировано в
    [`docs/benchmarks/06_production_load_stability_20260704.md`](docs/benchmarks/06_production_load_stability_20260704.md).
- **Валидация путей** — централизованная защита от `..`-traversal на upload/run.
- **Потоковая отдача файлов** — chunked + HTTP byte-range (`206`/`416`, `Accept-Ranges`), чтобы
  крупные артефакты не поднимали RAM.
- **Кэш `/api/status`** (`status_payload()`, `:2744`/`:7251`) с TTL 1 с — снимает
  GIL-сериализацию на постоянно опрашиваемом эндпоинте.
- **Опциональная парольная защита** — PBKDF2-SHA256, HttpOnly-сессия; `/api/openapi.json`
  остаётся открытым.
- **OpenAPI 3.1** — `build_openapi_document()` (`:6170`), проходит формальную валидацию
  `openapi-spec-validator`; guard синхронизирует таблицу маршрутов с хендлерами.

## 7. QA-инструменты (human-in-the-loop, не инференс)

- [`apps/talc_review_web.py`](apps/talc_review_web.py) — canvas-ревью масок талька; инструмент
  `Similar` калибруется по цветовой сигнатуре кликнутого зерна: `collectSimilarSeedSamples()`
  собирает r/g/b/luma/текстуру вокруг seed-пикселя, `similarStats()` агрегирует их, а
  `computeSimilarTalcPreview()` выводит допуски (файл активно меняется — ищите по имени функции,
  а не по номеру строки). Роль приложения — производство обучающих (silver) масок, **не** движок
  инференса.
- [`apps/grain_review_web.py`](apps/grain_review_web.py) — ревью зёрен (обычное/тонкое) с
  сортировкой «ценные для проверки» — обратная связь для grain-классификатора.

## 8. Тесты и как их запускать

- `tests/` — **40** модулей unit/интеграционных тестов (движок, правило, web, REST, MCP,
  метрики, конвертер талька, устойчивость resident-пайплайна, Docker-конфиг…).
- `tests/browser/` — Playwright-смоук трёх приложений.
- OpenAPI — формальная валидация 3.1 + guard синхронизации маршрутов.

```bash
python3 -m pytest tests/ -q
```

## 9. Честные замечания к коду

- `apps/ore_pipeline_web.py` крупный (7341 стр.) — это осознанный выбор в пользу
  zero-dependency `http.server` без веб-фреймворка; логика сгруппирована по разделам,
  но файл монолитный.
- Часть возможностей (правка маски, Серии, ГИС-экспорт, MCP) наросла в ходе хакатона — см.
  [ChangeLog.md](ChangeLog.md) для хронологии и мотивации каждого изменения.
- Дефолтные ML-чекпойнты талька/grade лежат под `outputs/`/`models/` и в свежем клоне
  отсутствуют → бэкенд аккуратно откатывается к эвристике (см. [MODEL_CARD.md](MODEL_CARD.md)).

## 10. Внешние ревью

Проект прогонялся через внешние ревью (Codex / Gemini) в ходе разработки; аудиты
производительности/безопасности зафиксированы в `docs/benchmarks/06_...` и закрыты кодом
(см. §6 и [ChangeLog.md](ChangeLog.md)).
