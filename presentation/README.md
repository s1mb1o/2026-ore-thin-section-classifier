# Презентация — «Скажи мне, кто твой шлиф»

Материалы для защиты решения перед жюри. Презентация — основной вход для оценки.

## Что здесь

| Файл | Что это |
| --- | --- |
| `presentation_ru.md` | **Контент** основной презентации (17 слайдов + приложение). Слайды разделены строкой `---`. Формат: заголовок → текст → в конце `*курсивом*` — заметка спикера. |
| `features_ru.md` | **Контент** отдельной страницы возможностей (одностраничник, карточки). |
| `theme/deck.css` | **Внешний вид** (тёмная тема, семантические цвета классов). Отдельный файл — правится независимо от контента. |
| `render_presentation.py` | Рендерер: `md` + `theme/deck.css` → один самодостаточный HTML (картинки встраиваются как data-URI, оффлайн). |
| `capture_screens.py` | Playwright-скрипт съёмки скриншотов обоих приложений. |
| `presentation.html` | **Готовая презентация** (самодостаточная, ~2.6 МБ). Открывается в браузере, печатается в PDF. |
| `features.html` | **Готовая страница возможностей**. |
| `assets/screens/` | Скриншоты приложений (главное + ревью талька). |
| `assets/artifacts/` | Артефакты пайплайна (маски, оверлеи, тальк). |

## Как смотреть

Открыть `presentation.html` в браузере. Навигация: `↑ ↓ ← →` / пробел; клавиша `N` прячет заметки спикера. Экспорт в PDF — печать браузера (тема адаптирована под печать).

## Как пересобрать после правок

Правим `presentation_ru.md` (контент) или `theme/deck.css` (вид), затем:

```bash
PY=../.venv/bin/python   # python 3.12 с markdown + Pillow
"$PY" render_presentation.py --content presentation_ru.md --css theme/deck.css \
  --out presentation.html --title "Скажи мне, кто твой шлиф — Норникель" --mode deck
"$PY" render_presentation.py --content features_ru.md --css theme/deck.css \
  --out features.html --title "Возможности решения — Скажи мне, кто твой шлиф" --mode page
```

## Как пересобрать скриншоты

1. Запустить приложения на фиксированных портах:
   ```bash
   ../.venv/bin/python ../apps/ore_pipeline_web.py --host 127.0.0.1 --port 8230
   ../.venv/bin/python ../apps/talc_review_web.py --conversion-dir ../outputs/talc_blue_line_conversion --host 127.0.0.1 --port 8231
   ```
2. Снять экраны (нужен `playwright` + chromium в системном python):
   ```bash
   python3 capture_screens.py
   ```
3. Пересобрать HTML (шаг выше). Скриншоты вьюера загружаются через существующие завершённые запуски в `outputs/ore_pipeline_ui/`.

## Заметки по фактам

- Развёрнутая сульфидная модель — **SegFormer-B2, дообучен с ImageNet-претрена** (`nvidia/mit-b2`), не «с нуля».
- Все метрики размечены по источнику: weak-label (псевдо), silver (ревью талька), proxy (image-level). Числа берутся из `docs/benchmarks/`, `docs/cards/`, ChangeLog.
- Порты 8230/8231 — временные, только для съёмки; постоянного сервиса не разворачивают.
