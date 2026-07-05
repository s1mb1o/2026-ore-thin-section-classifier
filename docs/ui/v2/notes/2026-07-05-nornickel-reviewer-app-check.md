# Проверка приложения как проверяющий Норникеля

Дата проверки: 2026-07-05.

Проверяемый каталог: `/Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2`.

Использованные документы: `README.md`, `SUBMISSION_README.md`, `QUICKSTART.md`, `TROUBLESHOOTING.md`, `SMOKE_TESTS.md`, `EVALUATION.md`, `MODEL_CARD.md`, `DATA_CARD.md`, `LIMITATIONS.md`.

## Итог

Локальный путь проверки из `QUICKSTART.md` работает. Приложение запускается как v2 ore-pipeline UI, отдает русскоязычный интерфейс, публикует OpenAPI, принимает реальную фотографию шлифа, выполняет расчет, формирует итоговую классификацию и скачиваемые артефакты `CSV`, `PDF` и `ZIP`.

Блокирующих дефектов в проверенном reviewer smoke не найдено.

Важная граница проверки: этот прогон проверял минимальный host-Python путь с heuristic backends, который нужен для воспроизводимого локального запуска без приватных весов. Полный ML-путь с mounted checkpoints должен оцениваться по Docker/deployed маршруту и метрикам из `EVALUATION.md`.

## Запуск

Проверялся уже поднятый quickstart-сервер:

```bash
python -u apps/ore_pipeline_web.py \
  --workspace-dir /tmp/nornikel_v2_quickstart_workspace \
  --host 127.0.0.1 \
  --port 0 \
  --backend heuristic \
  --talc-backend heuristic \
  --grain-backend heuristic
```

Фактический URL проверки: `http://127.0.0.1:60623`.

`/api/status` подтвердил:

- `app.version = v2`;
- `health.overall = ok`;
- `backend = heuristic`;
- `talc_backend = heuristic`;
- `grain_backend = heuristic`;
- `workspace_dir = /private/tmp/nornikel_v2_quickstart_workspace`.

Публичный production endpoint `https://nornickel-ai-hackathon.alola.ru/workspace` без учетных данных вернул `401`, то есть внешний reviewer route защищен basic auth. Закрытые учетные данные в этот отчет не включаются.

## HTTP и OpenAPI

Проверенные страницы локального приложения:

| URL | HTTP |
| --- | --- |
| `/` | `200` |
| `/workspace` | `200` |
| `/batch` | `200` |
| `/history` | `200` |
| `/settings` | `200` |
| `/status` | `200` |
| `/api` | `200` |
| `/api/openapi.json` | `200` |

`/api/openapi.json` отдает OpenAPI `3.1.0` с `33` путями. В документе присутствуют ключевые reviewer endpoints: `/api/status`, `/api/uploads`, `/api/runs/start`, `/api/runs/{runId}/metrics.csv`, `/api/runs/{runId}/report.pdf`, `/api/runs/{runId}/artifacts.zip`.

## End-to-end API smoke

Проверочная фотография из официального локального датасета:

```text
dataset/Фото руд по сортам. ч1/Рядовые руды/2539589-1.JPG
```

Загрузка прошла успешно:

- `upload_id = 20260705_024057_832437000_3cdae2790b`;
- размер изображения: `2272 x 1704`;
- размер файла: `813910` байт.

Запуск расчета:

- `run_id = run_20260705_024058_601036000_ad28ba27`;
- итоговый статус: `complete`;
- прогресс: `100`;
- время выполнения: `2.009` с.

Итоговый текст приложения:

> Руда классифицирована как труднообогатимая: содержание талька — 1.8%, преобладание тонких срастаний — 71.6%.

Ключевые значения из результата:

- класс: `труднообогатимая руда`;
- доля сульфидов: `21.919284%`;
- доля обычных срастаний среди сульфидов: `28.364180%`;
- доля тонких срастаний среди сульфидов: `71.635820%`;
- доля талька: `1.762062%`;
- компонентов сульфидов: `235`;
- `needs_expert_review = false`;
- `warnings = []`.

Скачанные артефакты:

| Артефакт | Размер |
| --- | ---: |
| `metrics.csv` | `1276` байт |
| `report.pdf` | `937466` байт |
| `artifacts.zip` | `42002388` байт |

`/api/runs/{runId}/files` и `artifacts.zip` содержат `43` файла. В составе есть обязательные артефакты для проверки результата:

- `run.json`;
- `reports/ore_summary.json`;
- `reports/metrics.csv`;
- `reports/runtime.json`;
- `reports/final_classes.geojson`;
- `reports/final_classes_shapefile.zip`;
- `reports/component_features.csv`;
- `reports/ore_report.pdf`;
- `masks/sulfide_mask.png`;
- `masks/final_mask.png`;
- `masks/talc_final_mask.png`;
- display overlays для original/sulfide/ordinary/fine/talc/talc-cluster.

## UI smoke

Через браузер проверены страницы `/workspace`, `/batch`, `/history`, `/settings`, `/status`, `/api`.

На `/workspace` отображаются:

- заголовок `Классификатор рудного шлифа`;
- русская навигация: `Рабочее место`, `Серии`, `История`, `Статус`, `API`, `Настройки`;
- загрузка изображения;
- кнопка `Старт`;
- переключатели слоев;
- экспортные действия `Сохранить CSV`, `Сохранить PDF-отчет`, `Просмотреть файлы`.

На `/api` отображается справочная страница с ключевыми upload/run/export endpoint. Страницы `/settings` и `/status` показывают runtime controls и health/OpenAPI сведения. Ошибок и предупреждений в browser console на проверенных страницах не обнаружено.

## Автоматизированные проверки

Выполнены команды:

```bash
python3 -m py_compile apps/ore_pipeline_web.py
python3 -m unittest discover -s tests -p 'test_ore_pipeline_web.py' -v
docker compose config --quiet
```

Результат:

- `py_compile` прошел;
- `docker compose config --quiet` прошел;
- `test_ore_pipeline_web.py`: `60` тестов прошли, `1` тест пропущен, потому что `openapi-spec-validator` не установлен из `requirements-dev.txt`.

Пропуск не блокирует reviewer smoke: внутренние OpenAPI-тесты на валидность документа, resolvable refs и доступность `/api/openapi.json` прошли.

## Замечания проверяющего

1. Локальный quickstart-путь честно проверяет воспроизводимость и работоспособность интерфейса/API, но не подтверждает ML-качество модели. Для ML-качества нужно смотреть `EVALUATION.md` и deployed/Docker путь с checkpoint-файлами.
2. Публичный endpoint защищен: без пароля `/workspace` возвращает `401`. Это соответствует документации, где учетные данные не хранятся в публичном репозитории.
3. `/api/status` показывает `health.overall = ok`; свободное место на диске около нижней границы. Для длинной демонстрации с большим числом загрузок стоит заранее очистить workspace или использовать отдельный временный каталог.
