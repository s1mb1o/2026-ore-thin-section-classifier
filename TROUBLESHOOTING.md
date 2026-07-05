# TROUBLESHOOTING — типовые проблемы запуска v2

Дата проверки: 2026-07-05

Этот файл относится только к v2-репозиторию:

```text
/Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2
```

Все команды из `README.md`, `SUBMISSION_README.md` и `QUICKSTART.md` нужно выполнять
из этого каталога.

## Сначала проверьте, что вы в v2

```bash
pwd
test -f apps/ore_pipeline_web.py
test -f compose.yaml
test -f requirements.txt
```

Ожидаемый `pwd`:

```text
/Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2
```

Если вы в другом каталоге, перейдите в v2:

```bash
cd /Volumes/T7_2TB/Projects-T7_2TB/2026_Nornikel_Hackaton_v2
```

## Если инструкция запуска не похожа на v2

В v2 нет отдельного пакета заявки внутри каталога `submissions` и нет отдельного
API-скрипта вне `apps/`. Если найденная команда создает окружение для старого
`ore_classifier`, ставит зависимости из вложенного каталога заявки или запускает
отдельный `serve`-скрипт, это устаревшая инструкция из другого репозитория.

Для этой заявки используйте только entrypoint v2:

```bash
python apps/ore_pipeline_web.py --host 127.0.0.1 --port 0
```

Быстрый локальный запуск v2 без GPU и без torch:

```bash
python3 -m venv /tmp/nornikel_v2_venv
source /tmp/nornikel_v2_venv/bin/activate
python -m pip install numpy Pillow opencv-python
rm -rf /tmp/nornikel_v2_quickstart_workspace
python apps/ore_pipeline_web.py \
  --workspace-dir /tmp/nornikel_v2_quickstart_workspace \
  --host 127.0.0.1 --port 0 \
  --backend heuristic \
  --talc-backend heuristic \
  --grain-backend heuristic
```

Для полного ML-окружения:

```bash
python3 -m venv /tmp/nornikel_v2_ml_venv
source /tmp/nornikel_v2_ml_venv/bin/activate
python -m pip install -r requirements.txt
git lfs pull
python apps/ore_pipeline_web.py --host 127.0.0.1 --port 0
```

После запуска откройте напечатанный адрес `/workspace` или `/api`.

## Ошибка из-за системного Homebrew Python

Если запускать `python3 -m pip install ...` прямо в системный Python, Homebrew может
вернуть:

```text
externally-managed-environment
```

Это ограничение системной среды, не ошибка проекта. Используйте виртуальное окружение:

```bash
python3 -m venv /tmp/nornikel_v2_venv
source /tmp/nornikel_v2_venv/bin/activate
python -m pip install numpy Pillow opencv-python
```

## Уведомление pip о новой версии

Сообщение вида:

```text
[notice] A new release of pip is available
```

не является ошибкой. Для проверки решения его можно игнорировать.

## Сервер запустился, но терминал не возвращает prompt

Команда:

```bash
python apps/ore_pipeline_web.py --host 127.0.0.1 --port 0
```

запускает локальный web/API-сервер и работает до `Ctrl+C`. Это нормальное поведение.
Для `curl`-проверок используйте второй терминал.

При `--port 0` приложение выбирает свободный порт и печатает фактический URL. Открывайте
напечатанный порт, а не буквальный `0`.

## Docker: `docker compose` не найден

Нужен Docker с плагином Compose v2. Проверка:

```bash
docker compose version
```

Если доступен только старый `docker-compose`, используйте установленный в системе способ
запуска Compose или поставьте Docker Desktop / Compose plugin.

Проверка конфигурации v2:

```bash
docker compose config --quiet
```

## ML-чекпойнты не найдены

Свежий клон может не содержать все тяжелые модели. Сначала подтяните Git LFS:

```bash
git lfs pull
```

Если часть чекпойнтов все равно отсутствует, v2 UI откатывается к эвристическому бэкенду.
Это ожидаемо для локальной CPU-проверки. Полный ML-путь доступен на развёрнутом демо и
на окружениях, где смонтированы модели.

## Пример изображения не найден

Если путь вида `dataset/...` не существует, значит официальный датасет не развернут
локально. UI и API можно запустить без датасета; для проверки загрузите любой доступный
`JPEG`, `PNG` или `TIFF` снимок шлифа через `/workspace`.

## Python слишком старый

Код v2 требует Python 3.10 или новее. Симптомы на старых версиях могут выглядеть как
ошибки синтаксиса вокруг `X | None` или `zip(..., strict=True)`.

Проверка:

```bash
python3 --version
```
