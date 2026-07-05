# QUICKSTART — как запустить решение

Три способа, от самого простого к самому гибкому. Для проверки жюри достаточно **способа 1**.

---

## Способ 1. Развёрнутое решение (ничего ставить не нужно) ⭐

Откройте в браузере и войдите:

- **Основной:** <https://nornickel-ai-hackathon.alola.ru/>
- **Резерв:** <https://nornickel-ai-hackathon.my.3simbio.ru/>

**Доступ:** логин и пароль — на слайде «Ссылки» презентации (не публикуются в открытом репозитории).

Дальше: `Рабочее место` → загрузите снимок шлифа (TIFF/PNG/JPEG) → `Старт` → смотрите маску,
доли фаз, заключение; экспорт — PDF/CSV/ZIP. Тестовые снимки есть в датасете под
`dataset/Фото руд по сортам. ч1/`.

---

## Способ 2. Docker (одна команда)

Требуется Docker с плагином Compose. Из корня репозитория:

```bash
# CPU / эвристический бэкенд (по умолчанию) — работает без GPU и без torch-моделей
docker compose up --build
# → откройте http://localhost:8080/workspace
```

```bash
# NVIDIA GPU / ML-бэкенд (SegFormer-B2 + SegFormer-B0), нужен nvidia-container-toolkit
docker compose --profile gpu up --build
# → откройте http://localhost:8210/workspace
```

- Образ CPU (`nornikel/ore-pipeline-ui:v2`) собирается из
  [`docker/ore-pipeline-ui/Dockerfile`](docker/ore-pipeline-ui/Dockerfile) — `python:3.11-slim`,
  собирается и на **x86_64**, и на **ARM64** (проверено на gx10, `NVIDIA GB10`).
- Образ GPU (`nornikel/ore-pipeline-ui:v2-gx10-ml`) — из
  [`docker/ore-pipeline-ui/Dockerfile.gx10-ml`](docker/ore-pipeline-ui/Dockerfile.gx10-ml)
  (база `nvcr.io/nvidia/pytorch`).
- Чекпойнты **не** «запекаются» в образ — они монтируются из `./models` (и
  `./outputs/talc_segformer_folds` для талька) во время запуска. Перед ML-запуском
  подтяните LFS: `git lfs pull` (см. ниже про доступность чекпойнтов).

Полезные переменные окружения (значения по умолчанию в
[`compose.yaml`](compose.yaml)): `ORE_UI_PUBLIC_PORT`, `ORE_UI_BACKEND`
(`heuristic`/`ml`), `ORE_UI_CHECKPOINT`, `ORE_UI_TALC_BACKEND`, `ORE_UI_TALC_CHECKPOINT`.

---

## Способ 3. Локально на хосте (Python)

Требуется **Python ≥ 3.10** (код использует `zip(strict=True)` и `X | None`).

### 3a. Эвристический бэкенд (минимум зависимостей, без GPU)

```bash
python3 -m pip install numpy Pillow opencv-python
python3 apps/ore_pipeline_web.py --host 127.0.0.1 --port 0
# порт 0 = случайный свободный порт; адрес печатается в консоли
```

Это объяснимый baseline: детерминированная сегментация без нейросетей. Годится для смоук-проверки
на любой машине.

### 3b. Полный ML-бэкенд (SegFormer + Grade-CNN)

```bash
python3 -m pip install -r requirements.txt   # torch, torchvision, transformers, ...
git lfs pull                                  # подтянуть чекпойнты (см. ниже)
python3 apps/ore_pipeline_web.py --host 127.0.0.1 --port 0
```

Приложение само выбирает ML-бэкенд, если чекпойнты на месте (иначе — эвристику).
Устройство определяется автоматически: NVIDIA CUDA, Apple MPS или CPU.

### 3c. Один снимок из CLI

```bash
python3 scripts/run_ore_pipeline.py \
  --image "dataset/Фото руд по сортам. ч1/Рядовые руды/2539589-1.JPG" \
  --checkpoint models/binary_sulfide/segformer_b2_dataset_v0_zelda_20260703_overnight_safetensors/best.pt \
  --talc-checkpoint outputs/talc_segformer_folds/segformer_b0_full_20260703/fold_00/segformer_b0/best.pt \
  --out-dir outputs/demo_ore_pipeline \
  --tile-size 1024 --stride 768 --batch-size 4
```

Артефакты (маски, `ore_summary.json`, признаки зёрен, оверлеи) появятся в `--out-dir`.

---

## Доступность чекпойнтов (важно для ML-запуска)

- **В репозитории (Git LFS, `git lfs pull`):** сульфидные `SegFormer-B0/B1/B2` и `ResUNet`,
  а также ResUNet-тальк. Дефолтный сульфидный чекпойнт **B2** доступен сразу.
- **Не в клоне (крупные/локальные):** развёрнутый тальк-`SegFormer-B0` (лежит под `outputs/`),
  `Grade-CNN` (EfficientNet-B3) и grain-классификатор. Они доступны **на развёрнутом демо**
  (способ 1) и по ссылкам; в свежем клоне тальк и grade автоматически откатываются к эвристике.
- Детали — [MODEL_CARD.md](MODEL_CARD.md) § «Доступность чекпойнтов».

**Вывод для жюри:** самый простой и полный путь — способ 1 (развёрнутое демо). Локально без
чекпойнтов гарантированно работает эвристический бэкенд (способ 3a / Docker CPU).

---

## Проверка окружения (по желанию)

```bash
python3 -m pytest tests/ -q          # 40 модулей unit/интеграционных тестов
```

Для больших панорам используйте загрузку по пути к файлу, а не base64 — так память
не расходуется на кодирование гигапикселя.
