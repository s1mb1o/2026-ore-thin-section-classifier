# Talc annotation v1 — 82 non-expert manual annotations

Ручная разметка талька (non-expert) поверх официального набора Норникеля, для
воспроизведения у другого разработчика, у которого есть **тот же исходный набор
данных организатора**. Сюда закоммичены только маски + манифест (сами исходные
снимки НЕ входят — они у разработчика уже есть).

## Состав

- `manifest.csv` / `manifest.json` — 82 записи. Каждая: `sample_id`,
  `source_relpath` (путь внутри данных организатора), `source_filename`,
  `source_sha256` (для точного сопоставления), размеры, `ore_class_gt`
  (talcose / row / fine — по папке-источнику), ручная доля талька.
- `masks/<sample_id>/`:
  - `talc_mask.png` — **ручная маска талька** (255 = тальк). Это и есть разметка.
  - `ignore_mask.png` — зоны неопределённости (исключать из loss).
  - `ore_mask.png` — эвристическая ore/opaque-маска (`sulfide_mask_final`),
    нужна, чтобы точно воспроизвести результат talcose-классификатора
    (matrix = кадр − ore). Не является разметкой, это рабочий артефакт.

## Как сопоставить с исходниками

Для каждого `sample_id`: найти в своих данных организатора файл по
`source_relpath` (или по `source_filename`), проверить `source_sha256`, наложить
`talc_mask.png` — размеры маски совпадают с исходным снимком. Пример на Python:

```python
import json, hashlib
from pathlib import Path
DATA = Path("<корень данных организатора>")   # содержит "Фото руд по сортам. ч1/..."
man = json.load(open("datasets/talc_annotation_v1/manifest.json"))
for s in man["samples"]:
    img = DATA / s["source_relpath"]
    assert img.exists(), img
    if s["source_sha256"]:
        assert hashlib.sha256(img.read_bytes()).hexdigest() == s["source_sha256"], img
    mask = Path("datasets/talc_annotation_v1/masks")/s["sample_id"]/"talc_mask.png"
    # img + mask -> обучение/оценка
```

## Провенанс разметки

- Конвенция: `docs/notes/2026-07-03-talc-manual-annotation-protocol.md`
  (плотные пятна-агрегаты талька; исчерпывающая разметка на каждом снимке).
- Стратификация: 62 talcose (ч1 «Оталькованные руды» + ч2 «оталькованные»),
  10 row (ч2 «рядовые»), 10 fine (ч2 «тонкие»). DSCN и нумерованные, разные
  увеличения. Это non-expert QA, не экспертная geologist-истина.
- Разработчик отметил, что расхождения классификатора с папочными метками —
  скорее ошибки исходной классификации экспертом (конфликты меток), см.
  `docs/notes/2026-07-04-heuristic-talcose-classifier.md`.

## Воспроизведение зафиксированного теста

С этими масками эвристический классификатор
(`src/ore_classifier/talc_zone_heuristic.py`) даёт зафиксированный результат:
**AUC 0.809, точность 87.8% (72/82)**. См. регресс-тест
`tests/test_talc_zone_heuristic.py` и заметку про классификатор.
