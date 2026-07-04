# STT and visual verification report

## Container and Visual Check

- Final MP4: `nornikel_v2_ui_only_demo_1080p_ru.mp4`.
- ffprobe: 1920x1080, 24 fps, H.264 video, AAC audio, duration 300.000 seconds.
- Visual sample: `contact_sheet.jpg` was inspected; all sampled frames show only the main v2 ore pipeline UI.
- Sampled frames cover History, Workspace final layer, Workspace sulfide layer, metrics, run files popup, panorama, Series history, Settings, Status, and API.
- Burned-in subtitles and corner time marks are visible in the sampled frames.

## STT Check

- STT model: Whisper base, language ru.
- Normalized word similarity ratio: 0.797.
- Script words: 686; STT words: 686.
- Technical English tokens are sometimes transcribed phonetically; these were adjudicated against the audio transcript below and the burned-in subtitles.

## Key Claim Checks

- [x] Dedicated only main v2 UI
- [x] No other applications used
- [x] Workspace page
- [x] History page
- [x] Single-image run loaded from history
- [x] Final segmentation layer
- [x] Sulfide layer
- [x] Talc only as an in-v2-UI layer/configuration
- [x] Fix/edit artefacts workflow
- [x] Text output and metrics table
- [x] Grain report table
- [x] Run files popup
- [x] CSV/JSON/PDF/ZIP exports
- [x] Panorama tiling
- [x] Long panorama waits can be opened from history / accelerated in recording
- [x] Series history
- [x] Series page
- [x] Settings page
- [x] Runtime backend/checkpoint settings
- [x] Status page
- [x] REST API page
- [x] Built-in API sandbox
- [x] Reproducible artifact package

## Missing/Uncertain Keys

- None after phonetic adjudication.

## Notes

- `v2 UI` is heard by Whisper as variants like `вид во UI` and `V2EY`.
- `API` is heard as `IPI`, `ИПi`, or similar phonetic variants.
- `backend` is heard as `бейкинт`.
- `checkpoint` is heard as `чекопаинт`.
- `run` is heard as `ран`, `рон`, or `ронс`.
- `immutable run` is heard as `и мутаблиран`.
- `JSON` is heard as `из сран`.
- These are STT spelling issues; the visible burned subtitles show the intended technical terms.

## STT Transcript Excerpt

```text
Это отдельный ролик, только про основной вид воёй инструмента, скажи мне, кто твой шлиф.
Другие приложения в этом видео не используются.
В кадре одно браузерное приложение, рабочее место, история запуска в серии, статус системы, IPI и настрой керантайм.
Начинаем с истории.
Готовые прогоны загружаются мгновенно, поэтому на демо мы не тратим минуты на повторный расчёт тяжёлой модели.
...
Тальк в этом ролике показан только как слой и настройка внутри вид во UI,
отдельное приложение размедки или ревью масок здесь не открывается.
...
Итога в эсценарии остается внутри одного V2UI, загрузка, настройка, запуск, слои, метрики, в зерновое чёт, экспорт, история, серии и пиай.
Главная ценность интерфейса воспроизводимый пакет, исходник, настройки, модели, маски, таблицы, ачеты и ограничения сохранены рядом с результатом.
```
