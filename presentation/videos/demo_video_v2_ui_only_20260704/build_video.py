#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import math
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WIDTH = 1920
HEIGHT = 1080
FPS = 24
TARGET_SECONDS = 300.0
VOICE = "ru-RU-DmitryNeural"
TTS_RATE = "-12%"
FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
if not FONT_BOLD.exists():
    FONT_BOLD = FONT_REGULAR


SENTENCES = [
    {
        "scene": "scene_single_final.png",
        "title": "Только v2 UI",
        "text": "Это отдельный ролик только про основной v2 UI инструмента «Скажи мне, кто твой шлиф». Другие приложения в этом видео не используются.",
    },
    {
        "scene": "scene_history.png",
        "title": "Один интерфейс",
        "text": "В кадре одно браузерное приложение: рабочее место, история запусков, серии, статус системы, API и настройки runtime.",
    },
    {
        "scene": "scene_history.png",
        "title": "История",
        "text": "Начинаем с истории. Готовые прогоны загружаются мгновенно, поэтому на демо мы не тратим минуты на повторный расчёт тяжёлой модели.",
    },
    {
        "scene": "scene_single_final.png",
        "title": "Рабочее место",
        "text": "После загрузки run открывается рабочее место: слева карточка входного изображения, метаданные, аугментация, предобработка и блок запуска.",
    },
    {
        "scene": "scene_single_final.png",
        "title": "Viewer",
        "text": "Основная область — viewer. В нём можно переключать оригинал, предобработку, сульфидную маску, финальную сегментацию и режим сравнения слоёв.",
    },
    {
        "scene": "scene_single_final.png",
        "title": "Финальный слой",
        "text": "Сейчас выбран финальный слой. Легенда слева показывает классы: обычные сростания, тонкие сросты, тальк, артефакты и фон.",
    },
    {
        "scene": "scene_single_final.png",
        "title": "Доли классов",
        "text": "Рядом с классами сразу видны доли по проанализированной области, поэтому оператор видит не только цветную маску, но и численную основу решения.",
    },
    {
        "scene": "scene_single_final.png",
        "title": "Тальк внутри UI",
        "text": "Тальк в этом ролике показан только как слой и настройка внутри v2 UI; отдельное приложение разметки или ревью масок здесь не открывается.",
    },
    {
        "scene": "scene_single_sulfide.png",
        "title": "Сульфиды",
        "text": "Переключаемся на слой сульфидов. Он отделяет сульфидную фазу от несульфидной основы и даёт базу для компонентного анализа.",
    },
    {
        "scene": "scene_single_sulfide.png",
        "title": "Контроль слоя",
        "text": "Под viewer находятся рабочие переключатели: показать тайлы, оставить только контуры и изменить прозрачность наложения поверх исходного шлифа.",
    },
    {
        "scene": "scene_single_sulfide.png",
        "title": "Исправление",
        "text": "Кнопка «Исправить» открывает правку артефактов подготовки. Такие зоны можно исключать, чтобы царапины и мусор не портили знаменатель метрик.",
    },
    {
        "scene": "scene_metrics.png",
        "title": "Текстовый вывод",
        "text": "Ниже viewer приложение пишет текстовый вывод: классификацию руды, доли сульфидов и талька, а также краткое объяснение применённого правила.",
    },
    {
        "scene": "scene_metrics.png",
        "title": "Таблица метрик",
        "text": "Таблица метрик показывает проанализированную область, доли классов, количество компонент и служебные величины, которые потом попадают в отчёт.",
    },
    {
        "scene": "scene_metrics.png",
        "title": "Масштаб",
        "text": "Если в метаданных задан масштаб, тот же интерфейс добавляет физические площади в квадратных микрометрах и миллиметрах, без отдельной обработки файла.",
    },
    {
        "scene": "scene_grains_files.png",
        "title": "Зерновой отчёт",
        "text": "Зерновая таблица разворачивает сульфидные компоненты: площадь, эквивалентный диаметр, периметр, долю сульфидов и proxy-показатели освобождения.",
    },
    {
        "scene": "scene_grains_files.png",
        "title": "Связь таблицы и картинки",
        "text": "Строки таблицы можно использовать как навигацию по изображению, чтобы эксперт быстро увидел, какие зёрна сильнее всего влияют на итоговый сорт.",
    },
    {
        "scene": "scene_files_popup.png",
        "title": "Файлы run",
        "text": "Список файлов run открывается прямо из интерфейса. Изображения, CSV, JSON, PDF и ZIP остаются привязаны к одному неизменяемому run id.",
    },
    {
        "scene": "scene_files_popup.png",
        "title": "Экспорт",
        "text": "Это удобно для передачи результата: можно открыть предпросмотр, скачать отдельный файл или забрать весь пакет артефактов одним ZIP архивом.",
    },
    {
        "scene": "scene_panorama_tiling.png",
        "title": "Панорама",
        "text": "Теперь тот же v2 UI показывает панорамный шлиф. Большие изображения обрабатываются через анализируемое представление и тайловую схему.",
    },
    {
        "scene": "scene_panorama_tiling.png",
        "title": "Тайлы",
        "text": "Включённая сетка тайлов показывает, как panorama разбивается на участки с перекрытием без ручного кадрирования пользователем.",
    },
    {
        "scene": "scene_panorama_final.png",
        "title": "Результат панорамы",
        "text": "Панорамный run сохраняет такие же слои, доли классов и отчётные файлы, как обычный шлиф, но визуально работает с гораздо большим исходником.",
    },
    {
        "scene": "scene_panorama_metrics.png",
        "title": "Долгие ожидания",
        "text": "Если расчёт большой панорамы длится долго, демо открывает готовый run из истории. В реальной записи такой участок можно ускорять без потери смысла.",
    },
    {
        "scene": "scene_history.png",
        "title": "Журнал запусков",
        "text": "История хранит одиночные runs с датой, статусом, прогрессом, классификацией, долями сульфидов и талька, а также действиями загрузки и удаления.",
    },
    {
        "scene": "scene_history_current.png",
        "title": "Активные состояния",
        "text": "Если run ещё выполняется или остановлен, UI показывает текстовый статус, процент, этап и счётчик тайлов, чтобы пользователь не гадал, что происходит.",
    },
    {
        "scene": "scene_history_series_current.png",
        "title": "История серий",
        "text": "В том же разделе есть режим серий. Он показывает batch id, число изображений, итоговый статус, прогресс и быстрый переход к серии.",
    },
    {
        "scene": "scene_series_current.png",
        "title": "Серии",
        "text": "Страница «Серии» нужна для групповой обработки. Сюда добавляют несколько изображений, задают общие настройки и запускают последовательный расчёт.",
    },
    {
        "scene": "scene_series_current.png",
        "title": "Пакетный сценарий",
        "text": "Каждый элемент серии становится обычным immutable run, поэтому его можно открыть отдельно, выгрузить CSV или включить в общий результат серии.",
    },
    {
        "scene": "scene_settings_current.png",
        "title": "Настройки",
        "text": "Настройки сохраняются на сервере приложения. Здесь выбираются язык, тема, backend, checkpoint, источник талька и параметры предобработки по умолчанию.",
    },
    {
        "scene": "scene_settings_current.png",
        "title": "Runtime",
        "text": "Переключение backend применяется к новым запускам, а уже созданные runs остаются с тем checkpoint и runtime provenance, с которыми они были рассчитаны.",
    },
    {
        "scene": "scene_settings_current.png",
        "title": "Проверка модели",
        "text": "Кнопка проверки runtime позволяет валидировать ML checkpoint без создания run, поэтому проблемы окружения видны до начала обработки изображения.",
    },
    {
        "scene": "scene_status.png",
        "title": "Статус",
        "text": "Страница статуса показывает здоровье сервиса: backend, checkpoint, CPU, GPU или Metal, память, диск, размер workspace, активные задачи и журнал событий.",
    },
    {
        "scene": "scene_status.png",
        "title": "Операционный экран",
        "text": "Этот экран полезен во время демонстрации: он объясняет, чем занят сервер, сколько есть ресурсов и почему run может идти медленнее обычного.",
    },
    {
        "scene": "scene_api.png",
        "title": "REST API",
        "text": "API страница документирует те же операции, что доступны в браузере: загрузка изображений, старт run, опрос статуса, артефакты, серии и настройки.",
    },
    {
        "scene": "scene_api.png",
        "title": "Песочница",
        "text": "Встроенная песочница отправляет запросы на этот же сервер, поэтому интегратор может проверить контракт API без отдельного Postman сценария.",
    },
    {
        "scene": "scene_single_final.png",
        "title": "Итог",
        "text": "Итоговый сценарий остаётся внутри одного v2 UI: загрузка, настройка, запуск, слои, метрики, зерновой отчёт, экспорт, история, серии и API.",
    },
    {
        "scene": "scene_files_popup.png",
        "title": "Итог",
        "text": "Главная ценность интерфейса — воспроизводимый пакет: исходник, настройки, модели, маски, таблицы, отчёты и ограничения сохранены рядом с результатом.",
    },
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.FreeTypeFont, fill, width: int, line_gap: int = 8) -> int:
    x, y = xy
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += int(font.size * 1.25) + line_gap
    return y


def make_slide(path: Path, title: str, subtitle: str, bullets: list[str], accent=(0, 137, 145)) -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), (247, 250, 252))
    draw = ImageDraw.Draw(img)
    title_font = load_font(64, True)
    sub_font = load_font(34)
    bullet_font = load_font(34)
    small_font = load_font(24)
    draw.rectangle((0, 0, WIDTH, 110), fill=(21, 32, 43))
    draw.rectangle((0, 110, WIDTH, 124), fill=accent)
    draw.text((70, 28), title, font=title_font, fill=(255, 255, 255))
    y = 170
    y = draw_wrapped(draw, (80, y), subtitle, sub_font, (35, 45, 58), 1600, 10)
    y += 26
    for bullet in bullets:
        draw.rounded_rectangle((82, y + 8, 100, y + 26), radius=4, fill=accent)
        y = draw_wrapped(draw, (124, y), bullet, bullet_font, (25, 35, 48), 1600, 8) + 12
    draw.text((80, HEIGHT - 72), "v2 UI demo · Nornickel AI Science Hack · 2026-07-04", font=small_font, fill=(90, 104, 122))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def make_arch_slide(path: Path) -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), (247, 250, 252))
    draw = ImageDraw.Draw(img)
    title_font = load_font(56, True)
    box_font = load_font(28, True)
    metric_font = load_font(30)
    draw.text((70, 55), "Архитектура и измеримые метрики", font=title_font, fill=(21, 32, 43))
    steps = [
        "Изображение",
        "Сульфиды\nSegFormer-B2",
        "Компоненты\nи признаки",
        "Тальк\nSegFormer-B0",
        "Правило сорта\nи отчёт",
    ]
    x = 90
    y = 210
    for idx, step in enumerate(steps):
        draw.rounded_rectangle((x, y, x + 300, y + 140), radius=18, fill=(255, 255, 255), outline=(198, 210, 222), width=2)
        draw_wrapped(draw, (x + 28, y + 34), step, box_font, (21, 32, 43), 250, 4)
        if idx < len(steps) - 1:
            draw.line((x + 310, y + 70, x + 385, y + 70), fill=(0, 137, 145), width=6)
            draw.polygon([(x + 385, y + 70), (x + 365, y + 58), (x + 365, y + 82)], fill=(0, 137, 145))
        x += 375
    metrics = [
        "Сульфиды: IoU 0.974 · F1 0.987 · AUC 0.999 · HD95 23.6 px",
        "Сорт руды: macro F1 0.744 · accuracy 0.742 · macro AUC OVR 0.880",
        "Тальк: mean IoU 0.644 · mean F1 0.782 · 42 reviewed masks",
        "Оговорка: weak supervision и review-маски не называются экспертной истиной",
    ]
    yy = 470
    for m in metrics:
        draw.rounded_rectangle((110, yy, WIDTH - 110, yy + 72), radius=14, fill=(233, 249, 250), outline=(0, 137, 145))
        draw.text((140, yy + 18), m, font=metric_font, fill=(21, 32, 43))
        yy += 96
    img.save(path)


def create_slides() -> None:
    make_slide(
        OUT / "slide_title.png",
        "Скажи мне, кто твой шлиф",
        "Пятиминутное демо v2 UI: загрузка шлифа, маски, метрики, тальк, панорама и воспроизводимые артефакты.",
        [
            "Один optical microscopy image -> сорт руды и отчёт.",
            "Immutable runs: исходник, настройки, маски, CSV, JSON, PDF и ZIP.",
            "В ролике показываются реальные сохранённые runs из локального v2 приложения.",
        ],
    )
    make_slide(
        OUT / "slide_data_audit.png",
        "Данные и защита от утечек",
        "Официальные папки дают image-level labels, поэтому перед метриками нужен аудит.",
        [
            "56 групп дубликатов по содержимому.",
            "24 группы с конфликтующими метками.",
            "Оценка идёт на deconflicted split с dedupe и group-level разделением.",
            "Тальковые контуры трактуются как weak labels и проверяются в review UI.",
        ],
        accent=(191, 88, 31),
    )
    make_arch_slide(OUT / "slide_architecture.png")
    make_slide(
        OUT / "slide_repro.png",
        "Развёртывание и воспроизводимость",
        "Демо-пакет должен быть не только видимым в браузере, но и повторяемым.",
        [
            "Локальный/offline запуск; Docker для x86 и ARM64.",
            "Обычное изображение: холодный старт около 18.6 s; простой контейнера около 301 MiB RAM.",
            "Каждый run сохраняет runtime provenance, checkpoint paths, masks, metrics and reports.",
            "Ограничения явно проговариваются: weak labels, talc fraction calibration, expert validation.",
        ],
        accent=(75, 100, 160),
    )


def srt_time(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


async def synthesize_sentence(text: str, out_path: Path) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, VOICE, rate=TTS_RATE)
    await communicate.save(str(out_path))


async def synthesize_all() -> None:
    audio_dir = OUT / "tts_sentences"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(SENTENCES, 1):
        path = audio_dir / f"{idx:03d}.mp3"
        if not path.exists() or path.stat().st_size < 1024:
            await synthesize_sentence(item["text"], path)


def build_audio_and_subtitles() -> tuple[Path, list[dict]]:
    audio_dir = OUT / "tts_sentences"
    timeline: list[dict] = []
    start = 0.0
    concat = OUT / "audio_concat.txt"
    with concat.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(SENTENCES, 1):
            mp3 = audio_dir / f"{idx:03d}.mp3"
            dur = ffprobe_duration(mp3)
            end = start + dur
            timeline.append({**item, "index": idx, "start": start, "end": end, "duration": dur, "audio": str(mp3)})
            f.write(f"file '{mp3.resolve()}'\n")
            start = end
    narration = OUT / "narration_raw.mp3"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(narration)])
    raw_dur = ffprobe_duration(narration)
    scale = 1.0
    audio = OUT / "narration_300s.wav"
    if raw_dur > TARGET_SECONDS - 0.2:
        scale = (TARGET_SECONDS - 1.0) / raw_dur
        tempo = raw_dur / (TARGET_SECONDS - 1.0)
        run([
            "ffmpeg",
            "-y",
            "-i",
            str(narration),
            "-filter:a",
            f"atempo={tempo:.6f},apad,atrim=0:{TARGET_SECONDS:.3f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(audio),
        ])
    else:
        run([
            "ffmpeg",
            "-y",
            "-i",
            str(narration),
            "-filter:a",
            f"apad,atrim=0:{TARGET_SECONDS:.3f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(audio),
        ])
    if scale != 1.0:
        for item in timeline:
            item["start"] *= scale
            item["end"] *= scale
            item["duration"] *= scale
    srt = OUT / "subtitles_ru.srt"
    with srt.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(timeline, 1):
            text = item["text"]
            wrapped = "\n".join(textwrap.wrap(text, width=72))
            f.write(f"{idx}\n{srt_time(item['start'])} --> {srt_time(item['end'])}\n{wrapped}\n\n")
    script = OUT / "script_ru.md"
    with script.open("w", encoding="utf-8") as f:
        f.write("# Сценарий озвучки v2 UI-only demo\n\n")
        f.write(f"Голос: {VOICE}, rate {TTS_RATE}. Итоговая длительность видео: 5:00.\n\n")
        for item in timeline:
            f.write(f"## {item['index']:02d}. {item['title']} [{item['start']:.2f}-{item['end']:.2f}s]\n\n{item['text']}\n\n")
    (OUT / "timeline.json").write_text(json.dumps({"target_seconds": TARGET_SECONDS, "voice": VOICE, "rate": TTS_RATE, "items": timeline}, ensure_ascii=False, indent=2), encoding="utf-8")
    return audio, timeline


def fit_image(path: Path) -> Image.Image:
    src = Image.open(path).convert("RGB")
    sw, sh = src.size
    scale = max(WIDTH / sw, HEIGHT / sh)
    nw, nh = int(sw * scale + 0.5), int(sh * scale + 0.5)
    src = src.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - WIDTH) // 2
    top = (nh - HEIGHT) // 2
    return src.crop((left, top, left + WIDTH, top + HEIGHT))


def scene_for_time(t: float, timeline: list[dict]) -> dict:
    for item in timeline:
        if item["start"] <= t < item["end"]:
            return item
    return timeline[-1]


def render_video(timeline: list[dict]) -> Path:
    images = {p.name: fit_image(p) for p in OUT.glob("*.png") if p.name.startswith(("scene_", "slide_"))}
    title_font = load_font(26, True)
    small_font = load_font(22)
    subtitle_font = load_font(34)
    video = OUT / "video_with_subs.mp4"
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    total_frames = int(TARGET_SECONDS * FPS)
    for frame_idx in range(total_frames):
        t = frame_idx / FPS
        item = scene_for_time(t, timeline)
        base = images[item["scene"]].copy()
        overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        mm = int(t // 60)
        ss = int(t % 60)
        ff = int((t - int(t)) * 10)
        stamp = f"{mm:02}:{ss:02}.{ff}  ·  {item['title']}"
        draw.rounded_rectangle((WIDTH - 560, 18, WIDTH - 28, 64), radius=10, fill=(0, 0, 0, 150))
        draw.text((WIDTH - 540, 29), stamp, font=small_font, fill=(255, 255, 255, 245))
        draw.rounded_rectangle((28, 18, 374, 64), radius=10, fill=(0, 137, 145, 210))
        draw.text((48, 29), "v2 UI-only demo", font=title_font, fill=(255, 255, 255, 255))
        subtitle_lines = textwrap.wrap(item["text"], width=78)
        if subtitle_lines:
            line_h = 46
            box_h = 34 + line_h * len(subtitle_lines)
            box_y = HEIGHT - box_h - 26
            draw.rounded_rectangle((220, box_y, WIDTH - 220, HEIGHT - 26), radius=16, fill=(0, 0, 0, 170))
            y = box_y + 18
            for line in subtitle_lines:
                tw = draw.textbbox((0, 0), line, font=subtitle_font)[2]
                draw.text(((WIDTH - tw) // 2, y), line, font=subtitle_font, fill=(255, 255, 255, 255))
                y += line_h
        base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        proc.stdin.write(base.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg video render failed")
    return video


def mux_final(video: Path, audio: Path) -> Path:
    final = OUT / "nornikel_v2_ui_only_demo_1080p_ru.mp4"
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-t",
        f"{TARGET_SECONDS:.3f}",
        str(final),
    ])
    return final


def main() -> None:
    asyncio.run(synthesize_all())
    audio, timeline = build_audio_and_subtitles()
    video = render_video(timeline)
    final = mux_final(video, audio)
    meta = {
        "final_mp4": str(final),
        "duration_seconds": ffprobe_duration(final),
        "audio": str(audio),
        "subtitles": str(OUT / "subtitles_ru.srt"),
        "script": str(OUT / "script_ru.md"),
    }
    (OUT / "build_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
