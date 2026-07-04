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
        "scene": "slide_title.png",
        "title": "Постановка задачи",
        "text": "Это демонстрация v2 интерфейса решения «Скажи мне, кто твой шлиф».",
    },
    {
        "scene": "slide_title.png",
        "title": "Постановка задачи",
        "text": "Задача простая для пользователя и сложная для модели: по одному оптическому изображению определить сорт руды.",
    },
    {
        "scene": "scene_single_final.png",
        "title": "Пример шлифа",
        "text": "Мы различаем обычные сростки, труднообогатимые тонкие сросты и оталькованные руды, где важна доля талька.",
    },
    {
        "scene": "scene_single_sulfide.png",
        "title": "Что видит модель",
        "text": "На экране виден реальный run из истории v2 UI: исходный шлиф, маски и результат уже сохранены как неизменяемые артефакты.",
    },
    {
        "scene": "slide_data_audit.png",
        "title": "Данные без утечек",
        "text": "Сначала мы проверяем данные, потому что официальные папки размечены на уровне изображения, а не пиксельными масками.",
    },
    {
        "scene": "slide_data_audit.png",
        "title": "Аудит датасета",
        "text": "Аудит нашёл пятьдесят шесть групп дубликатов по содержимому и двадцать четыре группы с конфликтующими метками.",
    },
    {
        "scene": "slide_data_audit.png",
        "title": "Честная оценка",
        "text": "Такие изображения исключены из оценки, а split делается по образцу, чтобы одинаковый шлиф не попадал одновременно в обучение и проверку.",
    },
    {
        "scene": "slide_data_audit.png",
        "title": "Оговорка по разметке",
        "text": "Поэтому все метрики в ролике трактуются аккуратно: weak supervision помогает учиться, но не заменяет экспертную геологическую истину.",
    },
    {
        "scene": "slide_architecture.png",
        "title": "Пайплайн",
        "text": "Пайплайн начинается с бинарной сегментации сульфидов, затем считает связные компоненты и признаки каждого зерна.",
    },
    {
        "scene": "slide_architecture.png",
        "title": "Пайплайн",
        "text": "После этого отдельная ветка оценивает тальк, а финальный сорт получается из долей, морфологии и явных правил принятия решения.",
    },
    {
        "scene": "slide_architecture.png",
        "title": "Метрики сегментации",
        "text": "Для сульфидов текущий SegFormer-B2 даёт IoU ноль целых девятьсот семьдесят четыре тысячных, F1 ноль целых девятьсот восемьдесят семь тысячных и AUC ноль целых девятьсот девяносто девять тысячных.",
    },
    {
        "scene": "slide_architecture.png",
        "title": "Метрики классификации",
        "text": "Классификация сорта на deconflicted split из трёхсот сорока пяти изображений даёт macro F1 ноль целых семьсот сорок четыре тысячных и macro AUC ноль целых восемьсот восемьдесят тысячных.",
    },
    {
        "scene": "scene_history.png",
        "title": "История запусков",
        "text": "Теперь переходим в приложение: оно хранит историю запусков, и каждый run можно загрузить, не пересчитывая тяжёлый ML заново.",
    },
    {
        "scene": "scene_single_final.png",
        "title": "Одиночный шлиф",
        "text": "В рабочем месте оператор видит загруженный шлиф, выбранную предобработку и доступные слои результата.",
    },
    {
        "scene": "scene_single_final.png",
        "title": "Финальная маска",
        "text": "В финальном слое зелёный цвет показывает обычные сростания, красный — тонкие, синий — тальк, а фон остаётся отдельным классом.",
    },
    {
        "scene": "scene_single_sulfide.png",
        "title": "Сульфидный слой",
        "text": "Переключаемся на слой сульфидов: можно менять прозрачность, включать контуры и быстро сверять маску с исходным изображением.",
    },
    {
        "scene": "scene_single_sulfide.png",
        "title": "Артефакты подготовки",
        "text": "Кнопка «Исправить» открывает ручную разметку артефактов шлифовки; отмеченные области исключаются из знаменателя метрик.",
    },
    {
        "scene": "scene_metrics.png",
        "title": "Результат",
        "text": "Ниже приложение пишет человекочитаемый вывод: сорт, долю сульфидов, долю талька и объяснение, почему сработало именно это правило.",
    },
    {
        "scene": "scene_metrics.png",
        "title": "Таблица метрик",
        "text": "Таблица метрик показывает проанализированную область, общую долю сульфидов, обычные и тонкие срастания, тальк и артефакты.",
    },
    {
        "scene": "scene_metrics.png",
        "title": "Компоненты",
        "text": "Если в метаданных задан калиброванный масштаб, в CSV и отчёте дополнительно появляются физические площади в квадратных микрометрах и миллиметрах.",
    },
    {
        "scene": "scene_grains_files.png",
        "title": "Зерновой отчёт",
        "text": "Отдельная таблица сульфидных зёрен показывает площадь, эквивалентный диаметр, периметр, долю сульфидов и proxy-показатели освобождения.",
    },
    {
        "scene": "scene_grains_files.png",
        "title": "Контроль зерен",
        "text": "Выбранные строки можно подсветить на изображении, чтобы эксперт быстро проверил, какие компоненты влияют на решение.",
    },
    {
        "scene": "scene_files_popup.png",
        "title": "Экспорт",
        "text": "Все файлы запуска открываются из встроенного списка: изображения, CSV, JSON, PDF-отчёт и ZIP-архив остаются связаны с тем же run id.",
    },
    {
        "scene": "scene_talc_2550381.png",
        "title": "Ревью талька",
        "text": "Тальк показываем отдельно, через приложение ревью масок, потому что исходные синие контуры — это примерные области, а не полная пиксельная истина.",
    },
    {
        "scene": "scene_talc_2550381.png",
        "title": "Проверенные маски",
        "text": "Здесь видны positive bag, подтверждённый talc node, защита от сульфидов и статус reviewed для образца.",
    },
    {
        "scene": "scene_talc_2550381.png",
        "title": "Метрики талька",
        "text": "После ручного ревью сорока двух образцов талковая SegFormer-B0 ветка даёт mean IoU ноль целых шестьсот сорок четыре тысячных и F1 ноль целых семьсот восемьдесят две тысячных.",
    },
    {
        "scene": "scene_talc_2550381.png",
        "title": "Оговорка по тальку",
        "text": "Важная оговорка: примерно восемьдесят три процента проверенного талька оказалось вне исходных синих мешков, поэтому мы не завышаем точность доли талька.",
    },
    {
        "scene": "scene_panorama_tiling.png",
        "title": "Панорама",
        "text": "Для больших панорам v2 UI использует анализ в уменьшенном представлении и тайловую обработку с перекрытием.",
    },
    {
        "scene": "scene_panorama_tiling.png",
        "title": "Тайлы",
        "text": "На кадре включена сетка тайлов: так видно, как большая панорама разбивается на участки без ручного кадрирования пользователем.",
    },
    {
        "scene": "scene_panorama_metrics.png",
        "title": "Панорамный результат",
        "text": "Панорамный run также сохраняет сорт, доли классов, маски и отчётные файлы, но долгий расчёт для демо открывается из истории и не тратит минуты в кадре.",
    },
    {
        "scene": "scene_status.png",
        "title": "Статус системы",
        "text": "Страница статуса показывает активный backend, checkpoint, состояние CPU, памяти, диска, историю запусков и предупреждения по среде.",
    },
    {
        "scene": "scene_api.png",
        "title": "REST API",
        "text": "Для интеграции есть REST API: загрузка изображений, запуск, опрос статуса, артефакты, настройки, серии и health checks.",
    },
    {
        "scene": "slide_repro.png",
        "title": "Воспроизводимость",
        "text": "Решение разворачивается локально и офлайн через Docker для x86 и ARM64; обычное изображение стартует примерно за восемнадцать целых шесть десятых секунды.",
    },
    {
        "scene": "slide_repro.png",
        "title": "Финальный вывод",
        "text": "Главная идея v2 UI — не просто показать красивую маску, а сохранить воспроизводимый пакет: исходник, настройки, модели, метрики, отчёты и ограничения.",
    },
    {
        "scene": "slide_repro.png",
        "title": "Финальный вывод",
        "text": "Следующий инженерный шаг — экспертная валидация масок талька и калибровка правила сорта, но текущий pipeline уже готов для понятного судейского демо.",
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
        f.write("# Сценарий озвучки v2 UI demo\n\n")
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
        draw.text((48, 29), "v2 UI pipeline demo", font=title_font, fill=(255, 255, 255, 255))
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
    final = OUT / "nornikel_v2_ui_demo_1080p_ru.mp4"
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
    create_slides()
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
