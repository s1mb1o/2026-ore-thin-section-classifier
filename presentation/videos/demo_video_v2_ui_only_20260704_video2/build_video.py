#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent
SCRIPT = OUT / "script.md"
WIDTH = 1920
HEIGHT = 1080
FPS = 24
LANGUAGE_CODE = "ru"
TTS_BACKEND = "edge_tts"
VOICE = "ru-RU-DmitryNeural"
TTS_RATE = "-12%"
OUTPUT_STEM = "nornikel_v2_ui_video2_1080p_ru"
FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
if not FONT_BOLD.exists():
    FONT_BOLD = FONT_REGULAR


RU_SCENES = {
    0: {
        "title": "Подготовка",
        "text": (
            "В этом видео показан основной интерфейс v2 для рудного пайплайна на одном "
            "реальном официальном образце. Мы отключаем аугментацию и предобработку, "
            "чтобы путь от изображения к маскам, метрикам и неизменяемым артефактам "
            "запуска было легко проверить."
        ),
    },
    1: {
        "title": "История как точка входа",
        "text": (
            "История удобна как стартовая точка записи. Каждый результат является "
            "неизменяемым запуском, поэтому мы можем открыть точно то же состояние "
            "без повторного расчета пайплайна."
        ),
    },
    2: {
        "title": "Открытие изображения в Workspace",
        "text": (
            "В рабочей области слева находится входное изображение, а справа - просмотрщик. "
            "Обратите внимание: аугментация и предобработка выключены. Соответствующие "
            "слои недоступны, потому что этот запуск намеренно их пропускает."
        ),
    },
    3: {
        "title": "Редактирование метаданных",
        "text": (
            "Метаданные уточняются до запуска и сохраняются вместе с результатом. "
            "В этом же окне видны сырые метаданные файла, поэтому ревьюер отделяет "
            "ввод оператора от фактов, извлеченных из изображения."
        ),
    },
    4: {
        "title": "Конфигурация",
        "text": (
            "Конфигурация применяется к следующему запуску. Здесь можно выбрать backend "
            "для сегментации сульфидов, определения талька и классификации зерен, "
            "не переписывая старые результаты. Завершенные запуски сохраняют свой provenance."
        ),
    },
    5: {
        "title": "Сквозной запуск пайплайна",
        "text": (
            "Кнопка Start создает неизменяемый запуск. Этот образец классифицирован как "
            "труднообогатимая руда: доля талька значительно ниже порога десять процентов, "
            "а доля тонких сростков немного выше доли обычных."
        ),
    },
    6: {
        "title": "Текстовый вывод и метрики",
        "text": (
            "Интерфейс не просто рисует маску. Он записывает краткое обоснование решения "
            "и показывает численную основу: анализируемую площадь, долю сульфидов, "
            "обычные и тонкие сростки, долю талька, площади кластеров и артефакты изображения."
        ),
    },
    7: {
        "title": "Таблица сульфидных зерен",
        "text": (
            "Таблица зерен раскладывает сульфидную маску на связные компоненты. "
            "Каждая строка - это зерновой proxy-отчет, а не химическое утверждение: "
            "площадь, форма, контакты, proxy раскрытия и признак locked или composite."
        ),
    },
    8: {
        "title": "Выбранное зерно обведено в просмотрщике",
        "text": (
            "Отмеченная строка зерна обводится прямо в просмотрщике. Это связь между "
            "табличными метриками и реальной областью изображения, из которой они получены."
        ),
    },
    9: {
        "title": "Интерфейс Fix Me",
        "text": (
            "Fix Me - это поверхность корректировки. В этом эпизоде используется слой "
            "Artefacts: царапину или дефект полировки можно исключить из анализируемой "
            "области до пересчета последующих долей."
        ),
    },
    10: {
        "title": "Fix and Restart создает новый запуск",
        "text": (
            "Коррекция никогда не перезаписывает исходный запуск. Fix and Restart создает "
            "новый запуск с ссылкой на родителя и записанной операцией правки. Старый "
            "запуск остается доступным для аудита и сравнения."
        ),
    },
    11: {
        "title": "Правка артефакта меняет дальнейшие метрики",
        "text": (
            "Поскольку маска артефакта меняет анализируемую область и маски, меняются "
            "и доли. В эталонном запуске доля артефакта изображения становится около "
            "ноль целых семь десятых процента; доли сульфидов и талька пересчитаны в дочернем запуске."
        ),
    },
    12: {
        "title": "Сравнение бок о бок",
        "text": (
            "Режим side-by-side быстрее всего объясняет, как итоговый слой классов связан "
            "с исходной сульфидной маской. Разделитель удерживает оба вида в одной "
            "системе координат изображения."
        ),
    },
    13: {
        "title": "Технические детали запуска",
        "text": (
            "Технические детали - это панель provenance. Результат в просмотрщике не "
            "является только скриншотом: id запуска, runtime, источник модели, tiling, "
            "маски, отчеты и runtime JSON привязаны к сохраненному запуску."
        ),
    },
    14: {
        "title": "Страница Series",
        "text": (
            "Series - это пакетный workflow. Каждый элемент становится таким же "
            "неизменяемым запуском, как одиночный сценарий, поэтому результат серии "
            "можно открыть и проверить по каждому изображению отдельно."
        ),
    },
    15: {
        "title": "Страница Status",
        "text": (
            "Страница Status - это операционный экран. Она показывает докладчику и "
            "оператору активный backend, доступность checkpoint-файлов, ресурсы машины "
            "и наличие выполняющихся запусков."
        ),
    },
    16: {
        "title": "Страница API - только открыть",
        "text": (
            "Страница API документирует те же операции, что доступны в браузере: upload, "
            "start, poll run, чтение артефактов, Series, settings и проверки runtime. "
            "В этом видео мы только открываем документацию и не запускаем playground."
        ),
    },
    17: {
        "title": "Страница Settings",
        "text": (
            "Settings - это серверные значения по умолчанию для новой работы. Они не "
            "переписывают завершенные запуски. Это разделение важно: ревьюер может "
            "изменить следующую конфигурацию, а provenance старых запусков останется неизменным."
        ),
    },
    18: {
        "title": "Финальный кадр",
        "text": (
            "Вся демонстрация остается внутри одного приложения: изображение, метаданные, "
            "конфигурация, запуск, маски, метрики, зерновые доказательства, Fix and Restart, "
            "Series, Status, API docs и Settings. Результат воспроизводим, потому что "
            "каждый итог сохранен как запуск с артефактами и provenance."
        ),
    },
}


@dataclass
class Scene:
    index: int
    title: str
    start: float
    end: float
    screenshot: str | None
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


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


def parse_timecode(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Unsupported timecode: {value}")
    minutes, seconds = parts
    return int(minutes) * 60 + int(seconds)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def fit_font_for_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    size: int,
    bold: bool = False,
    min_size: int = 18,
) -> ImageFont.FreeTypeFont:
    font = load_font(size, bold)
    while size > min_size and text_width(draw, text, font) > max_width:
        size -= 1
        font = load_font(size, bold)
    return font


def wrap_text_pixels(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if not current or text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def clean_text(value: str) -> str:
    return " ".join(value.replace("\n", " ").split())


def parse_script() -> list[Scene]:
    text = SCRIPT.read_text(encoding="utf-8")
    heading_re = re.compile(r"^## (\d+)\. (.+?) \[(\d\d:\d\d)-(\d\d:\d\d)\]\s*$", re.MULTILINE)
    matches = list(heading_re.finditer(text))
    scenes: list[Scene] = []
    for pos, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[pos + 1].start() if pos + 1 < len(matches) else len(text)
        body = text[body_start:body_end]
        shot_match = re.search(r"capture_filename:\s*(.+)", body)
        screenshot = None
        if shot_match:
            raw = shot_match.group(1).strip()
            if raw != "no screenshot":
                screenshot = raw
        narration_match = re.search(r'Narration:\s*"(.+?)"', body, re.DOTALL)
        if not narration_match:
            raise ValueError(f"Missing narration for scene {match.group(1)}")
        scene_index = int(match.group(1))
        override = RU_SCENES.get(scene_index, {})
        scenes.append(
            Scene(
                index=scene_index,
                title=override.get("title", match.group(2).strip()),
                start=parse_timecode(match.group(3)),
                end=parse_timecode(match.group(4)),
                screenshot=screenshot,
                text=clean_text(override.get("text", narration_match.group(1))),
            )
        )
    if not scenes:
        raise ValueError("No scenes parsed from script.md")
    return scenes


def draw_wrapped_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    y: int,
    max_width: int,
    fill: tuple[int, int, int, int],
    line_height: int,
) -> int:
    lines = textwrap.wrap(text, width=82)
    for line in lines:
        while draw.textbbox((0, 0), line, font=font)[2] > max_width and len(line) > 12:
            wrapped = textwrap.wrap(line, width=max(12, int(len(line) * 0.85)))
            if len(wrapped) <= 1:
                break
            lines = wrapped + lines[lines.index(line) + 1 :]
            line = lines[0]
            break
        width = draw.textbbox((0, 0), line, font=font)[2]
        draw.text(((WIDTH - width) // 2, y), line, font=font, fill=fill)
        y += line_height
    return y


def title_slide(scene: Scene) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), (245, 248, 250))
    draw = ImageDraw.Draw(img)
    title_font = load_font(66, True)
    subtitle_font = load_font(34)
    small_font = load_font(25)
    accent = (0, 137, 145)
    draw.rectangle((0, 0, WIDTH, 126), fill=(19, 30, 41))
    draw.rectangle((0, 126, WIDTH, 142), fill=accent)
    draw.text((72, 34), "ВИДЕО #2 - основной UI v2 ore pipeline", font=title_font, fill=(255, 255, 255))
    body = [
        "Пошаговый разбор apps/ore_pipeline_web.py",
        "Образец: 2550382-1 10x.JPG",
        "Augmentation: выключено",
        "Preprocessing: выключено",
        "Parent run, Fix and Restart child run, Series, Status, API и Settings",
    ]
    y = 230
    for item in body:
        draw.rounded_rectangle((96, y + 8, 116, y + 28), radius=4, fill=accent)
        for line in wrap_text_pixels(draw, item, subtitle_font, WIDTH - 260):
            draw.text((144, y), line, font=subtitle_font, fill=(24, 35, 48))
            y += 44
        y += 28
    draw.text((72, HEIGHT - 82), f"Сцена {scene.index:02d}: {scene.title}", font=small_font, fill=(85, 98, 115))
    return img


def load_scene_base(scene: Scene) -> Image.Image:
    if scene.screenshot is None:
        return title_slide(scene)
    path = OUT / scene.screenshot
    img = Image.open(path).convert("RGB")
    if img.size != (WIDTH, HEIGHT):
        raise ValueError(f"{path} has size {img.size}, expected {(WIDTH, HEIGHT)}")
    return img


def render_scene_frame(scene: Scene, out_path: Path) -> None:
    base = load_scene_base(scene).convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    small_font = load_font(22)
    subtitle_font = load_font(34)
    top_time = f"{format_srt_time(scene.start).replace(',', '.')[:8]} - {format_srt_time(scene.end).replace(',', '.')[:8]}"
    time_width = text_width(draw, top_time, small_font)
    top_label = f"ВИДЕО #2  |  {scene.index:02d}. {scene.title}"
    max_label_width = WIDTH - time_width - 150
    title_font = fit_font_for_width(draw, top_label, max_label_width, 28, True, 20)
    label_width = min(text_width(draw, top_label, title_font), max_label_width)
    draw.rounded_rectangle((22, 18, 64 + label_width, 70), radius=10, fill=(0, 0, 0, 155))
    draw.text((42, 31), top_label, font=title_font, fill=(255, 255, 255, 245))
    draw.rounded_rectangle((WIDTH - time_width - 66, 18, WIDTH - 22, 70), radius=10, fill=(0, 0, 0, 145))
    draw.text((WIDTH - time_width - 44, 34), top_time, font=small_font, fill=(255, 255, 255, 240))
    lines = wrap_text_pixels(draw, scene.text, subtitle_font, WIDTH - 460)
    line_height = 45
    box_h = 32 + line_height * len(lines)
    box_y = HEIGHT - box_h - 24
    draw.rounded_rectangle((190, box_y, WIDTH - 190, HEIGHT - 24), radius=16, fill=(0, 0, 0, 172))
    y = box_y + 18
    for line in lines:
        line_width = draw.textbbox((0, 0), line, font=subtitle_font)[2]
        draw.text(((WIDTH - line_width) // 2, y), line, font=subtitle_font, fill=(255, 255, 255, 255))
        y += line_height
    Image.alpha_composite(base, overlay).convert("RGB").save(out_path)


def format_srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3_600_000
    total_ms %= 3_600_000
    m = total_ms // 60_000
    total_ms %= 60_000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def build_subtitles(scenes: list[Scene]) -> Path:
    path = OUT / f"subtitles_{LANGUAGE_CODE}.srt"
    with path.open("w", encoding="utf-8") as handle:
        for idx, scene in enumerate(scenes, 1):
            wrapped = "\n".join(textwrap.wrap(scene.text, width=78))
            handle.write(
                f"{idx}\n"
                f"{format_srt_time(scene.start)} --> {format_srt_time(scene.end)}\n"
                f"{wrapped}\n\n"
            )
    return path


async def synthesize_sentence(text: str, out_path: Path) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, VOICE, rate=TTS_RATE)
    await communicate.save(str(out_path))


def synthesize_sentence_with_retries(text: str, out_path: Path, attempts: int = 5) -> None:
    for attempt in range(1, attempts + 1):
        try:
            asyncio.run(synthesize_sentence(text, out_path))
            return
        except Exception:
            if out_path.exists():
                out_path.unlink()
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)


def atempo_filter(tempo: float) -> str:
    parts: list[str] = []
    while tempo > 2.0:
        parts.append("atempo=2.000000")
        tempo /= 2.0
    while tempo < 0.5:
        parts.append("atempo=0.500000")
        tempo /= 0.5
    parts.append(f"atempo={tempo:.6f}")
    return ",".join(parts)


def synthesize_audio(scenes: list[Scene]) -> tuple[Path, list[dict]]:
    raw_dir = OUT / f"tts_sentences_{LANGUAGE_CODE}_dmitry"
    seg_dir = OUT / f"audio_segments_{LANGUAGE_CODE}"
    raw_dir.mkdir(exist_ok=True)
    seg_dir.mkdir(exist_ok=True)
    concat = OUT / f"audio_concat_{LANGUAGE_CODE}.txt"
    timeline: list[dict] = []
    with concat.open("w", encoding="utf-8") as concat_handle:
        for scene in scenes:
            mp3 = raw_dir / f"{scene.index:02d}.mp3"
            wav = seg_dir / f"{scene.index:02d}.wav"
            if not mp3.exists() or mp3.stat().st_size < 1024:
                synthesize_sentence_with_retries(scene.text, mp3)
            raw_duration = ffprobe_duration(mp3)
            audio_filter = f"apad,atrim=0:{scene.duration:.3f}"
            segment_tempo = 1.0
            if raw_duration > scene.duration - 0.1:
                segment_tempo = raw_duration / max(0.5, scene.duration - 0.1)
                audio_filter = f"{atempo_filter(segment_tempo)},apad,atrim=0:{scene.duration:.3f}"
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(mp3),
                    "-filter:a",
                    audio_filter,
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    str(wav),
                ]
            )
            concat_handle.write(f"file '{wav.resolve()}'\n")
            timeline.append(
                {
                    "index": scene.index,
                    "title": scene.title,
                    "start": scene.start,
                    "end": scene.end,
                    "duration": scene.duration,
                    "text": scene.text,
                    "screenshot": scene.screenshot,
                    "raw_audio_seconds": raw_duration,
                    "segment_tempo": segment_tempo,
                    "audio": str(wav),
                }
            )
    narration = OUT / f"narration_480s_{LANGUAGE_CODE}.wav"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(narration)])
    return narration, timeline


def render_video_frames(scenes: list[Scene]) -> tuple[Path, list[Path]]:
    frames_dir = OUT / f"video_frames_{LANGUAGE_CODE}"
    frames_dir.mkdir(exist_ok=True)
    frame_paths: list[Path] = []
    for scene in scenes:
        path = frames_dir / f"scene_{scene.index:02d}.png"
        render_scene_frame(scene, path)
        frame_paths.append(path)
    concat = OUT / f"video_concat_{LANGUAGE_CODE}.txt"
    with concat.open("w", encoding="utf-8") as handle:
        for scene, frame in zip(scenes, frame_paths):
            handle.write(f"file '{frame.resolve()}'\n")
            handle.write(f"duration {scene.duration:.3f}\n")
        handle.write(f"file '{frame_paths[-1].resolve()}'\n")
    return concat, frame_paths


def render_contact_sheet(frame_paths: list[Path]) -> Path:
    thumbs = [Image.open(path).convert("RGB").resize((480, 270), Image.Resampling.LANCZOS) for path in frame_paths]
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    gap = 16
    label_h = 34
    sheet = Image.new("RGB", (cols * 480 + (cols + 1) * gap, rows * (270 + label_h) + (rows + 1) * gap), (242, 245, 248))
    draw = ImageDraw.Draw(sheet)
    font = load_font(18, True)
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = gap + col * (480 + gap)
        y = gap + row * (270 + label_h + gap)
        draw.text((x, y + 6), f"{idx:02d}", font=font, fill=(20, 31, 43))
        sheet.paste(thumb, (x, y + label_h))
    out = OUT / f"contact_sheet_{LANGUAGE_CODE}.jpg"
    sheet.save(out, quality=90)
    return out


def mux_video(concat: Path, audio: Path, total_seconds: float) -> Path:
    draft = OUT / f"video_with_subs_{LANGUAGE_CODE}.mp4"
    final = OUT / f"{OUTPUT_STEM}.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(draft),
        ]
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(draft),
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
            f"{total_seconds:.3f}",
            str(final),
        ]
    )
    return final


def main() -> None:
    scenes = parse_script()
    total_seconds = scenes[-1].end
    subtitles = build_subtitles(scenes)
    audio, timeline = synthesize_audio(scenes)
    concat, frame_paths = render_video_frames(scenes)
    contact_sheet = render_contact_sheet(frame_paths)
    final = mux_video(concat, audio, total_seconds)
    metadata = {
        "final_mp4": str(final),
        "duration_seconds": ffprobe_duration(final),
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "tts_backend": TTS_BACKEND,
        "voice": VOICE,
        "tts_rate": TTS_RATE,
        "language": LANGUAGE_CODE,
        "audio": str(audio),
        "subtitles": str(subtitles),
        "script": str(SCRIPT),
        "timeline": str(OUT / f"timeline_{LANGUAGE_CODE}.json"),
        "contact_sheet": str(contact_sheet),
    }
    (OUT / f"timeline_{LANGUAGE_CODE}.json").write_text(json.dumps({"target_seconds": total_seconds, "items": timeline}, indent=2), encoding="utf-8")
    (OUT / f"build_metadata_{LANGUAGE_CODE}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
