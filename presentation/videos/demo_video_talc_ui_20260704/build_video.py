#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import subprocess
import textwrap
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WIDTH = 1920
HEIGHT = 1080
FPS = 24
TARGET_SECONDS = 300.0
NARRATION_SECONDS = 299.0
TTS_BACKEND = os.environ.get("TTS_BACKEND", "edge").strip().lower()
VOICE = os.environ.get("TTS_VOICE", "ru-RU-DmitryNeural")
TTS_RATE = os.environ.get("TTS_RATE", "-5%")
SILERO_SPEAKER = os.environ.get("SILERO_SPEAKER", "aidar")
SILERO_SAMPLE_RATE = int(os.environ.get("SILERO_SAMPLE_RATE", "48000"))


def first_existing(paths: list[str]) -> Path:
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of the font paths exists: {paths}")


FONT_REGULAR = first_existing(
    [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
)
FONT_BOLD = first_existing(
    [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        str(FONT_REGULAR),
    ]
)


SENTENCES = [
    {
        "scene": "scene_00_intro_original.png",
        "title": "Вступление",
        "text": "Это отдельное рабочее место для проверки и аккуратной правки масок талька.",
    },
    {
        "scene": "scene_00_intro_original.png",
        "title": "Синие линии",
        "text": "В исходных данных зоны оталькования размечены вручную: синими линиями поверх фотографии шлифа.",
    },
    {
        "scene": "scene_00_intro_original.png",
        "title": "Почему нужна проверка",
        "text": "Такая разметка помогает стартовать, но это еще не точная маска. Контуры бывают открытыми, иногда попадают сульфиды, а сам тальк лежит пятнами и прожилками.",
    },
    {
        "scene": "scene_00_intro_original.png",
        "title": "Рабочий процесс",
        "text": "Поэтому инструмент сначала строит черновую область, а человек быстро уточняет ее прямо в браузере.",
    },
    {
        "scene": "scene_01_ms_paint_annotation.png",
        "title": "Фоны для проверки",
        "text": "В верхнем списке можно переключать фоны: чистая фотография, исходная картинка с синими линиями и проверочный слой конвертера.",
    },
    {
        "scene": "scene_02_converter_qa_overlay.png",
        "title": "Проверочный слой",
        "text": "Проверочный слой показывает, как алгоритм замкнул линии, какие зоны считает стартовыми и где видит пересечение с сульфидами.",
    },
    {
        "scene": "scene_02_converter_qa_overlay.png",
        "title": "Редактируем маску",
        "text": "Правим не сам рисунок с линиями. Работа идет с маской, которая потом попадет в обучение и в расчет доли талька.",
    },
    {
        "scene": "scene_03_classes_clusters.png",
        "title": "Виджет классов",
        "text": "Слева поверх изображения находится виджет классов и видимости слоев.",
    },
    {
        "scene": "scene_03_classes_clusters.png",
        "title": "Черновая область",
        "text": "Первый класс задает положительную область-кандидат: где тальк ожидается. Это границы поиска, а не утверждение, что каждый пиксель внутри уже тальк.",
    },
    {
        "scene": "scene_03_classes_clusters.png",
        "title": "Подтвержденный тальк",
        "text": "Второй класс это подтвержденный тальк. Именно эти пиксели считаются надежной разметкой для модели и метрик.",
    },
    {
        "scene": "scene_03_classes_clusters.png",
        "title": "Скопления талька",
        "text": "Отдельная подсветка показывает локальные скопления талька. Это подсказка для обзора, а не новый обучающий класс.",
    },
    {
        "scene": "scene_03_classes_clusters.png",
        "title": "Проценты",
        "text": "Рядом сразу видны проценты по изображению. Нижний статус напоминает, близка ли разметка к порогу оталькованной руды.",
    },
    {
        "scene": "scene_04_brush_talc_target.png",
        "title": "Цель редактирования",
        "text": "Перед правкой выбираем активный класс. Если выбран подтвержденный тальк, кисть добавляет именно его.",
    },
    {
        "scene": "scene_04_brush_talc_target.png",
        "title": "Кисть",
        "text": "Левая кнопка мыши рисует, правая сразу стирает; отдельный ластик не нужен.",
    },
    {
        "scene": "scene_04_brush_talc_target.png",
        "title": "Размер кисти",
        "text": "Размер кисти меняется в панели инструментов, и круг под курсором показывает фактическую область рисования.",
    },
    {
        "scene": "scene_05_fill_tool.png",
        "title": "Заливка",
        "text": "Для замкнутых или почти замкнутых участков удобна заливка. Она растет до синей линии, текущей маски, сульфида или края изображения.",
    },
    {
        "scene": "scene_06_rectangle_tool.png",
        "title": "Фигуры",
        "text": "Крупные фрагменты проще наметить прямоугольником или полигоном.",
    },
    {
        "scene": "scene_06_rectangle_tool.png",
        "title": "Заполненные области",
        "text": "Это не обводка. Инструмент сразу создает заполненную область выбранного класса.",
    },
    {
        "scene": "scene_06_rectangle_tool.png",
        "title": "Правка прямоугольника",
        "text": "Прямоугольник можно растянуть или сдвинуть, потянув за углы и стороны.",
    },
    {
        "scene": "scene_07_polygon_tool.png",
        "title": "Полигон",
        "text": "Полигон строится точками: чтобы завершить фигуру, кликаем по первой точке.",
    },
    {
        "scene": "scene_07_polygon_tool.png",
        "title": "Редактирование фигур",
        "text": "Пока маска не сохранена и образец не сменился, фигуры остаются редактируемыми.",
    },
    {
        "scene": "scene_07_polygon_tool.png",
        "title": "Пиксельная маска",
        "text": "Можно двигать вершины, добавлять и убирать точки. При сохранении все превращается в обычную пиксельную маску.",
    },
    {
        "scene": "scene_08_similar_preview.png",
        "title": "Похожие пиксели",
        "text": "Если конкретное темное зерно выглядит как тальк, включаем поиск похожих пикселей.",
    },
    {
        "scene": "scene_08_similar_preview.png",
        "title": "Предпросмотр",
        "text": "Инструмент берет цвет и яркость вокруг клика и сразу показывает похожие места по всему изображению.",
    },
    {
        "scene": "scene_08_similar_preview.png",
        "title": "Применение подсказки",
        "text": "Это только предварительный просмотр. Маска изменится только после применения подсказки или сохранения активного результата.",
    },
    {
        "scene": "scene_08_similar_preview.png",
        "title": "Строгость поиска",
        "text": "Строгость регулирует, насколько узко искать похожие пиксели и сколько кандидатов попадет в подсказку.",
    },
    {
        "scene": "scene_09_sam2_dark_preview.png",
        "title": "Нейроассистент",
        "text": "Еще есть нейросетевой помощник: он может предложить область по точке или прямоугольнику, если модель подключена.",
    },
    {
        "scene": "scene_09_sam2_dark_preview.png",
        "title": "Темные пиксели",
        "text": "Ползунок предпросмотра темных пикселей тоже не меняет разметку. Он скрывает яркие участки и показывает, сколько изображения остается темным кандидатом.",
    },
    {
        "scene": "scene_10_quality_sulfides_clusters.png",
        "title": "Сульфиды",
        "text": "Главное правило качества: тальк не должен автоматически попадать в сульфидные зерна.",
    },
    {
        "scene": "scene_10_quality_sulfides_clusters.png",
        "title": "Защита от сульфидов",
        "text": "Поэтому защита включена по умолчанию. Все новые пиксели талька обрезаются по сульфидной маске.",
    },
    {
        "scene": "scene_10_quality_sulfides_clusters.png",
        "title": "Вычитание сульфидов",
        "text": "Если пересечение уже было в старой маске, отдельная команда вычитает сульфиды из текущей разметки.",
    },
    {
        "scene": "scene_10_quality_sulfides_clusters.png",
        "title": "Скопления",
        "text": "Для финального обзора включаем подсветку скоплений: она помогает увидеть, где тальк расположен кучно, и не создает новый обучающий класс.",
    },
    {
        "scene": "scene_11_save_controls_result.png",
        "title": "Сохранение",
        "text": "После правки нажимаем кнопку сохранения.",
    },
    {
        "scene": "scene_11_save_controls_result.png",
        "title": "Следующий образец",
        "text": "Если нужно сразу идти дальше, сохраняем и переходим к следующему. Отдельная кнопка перехода просто открывает следующий образец без записи изменений.",
    },
    {
        "scene": "slide_reviewed_masks.png",
        "title": "Проверенные маски",
        "text": "На выходе остаются три проверенных результата: область-кандидат, подтвержденный тальк и объединенная совместимая маска.",
    },
    {
        "scene": "slide_reviewed_masks.png",
        "title": "Итог",
        "text": "Эти маски идут дальше в обучение модели талька и в честный расчет доли оталькования по всему изображению.",
    },
]


SILERO_MODEL = None


HIGHLIGHTS = {
    "scene_01_ms_paint_annotation.png": [(1634, 120, 1892, 177, "Background: MS Paint annotation")],
    "scene_02_converter_qa_overlay.png": [(1634, 120, 1892, 177, "Background: Converter QA overlay")],
    "scene_03_classes_clusters.png": [(304, 120, 590, 300, "Segmentation classes")],
    "scene_04_brush_talc_target.png": [(552, 10, 796, 88, "Brush + Talc target"), (304, 120, 590, 300, "Talc edit radio")],
    "scene_05_fill_tool.png": [(620, 10, 672, 47, "Fill")],
    "scene_06_rectangle_tool.png": [(748, 10, 844, 47, "Rectangle")],
    "scene_07_polygon_tool.png": [(848, 10, 932, 47, "Polygon")],
    "scene_08_similar_preview.png": [(672, 10, 744, 47, "Similar preview")],
    "scene_09_sam2_dark_preview.png": [(938, 10, 1005, 47, "SAM2"), (1634, 196, 1892, 398, "Dark pixel preview")],
    "scene_10_quality_sulfides_clusters.png": [(1634, 407, 1892, 903, "Clusters + sulfide protection")],
    "scene_11_save_controls_result.png": [(1348, 10, 1537, 47, "Save / Save & Next"), (1634, 927, 1892, 1070, "Mask counts")],
}


def ffmpeg_exe() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg not found; install ffmpeg or imageio-ffmpeg") from exc


def run(cmd: list[str]) -> None:
    if cmd and cmd[0] == "ffmpeg":
        cmd = [ffmpeg_exe(), *cmd[1:]]
    subprocess.run(cmd, check=True)


def ffprobe_duration(path: Path) -> float:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        try:
            import imageio_ffmpeg

            _frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
            return float(seconds)
        except Exception as exc:
            raise RuntimeError(f"ffprobe not found and fallback duration failed for {path}") from exc

    out = subprocess.check_output(
        [
            ffprobe,
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


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill,
    width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    lines: list[str] = []
    line = ""
    for word in text.split():
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
        y += int(font.size * 1.18) + line_gap
    return y


def create_reviewed_masks_slide() -> None:
    bg = Image.open(OUT / "scene_11_save_controls_result.png").convert("RGB")
    bg = bg.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    veil = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 135))
    img = Image.alpha_composite(bg.convert("RGBA"), veil)
    draw = ImageDraw.Draw(img)
    title_font = load_font(62, True)
    body_font = load_font(38)
    mono_font = load_font(34, True)
    small_font = load_font(26)
    panel = (255, 255, 255, 235)
    draw.rounded_rectangle((260, 170, WIDTH - 260, 860), radius=18, fill=panel)
    draw.text((330, 230), "Сохранение -> проверенные маски -> обучение", font=title_font, fill=(17, 24, 39))
    draw_wrapped(
        draw,
        (330, 330),
        "После сохранения инструмент кладет проверенные маски рядом с артефактами образца.",
        body_font,
        (38, 51, 70),
        1220,
        10,
    )
    names = [
        "reviewed_positive_bag_mask",
        "reviewed_talc_node_mask",
        "reviewed_talc_mask",
    ]
    y = 460
    for name in names:
        draw.rounded_rectangle((340, y, WIDTH - 340, y + 76), radius=10, fill=(232, 246, 248), outline=(0, 137, 145), width=2)
        draw.text((380, y + 18), name, font=mono_font, fill=(17, 70, 78))
        y += 104
    draw.text((330, 780), "Проверенные маски используются для обучения модели талька и честной оценки доли оталькования.", font=small_font, fill=(70, 85, 105))
    img.convert("RGB").save(OUT / "slide_reviewed_masks.png")


def srt_time(seconds: float) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def synthesize_silero_sentence(text: str, out_path: Path) -> None:
    import numpy as np
    import torch
    from scipy.io import wavfile

    global SILERO_MODEL
    if SILERO_MODEL is None:
        SILERO_MODEL, _example_text = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v4_ru",
            trust_repo=True,
        )
        SILERO_MODEL.to(torch.device("cpu"))
    audio = SILERO_MODEL.apply_tts(text=text, speaker=SILERO_SPEAKER, sample_rate=SILERO_SAMPLE_RATE)
    data = audio.detach().cpu().numpy()
    data = np.clip(data, -1.0, 1.0)
    wavfile.write(str(out_path), SILERO_SAMPLE_RATE, (data * 32767).astype(np.int16))


async def synthesize_sentence(text: str, out_path: Path) -> None:
    if TTS_BACKEND == "silero":
        synthesize_silero_sentence(text, out_path)
        return

    import edge_tts

    communicate = edge_tts.Communicate(text, VOICE, rate=TTS_RATE)
    await communicate.save(str(out_path))


async def synthesize_all() -> None:
    audio_dir = OUT / "tts_sentences"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_ext = ".wav" if TTS_BACKEND == "silero" else ".mp3"
    for idx, item in enumerate(SENTENCES, 1):
        path = audio_dir / f"{idx:03d}{audio_ext}"
        text_path = audio_dir / f"{idx:03d}.txt"
        current_text = json.dumps(
            {
                "backend": TTS_BACKEND,
                "voice": VOICE,
                "rate": TTS_RATE,
                "silero_speaker": SILERO_SPEAKER,
                "sample_rate": SILERO_SAMPLE_RATE,
                "text": item["text"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        cached_text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
        if cached_text != current_text and path.exists():
            path.unlink()
        if not path.exists() or path.stat().st_size < 1024:
            print(f"TTS {idx:02d}/{len(SENTENCES)}: {item['title']}", flush=True)
            for attempt in range(1, 5):
                try:
                    await synthesize_sentence(item["text"], path)
                    text_path.write_text(current_text, encoding="utf-8")
                    break
                except Exception:
                    if path.exists():
                        path.unlink()
                    if attempt == 4:
                        raise
                    await asyncio.sleep(2 * attempt)


def atempo_filter(tempo: float) -> str:
    if 0.5 <= tempo <= 2.0:
        return f"atempo={tempo:.6f}"
    parts = []
    remaining = tempo
    while remaining > 2.0:
        parts.append("atempo=2.000000")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.500000")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def build_audio_and_subtitles() -> tuple[Path, list[dict]]:
    audio_dir = OUT / "tts_sentences"
    audio_ext = ".wav" if TTS_BACKEND == "silero" else ".mp3"
    timeline: list[dict] = []
    start = 0.0
    concat = OUT / "audio_concat.txt"
    with concat.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(SENTENCES, 1):
            audio_path = audio_dir / f"{idx:03d}{audio_ext}"
            dur = ffprobe_duration(audio_path)
            end = start + dur
            timeline.append({**item, "index": idx, "start": start, "end": end, "duration": dur, "audio": str(audio_path)})
            f.write(f"file '{audio_path.resolve()}'\n")
            start = end

    narration = OUT / ("narration_raw.wav" if TTS_BACKEND == "silero" else "narration_raw.mp3")
    if TTS_BACKEND == "silero":
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
                "-c:a",
                "pcm_s16le",
                str(narration),
            ]
        )
    else:
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(narration)])
    raw_dur = ffprobe_duration(narration)
    scale = NARRATION_SECONDS / raw_dur
    tempo = raw_dur / NARRATION_SECONDS
    audio = OUT / "narration_300s.wav"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(narration),
            "-filter:a",
            f"{atempo_filter(tempo)},apad,atrim=0:{TARGET_SECONDS:.3f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(audio),
        ]
    )
    for item in timeline:
        item["start"] *= scale
        item["end"] *= scale
        item["duration"] *= scale

    srt = OUT / "subtitles_ru.srt"
    with srt.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(timeline, 1):
            wrapped = "\n".join(textwrap.wrap(item["text"], width=72))
            f.write(f"{idx}\n{srt_time(item['start'])} --> {srt_time(item['end'])}\n{wrapped}\n\n")

    script = OUT / "script_ru.md"
    with script.open("w", encoding="utf-8") as f:
        f.write("# Сценарий озвучки Talc UI tool\n\n")
        f.write("Источник: docs/ui/v2/notes/2026-07-04-talc-ui-video-script-ru.md\n\n")
        f.write(
            f"Бэкенд TTS: {TTS_BACKEND}. Голос: {VOICE}; Silero speaker: {SILERO_SPEAKER}; "
            f"rate {TTS_RATE}. Итоговая длительность видео: 5:00.\n\n"
        )
        for item in timeline:
            f.write(f"## {item['index']:02d}. {item['title']} [{item['start']:.2f}-{item['end']:.2f}s]\n\n{item['text']}\n\n")

    (OUT / "timeline.json").write_text(
        json.dumps(
            {
                "target_seconds": TARGET_SECONDS,
                "narration_seconds": NARRATION_SECONDS,
                "raw_audio_seconds": raw_dur,
                "tempo": tempo,
                "tts_backend": TTS_BACKEND,
                "voice": VOICE,
                "rate": TTS_RATE,
                "silero_speaker": SILERO_SPEAKER,
                "items": timeline,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
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


def draw_highlights(draw: ImageDraw.ImageDraw, scene_name: str, title_font: ImageFont.FreeTypeFont, small_font: ImageFont.FreeTypeFont) -> None:
    for idx, (x1, y1, x2, y2, label) in enumerate(HIGHLIGHTS.get(scene_name, [])):
        color = (34, 211, 238, 235) if idx == 0 else (250, 204, 21, 230)
        fill = (0, 0, 0, 95)
        draw.rounded_rectangle((x1, y1, x2, y2), radius=8, outline=color, width=4)
        lx = min(x1, WIDTH - 430)
        ly = max(74, y1 - 42)
        draw.rounded_rectangle((lx, ly, lx + 400, ly + 34), radius=8, fill=fill)
        draw.text((lx + 12, ly + 6), label, font=small_font, fill=(255, 255, 255, 245))


def render_video(timeline: list[dict]) -> Path:
    create_reviewed_masks_slide()
    images = {p.name: fit_image(p) for p in OUT.glob("*.png") if p.name.startswith(("scene_", "slide_"))}
    missing = sorted({item["scene"] for item in timeline} - set(images))
    if missing:
        raise FileNotFoundError(f"Missing render scenes: {missing}")

    title_font = load_font(26, True)
    small_font = load_font(22)
    subtitle_font = load_font(34)
    video = OUT / "video_with_subs.mp4"
    proc = subprocess.Popen(
        [
            ffmpeg_exe(),
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
        if frame_idx and frame_idx % (FPS * 20) == 0:
            print(f"Rendered {frame_idx / FPS:.0f}s / {TARGET_SECONDS:.0f}s", flush=True)
        t = frame_idx / FPS
        item = scene_for_time(t, timeline)
        base = images[item["scene"]].copy()
        overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        mm = int(t // 60)
        ss = int(t % 60)
        ff = int((t - int(t)) * 10)
        stamp = f"{mm:02}:{ss:02}.{ff}  ·  {item['title']}"
        draw.rounded_rectangle((WIDTH - 590, 18, WIDTH - 28, 64), radius=10, fill=(0, 0, 0, 155))
        draw.text((WIDTH - 570, 29), stamp, font=small_font, fill=(255, 255, 255, 245))
        draw.rounded_rectangle((28, 18, 370, 64), radius=10, fill=(0, 137, 145, 220))
        draw.text((48, 29), "Talc UI tool", font=title_font, fill=(255, 255, 255, 255))
        draw_highlights(draw, item["scene"], title_font, small_font)

        subtitle_lines = textwrap.wrap(item["text"], width=78)
        if subtitle_lines:
            line_h = 46
            box_h = 34 + line_h * len(subtitle_lines)
            box_y = HEIGHT - box_h - 26
            draw.rounded_rectangle((220, box_y, WIDTH - 220, HEIGHT - 26), radius=16, fill=(0, 0, 0, 178))
            y = box_y + 18
            for line in subtitle_lines:
                bbox = draw.textbbox((0, 0), line, font=subtitle_font)
                tw = bbox[2] - bbox[0]
                draw.text(((WIDTH - tw) // 2, y), line, font=subtitle_font, fill=(255, 255, 255, 255))
                y += line_h

        base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        proc.stdin.write(base.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg video render failed")
    return video


def mux_final(video: Path, audio: Path) -> Path:
    final = OUT / "nornikel_talc_ui_demo_1080p_ru.mp4"
    run(
        [
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
        ]
    )
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
        "timeline": str(OUT / "timeline.json"),
    }
    (OUT / "build_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
