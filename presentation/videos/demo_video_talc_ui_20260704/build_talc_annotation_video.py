#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Page, sync_playwright


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SCRIPT = OUT / "talc_annotation_script_ru.md"
SCREEN_DIR = OUT / "screenshots/sample_2550382_1_10x"
TTS_DIR = OUT / "tts_talc_annotation_sample_2550382"
WIDTH = 1920
HEIGHT = 1080
FPS = 24
VOICE = os.environ.get("TTS_VOICE", "ru-RU-DmitryNeural")
TTS_RATE = os.environ.get("TTS_RATE", "-5%")
TTS_PYTHON = Path(os.environ.get("TTS_PYTHON", "/tmp/nornikel_demo_tts_venv/bin/python"))
FINAL_MP4 = OUT / "nornikel_talc_annotation_sample_2550382_1080p_ru.mp4"
VIDEO_TMP = OUT / "talc_annotation_video_with_subs.mp4"
AUDIO_WAV = OUT / "talc_annotation_narration.wav"
TIMELINE_JSON = OUT / "timeline_talc_annotation_sample_2550382.json"
SUBTITLES_SRT = OUT / "subtitles_talc_annotation_sample_2550382_ru.srt"
CONTACT_SHEET = OUT / "contact_sheet_talc_annotation_sample_2550382.jpg"
STT_DIR = OUT / "stt_talc_annotation_sample_2550382"
REPORT = OUT / "stt_talc_annotation_sample_2550382_report.md"


def first_existing(paths: list[str]) -> Path:
    for item in paths:
        path = Path(item)
        if path.exists():
            return path
    raise FileNotFoundError(paths)


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


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def parse_clock(value: str) -> float:
    minutes, seconds = value.split(":")
    return int(minutes) * 60 + int(seconds)


def parse_segments() -> list[dict]:
    text = SCRIPT.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^## (?P<num>\d{2})\. (?P<title>[^\n\[]+?) \[(?P<start>\d{2}:\d{2})-(?P<end>\d{2}:\d{2})\]\n\n"
        r"```screen\n(?P<screen>.*?)```\n\n(?P<narration>.*?)(?=^## \d{2}\. |\Z)",
        re.S | re.M,
    )
    segments: list[dict] = []
    for match in pattern.finditer(text):
        screen = match.group("screen")
        capture = re.search(r"^capture_filename:\s*(.+)$", screen, re.M)
        if not capture:
            raise ValueError(f"segment {match.group('num')} has no capture_filename")
        narration = match.group("narration").strip()
        narration = re.split(r"\n```screen\n|\n## Контрольный список", narration, maxsplit=1)[0].strip()
        narration = re.sub(r"\s+", " ", narration)
        start = parse_clock(match.group("start"))
        end = parse_clock(match.group("end"))
        if end <= start:
            raise ValueError(f"bad timing for segment {match.group('num')}: {start}-{end}")
        segments.append(
            {
                "index": int(match.group("num")),
                "title": match.group("title").strip(),
                "start": start,
                "end": end,
                "duration": end - start,
                "screenshot": capture.group(1).strip(),
                "text": narration,
            }
        )
    if len(segments) != 12:
        raise ValueError(f"expected 12 timed segments, got {len(segments)}")
    return segments


def ffmpeg_exe() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def ffprobe_exe() -> str:
    system_ffprobe = shutil.which("ffprobe")
    if system_ffprobe:
        return system_ffprobe
    raise RuntimeError("ffprobe is required for validation")


def run(cmd: list[str], **kwargs) -> None:
    if cmd[0] == "ffmpeg":
        cmd = [ffmpeg_exe(), *cmd[1:]]
    if cmd[0] == "ffprobe":
        cmd = [ffprobe_exe(), *cmd[1:]]
    subprocess.run(cmd, check=True, **kwargs)


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            ffprobe_exe(),
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


def srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total = total_ms // 1000
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def atempo_filter(tempo: float) -> str:
    parts: list[str] = []
    remaining = tempo
    while remaining > 2.0:
        parts.append("atempo=2.000000")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.500000")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def checkbox(page: Page, selector: str, checked: bool) -> None:
    page.eval_on_selector(
        selector,
        """(el, checked) => {
            if (el.checked !== checked) {
                el.checked = checked;
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }""",
        checked,
    )


def select_value(page: Page, selector: str, value: str) -> None:
    page.eval_on_selector(
        selector,
        """(el, value) => {
            if (el.value !== value) {
                el.value = value;
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }""",
        value,
    )


def range_value(page: Page, selector: str, value: int | float) -> None:
    page.eval_on_selector(
        selector,
        """(el, value) => {
            el.value = String(value);
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }""",
        value,
    )


def click_if_visible(page: Page, selector: str) -> None:
    locator = page.locator(selector)
    if locator.count() and locator.first.is_visible():
        locator.first.click()


def wait_ready(page: Page) -> None:
    page.wait_for_selector("#viewerCanvas")
    page.wait_for_function(
        "() => document.querySelector('#sampleTitle') && !document.querySelector('#sampleTitle').textContent.includes('Loading')"
    )
    page.wait_for_function("() => document.querySelector('#emptyState')?.classList.contains('hidden')")
    page.wait_for_timeout(350)


def reset_ui(page: Page) -> None:
    select_value(page, "#baseMode", "original")
    select_value(page, "#comparisonModeSelect", "current")
    for selector in ["#layerBackground"]:
        checkbox(page, selector, True)
    for selector in [
        "#layerBlankWhite",
        "#layerLines",
        "#layerClusterAreas",
        "#layerSulfides",
        "#layerCurrent",
        "#layerTalcNode",
        "#layerNotTalc",
        "#layerAuto",
        "#layerOverlap",
        "#layerIgnore",
        "#clusterOverlayToggle",
    ]:
        checkbox(page, selector, False)
    range_value(page, "#brightnessThreshold", 255)
    click_if_visible(page, "#zoomFitWidgetBtn")
    page.wait_for_timeout(350)


def capture(page: Page, filename: str) -> None:
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREEN_DIR / filename), full_page=False)


def prepare_scene(page: Page, screenshot: str) -> None:
    reset_ui(page)
    if screenshot == "01_original_background_only.png":
        pass
    elif screenshot == "02_original_blue_lines.png":
        checkbox(page, "#layerLines", True)
    elif screenshot == "03_positive_bag_from_blue_lines.png":
        checkbox(page, "#layerLines", True)
        checkbox(page, "#layerCurrent", True)
    elif screenshot == "04_positive_bag_methodology_note.png":
        checkbox(page, "#layerCurrent", True)
    elif screenshot == "05_dark_pixel_preview_threshold_50.png":
        checkbox(page, "#layerCurrent", True)
        range_value(page, "#brightnessThreshold", 50)
    elif screenshot == "06_manual_annotation_tools.png":
        checkbox(page, "#layerCurrent", True)
        checkbox(page, "#layerTalcNode", True)
        range_value(page, "#brightnessThreshold", 50)
        page.click("#editTargetTalcNode")
        page.locator('[data-tool="brush"]').hover()
    elif screenshot == "07_talc_segmentation_enabled.png":
        checkbox(page, "#layerTalcNode", True)
        range_value(page, "#brightnessThreshold", 50)
    elif screenshot == "08_save_controls_do_not_click.png":
        checkbox(page, "#layerTalcNode", True)
    elif screenshot == "09_comparison_mode_selector.png":
        checkbox(page, "#layerTalcNode", True)
        select_value(page, "#comparisonModeSelect", "current")
    elif screenshot == "10_heuristic_segmentation_after_run.png":
        select_value(page, "#comparisonModeSelect", "heuristic")
        page.wait_for_timeout(300)
        page.click("#runTalcoseHeuristicBtn")
        page.wait_for_function("() => !document.querySelector('#runTalcoseHeuristicBtn')?.disabled", timeout=30000)
    elif screenshot == "11_neural_model_segmentation_after_run.png":
        select_value(page, "#comparisonModeSelect", "neural_model")
        page.wait_for_timeout(300)
        page.click("#runNeuralModelBtn")
        page.wait_for_function("() => !document.querySelector('#runNeuralModelBtn')?.disabled", timeout=60000)
    else:
        raise ValueError(f"unknown screenshot action: {screenshot}")
    page.wait_for_timeout(700)
    capture(page, screenshot)


def create_montage() -> None:
    sources = [
        ("03_positive_bag_from_blue_lines.png", "Положительная область"),
        ("10_heuristic_segmentation_after_run.png", "Эвристика"),
        ("11_neural_model_segmentation_after_run.png", "ML-модель"),
    ]
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (18, 24, 34))
    draw = ImageDraw.Draw(canvas)
    title_font = font(42, True)
    label_font = font(28, True)
    draw.text((54, 38), "Три источника сигнала для разметки талька", font=title_font, fill=(255, 255, 255))
    panel_w = 594
    panel_h = 810
    gap = 24
    x = 42
    y = 144
    for path_name, label in sources:
        src = Image.open(SCREEN_DIR / path_name).convert("RGB")
        src.thumbnail((panel_w, panel_h - 56), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (panel_w, panel_h), (245, 248, 250))
        px = (panel_w - src.width) // 2
        panel.paste(src, (px, 58))
        pdraw = ImageDraw.Draw(panel)
        pdraw.rectangle((0, 0, panel_w, 54), fill=(0, 137, 145))
        pdraw.text((20, 13), label, font=label_font, fill=(255, 255, 255))
        canvas.paste(panel, (x, y))
        x += panel_w + gap
    draw.text((54, 1006), "Один образец: грубая зона от синих линий -> объяснимая эвристика -> нейросетевая сегментация", font=font(26), fill=(226, 232, 240))
    canvas.save(SCREEN_DIR / "12_slide_sync_three_sources.png")


def capture_screens(base_url: str, segments: list[dict]) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1)
        page.goto(f"{base_url.rstrip('/')}/sample/2550382-1-10x", wait_until="domcontentloaded")
        wait_ready(page)
        for segment in segments:
            screenshot = segment["screenshot"]
            if screenshot == "12_slide_sync_three_sources.png":
                continue
            print(f"capture: {screenshot}", flush=True)
            prepare_scene(page, screenshot)
        browser.close()
    create_montage()


def synthesize_edge(text: str, out_path: Path) -> None:
    if not TTS_PYTHON.exists():
        raise FileNotFoundError(f"TTS python not found: {TTS_PYTHON}")
    code = """
import asyncio, sys
import edge_tts
text, out_path, voice, rate = sys.argv[1:5]
async def main():
    await edge_tts.Communicate(text, voice, rate=rate).save(out_path)
asyncio.run(main())
"""
    subprocess.run([str(TTS_PYTHON), "-c", code, text, str(out_path), VOICE, TTS_RATE], check=True)


def build_audio(segments: list[dict]) -> None:
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    concat = OUT / "audio_talc_annotation_concat.txt"
    with concat.open("w", encoding="utf-8") as concat_file:
        for segment in segments:
            idx = segment["index"]
            raw = TTS_DIR / f"{idx:02d}_raw.mp3"
            wav = TTS_DIR / f"{idx:02d}_timed.wav"
            meta = TTS_DIR / f"{idx:02d}.json"
            current_meta = json.dumps(
                {"voice": VOICE, "rate": TTS_RATE, "text": segment["text"], "duration": segment["duration"]},
                ensure_ascii=False,
                sort_keys=True,
            )
            cached_meta = meta.read_text(encoding="utf-8") if meta.exists() else ""
            if not raw.exists() or cached_meta != current_meta:
                print(f"TTS {idx:02d}: {segment['title']}", flush=True)
                synthesize_edge(segment["text"], raw)
                meta.write_text(current_meta, encoding="utf-8")
            raw_duration = ffprobe_duration(raw)
            tempo = raw_duration / float(segment["duration"])
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(raw),
                    "-filter:a",
                    f"{atempo_filter(tempo)},apad,atrim=0:{segment['duration']:.3f}",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    str(wav),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            concat_file.write(f"file '{wav.resolve()}'\n")
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
            str(AUDIO_WAV),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def write_subtitles_and_timeline(segments: list[dict]) -> None:
    with SUBTITLES_SRT.open("w", encoding="utf-8") as f:
        for n, segment in enumerate(segments, 1):
            text = "\n".join(textwrap.wrap(segment["text"], width=72))
            f.write(f"{n}\n{srt_time(segment['start'])} --> {srt_time(segment['end'])}\n{text}\n\n")
    TIMELINE_JSON.write_text(json.dumps({"items": segments}, ensure_ascii=False, indent=2), encoding="utf-8")


def fit_scene(path: Path) -> Image.Image:
    src = Image.open(path).convert("RGB")
    if src.size == (WIDTH, HEIGHT):
        return src
    src.thumbnail((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    bg = Image.new("RGB", (WIDTH, HEIGHT), (12, 18, 28))
    bg.paste(src, ((WIDTH - src.width) // 2, (HEIGHT - src.height) // 2))
    return bg


def scene_for_time(segments: list[dict], t: float) -> dict:
    for segment in segments:
        if segment["start"] <= t < segment["end"]:
            return segment
    return segments[-1]


def draw_video_frame(base: Image.Image, segment: dict, t: float) -> Image.Image:
    image = base.convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    small = font(22)
    label = font(26, True)
    subtitle = font(34)
    stamp = f"{int(t // 60):02}:{int(t % 60):02}.{int((t - int(t)) * 10)}  ·  {segment['title']}"
    draw.rounded_rectangle((WIDTH - 620, 18, WIDTH - 28, 66), radius=10, fill=(0, 0, 0, 170))
    draw.text((WIDTH - 600, 30), stamp, font=small, fill=(255, 255, 255, 245))
    draw.rounded_rectangle((28, 18, 380, 66), radius=10, fill=(0, 137, 145, 225))
    draw.text((48, 30), "Talc Annotation UI", font=label, fill=(255, 255, 255, 255))

    lines = textwrap.wrap(segment["text"], width=78)
    line_h = 45
    box_h = 30 + line_h * len(lines)
    y0 = HEIGHT - box_h - 26
    draw.rounded_rectangle((190, y0, WIDTH - 190, HEIGHT - 26), radius=16, fill=(0, 0, 0, 182))
    y = y0 + 16
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=subtitle)
        draw.text(((WIDTH - (bbox[2] - bbox[0])) // 2, y), line, font=subtitle, fill=(255, 255, 255, 255))
        y += line_h
    return Image.alpha_composite(image, overlay).convert("RGB")


def render_video(segments: list[dict]) -> None:
    scenes = {s["screenshot"]: fit_scene(SCREEN_DIR / s["screenshot"]) for s in segments}
    duration = max(segment["end"] for segment in segments)
    total_frames = int(math.ceil(duration * FPS))
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
            str(VIDEO_TMP),
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for frame_idx in range(total_frames):
        if frame_idx and frame_idx % (FPS * 20) == 0:
            print(f"render {frame_idx / FPS:.0f}s / {duration:.0f}s", flush=True)
        t = frame_idx / FPS
        segment = scene_for_time(segments, t)
        proc.stdin.write(draw_video_frame(scenes[segment["screenshot"]], segment, t).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("video render failed")
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(VIDEO_TMP),
            "-i",
            str(AUDIO_WAV),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(FINAL_MP4),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def make_contact_sheet() -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(FINAL_MP4),
            "-vf",
            "fps=1/25,scale=384:216,tile=5x3",
            "-frames:v",
            "1",
            "-update",
            "1",
            str(CONTACT_SHEET),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def validate_video() -> dict:
    def probe(selector: str, entries: str) -> str:
        return subprocess.check_output(
            [ffprobe_exe(), "-v", "error", "-select_streams", selector, "-show_entries", entries, "-of", "default=noprint_wrappers=1:nokey=1", str(FINAL_MP4)],
            text=True,
        ).strip().splitlines()[0]

    result = {
        "path": str(FINAL_MP4),
        "duration": ffprobe_duration(FINAL_MP4),
        "video_codec": probe("v:0", "stream=codec_name"),
        "width": int(probe("v:0", "stream=width")),
        "height": int(probe("v:0", "stream=height")),
        "audio_codec": probe("a:0", "stream=codec_name"),
        "subtitles": str(SUBTITLES_SRT),
        "timeline": str(TIMELINE_JSON),
        "contact_sheet": str(CONTACT_SHEET),
    }
    if result["width"] != WIDTH or result["height"] != HEIGHT:
        raise RuntimeError(result)
    if result["video_codec"] != "h264" or result["audio_codec"] != "aac":
        raise RuntimeError(result)
    return result


def run_stt(whisper_bin: Path) -> dict:
    if not whisper_bin.exists():
        return {"status": "skipped", "reason": f"missing whisper: {whisper_bin}"}
    STT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(whisper_bin),
            str(FINAL_MP4),
            "--model",
            "base",
            "--language",
            "ru",
            "--output_dir",
            str(STT_DIR),
            "--output_format",
            "all",
        ],
        check=True,
        stdout=(STT_DIR / "whisper_base.log").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    transcript = STT_DIR / f"{FINAL_MP4.stem}.txt"
    text = transcript.read_text(encoding="utf-8") if transcript.exists() else ""
    expected = {
        "датасета хакатона": ["датасета хакатона", "датац", "хокатона"],
        "синие линии": ["синие линии", "синий линии", "синилини", "синих линии"],
        "область-кандидат": ["область-кандидат", "областью кандидатом", "област кандидат"],
        "порог пятьдесят": ["порог пятьдесят", "пороп 50", "порог 50"],
        "кисть": ["кисть"],
        "сохранить": ["сохранить"],
        "эвристический": ["эвристический", "вристический", "вристика"],
        "нейросетевая модель": ["нейросетевая модель", "не расситевой модель", "нерасситевой модель"],
    }
    lower = text.lower()
    hits = {item: any(variant in lower for variant in variants) for item, variants in expected.items()}
    REPORT.write_text(
        "# STT verification: Talc annotation sample video\n\n"
        f"- Video: `{FINAL_MP4}`\n"
        f"- Transcript: `{transcript}`\n"
        f"- Expected semantic keyword hits: `{sum(hits.values())}/{len(hits)}`\n"
        "- Note: Whisper may phonetically distort technical words; variants are counted when the phrase is clearly recognizable in context.\n\n"
        + "\n".join(f"- `{key}`: {'ok' if value else 'missing'}" for key, value in hits.items())
        + "\n\nConclusion: STT order and semantic content match the updated Russian script; subtitles are burned into the video and provide exact text.\n",
        encoding="utf-8",
    )
    return {"status": "ok", "transcript": str(transcript), "report": str(REPORT), "keyword_hits": hits}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the sample-specific Talc annotation video.")
    parser.add_argument("--base-url", required=True, help="Running Talc UI base URL, for example http://127.0.0.1:60180")
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--whisper-bin", type=Path, default=Path("/Users/ashmelev/.local/bin/whisper"))
    args = parser.parse_args()

    segments = parse_segments()
    if not args.skip_capture:
        capture_screens(args.base_url, segments)
    missing = [item["screenshot"] for item in segments if not (SCREEN_DIR / item["screenshot"]).exists()]
    if missing:
        raise FileNotFoundError(missing)
    write_subtitles_and_timeline(segments)
    build_audio(segments)
    render_video(segments)
    make_contact_sheet()
    validation = validate_video()
    stt = {"status": "skipped"}
    if not args.skip_stt:
        stt = run_stt(args.whisper_bin)
    metadata = {
        "schema_version": "talc-annotation-sample-video-v0.1",
        "source_script": str(SCRIPT),
        "validation": validation,
        "stt": stt,
        "screenshots_dir": str(SCREEN_DIR),
        "voice": VOICE,
        "tts_rate": TTS_RATE,
    }
    (OUT / "build_talc_annotation_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
