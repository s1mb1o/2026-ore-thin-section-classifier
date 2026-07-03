from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from ore_classifier.augmentation import apply_augmentation, normalize_augmentation_settings  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_SPLIT_JSON = ROOT / "outputs/official_balanced_eval_split_deconflicted.json"
DEFAULT_DATASET_ROOT = ROOT / "dataset"
DEFAULT_OUT_DIR = ROOT / "outputs/augmentation_review"


def neutral_settings(seed: int) -> dict[str, Any]:
    return {
        "enabled": True,
        "color": {
            "brightness_pct": 0.0,
            "contrast_pct": 0.0,
            "saturation_pct": 0.0,
            "hue_degrees": 0.0,
            "gamma": 1.0,
        },
        "acquisition": {
            "blur_radius": 0.0,
            "gaussian_noise_std": 0.0,
        },
        "surface_artifacts": {
            "scratch_count": 0,
            "scratch_intensity_pct": 0.0,
            "polishing_haze_pct": 0.0,
            "pit_count": 0,
            "pit_intensity_pct": 0.0,
        },
        "runtime": {
            "random_seed": seed,
        },
    }


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def variant_settings(seed: int, updates: dict[str, Any]) -> dict[str, Any]:
    return normalize_augmentation_settings(deep_update(neutral_settings(seed), updates))


def review_variants(seed: int) -> list[dict[str, Any]]:
    return [
        {
            "id": "original",
            "title": "Original preview",
            "description": "Downscaled source preview. No augmentation applied.",
            "settings": None,
        },
        {
            "id": "ui_default",
            "title": "UI default",
            "description": "Current v2 default augmentation balance.",
            "settings": normalize_augmentation_settings({"enabled": True, "runtime": {"random_seed": seed + 1}}),
        },
        {
            "id": "color_tone",
            "title": "Color and tone",
            "description": "Brightness, contrast, saturation, and gamma shift.",
            "settings": variant_settings(
                seed + 2,
                {
                    "color": {
                        "brightness_pct": 12.0,
                        "contrast_pct": 16.0,
                        "saturation_pct": 10.0,
                        "gamma": 1.08,
                    }
                },
            ),
        },
        {
            "id": "acquisition_noise",
            "title": "Acquisition noise",
            "description": "Mild blur plus sensor-like Gaussian noise.",
            "settings": variant_settings(
                seed + 3,
                {
                    "acquisition": {
                        "blur_radius": 0.8,
                        "gaussian_noise_std": 7.0,
                    }
                },
            ),
        },
        {
            "id": "scratches",
            "title": "Scratches",
            "description": "Bright and dark polishing/grinding scratches.",
            "settings": variant_settings(
                seed + 4,
                {
                    "surface_artifacts": {
                        "scratch_count": 14,
                        "scratch_intensity_pct": 32.0,
                    }
                },
            ),
        },
        {
            "id": "polishing_haze",
            "title": "Polishing haze",
            "description": "Low-frequency whitening haze over the polished section.",
            "settings": variant_settings(
                seed + 5,
                {
                    "surface_artifacts": {
                        "polishing_haze_pct": 18.0,
                    }
                },
            ),
        },
        {
            "id": "pits_dust",
            "title": "Pits and dust",
            "description": "Small dark pits and bright dust specks.",
            "settings": variant_settings(
                seed + 6,
                {
                    "surface_artifacts": {
                        "pit_count": 90,
                        "pit_intensity_pct": 28.0,
                    }
                },
            ),
        },
        {
            "id": "combined_moderate",
            "title": "Combined moderate",
            "description": "Balanced color, acquisition, scratch, haze, and dust stress.",
            "settings": variant_settings(
                seed + 7,
                {
                    "color": {
                        "brightness_pct": 8.0,
                        "contrast_pct": 12.0,
                        "saturation_pct": 5.0,
                    },
                    "acquisition": {
                        "blur_radius": 0.4,
                        "gaussian_noise_std": 4.0,
                    },
                    "surface_artifacts": {
                        "scratch_count": 8,
                        "scratch_intensity_pct": 20.0,
                        "polishing_haze_pct": 9.0,
                        "pit_count": 35,
                        "pit_intensity_pct": 18.0,
                    },
                },
            ),
        },
        {
            "id": "combined_stress",
            "title": "Combined stress",
            "description": "Stronger stress case for robustness review.",
            "settings": variant_settings(
                seed + 8,
                {
                    "color": {
                        "brightness_pct": -8.0,
                        "contrast_pct": 25.0,
                        "saturation_pct": -8.0,
                        "gamma": 0.9,
                    },
                    "acquisition": {
                        "blur_radius": 0.9,
                        "gaussian_noise_std": 9.0,
                    },
                    "surface_artifacts": {
                        "scratch_count": 20,
                        "scratch_intensity_pct": 40.0,
                        "polishing_haze_pct": 18.0,
                        "pit_count": 120,
                        "pit_intensity_pct": 30.0,
                    },
                },
            ),
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static HTML gallery for v2 runtime augmentation review.")
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--image", type=Path, action="append", default=[], help="Explicit image path. Can be repeated.")
    parser.add_argument("--per-label", type=int, default=1, help="Images per label when using --split-json.")
    parser.add_argument("--max-side", type=int, default=720, help="Maximum preview side in pixels.")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_split_images(split_json: Path, dataset_root: Path, per_label: int, seed: int) -> list[dict[str, Any]]:
    data = json.loads(split_json.read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in data.get("items", []):
        label = str(item.get("label") or "unlabelled")
        path = dataset_root / str(item.get("path") or "")
        if path.exists():
            groups.setdefault(label, []).append({**item, "source_path": path})
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for label in sorted(groups):
        candidates = list(groups[label])
        rng.shuffle(candidates)
        selected.extend(candidates[: max(0, per_label)])
    if not selected:
        raise SystemExit(f"No source images selected from {split_json}")
    return selected


def select_images(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.image:
        selected = []
        for path in args.image:
            source = path if path.is_absolute() else ROOT / path
            if not source.exists():
                raise SystemExit(f"Image does not exist: {source}")
            selected.append(
                {
                    "path": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
                    "label": "custom",
                    "source_path": source,
                }
            )
        return selected
    return select_split_images(args.split_json, args.dataset_root, args.per_label, args.seed)


def image_slug(source_path: Path, index: int) -> str:
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    stem = "".join(ch.lower() if ch.isalnum() else "-" for ch in source_path.stem)
    stem = "-".join(part for part in stem.split("-") if part)[:40] or f"image-{index}"
    return f"{index:02d}-{stem}-{digest}"


def load_preview(path: Path, max_side: int) -> Image.Image:
    with Image.open(path) as image:
        preview = image.convert("RGB")
        preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return preview.copy()


def save_gallery_assets(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    selected = select_images(args)
    variants = review_variants(args.seed)
    out_dir = args.out_dir
    images_dir = out_dir / "images"
    if out_dir.exists() and args.overwrite:
        for path in sorted(images_dir.glob("*")) if images_dir.exists() else []:
            path.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": "augmentation-review-gallery-v0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "max_side": args.max_side,
        "split_json": str(args.split_json),
        "dataset_root": str(args.dataset_root),
        "out_dir": str(out_dir),
        "images": [],
        "variants": [
            {
                "id": variant["id"],
                "title": variant["title"],
                "description": variant["description"],
                "settings": variant["settings"],
            }
            for variant in variants
        ],
    }

    for index, item in enumerate(selected, start=1):
        source_path = Path(item["source_path"])
        base = load_preview(source_path, args.max_side)
        source_slug = image_slug(source_path, index)
        source_entry = {
            "source_path": str(source_path),
            "dataset_path": str(item.get("path", source_path)),
            "label": str(item.get("label") or "unlabelled"),
            "width": int(base.width),
            "height": int(base.height),
            "slug": source_slug,
            "outputs": [],
        }
        for variant in variants:
            variant_image = base if variant["settings"] is None else apply_augmentation(base, variant["settings"])
            filename = f"{source_slug}__{variant['id']}.jpg"
            path = images_dir / filename
            variant_image.save(path, quality=92, optimize=True)
            source_entry["outputs"].append(
                {
                    "variant_id": variant["id"],
                    "title": variant["title"],
                    "description": variant["description"],
                    "image": str(path.relative_to(out_dir)),
                    "settings": variant["settings"],
                }
            )
        manifest["images"].append(source_entry)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path = out_dir / "index.html"
    html_path.write_text(render_html(manifest), encoding="utf-8")
    return manifest, html_path


def settings_summary(settings: dict[str, Any] | None) -> str:
    if settings is None:
        return "none"
    color = settings["color"]
    acquisition = settings["acquisition"]
    artifacts = settings["surface_artifacts"]
    return (
        f"brightness {color['brightness_pct']}%, contrast {color['contrast_pct']}%, "
        f"saturation {color['saturation_pct']}%, gamma {color['gamma']}; "
        f"blur {acquisition['blur_radius']}, noise {acquisition['gaussian_noise_std']}; "
        f"scratches {artifacts['scratch_count']} @ {artifacts['scratch_intensity_pct']}%, "
        f"haze {artifacts['polishing_haze_pct']}%, "
        f"pits {artifacts['pit_count']} @ {artifacts['pit_intensity_pct']}%"
    )


def render_html(manifest: dict[str, Any]) -> str:
    image_count = len(manifest["images"])
    variant_count = len(manifest["variants"])
    cards: list[str] = []
    source_nav: list[str] = []
    for source in manifest["images"]:
        source_id = f"source-{html.escape(source['slug'])}"
        label = html.escape(source["label"])
        dataset_path = html.escape(source["dataset_path"])
        source_nav.append(f'<a href="#{source_id}">{label}: {html.escape(Path(source["dataset_path"]).name)}</a>')
        output_cards = []
        for output in source["outputs"]:
            variant_id = html.escape(output["variant_id"])
            settings = output["settings"]
            settings_json = html.escape(json.dumps(settings, ensure_ascii=False, indent=2) if settings else "{}")
            output_cards.append(
                f"""
                <article class="card" data-variant="{variant_id}">
                  <img src="{html.escape(output['image'])}" alt="{html.escape(output['title'])}">
                  <div class="card-body">
                    <div class="variant">{html.escape(output['title'])}</div>
                    <p>{html.escape(output['description'])}</p>
                    <p class="summary">{html.escape(settings_summary(settings))}</p>
                    <details>
                      <summary>settings JSON</summary>
                      <pre>{settings_json}</pre>
                    </details>
                  </div>
                </article>
                """
            )
        cards.append(
            f"""
            <section class="source" id="{source_id}" data-label="{label}">
              <div class="source-head">
                <div>
                  <h2>{label}</h2>
                  <p>{dataset_path}</p>
                </div>
                <span>{source['width']} x {source['height']} preview</span>
              </div>
              <div class="grid">
                {''.join(output_cards)}
              </div>
            </section>
            """
        )
    variant_options = "\n".join(
        f'<option value="{html.escape(variant["id"])}">{html.escape(variant["title"])}</option>'
        for variant in manifest["variants"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Augmentation Review Gallery</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1d242d;
      --muted: #66717f;
      --line: #d9dee6;
      --accent: #2563eb;
      --shadow: 0 12px 28px rgba(20, 32, 47, 0.08);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #101418;
        --panel: #171d24;
        --ink: #ecf1f7;
        --muted: #9ca8b6;
        --line: #2b3541;
        --accent: #75a7ff;
        --shadow: none;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      border-bottom: 1px solid var(--line);
      background: color-mix(in srgb, var(--panel) 92%, transparent);
      backdrop-filter: blur(16px);
    }}
    .bar {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 14px 18px;
      display: grid;
      gap: 10px;
    }}
    h1, h2, p {{ margin: 0; }}
    h1 {{ font-size: 22px; letter-spacing: 0; }}
    h2 {{ font-size: 18px; letter-spacing: 0; }}
    .meta, .source-head p, .summary, .card p {{ color: var(--muted); font-size: 13px; }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    select, input {{
      min-height: 34px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 6px;
      padding: 6px 8px;
      font: inherit;
    }}
    main {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 18px;
      display: grid;
      gap: 18px;
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    nav a {{
      color: var(--accent);
      text-decoration: none;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 5px 8px;
      background: var(--panel);
      font-size: 13px;
    }}
    .source {{
      display: grid;
      gap: 12px;
    }}
    .source-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
    }}
    .source-head span {{
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 12px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: var(--shadow);
      min-width: 0;
    }}
    .card img {{
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background: #0b0f13;
      display: block;
    }}
    .card-body {{
      display: grid;
      gap: 6px;
      padding: 10px;
    }}
    .variant {{
      font-weight: 750;
      font-size: 14px;
    }}
    details {{
      border-top: 1px solid var(--line);
      padding-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }}
    summary {{ cursor: pointer; color: var(--accent); }}
    pre {{
      max-height: 280px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: var(--bg);
      color: var(--ink);
      font-size: 11px;
      white-space: pre-wrap;
    }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div>
        <h1>Augmentation Review Gallery</h1>
        <p class="meta">Generated {html.escape(manifest["created_at"])}. {image_count} source images x {variant_count} variants. Seed {manifest["seed"]}. Max side {manifest["max_side"]} px.</p>
      </div>
      <div class="controls">
        <label>Variant <select id="variantFilter"><option value="all">All variants</option>{variant_options}</select></label>
        <label>Search <input id="search" type="search" placeholder="label, filename, variant"></label>
      </div>
      <nav>{''.join(source_nav)}</nav>
    </div>
  </header>
  <main>{''.join(cards)}</main>
  <script>
    const variantFilter = document.getElementById('variantFilter');
    const search = document.getElementById('search');
    function applyFilters() {{
      const variant = variantFilter.value;
      const query = search.value.trim().toLowerCase();
      document.querySelectorAll('.card').forEach(card => {{
        const source = card.closest('.source');
        const text = (source.dataset.label + ' ' + source.innerText + ' ' + card.dataset.variant).toLowerCase();
        const visible = (variant === 'all' || card.dataset.variant === variant) && (!query || text.includes(query));
        card.classList.toggle('hidden', !visible);
      }});
      document.querySelectorAll('.source').forEach(source => {{
        const anyVisible = Array.from(source.querySelectorAll('.card')).some(card => !card.classList.contains('hidden'));
        source.classList.toggle('hidden', !anyVisible);
      }});
    }}
    variantFilter.addEventListener('change', applyFilters);
    search.addEventListener('input', applyFilters);
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    manifest, html_path = save_gallery_assets(args)
    print(f"Wrote {html_path}")
    print(f"Wrote {html_path.parent / 'manifest.json'}")
    print(f"Sources: {len(manifest['images'])}; variants per source: {len(manifest['variants'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
