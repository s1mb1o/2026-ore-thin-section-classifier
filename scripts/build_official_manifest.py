#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
Image.MAX_IMAGE_PIXELS = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a manifest for official image files.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--out", type=Path, default=Path("outputs/official_manifest.json"))
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    items = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
        except Exception as exc:
            print(f"skip unreadable image {path}: {exc}", file=sys.stderr)
            continue
        rel = path.relative_to(dataset_root)
        items.append(
            {
                "path": str(rel),
                "width": width,
                "height": height,
                "mode": mode,
                "bytes": path.stat().st_size,
                "label_hint": infer_label_hint(rel),
            }
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "count": len(items),
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out} with {len(items)} images")
    return 0


def infer_label_hint(path: Path) -> str:
    lower = "/".join(path.parts).lower()
    if "панорамы" in lower:
        return "panorama"
    if "области оталькования" in lower:
        return "talc_annotation"
    if "отальк" in lower:
        return "talcose"
    if "труднообогат" in lower or "/тонкие/" in lower:
        return "fine_intergrowth"
    if "рядовые" in lower:
        return "ordinary_intergrowth"
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
