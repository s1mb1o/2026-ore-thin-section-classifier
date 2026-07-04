#!/usr/bin/env python3
"""Build an image-level validation split from class-folder datasets.

The split is intentionally unbalanced: it keeps all usable images from the
requested class folders, but skips every content hash that appears under more
than one class label.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
LABELS = ("ordinary_intergrowth", "fine_intergrowth", "talcose")
DEFAULT_ROOTS = (
    Path("Фото руд по сортам. ч1"),
    Path("Фото руд по сортам. ч2"),
)

Image.MAX_IMAGE_PIXELS = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--class-root", type=Path, action="append", default=None)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/ch1_ch2_class_folder_eval_split.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/ch1_ch2_class_folder_eval_split.csv"))
    parser.add_argument("--conflicts-csv", type=Path, default=None)
    parser.add_argument("--duplicates-csv", type=Path, default=None)
    args = parser.parse_args()

    class_roots = args.class_root or list(DEFAULT_ROOTS)
    items, skipped = collect_items(args.dataset_root, class_roots)
    split = build_split(
        items,
        skipped=skipped,
        dataset_root=args.dataset_root,
        class_roots=class_roots,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_items_csv(args.out_csv, split["items"])
    write_groups_csv(args.conflicts_csv or args.out_json.with_name("class_folder_label_conflicts.csv"), split["label_conflict_groups"])
    write_groups_csv(args.duplicates_csv or args.out_json.with_name("class_folder_duplicate_groups.csv"), split["duplicate_groups"])

    print(
        json.dumps(
            {
                "items_total": split["items_total"],
                "selected_total": split["selected_total"],
                "selected_by_label": split["selected_by_label"],
                "skipped_conflict_items": split["skipped_conflict_items"],
                "label_conflict_group_count": split["label_conflict_group_count"],
                "duplicate_group_count": split["duplicate_group_count"],
                "out_json": str(args.out_json),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def collect_items(dataset_root: Path, class_roots: list[Path]) -> tuple[list[dict[str, Any]], Counter[str]]:
    items: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for class_root in class_roots:
        root = dataset_root / class_root
        if not root.exists():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rel = path.relative_to(dataset_root)
            label = infer_label(rel)
            if label is None:
                skipped["unlabelled_or_annotation"] += 1
                continue
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    mode = image.mode
            except Exception as exc:  # noqa: BLE001 - keep scanning remaining dataset files
                print(f"skip unreadable image {path}: {exc}", file=sys.stderr)
                skipped["unreadable"] += 1
                continue
            items.append(
                {
                    "path": str(rel),
                    "label": label,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "source_class_root": str(class_root),
                    "source_class_dir": rel.parts[1] if len(rel.parts) > 1 else "",
                }
            )
    return items, skipped


def infer_label(path: Path) -> str | None:
    lower = "/".join(path.parts).lower()
    if "области оталькования" in lower:
        return None
    if "отальк" in lower:
        return "talcose"
    if "труднообогат" in lower or "/тонкие/" in lower:
        return "fine_intergrowth"
    if "рядовые" in lower or "/рядовые/" in lower:
        return "ordinary_intergrowth"
    return None


def build_split(
    items: list[dict[str, Any]],
    *,
    skipped: Counter[str],
    dataset_root: Path,
    class_roots: list[Path],
) -> dict[str, Any]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_hash[item["sha256"]].append(item)

    duplicate_groups: list[dict[str, Any]] = []
    conflict_groups: list[dict[str, Any]] = []
    conflict_hashes: set[str] = set()
    duplicate_item_count = 0
    for digest, group_items in sorted(by_hash.items()):
        if len(group_items) <= 1:
            continue
        labels = sorted({item["label"] for item in group_items})
        group = {
            "sha256": digest,
            "labels": labels,
            "count": len(group_items),
            "items": group_items,
        }
        duplicate_groups.append(group)
        duplicate_item_count += len(group_items)
        if len(labels) > 1:
            conflict_groups.append(group)
            conflict_hashes.add(digest)

    selected = [item for item in items if item["sha256"] not in conflict_hashes]
    selected.sort(key=lambda item: (item["label"], item["path"]))
    selected_by_label = Counter(item["label"] for item in selected)
    source_by_label = Counter(item["label"] for item in items)
    skipped_conflict_items = len(items) - len(selected)

    return {
        "schema_version": "class-folder-eval-split-v0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "class_roots": [str(root) for root in class_roots],
        "labels": list(LABELS),
        "items_total": len(items),
        "source_by_label": dict(source_by_label),
        "selected_total": len(selected),
        "selected_by_label": dict(selected_by_label),
        "skipped": dict(skipped),
        "skip_policy": "Skip every image whose sha256 appears under more than one source class label.",
        "skipped_conflict_items": skipped_conflict_items,
        "label_conflict_group_count": len(conflict_groups),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_item_count": duplicate_item_count,
        "items": selected,
        "label_conflict_groups": conflict_groups,
        "duplicate_groups": duplicate_groups,
    }


def write_items_csv(path: Path, items: list[dict[str, Any]]) -> None:
    fieldnames = (
        "path",
        "label",
        "width",
        "height",
        "bytes",
        "sha256",
        "source_class_root",
        "source_class_dir",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: item.get(key, "") for key in fieldnames} for item in items)


def write_groups_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    fieldnames = ("sha256", "labels_in_group", "path", "label", "source_class_root", "source_class_dir")
    rows = []
    for group in groups:
        for item in group["items"]:
            rows.append(
                {
                    "sha256": group["sha256"],
                    "labels_in_group": "|".join(group["labels"]),
                    "path": item["path"],
                    "label": item["label"],
                    "source_class_root": item.get("source_class_root", ""),
                    "source_class_dir": item.get("source_class_dir", ""),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
