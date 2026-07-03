#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


LABELS = ("ordinary_intergrowth", "fine_intergrowth", "talcose")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build balanced official image-level eval split.")
    parser.add_argument("--official-manifest", type=Path, default=Path("outputs/official_manifest.json"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/official_balanced_eval_split.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/official_balanced_eval_split.csv"))
    parser.add_argument("--label-audit-json", type=Path, default=None)
    parser.add_argument("--exclude-conflicts", action="store_true")
    parser.add_argument("--dedupe-sha256", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    manifest = json.loads(args.official_manifest.read_text(encoding="utf-8"))
    audit = load_label_audit(args.label_audit_json)
    conflict_paths = set(audit.get("conflict_paths", [])) if args.exclude_conflicts else set()
    path_to_sha256 = audit.get("path_to_sha256", {}) if args.dedupe_sha256 else {}
    seen_hashes: set[str] = set()
    by_label: dict[str, list[dict]] = defaultdict(list)
    panorama_items = []
    excluded = Counter()
    excluded_conflicts = 0
    excluded_duplicates = 0
    for item in manifest["items"]:
        label = item.get("label_hint", "unknown")
        if label in LABELS:
            if item["path"] in conflict_paths:
                excluded_conflicts += 1
                continue
            digest = path_to_sha256.get(item["path"])
            if digest:
                if digest in seen_hashes:
                    excluded_duplicates += 1
                    continue
                seen_hashes.add(digest)
            by_label[label].append(item)
        elif label == "panorama":
            panorama_items.append(item)
        else:
            excluded[label] += 1

    available_counts = {label: len(by_label[label]) for label in LABELS}
    per_class = min(available_counts.values())
    if args.max_per_class > 0:
        per_class = min(per_class, args.max_per_class)
    rng = random.Random(args.seed)
    selected = []
    for label in LABELS:
        candidates = list(by_label[label])
        rng.shuffle(candidates)
        for item in sorted(candidates[:per_class], key=lambda x: x["path"]):
            selected.append(
                {
                    "path": item["path"],
                    "label": label,
                    "width": item["width"],
                    "height": item["height"],
                    "bytes": item["bytes"],
                }
            )
    selected.sort(key=lambda x: (x["label"], x["path"]))
    output = {
        "schema_version": "official-balanced-eval-split-v0.2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_manifest": str(args.official_manifest),
        "seed": args.seed,
        "labels": list(LABELS),
        "available_counts": available_counts,
        "label_audit_json": str(args.label_audit_json) if args.label_audit_json else None,
        "exclude_conflicts": bool(args.exclude_conflicts),
        "dedupe_sha256": bool(args.dedupe_sha256),
        "excluded_conflict_paths": excluded_conflicts,
        "excluded_duplicate_paths": excluded_duplicates,
        "selected_per_class": per_class,
        "selected_total": len(selected),
        "panorama_count_unlabelled": len(panorama_items),
        "excluded_counts": dict(excluded),
        "items": selected,
        "panorama_items_unlabelled": [
            {
                "path": item["path"],
                "width": item["width"],
                "height": item["height"],
                "bytes": item["bytes"],
            }
            for item in panorama_items
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.out_csv, selected)
    print(json.dumps({k: output[k] for k in ("selected_per_class", "selected_total", "available_counts", "panorama_count_unlabelled")}, ensure_ascii=False, indent=2))
    return 0


def load_label_audit(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("path", "label", "width", "height", "bytes"))
        writer.writeheader()
        writer.writerows(items)


if __name__ == "__main__":
    raise SystemExit(main())
