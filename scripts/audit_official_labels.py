#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABELS = {"ordinary_intergrowth", "fine_intergrowth", "talcose"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit official image labels for duplicate-content conflicts.")
    parser.add_argument("--official-manifest", type=Path, default=Path("outputs/official_manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/official_label_audit"))
    args = parser.parse_args()

    manifest = json.loads(args.official_manifest.read_text(encoding="utf-8"))
    audit = build_label_audit(manifest["items"], dataset_root=args.dataset_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_conflicts_csv(args.out_dir / "label_conflicts.csv", audit["label_conflict_groups"])
    write_duplicates_csv(args.out_dir / "duplicate_groups.csv", audit["duplicate_groups"])
    print(
        json.dumps(
            {
                "labelled_items": audit["labelled_items"],
                "unique_hashes": audit["unique_hashes"],
                "duplicate_group_count": audit["duplicate_group_count"],
                "label_conflict_group_count": audit["label_conflict_group_count"],
                "conflict_path_count": len(audit["conflict_paths"]),
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_label_audit(items: list[dict[str, Any]], *, dataset_root: Path) -> dict[str, Any]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    path_to_sha256: dict[str, str] = {}
    labelled_items = 0
    skipped_by_label = Counter()
    for item in items:
        label = item.get("label_hint", "unknown")
        if label not in LABELS:
            skipped_by_label[label] += 1
            continue
        labelled_items += 1
        rel_path = str(item["path"])
        digest = sha256_file(dataset_root / rel_path)
        path_to_sha256[rel_path] = digest
        by_hash[digest].append(
            {
                "path": rel_path,
                "label": label,
                "width": item.get("width"),
                "height": item.get("height"),
                "bytes": item.get("bytes"),
            }
        )

    duplicate_groups = []
    conflict_groups = []
    conflict_paths: set[str] = set()
    duplicate_paths: set[str] = set()
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
        duplicate_paths.update(item["path"] for item in group_items)
        if len(labels) > 1:
            conflict_groups.append(group)
            conflict_paths.update(item["path"] for item in group_items)

    return {
        "schema_version": "official-label-audit-v0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labelled_items": labelled_items,
        "unique_hashes": len(by_hash),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_path_count": len(duplicate_paths),
        "label_conflict_group_count": len(conflict_groups),
        "label_conflict_path_count": len(conflict_paths),
        "skipped_by_label": dict(skipped_by_label),
        "path_to_sha256": path_to_sha256,
        "duplicate_paths": sorted(duplicate_paths),
        "conflict_paths": sorted(conflict_paths),
        "duplicate_groups": duplicate_groups,
        "label_conflict_groups": conflict_groups,
    }


def write_conflicts_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    rows = []
    for group in groups:
        for item in group["items"]:
            rows.append(
                {
                    "sha256": group["sha256"],
                    "labels_in_group": "|".join(group["labels"]),
                    "path": item["path"],
                    "label": item["label"],
                    "width": item.get("width", ""),
                    "height": item.get("height", ""),
                    "bytes": item.get("bytes", ""),
                }
            )
    write_rows(path, rows, ["sha256", "labels_in_group", "path", "label", "width", "height", "bytes"])


def write_duplicates_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    rows = []
    for group in groups:
        for item in group["items"]:
            rows.append(
                {
                    "sha256": group["sha256"],
                    "labels_in_group": "|".join(group["labels"]),
                    "path": item["path"],
                    "label": item["label"],
                    "group_count": group["count"],
                }
            )
    write_rows(path, rows, ["sha256", "labels_in_group", "path", "label", "group_count"])


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
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
