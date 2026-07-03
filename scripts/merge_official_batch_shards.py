#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge multiple run_official_batch.py shard outputs.")
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    result = merge_shards(args.shard_dirs, out_dir=args.out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def merge_shards(shard_dirs: list[Path], *, out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()

    for shard_dir in shard_dirs:
        shard_rows = read_rows(shard_dir / "summary.csv")
        duplicate_run_ids = sorted(
            run_id
            for run_id in (str(row.get("run_id", "")) for row in shard_rows)
            if run_id and run_id in seen_run_ids
        )
        if duplicate_run_ids:
            raise ValueError(f"duplicate run_id values across shards: {', '.join(duplicate_run_ids)}")
        seen_run_ids.update(str(row.get("run_id", "")) for row in shard_rows if row.get("run_id"))
        for row in shard_rows:
            rows.append({**row, "shard_dir": str(shard_dir)})

        shard_failures = read_failures(shard_dir / "failures.json")
        for failure in shard_failures:
            failures.append({**failure, "shard_dir": str(shard_dir)})

        shards.append(
            {
                "shard_dir": str(shard_dir),
                "rows": len(shard_rows),
                "failures": len(shard_failures),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "summary.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps({"rows": rows, "failures": failures, "shards": shards}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"rows": len(rows), "failures": len(failures), "shards": shards, "out_dir": str(out_dir)}


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_failures(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"failures JSON must contain a list: {path}")
    return payload


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = ["run_id", "source_label", "expected_ore_class", "source_rel_path", "predicted_ore_class"]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
