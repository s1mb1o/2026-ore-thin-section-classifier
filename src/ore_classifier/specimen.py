"""Аншлиф (polished-section / specimen) grouping for leak-free grouped CV.

There is no stored specimen id anywhere in the dataset, manifest, audit, or
split — only file paths. This module derives a best-effort specimen group key
from a dataset-relative image path so that all photos of one physical polished
section stay on one side of a train/val split or CV fold.

Convention mirrors `scripts/train_grade_classifier.py` (path A): a leading run of
>=3 digits in the filename is the specimen number (`spec:<digits>`), otherwise
the file stem is its own singleton group (`file:<stem>`). The key is additionally
scoped by the image's parent directory (the grade folder) so that the ч2
subfolders' per-folder sequential counters (e.g. `150_.jpg` in `оталькованные`
vs `150.JPG` in `рядовые`) are NOT wrongly merged into one group.

Known limitations (documented, not bugs):
  - Coverage is partial: DSCN camera names and bare indices have no numeric
    specimen prefix, so each falls to its own per-file key. This does NOT prevent
    specimen leakage — several DSCN frames of the SAME physical section get
    distinct keys and grouped CV may place them on opposite sides of a fold. We
    cannot recover the specimen id from a Nikon frame counter, so this residual
    leak is accepted for the ~half of images with letter-prefixed names; treat the
    grouped-CV number as slightly optimistic for those. (Numeric-prefixed names,
    the ч1 series, ARE correctly grouped.)
  - Genuine cross-grade specimens (the same real аншлиф number filed under two
    grade folders) become two groups here because the key is folder-scoped; that
    is the safe choice for grouped CV (they carry different labels anyway).
"""
from __future__ import annotations

import re
from pathlib import Path

SPECIMEN_RE = re.compile(r"^\s*(\d{3,})")


def specimen_group(rel_path: str) -> str:
    """Return a specimen group key for a dataset-relative image path.

    All photos of one specimen (as far as the filename reveals) filed in the same
    folder share a key; photos in different folders never share a key.
    """
    p = Path(rel_path)
    match = SPECIMEN_RE.match(p.name)
    key = f"spec:{match.group(1)}" if match else f"file:{p.stem}"
    parent = p.parent.as_posix()
    return f"{parent}|{key}" if parent not in ("", ".") else key
