from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    width: int
    height: int


def iter_tiles(width: int, height: int, tile_size: int, stride: int) -> list[Tile]:
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    xs = _starts(width, tile_size, stride)
    ys = _starts(height, tile_size, stride)
    return [Tile(x=x, y=y, width=tile_size, height=tile_size) for y in ys for x in xs]


def crop_array_with_pad(
    array: np.ndarray,
    tile: Tile,
    fill_value: int | tuple[int, int, int] = 0,
) -> np.ndarray:
    h, w = array.shape[:2]
    x0, y0 = tile.x, tile.y
    x1, y1 = min(x0 + tile.width, w), min(y0 + tile.height, h)
    crop = array[y0:y1, x0:x1]
    pad_h = tile.height - crop.shape[0]
    pad_w = tile.width - crop.shape[1]
    if pad_h == 0 and pad_w == 0:
        return crop
    if array.ndim == 2:
        out = np.full((tile.height, tile.width), fill_value, dtype=array.dtype)
        out[: crop.shape[0], : crop.shape[1]] = crop
    else:
        out = np.full((tile.height, tile.width, array.shape[2]), fill_value, dtype=array.dtype)
        out[: crop.shape[0], : crop.shape[1], :] = crop
    return out


def save_rgb(path: Path, array: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(
        path, quality=quality, optimize=True
    )


def save_gray(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def _starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts
