from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BinarySulfideTileDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        augment: bool = False,
        normalize: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        base_dir = self.manifest_path.parent
        self.items = [item for item in manifest["items"] if item["split"] == split]
        if not self.items:
            raise ValueError(f"manifest has no items for split={split!r}")
        self.base_dir = base_dir
        self.augment = augment
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = Image.open(self.base_dir / item["image"]).convert("RGB")
        mask = Image.open(self.base_dir / item["mask"]).convert("L")
        ignore = Image.open(self.base_dir / item["ignore"]).convert("L")

        if self.augment:
            image, mask, ignore = self._augment(image, mask, ignore)

        image_tensor = TF.to_tensor(image)
        if self.normalize:
            image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)

        mask_np = np.asarray(mask, dtype=np.uint8)
        ignore_np = np.asarray(ignore, dtype=np.uint8)
        target = np.where(ignore_np > 0, 255, np.where(mask_np > 0, 1, 0)).astype(np.int64)

        return {
            "image": image_tensor,
            "mask": torch.from_numpy(target),
            "meta": json.dumps(item, ensure_ascii=False),
        }

    @staticmethod
    def _augment(image: Image.Image, mask: Image.Image, ignore: Image.Image):
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            ignore = TF.hflip(ignore)
        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            ignore = TF.vflip(ignore)
        rotations = random.randint(0, 3)
        if rotations:
            image = image.rotate(90 * rotations, expand=False)
            mask = mask.rotate(90 * rotations, expand=False)
            ignore = ignore.rotate(90 * rotations, expand=False)
        if random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.2))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        return image, mask, ignore
