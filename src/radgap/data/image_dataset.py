"""Dataset PyTorch : lit une ligne de manifest -> (pixel_values, labels, image_path).

Résout le chemin via `radgap.data.paths.resolve_image_path` (gère le variant PNG de CheXpert,
NIH, MURA), charge l'image en RGB, applique la transform du backbone (processor HF ou
preprocess open_clip). Les labels NaN (pathologie non annotée) sont conservés tels quels —
le masquage se fait dans la loss (cf. SKILL `foundation-model-adaptation`).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from radgap.data.paths import resolve_image_path


class CXRImageDataset(Dataset):
    def __init__(self, df, data_root: str | Path, transform, label_cols: list[str]):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.label_cols = label_cols

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        path = resolve_image_path(str(row["image_path"]), str(row["dataset"]), self.data_root)
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(
            row[self.label_cols].to_numpy(dtype="float32"), dtype=torch.float32
        )
        return x, y, str(row["image_path"])
