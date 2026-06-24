"""Chargement des embeddings cachés sur disque (produits par scripts/extract_embeddings.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def embeddings_dir(backbone: str, dataset: str, split: str, data_root: str | Path) -> Path:
    return Path(data_root) / "embeddings" / backbone / dataset / split


def load_cached_embeddings(backbone: str, dataset: str, split: str, data_root: str | Path):
    """Renvoie (embeddings (N,D), labels (N,L), ids list[str], meta dict)."""
    d = embeddings_dir(backbone, dataset, split, data_root)
    emb = np.load(d / "embeddings.npy")
    lab = np.load(d / "labels.npy")
    ids = (d / "ids.txt").read_text().splitlines()
    meta = json.loads((d / "meta.json").read_text())
    return emb, lab, ids, meta
