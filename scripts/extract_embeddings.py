"""Extrait et cache les embeddings d'un backbone gelé sur un manifest (M3).

Cœur du protocole RadGap : on passe chaque image dans le backbone gelé **une seule fois**,
on sauvegarde l'embedding sur disque, puis l'entraînement des têtes ne retouche jamais le GPU.
Cache : `<data_root>/embeddings/<backbone>/<dataset>/<split>/{embeddings.npy, labels.npy,
ids.txt, meta.json}`. Ré-exécution idempotente (un split déjà extrait est sauté).

Usage :
  export RADGAP_DATA_ROOT=/chemin/vers/data
  uv run python scripts/extract_embeddings.py                       # backbone=rad_dino (défaut)
  uv run python scripts/extract_embeddings.py backbone=dinov2
  uv run python scripts/extract_embeddings.py backbone=biomedclip batch_size=128
  uv run python scripts/extract_embeddings.py splits=[val] overwrite=true   # ré-extraction ciblée
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from radgap.data import label_columns, load_manifest
from radgap.data.image_dataset import CXRImageDataset
from radgap.models.backbones import Backbone
from radgap.utils import get_logger, set_determinism

log = get_logger("radgap.embed")


def _iter_progress(loader):
    try:
        from tqdm import tqdm

        return tqdm(loader, leave=False)
    except ImportError:
        return loader


@torch.no_grad()
def _extract_split(backbone: Backbone, loader: DataLoader, device: str) -> tuple:
    embs, labels, ids = [], [], []
    use_amp = device == "cuda"
    for pixel_values, y, image_id in _iter_progress(loader):
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            e = backbone.embed(pixel_values)
        embs.append(e.float().cpu())
        labels.append(y)
        ids.extend(image_id)
    return torch.cat(embs).numpy(), torch.cat(labels).numpy(), ids


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_determinism(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(cfg.data_root)

    manifest_name = cfg.get("manifest_name", "available")
    manifest = root / "manifests" / f"{manifest_name}.parquet"
    if not manifest.exists():
        raise SystemExit(f"Manifest absent : {manifest} (lancer build_available_manifest.py).")
    df = load_manifest(manifest)
    label_cols = label_columns(df)
    splits = list(cfg.get("splits", ["train", "val", "test"]))
    batch_size = int(cfg.get("batch_size", 256))
    num_workers = int(cfg.get("num_workers", 8))
    overwrite = bool(cfg.get("overwrite", False))

    bb_name = cfg.backbone.name
    hf_id = cfg.backbone.get("hf_id")
    log.info("Backbone %s (hf_id=%s) sur %s", bb_name, hf_id, device)
    backbone = Backbone(bb_name, hf_id=hf_id, device=device)
    log.info("dim embedding = %d | %d labels", backbone.dim, len(label_cols))

    dataset_name = cfg.dataset.name
    for split in splits:
        sub = df[df["split"] == split]
        if sub.empty:
            log.warning("split %s vide, sauté", split)
            continue
        out_dir = root / "embeddings" / bb_name / dataset_name / split
        if (out_dir / "embeddings.npy").exists() and not overwrite:
            log.info("split %s déjà extrait (%s), sauté", split, out_dir)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        ds = CXRImageDataset(sub, root, backbone.transform, label_cols)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )
        log.info("extraction %s : %d images (batch=%d)…", split, len(ds), batch_size)
        emb, lab, ids = _extract_split(backbone, loader, device)

        np.save(out_dir / "embeddings.npy", emb)
        np.save(out_dir / "labels.npy", lab)
        (out_dir / "ids.txt").write_text("\n".join(ids))
        meta = {
            "backbone": bb_name,
            "hf_id": hf_id,
            "dataset": dataset_name,
            "split": split,
            "n": int(emb.shape[0]),
            "dim": int(emb.shape[1]),
            "label_cols": label_cols,
            "manifest": str(manifest),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        log.info("  ✓ %s : %s -> %s", split, emb.shape, out_dir)

    log.info("Terminé. Config : %s", OmegaConf.to_container(cfg.backbone, resolve=True))


if __name__ == "__main__":
    main()
