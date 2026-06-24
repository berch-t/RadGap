"""Résolution des chemins d'images : manifest (relatif, agnostique) -> fichier réel sur disque.

Le manifest stocke un `image_path` relatif et **agnostique du variant** (ex. CheXpert :
`train/patientX/studyN/view1_frontal.jpg`, hérité des métadonnées d'origine). Les fichiers
réellement téléchargés vivent ailleurs et dans un autre format selon le variant :

  - chexpert_plus (variant PNG) : `raw/chexpert_plus/PNG/<image_path, .jpg -> .png>`
  - mura                        : `raw/mura/<image_path>`
  - nih_cxr14                   : `raw/nih_cxr14/.../<basename>` (images réparties en images_*/)

Ce module centralise ce mapping pour que loaders, preprocessing et extraction d'embeddings
résolvent les chemins de façon identique. Voir SKILL `medical-imaging-data`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Sous-dossier racine des images de chaque dataset, relatif au data_root.
DATASET_IMAGE_ROOT: dict[str, str] = {
    "chexpert_plus": "raw/chexpert_plus/PNG",
    "mura": "raw/mura",
    "nih_cxr14": "raw/nih_cxr14",
}


def resolve_image_path(image_path: str, dataset: str, data_root: str | Path) -> Path:
    """Chemin absolu du fichier image pour une ligne de manifest.

    Ne vérifie pas l'existence (rapide) ; utiliser `Path.exists()` ou `filter_available`
    pour cela. Pour NIH, les images sont réparties dans des sous-dossiers `images_*/images/`,
    donc on résout via un index construit une fois (cf. `_nih_index`).
    """
    data_root = Path(data_root)
    if dataset == "chexpert_plus":
        rel = image_path[:-4] + ".png" if image_path.endswith(".jpg") else image_path
        return data_root / DATASET_IMAGE_ROOT["chexpert_plus"] / rel
    if dataset == "mura":
        return data_root / DATASET_IMAGE_ROOT["mura"] / image_path
    if dataset == "nih_cxr14":
        nih_root = data_root / DATASET_IMAGE_ROOT["nih_cxr14"]
        index = _nih_index(str(nih_root))
        return index.get(Path(image_path).name, nih_root / image_path)
    return data_root / image_path


@lru_cache(maxsize=8)
def _nih_index(nih_root: str) -> dict[str, Path]:
    """Index basename -> chemin (NIH éclate les 112k images en images_001..012/images/)."""
    root = Path(nih_root)
    if not root.exists():
        return {}
    return {p.name: p for p in root.rglob("*.png")}


def present_image_keys(dataset: str, data_root: str | Path) -> set[str]:
    """Ensemble des `image_path` (convention manifest) effectivement présents sur disque.

    Parcourt l'arborescence du dataset **une seule fois** (bien plus rapide qu'un `stat`
    par ligne sur un mount réseau), et renvoie les clés au format du manifest pour permettre
    un `df['image_path'].isin(...)` vectorisé.
    """
    data_root = Path(data_root)
    root = data_root / DATASET_IMAGE_ROOT.get(dataset, "")
    if not root.exists():
        return set()
    if dataset == "chexpert_plus":
        # fichiers .png -> reconvertis en clé .jpg (convention manifest)
        return {p.relative_to(root).as_posix()[:-4] + ".jpg" for p in root.rglob("*.png")}
    if dataset == "nih_cxr14":
        # manifest NIH = basename (Image Index)
        return {p.name for p in root.rglob("*.png")}
    # défaut : chemin relatif tel quel
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def filter_available(df, data_root: str | Path):
    """Sous-ensemble des lignes dont le fichier image est présent sur disque (tous datasets)."""
    import pandas as pd

    parts = []
    for dataset, sub in df.groupby("dataset"):
        keys = present_image_keys(str(dataset), data_root)
        parts.append(sub[sub["image_path"].astype(str).isin(keys)])
    return pd.concat(parts) if parts else df.iloc[0:0]
