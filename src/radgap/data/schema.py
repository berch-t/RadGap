"""Schéma du manifest unifié et checks d'intégrité (le contrat dont dépend tout le code aval).

Chaque ligne = une image. Voir SKILL `medical-imaging-data`.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Colonnes de métadonnées obligatoires (hors colonnes de labels `label_*`).
METADATA_COLUMNS: list[str] = [
    "image_path",  # chemin relatif au data_root (jamais absolu/machine-specific dans le parquet)
    "dataset",  # chexpert_plus / nih_cxr14 / padchest / vindr_cxr
    "patient_id",  # id stable, préfixé par dataset pour éviter les collisions
    "split",  # train / val / test (assigné par patient)
    "view",  # PA / AP / LATERAL / unknown
    "sex",  # M / F / unknown (pour C3)
    "age",  # années, ou NaN
    "race",  # auto-déclarée, CheXpert uniquement (pour C3) — à manier avec précaution
]

VALID_SPLITS = {"train", "val", "test"}


def label_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("label_")]


def validate_manifest(df: pd.DataFrame, *, data_root: str | None = None) -> list[str]:
    """Renvoie la liste des problèmes détectés (vide = manifest valide).

    Checks (cf. DoD M1) :
      - colonnes de métadonnées présentes
      - 0 fuite patient inter-split
      - valeurs de labels dans {0, 1, NaN} uniquement
      - splits valides
      - (si `data_root` fourni) 0 chemin image cassé
    """
    problems: list[str] = []

    # Métadonnées présentes
    missing = [c for c in METADATA_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"colonnes de métadonnées manquantes : {missing}")

    # Splits valides
    if "split" in df.columns:
        bad = set(df["split"].dropna().unique()) - VALID_SPLITS
        if bad:
            problems.append(f"valeurs de split invalides : {bad}")

    # Fuite patient inter-split (un patient ne doit apparaître que dans un seul split)
    if {"patient_id", "split"} <= set(df.columns):
        per_patient_splits = df.groupby("patient_id")["split"].nunique()
        leaked = per_patient_splits[per_patient_splits > 1]
        if len(leaked) > 0:
            problems.append(
                f"fuite patient : {len(leaked)} patient(s) dans plusieurs splits "
                f"(ex. {list(leaked.index[:3])})"
            )

    # Valeurs de labels dans {0, 1, NaN}
    for col in label_columns(df):
        vals = df[col].to_numpy(dtype=float)
        finite = vals[~np.isnan(vals)]
        bad_vals = set(np.unique(finite)) - {0.0, 1.0}
        if bad_vals:
            problems.append(f"colonne {col} : valeurs hors {{0,1,NaN}} -> {bad_vals}")

    # Chemins images (optionnel, nécessite les fichiers sur disque)
    if data_root is not None and "image_path" in df.columns:
        n_broken = 0
        for rel in df["image_path"]:
            if not os.path.exists(os.path.join(data_root, str(rel))):
                n_broken += 1
        if n_broken:
            problems.append(f"{n_broken} chemin(s) image cassé(s) sous {data_root}")

    return problems
