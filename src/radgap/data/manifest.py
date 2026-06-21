"""Construction, sauvegarde et chargement du manifest unifié (data/manifests/unified.parquet)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from radgap.data.label_map import LABEL_COLUMNS
from radgap.data.schema import METADATA_COLUMNS, validate_manifest


def build_unified_manifest(per_dataset: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatène les manifests par dataset en un manifest unifié au schéma standard.

    Chaque DataFrame d'entrée doit déjà contenir les colonnes de métadonnées + `label_*`
    (colonnes de labels absentes -> NaN, sémantique "non labellisé", jamais 0).
    """
    if not per_dataset:
        raise ValueError("aucun manifest à unifier")

    all_cols = METADATA_COLUMNS + LABEL_COLUMNS
    frames = []
    for df in per_dataset:
        df = df.copy()
        for col in all_cols:
            if col not in df.columns:
                df[col] = pd.NA if col in METADATA_COLUMNS else float("nan")
        frames.append(df[all_cols])

    unified = pd.concat(frames, ignore_index=True)
    return unified


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_validate_save(
    per_dataset: list[pd.DataFrame],
    out_path: str | Path,
    *,
    data_root: str | None = None,
) -> pd.DataFrame:
    """Pipeline M1 : unifie -> valide -> sauve. Lève si le manifest est invalide."""
    unified = build_unified_manifest(per_dataset)
    problems = validate_manifest(unified, data_root=data_root)
    if problems:
        raise ValueError("manifest invalide :\n  - " + "\n  - ".join(problems))
    save_manifest(unified, out_path)
    return unified
