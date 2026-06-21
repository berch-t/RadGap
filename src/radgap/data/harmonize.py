"""Transforme les labels bruts d'un dataset en colonnes canoniques `label_*`.

Pure et testable : prend un DataFrame de labels bruts, renvoie un DataFrame de colonnes
canoniques (float, valeurs dans {0, 1, NaN}). C'est ici que vit la sémantique du
croisement cross-dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from radgap.data.label_map import LABEL_COLUMNS, LABEL_MAP, canonical_column
from radgap.data.uncertainty import apply_uncertainty_policy, resolve_policy


def apply_uncertainty_to_frame(
    df: pd.DataFrame,
    raw_columns: list[str],
    *,
    global_policy: str | None = None,
) -> pd.DataFrame:
    """Applique la politique d'incertitude CheXpert (-1) colonne par colonne.

    `global_policy` force la même politique partout ; sinon on utilise le défaut par
    pathologie (U-Ones pour Atelectasis/Edema, U-Zeros ailleurs).
    """
    df = df.copy()
    for col in raw_columns:
        if col not in df.columns:
            continue
        policy = resolve_policy(col, override=global_policy)

        def _apply(v, p=policy):
            return apply_uncertainty_policy(float(v), p) if pd.notna(v) else np.nan

        df[col] = df[col].map(_apply)
    return df


def harmonize_labels(df_raw: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Renvoie un DataFrame indexé comme `df_raw` avec les colonnes `label_<canonique>`.

    Agrégation OR (max en ignorant les NaN) quand plusieurs labels bruts pointent vers
    la même pathologie canonique (ex. VinDr `Infiltration` + `Lung Opacity`).
    """
    if dataset not in LABEL_MAP:
        raise ValueError(f"dataset inconnu pour l'harmonisation : {dataset!r}")

    out = pd.DataFrame(
        {col: np.full(len(df_raw), np.nan, dtype=float) for col in LABEL_COLUMNS},
        index=df_raw.index,
    )

    for raw_col, canon in LABEL_MAP[dataset].items():
        if canon is None or raw_col not in df_raw.columns:
            continue
        col = canonical_column(canon)
        raw_vals = pd.to_numeric(df_raw[raw_col], errors="coerce").to_numpy(dtype=float)
        # fmax ignore les NaN : combine plusieurs sources brutes vers une canonique en OR.
        out[col] = np.fmax(out[col].to_numpy(dtype=float), raw_vals)

    return out
