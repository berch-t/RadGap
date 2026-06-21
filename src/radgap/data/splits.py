"""Splits train/val/test **par patient** — jamais par image (sinon fuite et métriques gonflées).

Un patient avec plusieurs études doit vivre dans un seul split. Voir SKILL
`medical-imaging-data`.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_by_patient(
    df: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Assigne `train`/`val`/`test` dans une colonne `split`, groupé par patient.

    Renvoie une copie ; l'entrée n'est pas mutée.
    """
    df = df.copy().reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss.split(df, groups=df[patient_col]))

    trainval = df.iloc[trainval_idx]
    rel_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    train_local, val_local = next(gss2.split(trainval, groups=trainval[patient_col]))

    df["split"] = "train"
    df.loc[trainval.iloc[val_local].index, "split"] = "val"
    df.loc[df.index[test_idx], "split"] = "test"
    return df
