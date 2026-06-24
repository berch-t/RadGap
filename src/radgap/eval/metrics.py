"""AUROC multi-label avec masquage des NaN et intervalles de confiance bootstrap.

Règles non négociables (cf. SKILL `medical-ml-evaluation`) :
  - tout chiffre a un IC (bootstrap) ;
  - métriques par pathologie d'abord, puis macro-AUROC en tête de gondole ;
  - les NaN (non annotés) sont exclus de chaque métrique, jamais comptés négatifs ;
  - une pathologie sans les deux classes présentes -> AUROC = NaN (non définie).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _auroc_vector(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """AUROC par colonne (L,), NaN si non définie (classe unique ou aucun label)."""
    n_labels = y_true.shape[1]
    out = np.full(n_labels, np.nan)
    for j in range(n_labels):
        mask = ~np.isnan(y_true[:, j])
        yt = y_true[mask, j]
        if yt.size == 0 or np.unique(yt).size < 2:
            continue
        out[j] = roc_auc_score(yt, y_score[mask, j])
    return out


def auroc_per_label(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    return _auroc_vector(y_true, y_score)


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(np.nanmean(_auroc_vector(y_true, y_score)))


def bootstrap_ci(y_true, y_score, metric_fn, *, n=1000, alpha=0.05, seed=0):
    """IC bootstrap (percentile) d'un scalaire ; renvoie (moyenne, lo, hi)."""
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    stats = [
        metric_fn(y_true[idx], y_score[idx])
        for idx in (rng.integers(0, n_samples, n_samples) for _ in range(n))
    ]
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.nanmean(stats)), float(lo), float(hi)


def evaluate_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Table de résultats : une ligne par pathologie (AUROC + IC + tailles) + une ligne macro.

    Une seule passe de bootstrap calcule à la fois les IC par pathologie et l'IC macro
    (rééchantillonnage cohérent : mêmes indices pour toutes les colonnes à chaque tirage).
    """
    rng = np.random.default_rng(seed)
    n_samples, n_labels = y_true.shape
    point = _auroc_vector(y_true, y_score)

    boot = np.empty((n_boot, n_labels))
    macro_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_samples, n_samples)
        v = _auroc_vector(y_true[idx], y_score[idx])
        boot[b] = v
        macro_boot[b] = np.nanmean(v)

    lo, hi = np.nanpercentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    mlo, mhi = np.nanpercentile(macro_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    rows = []
    for j, name in enumerate(label_names):
        labeled = ~np.isnan(y_true[:, j])
        rows.append(
            {
                "pathology": name,
                "auroc": point[j],
                "ci_lo": lo[j],
                "ci_hi": hi[j],
                "n_labeled": int(labeled.sum()),
                "n_pos": int(np.nansum(y_true[:, j] == 1)),
            }
        )
    rows.append(
        {
            "pathology": "MACRO",
            "auroc": float(np.nanmean(point)),
            "ci_lo": float(mlo),
            "ci_hi": float(mhi),
            "n_labeled": int((~np.isnan(y_true)).any(axis=1).sum()),
            "n_pos": int(np.nansum(y_true == 1)),
        }
    )
    return pd.DataFrame(rows)
