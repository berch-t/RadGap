"""Loaders standardisés : fichiers bruts d'un dataset -> DataFrame au schéma unifié.

Chaque loader renvoie un DataFrame contenant les colonnes de métadonnées
(`image_path`, `dataset`, `patient_id`, `view`, `sex`, `age`, `race`) + les colonnes de
labels canoniques `label_*`. Le `split` est assigné ensuite par `splits.split_by_patient`.

Note : les formats de fichiers bruts dépendent de chaque dataset. Les fonctions pures de
transformation (harmonisation, expansion des findings NIH) sont testées sur données
synthétiques ; le chargement disque est documenté ici et exercé à M1 avec les vraies données.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from radgap.data.harmonize import apply_uncertainty_to_frame, harmonize_labels
from radgap.data.label_map import LABEL_MAP

# Observations CheXpert présentes dans le CSV (sous-ensemble géré dans LABEL_MAP).
_CHEXPERT_RAW_COLS = list(LABEL_MAP["chexpert_plus"].keys())
_NIH_RAW_FINDINGS = [c for c in LABEL_MAP["nih_cxr14"] if c != "No Finding"]


def _normalize_view(value: str) -> str:
    v = str(value).strip().upper()
    if v in {"PA", "AP", "LATERAL", "LL", "L"}:
        return "LATERAL" if v in {"LL", "L"} else v
    if "LAT" in v:
        return "LATERAL"
    return "unknown"


def _chexpert_view(frontal_lateral: pd.Series, ap_pa: pd.Series) -> np.ndarray:
    """`frontal_lateral` + `ap_pa` -> PA / AP / LATERAL / unknown."""
    fl = frontal_lateral.astype(str).str.lower()
    appa = ap_pa.astype(str).str.upper()
    return np.where(
        fl.str.startswith("lat"),
        "LATERAL",
        np.where(appa.isin(["AP", "PA"]), appa, "unknown"),
    )


def load_chexpert_plus(
    meta_csv: str | Path,
    labels_jsonl: str | Path,
    *,
    global_policy: str | None = None,
) -> pd.DataFrame:
    """Charge CheXpert Plus au schéma unifié.

    CheXpert Plus sépare métadonnées et labels :
      - `meta_csv`     : table structurée `df_chexpert_plus_240401` (chemin, démographie, vue)
      - `labels_jsonl` : labels CheXbert (un objet par ligne, `path_to_image` + 14 observations),
                         p.ex. `impression_fixed.json` (section impression, défaut CheXpert)

    On joint sur `path_to_image`, applique la politique d'incertitude, puis harmonise.
    `global_policy` : politique d'incertitude (None = défaut par pathologie).
    """
    meta = pd.read_csv(meta_csv)
    labels_raw = pd.read_json(labels_jsonl, lines=True)
    df = meta.merge(labels_raw, on="path_to_image", how="inner")

    # Labels : politique d'incertitude PUIS harmonisation.
    df_unc = apply_uncertainty_to_frame(df, _CHEXPERT_RAW_COLS, global_policy=global_policy)
    labels = harmonize_labels(df_unc, "chexpert_plus")

    empty = pd.Series("", index=df.index)
    out = pd.DataFrame(
        {
            "image_path": df["path_to_image"].astype(str),
            "dataset": "chexpert_plus",
            "patient_id": "chexpert_plus_" + df["deid_patient_id"].astype(str),
            "view": _chexpert_view(df.get("frontal_lateral", empty), df.get("ap_pa", empty)),
            "sex": df.get("sex", empty).astype(str).str[0].str.upper(),
            "age": pd.to_numeric(df.get("age"), errors="coerce"),
            "race": df.get("race", pd.Series(np.nan, index=df.index)),
        }
    )
    return pd.concat([out, labels], axis=1)


def expand_nih_findings(finding_labels: pd.Series) -> pd.DataFrame:
    """`"Cardiomegaly|Effusion"` -> colonnes binaires 0/1 par finding NIH (pure, testée).

    Les labels NIH sont exhaustifs : l'absence d'un finding vaut négatif (0), pas NaN.
    """
    out = {f: np.zeros(len(finding_labels), dtype=float) for f in _NIH_RAW_FINDINGS}
    out["No Finding"] = np.zeros(len(finding_labels), dtype=float)
    for i, raw in enumerate(finding_labels.fillna("").astype(str)):
        findings = {f.strip() for f in raw.split("|") if f.strip()}
        if "No Finding" in findings:
            out["No Finding"][i] = 1.0
        for f in findings:
            if f in out:
                out[f][i] = 1.0
    return pd.DataFrame(out, index=finding_labels.index)


def load_nih(csv_path: str | Path) -> pd.DataFrame:
    """Charge `Data_Entry_2017.csv` (NIH ChestX-ray14) au schéma unifié."""
    df = pd.read_csv(csv_path)
    raw = expand_nih_findings(df["Finding Labels"])
    labels = harmonize_labels(raw, "nih_cxr14")

    view = df["View Position"].map(_normalize_view) if "View Position" in df else "unknown"
    sex_raw = df.get("Patient Gender", pd.Series("unknown", index=df.index))
    out = pd.DataFrame(
        {
            "image_path": df["Image Index"].astype(str),
            "dataset": "nih_cxr14",
            "patient_id": "nih_" + df["Patient ID"].astype(str),
            "view": view,
            "sex": sex_raw.astype(str).str[0].str.upper(),
            "age": pd.to_numeric(df.get("Patient Age"), errors="coerce"),
            "race": np.nan,  # non fourni par NIH
        }
    )
    return pd.concat([out, labels], axis=1)


def load_mura(csv_path: str | Path) -> pd.DataFrame:
    """Charge MURA (tâche binaire normal/anormal par étude, 7 régions osseuses).

    MURA est une tâche disjointe des pathologies CXR -> manifest dédié (C4), pas fusionné
    dans le manifest CXR unifié. Colonnes : image_path, dataset, patient_id, body_part,
    label_abnormal.
    """
    df = pd.read_csv(csv_path, header=None, names=["path"])
    # Chemin type : MURA-v1.1/train/XR_ELBOW/patient00011/study1_positive/image1.png
    body_part = df["path"].str.extract(r"XR_([A-Z]+)")[0]
    patient = df["path"].str.extract(r"(patient\d+)")[0]
    abnormal = df["path"].str.contains("positive").astype(float)
    return pd.DataFrame(
        {
            "image_path": df["path"].astype(str),
            "dataset": "mura",
            "patient_id": "mura_" + patient.fillna("unknown"),
            "body_part": body_part.fillna("unknown"),
            "label_abnormal": abnormal,
        }
    )
