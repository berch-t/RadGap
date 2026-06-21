"""Tests de la table d'harmonisation des labels (artefact central C1)."""

import pandas as pd

from radgap.data import (
    CANONICAL,
    LABEL_COLUMNS,
    canonical_column,
    canonical_pathologies,
    harmonize_labels,
)


def test_canonical_column():
    assert canonical_column("Pleural Effusion") == "label_pleural_effusion"
    assert canonical_column("No Finding") == "label_no_finding"


def test_label_columns_match_canonical():
    assert LABEL_COLUMNS == [canonical_column(p) for p in CANONICAL]


def test_canonical_pathologies_shared_subset():
    shared = canonical_pathologies("chexpert_plus", "nih_cxr14")
    # NIH n'a pas Fracture ; CheXpert oui -> Fracture exclu du sous-ensemble commun
    assert "Fracture" not in shared
    assert "Pleural Effusion" in shared  # NIH "Effusion" -> Pleural Effusion
    assert "Lung Opacity" in shared  # NIH "Infiltration" -> Lung Opacity
    # ordre canonique préservé
    assert shared == [p for p in CANONICAL if p in shared]


def test_harmonize_nih_name_remap():
    raw = pd.DataFrame({"Effusion": [1.0, 0.0], "Infiltration": [0.0, 1.0]})
    out = harmonize_labels(raw, "nih_cxr14")
    assert out["label_pleural_effusion"].tolist() == [1.0, 0.0]
    assert out["label_lung_opacity"].tolist() == [0.0, 1.0]


def test_harmonize_or_aggregation():
    # VinDr : Lung Opacity ET Infiltration -> tous deux vers label_lung_opacity (OR/max)
    raw = pd.DataFrame({"Lung Opacity": [1.0, 0.0, 0.0], "Infiltration": [0.0, 1.0, 0.0]})
    out = harmonize_labels(raw, "vindr_cxr")
    assert out["label_lung_opacity"].tolist() == [1.0, 1.0, 0.0]


def test_harmonize_unmapped_is_nan():
    raw = pd.DataFrame({"Cardiomegaly": [1.0]})
    out = harmonize_labels(raw, "chexpert_plus")
    # une pathologie non présente dans le brut reste NaN (non labellisée), jamais 0
    assert out["label_edema"].isna().all()
