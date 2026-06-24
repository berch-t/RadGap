"""Tests des métriques AUROC : masquage NaN, classe unique, table + IC bootstrap."""

from __future__ import annotations

import numpy as np

from radgap.eval.metrics import auroc_per_label, evaluate_auroc, macro_auroc


def test_perfect_and_masked():
    # 2 pathologies : la 1re parfaitement séparée, la 2e non annotée (que des NaN)
    y_true = np.array([[0.0, np.nan], [0.0, np.nan], [1.0, np.nan], [1.0, np.nan]])
    y_score = np.array([[0.1, 0.5], [0.2, 0.5], [0.8, 0.5], [0.9, 0.5]])
    auc = auroc_per_label(y_true, y_score)
    assert auc[0] == 1.0  # séparation parfaite
    assert np.isnan(auc[1])  # aucune annotation -> NaN
    assert macro_auroc(y_true, y_score) == 1.0  # nanmean ignore la colonne NaN


def test_single_class_is_nan():
    # une seule classe présente -> AUROC non définie
    y_true = np.array([[1.0], [1.0], [1.0]])
    y_score = np.array([[0.2], [0.7], [0.9]])
    assert np.isnan(auroc_per_label(y_true, y_score)[0])


def test_evaluate_table_has_macro_and_ci():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, (200, 3)).astype(float)
    y_score = rng.random((200, 3))
    table = evaluate_auroc(y_true, y_score, ["a", "b", "c"], n_boot=50, seed=0)
    assert set(table["pathology"]) == {"a", "b", "c", "MACRO"}
    assert (table["ci_lo"] <= table["auroc"]).all()
    assert (table["auroc"] <= table["ci_hi"]).all()
