"""Tests des splits par patient : aucune fuite, reproductibilité."""

import pandas as pd

from radgap.data import split_by_patient


def _toy(n_patients=50, studies_per_patient=3):
    rows = []
    for p in range(n_patients):
        for s in range(studies_per_patient):
            rows.append({"patient_id": f"p{p}", "study": s})
    return pd.DataFrame(rows)


def test_no_patient_leak():
    df = split_by_patient(_toy(), seed=42)
    per_patient = df.groupby("patient_id")["split"].nunique()
    assert (per_patient == 1).all()


def test_all_splits_present():
    df = split_by_patient(_toy(), seed=42)
    assert set(df["split"].unique()) == {"train", "val", "test"}


def test_reproducible():
    a = split_by_patient(_toy(), seed=123)
    b = split_by_patient(_toy(), seed=123)
    pd.testing.assert_series_equal(a["split"], b["split"])


def test_input_not_mutated():
    df = _toy()
    assert "split" not in df.columns
    _ = split_by_patient(df, seed=1)
    assert "split" not in df.columns  # entrée intacte
