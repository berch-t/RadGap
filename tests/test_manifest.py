"""Tests du manifest unifié : schéma, validation, round-trip parquet, expansion NIH."""

import numpy as np
import pandas as pd

from radgap.data import (
    LABEL_COLUMNS,
    METADATA_COLUMNS,
    build_unified_manifest,
    canonical_column,
    expand_nih_findings,
    load_manifest,
    save_manifest,
    split_by_patient,
    validate_manifest,
)


def _toy_dataset(name, n=20, leak=False):
    rows = []
    for i in range(n):
        rows.append(
            {
                "image_path": f"{name}/img{i}.png",
                "dataset": name,
                "patient_id": f"{name}_p{i // 2}",
                "view": "PA",
                "sex": "M" if i % 2 else "F",
                "age": 40.0 + i,
                "race": np.nan,
                canonical_column("Cardiomegaly"): float(i % 2),
            }
        )
    df = pd.DataFrame(rows)
    df = split_by_patient(df, seed=0)
    if leak:
        # force un patient dans deux splits
        df.loc[0, "split"] = "train"
        df.loc[1, "split"] = "test"  # même patient _p0
    return df


def test_unified_schema_complete():
    uni = build_unified_manifest([_toy_dataset("chexpert_plus"), _toy_dataset("nih_cxr14")])
    for col in METADATA_COLUMNS + LABEL_COLUMNS:
        assert col in uni.columns
    assert len(uni) == 40


def test_validate_clean_manifest():
    uni = build_unified_manifest([_toy_dataset("chexpert_plus")])
    assert validate_manifest(uni) == []


def test_validate_detects_patient_leak():
    leaked = _toy_dataset("chexpert_plus", leak=True)
    problems = validate_manifest(leaked)
    assert any("fuite patient" in p for p in problems)


def test_validate_detects_bad_label_value():
    uni = build_unified_manifest([_toy_dataset("chexpert_plus")])
    uni.loc[0, canonical_column("Cardiomegaly")] = 2.0  # hors {0,1,NaN}
    problems = validate_manifest(uni)
    assert any("hors {0,1,NaN}" in p for p in problems)


def test_parquet_roundtrip(tmp_path):
    uni = build_unified_manifest([_toy_dataset("chexpert_plus")])
    path = tmp_path / "unified.parquet"
    save_manifest(uni, path)
    back = load_manifest(path)
    assert list(back.columns) == list(uni.columns)
    assert len(back) == len(uni)


def test_expand_nih_findings():
    s = pd.Series(["Cardiomegaly|Effusion", "No Finding", "Infiltration"])
    out = expand_nih_findings(s)
    assert out["Cardiomegaly"].tolist() == [1.0, 0.0, 0.0]
    assert out["Effusion"].tolist() == [1.0, 0.0, 0.0]
    assert out["No Finding"].tolist() == [0.0, 1.0, 0.0]
    assert out["Infiltration"].tolist() == [0.0, 0.0, 1.0]
