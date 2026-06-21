"""Tests des loaders sur CSV synthétiques imitant les vrais formats."""

import pandas as pd

from radgap.data import (
    canonical_column,
    load_chexpert_plus,
    load_mura,
    load_nih,
    validate_manifest,
)


def test_load_chexpert_plus(tmp_path):
    paths = [
        "train/patient00001/study1/view1_frontal.jpg",
        "train/patient00002/study1/view1_lateral.jpg",
    ]
    meta = tmp_path / "df_chexpert_plus_240401.csv"
    pd.DataFrame(
        {
            "path_to_image": paths,
            "deid_patient_id": ["patient00001", "patient00002"],
            "frontal_lateral": ["Frontal", "Lateral"],
            "ap_pa": ["PA", ""],
            "sex": ["Male", "Female"],
            "age": [55, 70],
            "race": ["White", "Asian"],
            "split": ["train", "valid"],
        }
    ).to_csv(meta, index=False)

    labels = tmp_path / "impression_fixed.json"
    pd.DataFrame(
        {
            "path_to_image": paths,
            "Cardiomegaly": [1.0, 0.0],
            "Atelectasis": [-1.0, 0.0],  # incertain -> u_ones par défaut
            "Edema": [-1.0, 1.0],  # incertain -> u_ones par défaut
            "No Finding": [0.0, 1.0],
        }
    ).to_json(labels, orient="records", lines=True)

    df = load_chexpert_plus(meta, labels)
    assert df["dataset"].unique().tolist() == ["chexpert_plus"]
    assert df["patient_id"].tolist() == ["chexpert_plus_patient00001", "chexpert_plus_patient00002"]
    assert df["view"].tolist() == ["PA", "LATERAL"]
    assert df["sex"].tolist() == ["M", "F"]
    # politique d'incertitude par pathologie appliquée (join metadata + labels)
    assert df[canonical_column("Atelectasis")].iloc[0] == 1.0
    assert df[canonical_column("Edema")].iloc[0] == 1.0
    assert df[canonical_column("Cardiomegaly")].tolist() == [1.0, 0.0]
    # le loader ne crée pas `split` ; il est assigné ensuite par split_by_patient.
    # Ici on l'assigne à la main (2 patients ne suffisent pas à un split 3-way).
    assert validate_manifest(df.assign(split="train")) == []


def test_load_nih(tmp_path):
    csv = tmp_path / "Data_Entry_2017.csv"
    pd.DataFrame(
        {
            "Image Index": ["00000001_000.png", "00000002_000.png"],
            "Finding Labels": ["Cardiomegaly|Effusion", "No Finding"],
            "Patient ID": [1, 2],
            "Patient Age": [58, 41],
            "Patient Gender": ["M", "F"],
            "View Position": ["PA", "AP"],
        }
    ).to_csv(csv, index=False)

    df = load_nih(csv)
    assert df["patient_id"].tolist() == ["nih_1", "nih_2"]
    assert df[canonical_column("Cardiomegaly")].tolist() == [1.0, 0.0]
    assert df[canonical_column("Pleural Effusion")].iloc[0] == 1.0
    assert df[canonical_column("No Finding")].tolist() == [0.0, 1.0]
    assert df["view"].tolist() == ["PA", "AP"]


def test_load_mura(tmp_path):
    csv = tmp_path / "train_image_paths.csv"
    pd.DataFrame(
        ["MURA-v1.1/train/XR_ELBOW/patient00011/study1_positive/image1.png",
         "MURA-v1.1/train/XR_HAND/patient00012/study1_negative/image1.png"]
    ).to_csv(csv, index=False, header=False)

    df = load_mura(csv)
    assert df["body_part"].tolist() == ["ELBOW", "HAND"]
    assert df["label_abnormal"].tolist() == [1.0, 0.0]
    assert df["patient_id"].tolist() == ["mura_patient00011", "mura_patient00012"]
