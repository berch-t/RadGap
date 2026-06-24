"""Tests du résolveur de chemins (manifest -> fichier réel) et du filtrage disque."""

from __future__ import annotations

import pandas as pd

from radgap.data import filter_available, resolve_image_path


def test_resolve_chexpert_jpg_to_png(tmp_path):
    rel = "train/patient00001/study1/view1_frontal"
    p = resolve_image_path(rel + ".jpg", "chexpert_plus", tmp_path)
    assert p == tmp_path / "raw" / "chexpert_plus" / "PNG" / (rel + ".png")


def test_resolve_mura_passthrough(tmp_path):
    rel = "MURA-v1.1/train/XR_ELBOW/patient00011/study1_positive/image1.png"
    assert resolve_image_path(rel, "mura", tmp_path) == tmp_path / "raw" / "mura" / rel


def test_filter_available_keeps_only_present(tmp_path):
    # Crée une image présente et une absente
    png_root = tmp_path / "raw" / "chexpert_plus" / "PNG" / "train" / "patientA" / "study1"
    png_root.mkdir(parents=True)
    (png_root / "view1_frontal.png").write_bytes(b"x")

    df = pd.DataFrame(
        {
            "image_path": [
                "train/patientA/study1/view1_frontal.jpg",  # présent (en .png sur disque)
                "train/patientZ/study1/view1_frontal.jpg",  # absent
            ],
            "dataset": ["chexpert_plus", "chexpert_plus"],
        }
    )
    avail = filter_available(df, tmp_path)
    assert len(avail) == 1
    assert avail.iloc[0]["image_path"] == "train/patientA/study1/view1_frontal.jpg"
