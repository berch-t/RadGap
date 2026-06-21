"""Télécharge NIH ChestX-ray14 depuis Kaggle (sans accréditation — test OOD de repli).

Prérequis : identifiants Kaggle (`~/.kaggle/kaggle.json` ou variables KAGGLE_USERNAME /
KAGGLE_KEY). Dataset : `nih-chest-xrays/data`.

Usage :
  export RADGAP_DATA_ROOT=/chemin/vers/data
  uv run python scripts/download_nih.py
"""

from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    root = os.environ.get("RADGAP_DATA_ROOT")
    if not root:
        raise SystemExit("RADGAP_DATA_ROOT non défini (cf. env.example).")
    out_dir = Path(root) / "raw" / "nih_cxr14"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub absent. `uv add kagglehub` ou télécharger manuellement le dataset "
            "Kaggle `nih-chest-xrays/data` vers " + str(out_dir)
        ) from None

    print(f"Téléchargement de NIH ChestX-ray14 -> {out_dir}")
    path = kagglehub.dataset_download("nih-chest-xrays/data")
    print(f"  ✓ téléchargé dans le cache kagglehub : {path}")
    print(f"  → lier/copier le contenu vers {out_dir} (Data_Entry_2017.csv + images/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
