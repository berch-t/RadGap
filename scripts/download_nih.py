"""Télécharge NIH ChestX-ray14 depuis Kaggle (sans accréditation — test OOD de repli).

Prérequis : identifiants Kaggle. kagglehub 1.x accepte, par ordre de priorité :
  - `~/.kaggle/access_token` (nouveau token d'accès, fichier brut) — recommandé ;
  - variables `KAGGLE_USERNAME` + `KAGGLE_KEY` ;
  - `~/.kaggle/kaggle.json` (format historique).
Dataset Kaggle : `nih-chest-xrays/data` (~45 Go, 112 120 images).

Ce script force le cache kagglehub sous `RADGAP_DATA_ROOT/raw/kagglehub_cache` (même
disque que les données — évite 45 Go sur le disque virtuel WSL et une copie cross-device),
puis crée un lien de commodité `raw/nih_cxr14` -> dossier téléchargé.

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
    raw = Path(root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # Doit être défini AVANT l'import de kagglehub (lu à la résolution du cache).
    os.environ.setdefault("KAGGLEHUB_CACHE", str(raw / "kagglehub_cache"))

    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub absent. `uv add kagglehub` ou télécharger manuellement le dataset "
            "Kaggle `nih-chest-xrays/data` vers " + str(raw / "nih_cxr14")
        ) from None

    print(f"Téléchargement de NIH ChestX-ray14 (cache : {os.environ['KAGGLEHUB_CACHE']})")
    path = Path(kagglehub.dataset_download("nih-chest-xrays/data"))
    print(f"  ✓ téléchargé : {path}")

    # Lien de commodité vers l'emplacement canonique attendu par le pipeline.
    link = raw / "nih_cxr14"
    if link.is_symlink():
        link.unlink()
    elif link.is_dir() and not any(link.iterdir()):
        link.rmdir()
    if link.exists():
        print(f"  ⚠ {link} existe déjà (non vide) — lien non créé, vérifier manuellement.")
    else:
        link.symlink_to(path, target_is_directory=True)
        print(f"  ✓ lien {link} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
