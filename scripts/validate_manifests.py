"""Valide le manifest unifié (DoD M1) : schéma, fuite patient, valeurs de labels, chemins.

Usage :
  uv run python scripts/validate_manifests.py
  uv run python scripts/validate_manifests.py --check-paths   # vérifie aussi les chemins images
Sortie non nulle si un problème est détecté (utilisable en CI).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from radgap.data import canonical_pathologies, load_manifest, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=None,
        help="chemin du parquet (défaut: $RADGAP_DATA_ROOT/manifests/unified.parquet)",
    )
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="vérifier l'existence des fichiers images",
    )
    args = parser.parse_args()

    path = args.manifest
    if path is None:
        root = os.environ.get("RADGAP_DATA_ROOT")
        if not root:
            raise SystemExit("RADGAP_DATA_ROOT non défini et --manifest non fourni.")
        path = str(Path(root) / "manifests" / "unified.parquet")

    df = load_manifest(path)
    data_root = os.environ.get("RADGAP_DATA_ROOT") if args.check_paths else None
    problems = validate_manifest(df, data_root=data_root)

    print(f"Manifest : {path} — {len(df)} lignes, datasets {sorted(df['dataset'].unique())}")
    cxr = {"chexpert_plus", "nih_cxr14", "padchest", "vindr_cxr"}
    datasets = [d for d in df["dataset"].unique() if d in cxr]
    if "chexpert_plus" in datasets:
        for other in datasets:
            if other != "chexpert_plus":
                shared = canonical_pathologies("chexpert_plus", other)
                print(f"  pathologies communes chexpert_plus ∩ {other} ({len(shared)}) : {shared}")

    if problems:
        print("\n❌ PROBLÈMES :")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\n✅ Manifest valide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
