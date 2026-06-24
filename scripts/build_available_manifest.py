"""Construit un manifest restreint aux images réellement présentes sur disque.

Tant que le dataset complet n'est pas téléchargé (limite d'egress Redivis 100 Go / 30 j),
on travaille sur le sous-ensemble disponible. Ce script lit le manifest unifié, filtre les
lignes dont le fichier image existe (résolution par dataset, cf. `radgap.data.paths`), et
écrit `manifests/available.parquet`. C'est ce manifest que consomment l'extraction
d'embeddings (M3) et les notebooks 02->05 en attendant le full dataset.

Usage :
  export RADGAP_DATA_ROOT=/chemin/vers/data
  uv run python scripts/build_available_manifest.py
"""

from __future__ import annotations

import os
from pathlib import Path

from radgap.data import filter_available, label_columns, load_manifest


def main() -> int:
    root = os.environ.get("RADGAP_DATA_ROOT")
    if not root:
        raise SystemExit("RADGAP_DATA_ROOT non défini (cf. env.example).")
    root = Path(root)
    src = root / "manifests" / "unified.parquet"
    if not src.exists():
        raise SystemExit(f"Manifest unifié absent : {src} (lancer build_manifest.py).")

    df = load_manifest(src)
    print(f"Manifest unifié : {len(df):,} lignes")
    print("Filtrage sur les images présentes (scan disque)…")
    avail = filter_available(df, root)

    pct = 100 * len(avail) / len(df) if len(df) else 0
    print(f"Disponibles : {len(avail):,} / {len(df):,} lignes ({pct:.1f} %)")
    if len(avail):
        print("  splits :", avail["split"].value_counts().to_dict())
        print("  patients :", avail["patient_id"].nunique())
        print("  datasets :", avail["dataset"].value_counts().to_dict())
        # Prévalences (sanity, sur labels non-NaN)
        for col in label_columns(avail):
            n = int(avail[col].notna().sum())
            print(f"    {col}: prévalence {avail[col].mean():.3f} (n={n})")

    out = root / "manifests" / "available.parquet"
    avail.to_parquet(out, index=False)
    print(f"✓ écrit {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
