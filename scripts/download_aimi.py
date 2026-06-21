"""Télécharge les datasets Stanford AIMI (CheXpert Plus, MURA) depuis Redivis.

Stanford AIMI distribue ses datasets via Redivis (Stanford Data Farm). La clé d'API
Stanford AIMI est un **token Redivis** : on l'expose sous `REDIVIS_API_TOKEN`.

Auth (par ordre de priorité) :
  1. variable d'env `REDIVIS_API_TOKEN`
  2. variable d'env `AIMI_API_KEY`
  3. fichier `AIMI_API_KEY.txt` à la racine du repo (gitignored)

Usage :
  export RADGAP_DATA_ROOT=/chemin/vers/data
  uv run python scripts/download_aimi.py --dataset chexpert_plus
  uv run python scripts/download_aimi.py --dataset mura
  uv run python scripts/download_aimi.py --list            # liste les tables disponibles

Repli manuel : si Redivis échoue, télécharger depuis le portail web (research use
agreement à accepter) — https://aimi.stanford.edu/datasets/chexpert-plus et
https://aimi.stanford.edu/datasets/mura-msk-xrays — puis dézipper sous
`$RADGAP_DATA_ROOT/raw/<dataset>/`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Références Redivis (organisation "stanford" / Stanford Data Farm).
REDIVIS_ORG = "stanford"
DATASETS = {
    "chexpert_plus": "5yyj-1a9f6ap0x",  # stanford.redivis.com/datasets/5yyj-1a9f6ap0x
    "mura": "cv1a-apytk3j44",  # stanford.redivis.com/datasets/cv1a-apytk3j44
}


def _resolve_token() -> str:
    token = os.environ.get("REDIVIS_API_TOKEN") or os.environ.get("AIMI_API_KEY")
    if not token:
        key_file = Path(__file__).resolve().parents[1] / "AIMI_API_KEY.txt"
        if key_file.exists():
            token = key_file.read_text().strip()
    if not token:
        raise SystemExit(
            "Token Redivis introuvable. Définir REDIVIS_API_TOKEN ou AIMI_API_KEY, "
            "ou placer la clé dans AIMI_API_KEY.txt."
        )
    return token


def _data_root() -> Path:
    root = os.environ.get("RADGAP_DATA_ROOT")
    if not root:
        raise SystemExit("RADGAP_DATA_ROOT non défini (cf. env.example).")
    return Path(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), help="dataset à télécharger")
    parser.add_argument("--list", action="store_true", help="lister les tables du dataset")
    args = parser.parse_args()

    os.environ["REDIVIS_API_TOKEN"] = _resolve_token()
    import redivis  # import tardif : la dépendance n'est requise que pour le download

    if not args.dataset and not args.list:
        parser.error("préciser --dataset, et/ou --list")

    target = args.dataset or "chexpert_plus"
    ds = redivis.organization(REDIVIS_ORG).dataset(DATASETS[target])

    if args.list:
        print(f"Tables de {target} :")
        for t in ds.list_tables():
            print(f"  - {t.name}")
        return 0

    out_dir = _data_root() / "raw" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement de {target} -> {out_dir}")
    for table in ds.list_tables():
        try:
            table.download_files(path=str(out_dir), overwrite=False)
            print(f"  ✓ {table.name}")
        except Exception as exc:  # noqa: BLE001  (on continue les autres tables)
            print(f"  ⚠ {table.name} : {exc}")
    print("Terminé. Vérifier le contenu puis lancer scripts/build_manifest.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
