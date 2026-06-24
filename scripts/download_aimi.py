"""Télécharge les datasets Stanford AIMI (CheXpert Plus, MURA) depuis Redivis.

Stanford AIMI distribue ses datasets via Redivis, organisation **AIMI**. La clé d'API
Stanford AIMI est un **token Redivis** : on l'expose sous `REDIVIS_API_TOKEN`.

Structure CheXpert Plus (org AIMI) :
  - `df_chexpert_plus_240401` : table STRUCTURÉE (labels + démographie) -> récupérée en DataFrame
  - `PNG_train` / `PNG_valid`  : images PNG (download fichier par fichier, ~223k fichiers)
  - `PNG_compressed`           : 5 archives .zip (~720 Go) — download groupé bien plus rapide ;
                                 ce script les télécharge ET les extrait une par une (pic disque
                                 ~155 Go, l'archive est supprimée après extraction)
  - `DICOM_train` / `DICOM_valid` : DICOM haute-résolution (résultats finaux)

Auth (par ordre de priorité) : env `REDIVIS_API_TOKEN`, env `AIMI_API_KEY`, fichier
`AIMI_API_KEY.txt` (gitignored).

Usage :
  export RADGAP_DATA_ROOT=/chemin/vers/data
  uv run python scripts/download_aimi.py --dataset chexpert_plus --list
  uv run python scripts/download_aimi.py --dataset chexpert_plus --labels-only
  uv run python scripts/download_aimi.py --dataset chexpert_plus --variant png_compressed
  uv run python scripts/download_aimi.py --dataset mura
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

REDIVIS_ORG = "AIMI"

# Références version-pinnées (reproductibilité).
DATASETS: dict[str, dict] = {
    "chexpert_plus": {
        "ref": "chexpert_plus:5yyj:v1_0",
        "structured_table": "df_chexpert_plus_240401",  # métadonnées + démographie -> CSV
        "label_files_table": "CheXpert Labels",  # labels CheXbert (JSONL par section)
        "variants": {
            "png": ["PNG_train", "PNG_valid"],
            "png_compressed": ["PNG_compressed"],
            "dicom": ["DICOM_train", "DICOM_valid"],
        },
        "default_variant": "png",
    },
    "mura": {
        "ref": "mura_msk_xrays:cv1a:v1_0",
        "structured_table": None,
        "label_files_table": None,  # labels inclus dans l'arborescence de fichiers
        "variants": {"all": ["MURA-v1.1"]},
        "default_variant": "all",
    },
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


def _download_and_extract_zips(table, out_dir: Path) -> None:
    """Télécharge puis extrait des archives .zip une par une.

    Stratégie économe en disque : download d'une archive -> extraction -> suppression
    de l'archive -> marqueur `.extracted`. Le pic disque reste de l'ordre d'une archive
    (~155 Go) au lieu de la somme des archives (~720 Go). Idempotent et reprenable :
    une archive déjà extraite (marqueur présent) est ignorée.
    """
    files = list(table.list_files())
    print(f"  {len(files)} archive(s) à télécharger puis extraire")
    for f in files:
        name = f.name
        if not name.endswith(".zip"):
            print(f"  ⚠ {name} : pas une archive .zip, ignoré")
            continue
        marker = out_dir / f".{name}.extracted"
        if marker.exists():
            print(f"  ↷ {name} déjà extrait, skip")
            continue
        zip_path = out_dir / name
        print(f"  ↓ {name} …")
        f.download(path=str(out_dir), overwrite=False)
        print(f"  ⇲ extraction {name} …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        zip_path.unlink(missing_ok=True)
        marker.write_text("ok\n")
        print(f"  ✓ {name} extrait, archive supprimée")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--variant", default=None, help="variant d'images (cf. --list)")
    parser.add_argument("--list", action="store_true", help="lister les tables du dataset")
    parser.add_argument(
        "--labels-only", action="store_true", help="exporter seulement la table de labels"
    )
    args = parser.parse_args()

    os.environ["REDIVIS_API_TOKEN"] = _resolve_token()
    import redivis  # import tardif : dépendance requise uniquement pour le download

    spec = DATASETS[args.dataset]
    ds = redivis.organization(REDIVIS_ORG).dataset(spec["ref"])

    if args.list:
        print(f"Tables de {args.dataset} ({spec['ref']}) :")
        for t in ds.list_tables():
            p = t.properties
            print(f"  - {t.name}  (rows={p.get('numRows')}, fichiers={p.get('isFileIndex')})")
        print(f"Variants d'images : {list(spec['variants'])} (défaut : {spec['default_variant']})")
        return 0

    out_dir = _data_root() / "raw" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Table structurée (métadonnées + démographie) -> CSV
    if spec["structured_table"]:
        print(f"Export métadonnées : {spec['structured_table']} -> {out_dir}")
        meta_df = ds.table(spec["structured_table"]).to_pandas_dataframe()
        csv_path = out_dir / f"{spec['structured_table']}.csv"
        meta_df.to_csv(csv_path, index=False)
        print(f"  ✓ {len(meta_df):,} lignes -> {csv_path}")

    # 2) Table de labels (fichiers JSONL CheXbert : findings/impression/report)
    if spec["label_files_table"]:
        print(f"Téléchargement des labels : {spec['label_files_table']} -> {out_dir}")
        try:
            ds.table(spec["label_files_table"]).download_files(path=str(out_dir), overwrite=False)
            print("  ✓ labels JSONL")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ {spec['label_files_table']} : {exc}")

    if args.labels_only:
        return 0

    # 3) Images (variant choisi)
    variant = args.variant or spec["default_variant"]
    if variant not in spec["variants"]:
        raise SystemExit(f"variant inconnu {variant!r} (dispo : {list(spec['variants'])})")
    print(f"Téléchargement des images (variant '{variant}') -> {out_dir}")
    is_compressed = "compressed" in variant
    for table_name in spec["variants"][variant]:
        try:
            if is_compressed:
                _download_and_extract_zips(ds.table(table_name), out_dir)
            else:
                ds.table(table_name).download_files(path=str(out_dir), overwrite=False)
            print(f"  ✓ {table_name}")
        except Exception as exc:  # noqa: BLE001  (on continue les autres tables)
            print(f"  ⚠ {table_name} : {exc}")

    print("Terminé. Vérifier le contenu puis lancer scripts/build_manifest.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
