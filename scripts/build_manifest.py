"""Construit le manifest unifié à partir des datasets téléchargés (M1).

Pipeline : loaders par dataset -> split par patient -> manifest CXR unifié + manifest MURA.
N'inclut que les datasets dont les fichiers bruts sont présents sous `$RADGAP_DATA_ROOT/raw/`.

Usage :
  uv run python scripts/build_manifest.py
  uv run python scripts/build_manifest.py seed=7 uncertainty_policy=u_ones
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from radgap.data import (
    build_validate_save,
    load_chexpert_plus,
    load_mura,
    load_nih,
    save_manifest,
    split_by_patient,
)
from radgap.utils import get_logger, set_determinism

log = get_logger("radgap.manifest")


def _first_existing(*paths: Path) -> Path | None:
    return next((p for p in paths if p.exists()), None)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_determinism(cfg.seed)
    root = Path(cfg.data_root)
    raw = root / "raw"
    policy = None if cfg.uncertainty_policy in (None, "auto") else cfg.uncertainty_policy

    cxr_frames: list[pd.DataFrame] = []

    section = cfg.get("label_section", "impression")  # findings | impression | report
    chex_meta = _first_existing(
        raw / "chexpert_plus" / "df_chexpert_plus_240401.csv",
        *raw.glob("chexpert_plus/df_chexpert*.csv"),
    )
    chex_labels = _first_existing(raw / "chexpert_plus" / f"{section}_fixed.json")
    if chex_meta and chex_labels:
        log.info("CheXpert Plus : meta=%s labels=%s (section=%s)", chex_meta, chex_labels, section)
        df = load_chexpert_plus(chex_meta, chex_labels, global_policy=policy)
        cxr_frames.append(split_by_patient(df, seed=cfg.seed))
    else:
        log.warning("CheXpert Plus incomplet (meta=%s, labels=%s) — sauté", chex_meta, chex_labels)

    nih_csv = _first_existing(raw / "nih_cxr14" / "Data_Entry_2017.csv")
    if nih_csv:
        log.info("NIH ChestX-ray14 : %s", nih_csv)
        df = load_nih(nih_csv)
        cxr_frames.append(split_by_patient(df, seed=cfg.seed))
    else:
        log.warning("NIH absent — sauté")

    if not cxr_frames:
        raise SystemExit(f"Aucun dataset CXR trouvé sous {raw}. Lancer les scripts de download.")

    out = root / "manifests" / "unified.parquet"
    unified = build_validate_save(cxr_frames, out)
    log.info("Manifest unifié : %d lignes -> %s", len(unified), out)
    log.info("Datasets : %s", dict(unified["dataset"].value_counts()))

    # MURA : tâche disjointe (C4), manifest dédié.
    mura_csv = _first_existing(
        *raw.glob("mura*/**/train_image_paths.csv"),
        raw / "mura" / "train_image_paths.csv",
    )
    if mura_csv:
        mura = split_by_patient(load_mura(mura_csv), seed=cfg.seed)
        save_manifest(mura, root / "manifests" / "mura.parquet")
        log.info("Manifest MURA : %d lignes", len(mura))


if __name__ == "__main__":
    main()
