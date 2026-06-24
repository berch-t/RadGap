"""Entraîne une tête légère sur embeddings gelés et évalue l'AUROC in-distribution (M3).

Charge les embeddings cachés (train/val/test) d'un backbone, standardise (stats du train),
entraîne un linear probe (ou MLP), sélectionne sur la macro-AUROC de validation, puis évalue
sur le test avec IC bootstrap par pathologie. Résultats versionnés dans
`experiments/results/auroc_in_distribution.csv`.

Usage :
  export RADGAP_DATA_ROOT=/chemin/vers/data
  uv run python scripts/train_probe.py                       # rad_dino, linear (défauts)
  uv run python scripts/train_probe.py backbone=dinov2
  uv run python scripts/train_probe.py experiment.head=mlp
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from radgap.eval.metrics import evaluate_auroc, macro_auroc
from radgap.models.embeddings import load_cached_embeddings
from radgap.models.heads import Head, masked_bce
from radgap.utils import get_logger, set_determinism

log = get_logger("radgap.probe")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _standardize(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(0, keepdims=True)
    sd = train.std(0, keepdims=True) + 1e-6
    return tuple((a - mu) / sd for a in (train, *others))


def _train_head(
    head, xtr, ytr, xva, yva, label_cols, *, epochs, batch_size, lr, wd, device
):
    head.to(device)
    xtr_t = torch.from_numpy(xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).float().to(device)
    xva_t = torch.from_numpy(xva).float().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)

    n = len(xtr_t)
    best_macro, best_state, best_epoch = -1.0, None, -1
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = masked_bce(head(xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            va_scores = torch.sigmoid(head(xva_t)).cpu().numpy()
        macro = macro_auroc(yva, va_scores)
        if macro > best_macro:
            best_macro = macro
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    log.info("  meilleure macro-AUROC val = %.4f (epoch %d)", best_macro, best_epoch)
    return head


def _save_results(table: pd.DataFrame, backbone: str, head_kind: str, dataset: str, split: str):
    out_dir = REPO_ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "auroc_in_distribution.csv"
    table = table.assign(backbone=backbone, head=head_kind, dataset=dataset, eval_split=split)
    keys = ["backbone", "head", "dataset", "eval_split", "pathology"]
    if out.exists():
        prev = pd.read_csv(out)
        merged = prev[~prev.set_index(keys).index.isin(table.set_index(keys).index)]
        table = pd.concat([merged, table], ignore_index=True)
    table.to_csv(out, index=False)
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_determinism(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(cfg.data_root)
    bb, ds = cfg.backbone.name, cfg.dataset.name

    xtr, ytr, _, meta = load_cached_embeddings(bb, ds, "train", root)
    xva, yva, _, _ = load_cached_embeddings(bb, ds, "val", root)
    xte, yte, _, _ = load_cached_embeddings(bb, ds, "test", root)
    label_cols = meta["label_cols"]
    log.info(
        "%s/%s : train=%d val=%d test=%d, dim=%d",
        bb, ds, len(xtr), len(xva), len(xte), meta["dim"],
    )

    xtr, xva, xte = _standardize(xtr, xva, xte)

    head_kind = cfg.experiment.get("head", "linear")
    hidden = 0 if head_kind == "linear" else int(cfg.get("hidden", 512))
    head = Head(meta["dim"], len(label_cols), hidden=hidden)
    head = _train_head(
        head, xtr, ytr, xva, yva, label_cols,
        epochs=int(cfg.get("epochs", 100)),
        batch_size=int(cfg.get("batch_size", 1024)),
        lr=float(cfg.get("lr", 1e-3)),
        wd=float(cfg.get("weight_decay", 1e-4)),
        device=device,
    )

    head.eval()
    with torch.no_grad():
        te_scores = torch.sigmoid(head(torch.from_numpy(xte).float().to(device))).cpu().numpy()
    n_boot = int(cfg.experiment.get("n_bootstrap", 1000))
    table = evaluate_auroc(yte, te_scores, label_cols, n_boot=n_boot, seed=cfg.seed)

    macro_row = table[table["pathology"] == "MACRO"].iloc[0]
    log.info(
        "TEST macro-AUROC = %.4f (95%% IC %.4f-%.4f)",
        macro_row.auroc, macro_row.ci_lo, macro_row.ci_hi,
    )
    for _, r in table[table["pathology"] != "MACRO"].iterrows():
        log.info(
            "  %-18s %.4f (%.4f-%.4f)  n=%d pos=%d",
            r.pathology, r.auroc, r.ci_lo, r.ci_hi, r.n_labeled, r.n_pos,
        )

    out = _save_results(table, bb, head_kind, ds, "test")
    log.info("✓ résultats -> %s", out)


if __name__ == "__main__":
    main()
