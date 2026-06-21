# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (radgap)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 04 — Generalization gap cross-dataset (M4 · contribution C1 ★)
#
# **Phase ML :** évaluation hors distribution (OOD), la contribution scientifique centrale.
# **Skill associé :** `.claude/skills/medical-ml-evaluation`.
#
# > **Statut jalon :** s'exécute à M4 (têtes entraînées à M3, datasets OOD disponibles).
#
# ## Question scientifique
#
# Les foundation models médicaux (RAD-DINO) présentent-ils un **generalization gap** plus
# faible que les baselines ImageNet/supervisées quand on les évalue sur des distributions non
# vues (NIH, PadChest, VinDr) ?
#
# Protocole (figé avant de regarder les résultats) :
# ```
# Pour chaque backbone :
#   entraîner la tête sur CheXpert Plus (train)
#   pour cible in {CheXpert Plus test (in-dist), NIH, PadChest, VinDr (OOD)} :
#       restreindre aux pathologies communes canonical_pathologies(chexpert_plus, cible)
#       AUROC macro + IC bootstrap
#   gap = AUROC(in-dist) - AUROC(cible)
# ```

# %% [markdown]
# ## 1. Métrique : AUROC par pathologie + IC bootstrap
#
# Chaque AUROC est livré avec un **intervalle de confiance 95%** (bootstrap). Pas de point
# estimate nu — cela se fait démolir en soutenance.

# %%
import numpy as np
from sklearn.metrics import roc_auc_score


def auroc_macro(y_true: np.ndarray, y_score: np.ndarray) -> float:
    aucs = []
    for j in range(y_true.shape[1]):
        m = ~np.isnan(y_true[:, j])
        if len(np.unique(y_true[m, j])) == 2:
            aucs.append(roc_auc_score(y_true[m, j], y_score[m, j]))
    return float(np.nanmean(aucs)) if aucs else float("nan")


def bootstrap_ci(y_true, y_score, fn, n=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y_true)
    stats = [fn(y_true[idx := rng.integers(0, N, N)], y_score[idx]) for _ in range(n)]
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.nanmean(stats)), float(lo), float(hi)


# Démo sur données synthétiques (à M4 : vraies prédictions chargées depuis experiments/)
rng = np.random.default_rng(0)
y = rng.integers(0, 2, size=(500, 3)).astype(float)
s = y * 0.6 + rng.normal(0, 0.5, size=y.shape)  # scores corrélés au vrai label
mean, lo, hi = bootstrap_ci(y, s, auroc_macro)
print(f"AUROC macro (démo) = {mean:.3f} (IC95% {lo:.3f}–{hi:.3f})")

# %% [markdown]
# ## 2. Le generalization gap, visualisé
#
# Esquisse de la figure cible (remplie à M4 depuis `experiments/results/generalization_gap.csv`).

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Données illustratives — REMPLACER par le CSV réel à M4.
demo = pd.DataFrame(
    {
        "backbone": ["RAD-DINO", "DINOv2-IN", "DenseNet"],
        "in_dist": [0.88, 0.84, 0.86],
        "nih": [0.79, 0.71, 0.74],
        "padchest": [0.77, 0.69, 0.72],
    }
)
for col in ("nih", "padchest"):
    demo[f"gap_{col}"] = demo["in_dist"] - demo[col]

ax = demo.set_index("backbone")[["gap_nih", "gap_padchest"]].plot(
    kind="bar", figsize=(8, 4), title="Generalization gap (Δ AUROC) — ILLUSTRATIF"
)
ax.set_ylabel("Δ AUROC (in-dist − OOD)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Comparaison statistique (DeLong)
#
# « RAD-DINO bat la baseline de 0.04 AUROC » n'a de sens qu'avec un test de DeLong sur le
# **même** test set (prédictions corrélées). Implémenté une fois dans `radgap.eval.delong`.
#
# ## Conclusion attendue (gabarit)
#
# > « RAD-DINO perd X points hors distribution contre Y pour la baseline supervisée
# > (Δgap = Z, p < 0.05) — le pré-entraînement médical self-supervised réduit le
# > generalization gap. »
#
# **Prochaine étape :** `05_label_efficiency` (C2).
