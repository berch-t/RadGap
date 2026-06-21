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
# # 05 — Courbes d'efficacité-label (M5 · contribution C2)
#
# **Phase ML :** combien d'annotations chaque backbone « économise » ?
# **Skills associés :** `foundation-model-adaptation`, `medical-ml-evaluation`, `reproducible-ml-research`.
#
# > **Statut jalon :** s'exécute à M5 (embeddings cachés à M3 disponibles).
#
# ## Question scientifique
#
# Les annotations radiologiques coûtent cher. Un bon foundation model doit atteindre une
# performance donnée avec **moins** de labels. On sous-échantillonne le train CheXpert Plus à
# {1%, 10%, 100%} (stratifié, seed fixé, **n répétitions**) et on retrace AUROC = f(fraction).
#
# > **Rigueur reproductibilité :** un seul seed est une anecdote. On lance plusieurs seeds et
# > on reporte moyenne ± std (les barres d'erreur).

# %% [markdown]
# ## 1. Sous-échantillonnage stratifié multi-seed (esquisse)

# %%
import numpy as np

FRACTIONS = [0.01, 0.10, 1.00]
SEEDS = [0, 1, 2, 3, 4]


def subsample_indices(n: int, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = max(1, int(round(n * fraction)))
    return rng.choice(n, size=k, replace=False)


print({f"{int(f*100)}%": len(subsample_indices(10_000, f, 0)) for f in FRACTIONS})

# %% [markdown]
# ## 2. Courbe d'efficacité-label (gabarit de figure)
#
# Rempli à M5 depuis `experiments/results/label_efficiency.csv`.

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Données illustratives — REMPLACER par le CSV réel à M5.
rows = []
rng = np.random.default_rng(0)
for backbone, ceiling in [("RAD-DINO", 0.88), ("DINOv2-IN", 0.84), ("DenseNet", 0.86)]:
    for f in FRACTIONS:
        base = ceiling - 0.12 * (1 - f) ** 0.5
        for s in SEEDS:
            rows.append({"backbone": backbone, "fraction": f, "auroc": base + rng.normal(0, 0.005)})
demo = pd.DataFrame(rows)

agg = demo.groupby(["backbone", "fraction"])["auroc"].agg(["mean", "std"]).reset_index()
plt.figure(figsize=(8, 5))
for backbone, g in agg.groupby("backbone"):
    plt.errorbar(g["fraction"], g["mean"], yerr=g["std"], marker="o", capsize=3, label=backbone)
plt.xscale("log")
plt.xlabel("fraction de labels (échelle log)")
plt.ylabel("AUROC macro")
plt.title("Efficacité-label — ILLUSTRATIF (n=5 seeds)")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Conclusion attendue (gabarit)
#
# > « RAD-DINO @1% de labels ≈ baseline @10% » — quantifie l'économie d'annotations apportée
# > par le pré-entraînement self-supervised.
#
# **Prochaine étape :** `06_fairness_audit` (C3).
