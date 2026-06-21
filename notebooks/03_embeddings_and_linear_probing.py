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
# # 03 — Embeddings des foundation models & linear probing (M3)
#
# **Phase ML/DL :** extraction de features gelées + entraînement de têtes légères (in-distribution).
# **Skill associé :** `.claude/skills/foundation-model-adaptation`.
#
# > **Statut jalon :** s'exécute à M3 (backbones téléchargés, `radgap.models` implémenté).
#
# ## Stratégie : geler, extraire une fois, entraîner vite
#
# Pour RAD-DINO, le fine-tuning est généralement inutile : un classifieur sur le token CLS
# performe bien. Le protocole efficace **et comparable entre backbones** :
#
# 1. **geler** le backbone ;
# 2. **extraire** les embeddings une fois, les **cacher sur disque** ;
# 3. entraîner des têtes bon marché (minutes, voire CPU) ;
# 4. ne fine-tuner en LoRA que la **config gagnante** (M7).
#
# Cela garde chaque backbone sur un pied d'égalité et rend l'itération rapide (TDAH-friendly).

# %% [markdown]
# ## 1. Backbones évalués
#
# | Backbone | Dim | Rôle |
# |---|---|---|
# | RAD-DINO | 768 | foundation model principal (CXR) |
# | DINOv2-ImageNet | 768 | contrôle « domaine général » |
# | BiomedCLIP | 512 | baseline vision-langage |
# | DenseNet-121 (CheXpert) | — | baseline supervisée historique |
#
# > Les dimensions diffèrent (768 vs 512) → la tête est dimensionnée **par backbone**.

# %% [markdown]
# ## 2. Extraction + cache disque (esquisse — voir scripts/extract_embeddings.py à M3)

# %%
# from radgap.models import Backbone          # wrapper unifié (.embed -> (B, D))
# backbone = Backbone("rad-dino")
# # scripts/extract_embeddings.py écrit experiments/embeddings/<backbone>/<dataset>/<split>/
# #   embeddings.npy, labels.npy, ids.txt
# # Re-lancer l'entraînement ne retouche plus jamais le GPU une fois le cache écrit.

# %% [markdown]
# ## 3. Tête multi-label : sigmoïde + BCE masquée
#
# Multi-label (les pathologies co-occurrent) → sortie sigmoïde + `BCEWithLogitsLoss`, avec
# **masquage des labels NaN** (non annotés). Le masque est non négociable en cross-dataset.

# %%
import numpy as np


def masked_bce_numpy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Illustration NumPy de la BCE masquée (la vraie version est en torch)."""
    mask = ~np.isnan(targets)
    p = 1 / (1 + np.exp(-logits[mask]))
    t = targets[mask]
    eps = 1e-7
    return float(-(t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps)).mean())


rng = np.random.default_rng(0)
logits = rng.normal(size=(4, 3))
targets = np.array([[1, 0, np.nan], [0, 1, 1], [np.nan, np.nan, 0], [1, 1, 0]], dtype=float)
print("BCE masquée :", masked_bce_numpy(logits, targets))

# %% [markdown]
# ## 4. Évaluation in-distribution (AUROC + IC bootstrap)
#
# À M3 on produira `experiments/results/auroc_in_distribution.csv` et la table comparative
# « AUROC in-distribution par backbone (± IC 95%) », avec test de DeLong entre backbones.
# Voir notebook `04` (réutilise les mêmes utilitaires `radgap.eval`).
#
# **Prochaine étape :** `04_cross_dataset_generalization_gap` (la contribution cœur C1).
