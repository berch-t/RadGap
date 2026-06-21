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
# # 01 — Exploration des données & harmonisation des labels (M1)
#
# **Phase ML :** acquisition, analyse exploratoire (EDA), harmonisation cross-dataset.
# **Contribution visée :** prépare C1 (generalization gap) — voir `PLAN.md` M1.
# **Skill associé :** `.claude/skills/medical-imaging-data`.
#
# ## Objectif scientifique
#
# Le cœur d'un benchmark cross-dataset n'est pas de charger des pixels, c'est de rendre
# les labels **comparables** entre datasets aux vocabulaires différents (CheXpert,
# NIH, PadChest, VinDr). Ce notebook :
#
# 1. charge le manifest unifié construit par `scripts/build_manifest.py` ;
# 2. audite sa qualité (complétude, équilibre des classes, démographie) ;
# 3. vérifie l'**intégrité** (aucune fuite patient inter-split, valeurs de labels valides) ;
# 4. expose la **table d'harmonisation** et le sous-ensemble de pathologies communes —
#    l'artefact central sur lequel C1 sera évalué.
#
# > **Rigueur :** on ne regarde JAMAIS le split `test` au-delà de statistiques agrégées de
# > sanity-check. Le protocole d'évaluation est figé avant de regarder les résultats.

# %% [markdown]
# ## 0. Configuration

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from radgap.data import (
    LABEL_COLUMNS,
    canonical_pathologies,
    load_manifest,
    validate_manifest,
)

sns.set_theme(context="notebook", style="whitegrid")
pd.set_option("display.max_columns", 50)

DATA_ROOT = Path(os.environ.get("RADGAP_DATA_ROOT", "../data"))
MANIFEST = DATA_ROOT / "manifests" / "unified.parquet"

# %% [markdown]
# ## 1. Chargement du manifest unifié
#
# Si le manifest n'existe pas encore, lancer d'abord :
# ```bash
# uv run python scripts/download_aimi.py --dataset chexpert_plus
# uv run python scripts/download_nih.py
# uv run python scripts/build_manifest.py
# ```

# %%
if MANIFEST.exists():
    df = load_manifest(MANIFEST)
    print(f"Manifest : {len(df):,} images, {df['dataset'].nunique()} datasets")
    display(df.head())
else:
    print(f"⚠️ Manifest absent ({MANIFEST}). Voir les commandes ci-dessus.")
    df = None

# %% [markdown]
# ## 2. Intégrité (DoD M1)
#
# Checks automatiques : schéma complet, 0 fuite patient inter-split, labels dans {0,1,NaN}.

# %%
if df is not None:
    problems = validate_manifest(df)
    print("✅ Manifest valide" if not problems else "❌ Problèmes :")
    for p in problems:
        print("  -", p)

# %% [markdown]
# ## 3. Répartition par dataset et par split
#
# On vérifie que chaque dataset a bien des images, et que le split par patient est cohérent.

# %%
if df is not None:
    pivot = df.pivot_table(index="dataset", columns="split", values="image_path", aggfunc="count", fill_value=0)
    display(pivot)
    pivot.plot(kind="bar", stacked=True, figsize=(8, 4), title="Images par dataset et split")
    plt.ylabel("nombre d'images")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. Prévalence des pathologies (déséquilibre de classes)
#
# Les pathologies CXR sont rares et déséquilibrées : c'est pourquoi l'AUROC (insensible au
# seuil et au déséquilibre) est la métrique primaire, pas l'accuracy.

# %%
if df is not None:
    prevalence = (
        df[LABEL_COLUMNS]
        .apply(lambda c: c.mean())  # moyenne en ignorant les NaN
        .sort_values(ascending=False)
    )
    prevalence.plot(kind="barh", figsize=(8, 5), title="Prévalence (proportion de positifs) par pathologie")
    plt.xlabel("prévalence")
    plt.tight_layout()
    plt.show()
    display(prevalence.to_frame("prévalence"))

# %% [markdown]
# ## 5. Taux de labels manquants (NaN ≠ négatif)
#
# Différents datasets labellisent différentes pathologies. Les NaN sont **masqués** dans la
# loss et les métriques — jamais comptés comme négatifs (cela corromprait silencieusement
# l'AUROC).

# %%
if df is not None:
    missing = df.groupby("dataset")[LABEL_COLUMNS].apply(lambda g: g.isna().mean())
    plt.figure(figsize=(10, 3))
    sns.heatmap(missing, annot=True, fmt=".2f", cmap="rocket_r", cbar_kws={"label": "taux de NaN"})
    plt.title("Taux de labels manquants par dataset × pathologie")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. Démographie (préparation de l'audit d'équité C3)

# %%
if df is not None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df["sex"].value_counts().plot(kind="bar", ax=axes[0], title="Sexe")
    df["age"].dropna().plot(kind="hist", bins=30, ax=axes[1], title="Âge")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 7. La table d'harmonisation — sous-ensembles de pathologies communes
#
# C1 (generalization gap) est évalué UNIQUEMENT sur les pathologies partagées entre
# CheXpert Plus (source) et chaque dataset OOD (cible).

# %%
for target in ("nih_cxr14", "padchest", "vindr_cxr"):
    shared = canonical_pathologies("chexpert_plus", target)
    print(f"chexpert_plus ∩ {target} ({len(shared)}) : {shared}")

# %% [markdown]
# ## Conclusion & prochaine étape
#
# - Le manifest unifié relie chaque image à ses labels canoniques, sa démographie et son split.
# - Les sous-ensembles communs sont figés : ils définissent le terrain d'évaluation de C1.
# - **Suite (M3) :** `02_preprocessing_and_augmentation` puis `03_embeddings_and_linear_probing`.
