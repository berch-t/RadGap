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
# # 02 — Préprocessing radiologique & augmentation cliniquement valide (M3, prep)
#
# **Phase ML :** transformation des radiographies brutes en tenseurs prêts pour les backbones.
# **Skill associé :** `.claude/skills/radiology-preprocessing`.
#
# > **Statut jalon :** ce notebook s'exécute pleinement à partir de M3 (quand les images sont
# > téléchargées et le module `radgap.preprocessing` implémenté). Le contenu pédagogique et les
# > pièges documentés sont valables dès maintenant.
#
# ## Pourquoi un notebook dédié
#
# L'imagerie médicale a des pièges que les pipelines de vision génériques ratent. Deux
# principes gouvernent tout :
#
# 1. **Matcher le préprocessing d'entraînement du backbone.** Un foundation model gelé attend
#    des entrées distribuées comme ses données de pré-entraînement. Un préprocessing différent
#    détruit silencieusement la performance → on utilise le `AutoImageProcessor` officiel.
# 2. **L'augmentation ne doit pas changer le sens clinique.** Le flip horizontal d'une radio
#    thoracique inverse la latéralité (le cœur est à gauche) et casse le modèle.

# %% [markdown]
# ## 1. Lecture DICOM : le piège MONOCHROME1
#
# Oublier l'inversion `MONOCHROME1` contraste-inverse une fraction des images et fait chuter
# l'AUROC sans raison apparente. C'est le bug silencieux classique.

# %%
import numpy as np


# Démonstration pédagogique de la logique (sans fichier réel) :
def normalize_intensity(arr: np.ndarray, photometric: str = "MONOCHROME2") -> np.ndarray:
    """min-max -> [0,1], avec inversion si MONOCHROME1 (valeur haute = sombre)."""
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr
    arr = arr.astype(np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


demo = np.array([[0, 128, 255]], dtype=np.float32)
print("MONOCHROME2 :", normalize_intensity(demo, "MONOCHROME2"))
print("MONOCHROME1 :", normalize_intensity(demo, "MONOCHROME1"))  # inversé

# %% [markdown]
# ## 2. Utiliser le processor officiel du backbone (recommandé)
#
# Plutôt que de réinventer resize + normalisation, on récupère le processor publié.
# Décommenter à M3 (nécessite le téléchargement des poids HF) :

# %%
# from transformers import AutoImageProcessor
# processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
# inputs = processor(images=pil_image, return_tensors="pt")  # resize + normalize corrects

# %% [markdown]
# ## 3. Augmentation CXR-safe vs interdite
#
# | Sûr (CXR) | Interdit (CXR) |
# |---|---|
# | rotations ±5–10° | ❌ flip horizontal (casse la latéralité) |
# | translation/scale ±5–10% | ❌ flip vertical (non anatomique) |
# | jitter luminosité/contraste léger | ❌ élastique agressif / cutout sur la lésion |
# | random resized crop conservateur (0.8–1.0) | ❌ jitter de teinte (domaine niveaux de gris) |
#
# > MURA (os) tolère le flip horizontal (membres ~symétriques) — d'où un flag de config
# > **par dataset**, jamais un défaut global.

# %%
# import torchvision.transforms.v2 as T
# train_tf = T.Compose([
#     T.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
#     T.RandomRotation(degrees=7),
#     T.ColorJitter(brightness=0.1, contrast=0.1),
#     # PAS de RandomHorizontalFlip pour la radio thoracique
# ])
# eval_tf = T.Compose([T.Resize(224, antialias=True), T.CenterCrop(224)])

# %% [markdown]
# ## 4. Cohérence train/éval & embeddings cachés
#
# Pour le protocole « extraire une fois, entraîner la tête » (cf. notebook 03), on extrait les
# embeddings avec la transform **déterministe** d'éval, pour qu'ils soient stables et
# comparables entre backbones. L'augmentation n'intervient que dans le rare cas du fine-tuning
# end-to-end (LoRA, M7).
#
# **Prochaine étape :** `03_embeddings_and_linear_probing`.
