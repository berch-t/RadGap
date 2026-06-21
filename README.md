<div align="center">

# RadGap

**Benchmark reproductible de la _généralisation_, de l'_efficacité-label_ et de l'_équité_ des foundation models en radiologie**

_Comparer plusieurs encodeurs de vision **gelés** sur trois axes scientifiques — généralisation cross-dataset, efficacité-label, équité démographique — de façon rigoureuse et reproductible, sur **un seul GPU** (RTX 4080 Super 16 Go)._

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-ee4c2c.svg)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/deps-uv-261230.svg)](https://github.com/astral-sh/uv)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/berch-t/radgap/actions/workflows/ci.yml/badge.svg)](https://github.com/berch-t/radgap/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

> [!WARNING]
> **Projet de recherche en cours** (jalon M0 — bootstrap terminé). Le code modèle/données arrive aux jalons M1+.
> Suivi de l'avancement : [`PLAN.md`](PLAN.md) · Contexte technique complet : [`CLAUDE.md`](CLAUDE.md).
> Usage **recherche uniquement** — ce n'est pas un dispositif médical.

## Table des matières

- [Pourquoi RadGap](#pourquoi-radgap)
- [Contributions scientifiques](#contributions-scientifiques)
- [Model zoo](#model-zoo)
- [Datasets](#datasets)
- [Installation](#installation)
- [Démarrage rapide](#démarrage-rapide)
- [Reproduire un résultat](#reproduire-un-résultat)
- [Structure du dépôt](#structure-du-dépôt)
- [Méthodologie & rigueur](#méthodologie--rigueur)
- [Résultats](#résultats)
- [Éthique, licences & données](#éthique-licences--données)
- [Feuille de route](#feuille-de-route)
- [Citation](#citation)
- [Remerciements](#remerciements)

## Pourquoi RadGap

L'état de l'art 2026 en radiographie thoracique repose sur des **Vision Transformers pré-entraînés en self-supervised** (foundation models), adaptés en aval par des têtes légères. On n'entraîne plus de réseau _from scratch_. Les vrais problèmes ouverts ne sont pas la précision brute mais :

1. **La généralisation** — l'effondrement des performances hors de la distribution d'entraînement.
2. **L'efficacité-label** — le coût des annotations radiologiques.
3. **L'équité** — des biais démographiques documentés.

RadGap mesure ces trois axes de façon **comparative, statistiquement rigoureuse et reproductible**, en gardant chaque backbone sur un pied d'égalité : on **gèle** l'encodeur, on **pré-calcule et cache** les embeddings, puis on entraîne des têtes légères en quelques minutes.

> **Ce que RadGap n'est pas :** une course au leaderboard (+0,5 % d'AUROC), un pré-entraînement _from scratch_, un produit clinique, ou de la génération de compte-rendu. Voir les non-goals dans [`CLAUDE.md`](CLAUDE.md).

## Contributions scientifiques

| ID | Contribution | Métrique clé | Jalon |
|----|--------------|--------------|-------|
| **C1** | Generalization gap cross-dataset | Δ AUROC (in-distribution → out-of-distribution) | M4 ★ |
| **C2** | Efficacité-label | Courbes AUROC @ {1 %, 10 %, 100 %} labels × backbone | M5 |
| **C3** | Audit d'équité | Écarts AUROC / TPR par sexe, âge, race | M6 |
| **C4** | _(stretch)_ Transfert thorax → os (MURA) + LoRA | AUROC transfert vs in-domain ; gain LoRA vs coût | M7 |

## Model zoo

Tous les backbones sont évalués **gelés** sous le même protocole.

| Backbone | Type | Source | Dim | Rôle |
|----------|------|--------|-----|------|
| **RAD-DINO** | ViT-B/14 DINOv2 (CXR) | `microsoft/rad-dino` | 768 | Foundation model principal |
| RAD-DINO-MAIRA-2 | RAD-DINO + données | `microsoft/rad-dino-maira-2` | 768 | Variante (plus de données) |
| BiomedCLIP | CLIP biomédical | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | 512 | Baseline vision-langage |
| DINOv2-ImageNet | ViT-B/14 naturel | `facebook/dinov2-base` | 768 | Contrôle « domaine général » |
| DenseNet-121 (CheXpert) | CNN supervisé | `torchxrayvision` | — | Baseline supervisée historique |
| Medical MAE | ViT-B/16 MAE médical | `lambert-x/medical_mae` | 768 | Baseline SSL alternative |

## Datasets

| Dataset | Modalité | Rôle | Accès | Licence |
|---------|----------|------|-------|---------|
| **CheXpert Plus** | Radio thorax + texte + démo. | Train principal (C1) + équité (C3) | Stanford AIMI | Research Use Agreement |
| MURA | Radio os | Transfert (C4) | Stanford AIMI | Research Use Agreement |
| NIH ChestX-ray14 | Radio thorax | Test OOD (C1) | Kaggle | CC0-ish |
| PadChest | Radio thorax | Test OOD (C1) | BIMCV | Research |
| VinDr-CXR | Radio thorax | Test OOD (C1) | PhysioNet | Credentialed |
| MIMIC-CXR | Radio thorax + texte | Test OOD / VLM | PhysioNet | Credentialed |

> **Aucune donnée n'est versionnée dans ce dépôt** (licences recherche). On commit les _scripts_ de téléchargement, jamais les pixels. CheXpert Plus subsume CheXpert (non proposé seul par Stanford AIMI) et fournit la démographie pour C3.

## Installation

Prérequis : [`uv`](https://github.com/astral-sh/uv), un GPU NVIDIA avec CUDA 12.x (testé sur RTX 4080 Super, driver 591.86, WSL2).

```bash
git clone git@github.com:berch-t/radgap.git
cd radgap

uv python install 3.11
uv sync                                # crée .venv avec torch 2.6 (cu124)

cp env.example .env                    # renseigner RADGAP_DATA_ROOT + AIMI_API_KEY
uv run python scripts/check_env.py     # vérifie GPU + forward AMP
```

`check_env.py` doit afficher le GPU, `CUDA dispo : True`, et `Forward AMP : OK`.

## Démarrage rapide

```bash
uv run pytest -q                       # suite de tests
uv run ruff check .                    # lint

# Hydra : tout est paramétré, rien en dur. Exemple (jalons M3+) :
uv run python scripts/train.py backbone=rad_dino dataset=chexpert_plus experiment=in_dist
uv run python scripts/train.py -m backbone=rad_dino,dinov2,biomedclip   # sweep multi-backbone
```

## Reproduire un résultat

Le pipeline complet (disponible au fil des jalons) régénère chaque table et figure depuis zéro :

```
download → manifest unifié → extract embeddings (cache disque) → train têtes → evaluate → figures
```

```bash
uv run python scripts/download_aimi.py --dataset chexpert_plus   # M1 (Redivis, token AIMI)
uv run python scripts/download_aimi.py --dataset mura            # M1
uv run python scripts/download_nih.py                            # M1 (Kaggle, OOD)
uv run python scripts/build_manifest.py                          # M1 → data/manifests/unified.parquet
uv run python scripts/validate_manifests.py                      # M1 (0 fuite patient, labels valides)
uv run python scripts/extract_embeddings.py -m backbone=rad_dino,dinov2,biomedclip   # M3
uv run python scripts/evaluate.py experiment=gen_gap                                  # M4
uv run python scripts/make_figures.py               # M4+ → experiments/figures/*.png
```

Chaque figure se régénère à partir d'un CSV versionné dans `experiments/results/` — **exigence de reproductibilité**.

### Notebooks (un par phase ML/DL)

Notebooks académiques commentés au format **percent / jupytext** (`.py` versionnables, convertis en `.ipynb` à la volée) :

```bash
uv run jupytext --to notebook notebooks/01_eda_and_label_harmonization.py   # puis ouvrir le .ipynb
```

| Notebook | Phase | Jalon |
|----------|-------|:-----:|
| `01_eda_and_label_harmonization` | EDA + harmonisation des labels | M1 |
| `02_preprocessing_and_augmentation` | Préprocessing radiologique CXR-safe | M3 |
| `03_embeddings_and_linear_probing` | Embeddings gelés + têtes légères | M3 |
| `04_cross_dataset_generalization_gap` | Generalization gap (C1 ★) | M4 |
| `05_label_efficiency` | Courbes d'efficacité-label (C2) | M5 |
| `06_fairness_audit` | Audit d'équité démographique (C3) | M6 |

## Structure du dépôt

```
radgap/
├── CLAUDE.md PLAN.md README.md      # contexte IA · avancement · doc publique
├── pyproject.toml .python-version   # déps pinnées (uv) · Python 3.11
├── env.example                      # template d'env (jamais de secret committé)
├── configs/                         # Hydra : config.yaml + dataset/ backbone/ experiment/
├── src/radgap/                      # package : data · preprocessing · models · eval · utils
├── scripts/                         # CLI : download · extract · train · evaluate · figures
├── experiments/                     # sorties : embeddings (gitignored), résultats CSV, figures
├── notebooks/                       # 6 notebooks académiques (.py percent / jupytext), 1 par phase
├── tests/                           # pytest (CI)
└── .claude/skills/                  # 5 skills : data · preprocessing · adaptation · eval · repro
```

Règle : **la logique vit dans `src/`, la configuration dans `configs/`, les points d'entrée dans `scripts/`, les sorties dans `experiments/`.**

## Méthodologie & rigueur

- **Protocole d'éval figé _avant_ de regarder les résultats** (pas de p-hacking).
- **Chaque AUROC est livré avec un IC 95 %** (bootstrap). Pas de point estimate nu.
- **Comparaisons de modèles par test de DeLong**, pas à l'œil.
- **Multi-label = métriques par pathologie** d'abord, puis agrégation macro.
- **Split par patient** (jamais par image), **masquage des labels non annotés** (NaN ≠ négatif).
- **Backbones gelés + embeddings cachés** : protocole identique pour tous, itération rapide.
- **Runs pilotés par config + seed**, hash de commit loggé, multi-seed pour les barres d'erreur.

Détails dans les skills [`.claude/skills/`](.claude/skills/) (`medical-ml-evaluation`, `reproducible-ml-research`, etc.).

## Résultats

> _À venir._ Les tables et figures seront générées aux jalons M3 (in-distribution), M4 (generalization gap, C1), M5 (efficacité-label, C2), M6 (équité, C3). Aperçu de la structure attendue :

| Backbone | AUROC in-dist (CheXpert Plus) | Δ AUROC OOD (NIH / PadChest / VinDr) |
|----------|:----------------------------:|:------------------------------------:|
| RAD-DINO | _M3_ | _M4_ |
| DINOv2-ImageNet | _M3_ | _M4_ |
| BiomedCLIP | _M3_ | _M4_ |
| DenseNet-121 (baseline) | _M3_ | _M4_ |

Question scientifique centrale (C1) : _les foundation models médicaux (RAD-DINO) présentent-ils un generalization gap plus faible que les baselines ImageNet/supervisées ?_

## Éthique, licences & données

- **Code** : licence MIT (voir [`LICENSE`](LICENSE)).
- **Données & poids de modèles** : conservent leurs **propres licences recherche non-commerciales** (Stanford Research Use Agreement, PhysioNet credentialing + CITI). Rien n'est redistribué ici.
- **Équité** : les écarts démographiques sont reportés **de façon descriptive**, sans claim causal, avec tailles de sous-groupes et limites explicites (intersectionnalité non modélisée).
- **Model card** + **datasheet** fournis au jalon M8.

## Feuille de route

| Jalon | Objectif | Statut |
|-------|----------|:------:|
| M0 | Bootstrap & infrastructure | ✅ |
| M1 | Acquisition & harmonisation des données | 🟡 |
| M2 | Baseline supervisée (DenseNet-121) | ⬜ |
| M3 | Embeddings FMs + linear probes (in-dist) | ⬜ |
| M4 ★ | Generalization gap cross-dataset (C1) | ⬜ |
| M5 | Courbes d'efficacité-label (C2) | ⬜ |
| M6 | Audit d'équité (C3) | ⬜ |
| M7 | _(stretch)_ Transfert thorax → os + LoRA (C4) | ⬜ |
| M8 | Rédaction, repo public, soutenance | ⬜ |

Détail complet et _Definition of Done_ par jalon : [`PLAN.md`](PLAN.md).

## Citation

```bibtex
@software{berchet_radgap_2026,
  author  = {Berchet, Thomas},
  title   = {RadGap: a reproducible benchmark of generalization, label-efficiency
             and fairness for radiology foundation models},
  year    = {2026},
  url     = {https://github.com/berch-t/radgap}
}
```

## Remerciements

Appuyé sur RAD-DINO (Pérez-García et al., 2025), CheXpert / CheXpert Plus (Stanford AIMI, Chambon et al., 2024), DINOv2, BiomedCLIP, et `torchxrayvision`. Voir les références complètes dans [`CLAUDE.md`](CLAUDE.md) §11.

---

<div align="center">
<sub>Projet de recherche — Université Grenoble Alpes (UGA). Usage recherche uniquement, non clinique.</sub>
</div>
