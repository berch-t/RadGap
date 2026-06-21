# CLAUDE.md - RadGap
> Benchmark reproductible de la généralisation, de l'efficacité-label et de l'équité des foundation models en radiologie
> Dernière mise à jour : 2026-06-19 | Auteur : Thomas "Tonton" Berchet (berch-t) | Cible : soutenance UGA + repo GitHub public

---

## 0. NORTH STAR (à relire à chaque session)

**Une seule phrase :** comparer plusieurs foundation models radiologiques **gelés** sur trois axes scientifiques — généralisation cross-dataset, efficacité-label, équité démographique — de façon **rigoureuse et reproductible**, sur un seul GPU (RTX 4080 Super 16 Go).

**Ce que ce projet N'EST PAS (non-goals, anti-scope-creep) :**
- ❌ Battre un leaderboard de 0.5% d'AUROC (sans intérêt scientifique)
- ❌ Pré-entraîner un foundation model from scratch (impossible sur 1 GPU)
- ❌ Construire un produit clinique / dispositif médical (licences recherche uniquement)
- ❌ Faire de la génération de compte-rendu ou du VLM zero-shot (extensions futures, hors MVP)
- ❌ Toucher à l'IRM 3D / l'écho avant que les 4 contributions CXR soient bouclées

> RÈGLE TDAH : si une nouvelle idée surgit, la noter dans la section "Idées parking" en bas, PAS l'implémenter. On finit le jalon en cours d'abord (cf. PLAN.md).

> 📄 RÈGLE DOC VIVANTE : à **chaque changement structurant** (nouveau script/module, dépendance, jalon avancé, décision technique, résultat), **mettre à jour le `README.md`** dans le même mouvement — il doit toujours refléter l'état réel du projet. Détail en §13.

---

## 1. Contexte

Projet de recherche en machine learning appliqué à la radiologie assistée par IA, destiné à être présenté à l'Université Grenoble Alpes (UGA) et publié en open-source sur GitHub.

L'état de l'art 2026 en radiographie thoracique repose sur des **Vision Transformers pré-entraînés en self-supervised** (foundation models), adaptés en downstream par des têtes légères. On n'entraîne plus de réseau from scratch. Les vrais problèmes ouverts ne sont pas la précision brute mais :
1. La **généralisation** (effondrement des perfs hors distribution d'entraînement)
2. L'**efficacité-label** (coût des annotations radiologiques)
3. L'**équité** (biais démographiques documentés)

RadGap mesure ces trois axes de façon comparative et reproductible.

## 2. Contributions scientifiques (le livrable de fond)

| ID | Contribution | Métrique clé | Statut |
|----|--------------|--------------|--------|
| C1 | Generalization gap cross-dataset | Δ AUROC (in-distribution → out-of-distribution) | [ ] |
| C2 | Label efficiency | Courbes AUROC @ {1%, 10%, 100%} labels × backbone | [ ] |
| C3 | Fairness audit | Écarts AUROC / TPR par sexe, âge, race | [ ] |
| C4 | (stretch) Transfert cross-région thorax → os (MURA) | AUROC transfert vs in-domain | [ ] |

## 3. Stack technique

| Couche | Choix | Note |
|--------|-------|------|
| Langage | Python 3.11 | venv ou conda, pinné |
| DL | PyTorch 2.x + CUDA 12.x | RTX 4080 Super, AMP activé |
| Modèles | `transformers`, `open_clip_torch`, `torchxrayvision` | backbones gelés |
| Adaptation | `peft` (LoRA) | config gagnante uniquement |
| Données | DICOM via `pydicom`, images via `Pillow`/`opencv` | + `pandas` pour les manifests |
| Stats | `scikit-learn`, `scipy`, `statsmodels` | AUROC, bootstrap CI, DeLong |
| Config | `hydra-core` + `omegaconf` | tout est paramétré, rien en dur |
| Tracking | `wandb` (mode offline possible) | + CSV de résultats versionnés |
| Viz | `matplotlib`, `seaborn` | figures pour la soutenance |

## 4. Backbones évalués (model zoo)

| Nom | Type | ID / source | Rôle |
|-----|------|-------------|------|
| RAD-DINO | ViT-B/14 DINOv2 (CXR) | `microsoft/rad-dino` (HF) | Foundation model principal |
| RAD-DINO-MAIRA-2 | RAD-DINO + données | `microsoft/rad-dino-maira-2` (HF) | Variante (plus de données) |
| BiomedCLIP | CLIP biomédical | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (HF) | Baseline vision-langage |
| DINOv2-ImageNet | ViT-B/14 naturel | `facebook/dinov2-base` (HF) | Baseline "domaine général" |
| DenseNet-121 (CheXpert) | CNN supervisé | `torchxrayvision` | Baseline supervisée historique |
| Medical MAE | ViT-B/16 MAE médical | GitHub `lambert-x/medical_mae` | Baseline SSL alternative |

## 5. Datasets

| Dataset | Modalité | Rôle | Accès | Licence |
|---------|----------|------|-------|---------|
| CheXpert Plus | Radio thorax + texte + démographie | **Train principal (C1) + Fairness (C3)** | Stanford AIMI ✅ **accès obtenu** | Research Use Agreement |
| MURA | Radio os | Transfert (C4) | Stanford AIMI ✅ **accès obtenu** | Research Use Agreement |
| NIH ChestX-ray14 | Radio thorax | Test OOD (C1) | Kaggle | CC0-ish (vérifier) |
| PadChest | Radio thorax | Test OOD (C1) | BIMCV Valence | Research |
| VinDr-CXR | Radio thorax | Test OOD (C1) | PhysioNet | PhysioNet credentialed |
| MIMIC-CXR | Radio thorax + texte | Test OOD / VLM | PhysioNet | PhysioNet credentialed |

> ℹ️ **CheXpert standalone n'est plus proposé séparément** sur Stanford AIMI : seuls **CheXpert Plus** (qui inclut les images CheXpert + rapports + démographie) et **MURA** sont accessibles. On utilise donc CheXpert Plus comme dataset in-distribution principal — il subsume CheXpert et fournit en bonus la démographie pour C3. *Hypothèse à vérifier au M1 : que les images/labels CheXpert Plus correspondent bien au CheXpert d'origine pour le sanity-check baseline M2.*
> 🔑 **Clé API Stanford AIMI** : stockée dans `AIMI_API_KEY.txt` (à la racine, **gitignored**). Chargée via la variable d'env `AIMI_API_KEY` (cf. `env.example`). Ne jamais committer la clé.
> ⚠️ **PhysioNet** (VinDr, MIMIC) : accréditation + formation CITI obligatoires, délai de plusieurs jours. **À lancer au jour 1** (cf. PLAN.md M0).

## 6. Structure du repo

```
radgap/
├── CLAUDE.md                 # Ce fichier (contexte IA)
├── PLAN.md                   # Plan jalonné (source de vérité de l'avancement)
├── README.md                 # Doc publique (généré à M8)
├── pyproject.toml            # Dépendances pinnées
├── .gitignore                # IGNORE data/ ET checkpoints lourds (licences !)
├── configs/                  # Configs Hydra (datasets, modèles, expés)
│   ├── config.yaml
│   ├── dataset/
│   ├── backbone/
│   └── experiment/
├── data/                     # JAMAIS commit (raw/, processed/, manifests/)
├── src/radgap/
│   ├── data/                 # Loaders + harmonisation labels
│   ├── preprocessing/        # Pipeline images (cf. SKILL radiology-preprocessing)
│   ├── models/               # Wrappers backbones + têtes (cf. SKILL foundation-model-adaptation)
│   ├── eval/                 # Métriques, CI, fairness (cf. SKILL medical-ml-evaluation)
│   └── utils/                # Seeding, logging, IO
├── scripts/                  # CLI : download, extract_embeddings, train, evaluate
├── experiments/              # Sorties : embeddings cachés, résultats CSV, figures
├── notebooks/                # EDA uniquement (pas de logique métier)
├── tests/                    # pytest
└── .claude/skills/           # 5 skills (data, preprocessing, adaptation, eval, repro)
```

## 7. Décisions techniques

| Date | Décision | Raison |
|------|----------|--------|
| 2026-06-19 | Backbones **gelés** + têtes légères comme protocole principal | RAD-DINO n'a pas besoin de fine-tuning pour de bonnes perfs ; tient sur 16 Go ; comparable entre modèles |
| 2026-06-19 | **Pré-calcul et cache des embeddings** sur disque | Réduit l'entraînement des têtes de heures à minutes ; itération rapide (TDAH-friendly) |
| 2026-06-19 | CheXpert = train ; NIH/PadChest/VinDr = test OOD | Sépare in-distribution et out-of-distribution proprement |
| 2026-06-19 | LoRA réservé à la **config gagnante** uniquement | Évite l'explosion combinatoire d'expés |
| 2026-06-19 | Protocole d'éval figé AVANT de regarder les résultats | Intégrité scientifique (pas de p-hacking) |
| 2026-06-21 | **CheXpert Plus** (et non CheXpert standalone) = dataset in-distribution principal | CheXpert standalone non proposé séparément par Stanford AIMI ; CheXpert Plus le subsume et apporte la démographie pour C3 en un seul dataset |

## 8. Problèmes & solutions

### Harmonisation des labels cross-dataset
- **Contexte** : les 14 pathologies de CheXpert ≠ celles de NIH/PadChest (noms, granularité différents)
- **Solution** : table de correspondance canonique dans le skill `medical-imaging-data` ; on évalue C1 sur le sous-ensemble de pathologies communes (typiquement Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion, Pneumothorax)

### Labels incertains de CheXpert
- **Contexte** : CheXpert a des labels {positif, négatif, incertain}
- **Solution** : politique U-Zeros / U-Ones documentée et figée dès M2

## 9. État actuel

- [x] Cadrage scientifique et choix de la direction (RadGap)
- [x] Scaffold généré (CLAUDE.md, PLAN.md, 5 SKILL.md)
- [x] **Accès Stanford AIMI obtenu** (CheXpert Plus + MURA ; clé API en place)
- [~] **M0 bootstrap** : repo (uv + Python 3.11), `src/radgap/`, `configs/` Hydra, `scripts/check_env.py`, `tests/` (4 verts), CI ; CUDA 4080 vérifié (torch 2.6.0+cu124, AMP OK)
- [ ] **PROCHAINE ACTION** : (1) lancer l'accréditation PhysioNet [utilisateur], (2) push GitHub pour CI verte, (3) démarrer M1 (download CheXpert Plus + MURA)

## 10. Prochaine session

1. Lancer la demande d'accréditation PhysioNet (délai long, priorité absolue)
2. Initialiser l'environnement (`pyproject.toml`, venv, vérif CUDA sur la 4080)
3. ~~Demander l'accès Stanford AIMI~~ ✅ fait — télécharger CheXpert Plus + MURA via la clé API (M1)

## 11. Références clés

- RAD-DINO : Pérez-García et al., *Exploring Scalable Medical Image Encoders Beyond Text Supervision* (2025) — arXiv 2401.10815 ; poids `microsoft/rad-dino`
- CheXFound : *Chest X-ray Foundation Model with Global and Local Representations Integration* — arXiv 2502.05142
- EVA-X : *EVA-X: a foundation model for general chest x-ray analysis with self-supervised learning* — npj Digital Medicine (2025)
- CheXpert Plus : Chambon et al. — arXiv 2405.19538
- Biais : Glocker et al., *Risk of bias in chest radiography deep learning foundation models*, Radiology: AI (2023)
- Comparaison FMs : Li et al., *From embeddings to accuracy: Comparing foundation models for radiographic classification* — arXiv 2505.10823
- Stanford AIMI : https://aimi.stanford.edu/datasets

## 12. Idées parking (ne PAS implémenter sans finir le MVP)

> Capture ici toute idée hors-scope pour ne rien perdre (anti-FOMO).
- Extension zero-shot vision-langage (CheXzero-style) avec les rapports CheXpert Plus
- Génération de compte-rendu (MAIRA / MedGemma)
- IRM 3D (MRNet) avec un encodeur volumique
- Test-time adaptation pour réduire le generalization gap (méthode, pas juste mesure)

---

## 13. Documentation vivante — mettre à jour le README à chaque évolution

**Commande / réflexe obligatoire** : le `README.md` est la vitrine publique du projet et doit **toujours** être synchrone avec le code. À chaque changement structurant, le mettre à jour **dans le même commit que le changement**.

**Déclencheurs (quand mettre à jour le README) :**
- Un jalon change de statut → mettre à jour la table **Feuille de route** + les badges/statuts.
- Nouveau script CLI, module, ou dépendance → mettre à jour **Installation / Démarrage rapide / Structure du dépôt**.
- Nouvelle décision technique (cf. §7) ou changement de dataset/backbone → répercuter dans **Datasets / Model zoo / Méthodologie**.
- Premiers résultats (M3+) → remplir la table **Résultats** (avec IC, jamais de chiffre nu).
- Changement de licence, d'éthique, ou de procédure d'accès aux données → section **Éthique, licences & données**.

**Comment :**
1. Éditer `README.md` à la main pour les changements ciblés (préféré).
2. Pour une passe complète post-jalon, utiliser le skill gstack **`/document-release`** (synchronise README/ARCHITECTURE/CHANGELOG avec le diff).
3. Vérifier que la commande de repro du README tourne encore (`uv run python scripts/check_env.py`, `uv run pytest -q`).

> Garde-fou : un README qui ment (commande cassée, jalon faux, résultat périmé) est pire que pas de README. Si tu n'as pas le temps de le mettre à jour, note-le dans `PLAN.md` plutôt que de laisser une info fausse.
