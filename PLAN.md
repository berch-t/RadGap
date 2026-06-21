# PLAN.md - RadGap
> Source de vérité de l'avancement. Un jalon = un résultat concret et défendable.
> Dernière mise à jour : 2026-06-19

---

## Principe de pilotage (TDAH-aware)

1. **Un jalon à la fois.** On ne démarre M(n+1) que si la *Definition of Done* de M(n) est cochée.
2. **Chaque jalon produit un livrable visible** (une table, une courbe, un script qui tourne). Pas de jalon "invisible".
3. **Garde-fou de scope** : chaque jalon a une section "❌ Hors-scope" explicite. Si ça n'y est pas listé comme objectif, on ne le fait pas maintenant.
4. **Time-box** : si un jalon dépasse 1.5× son estimation, on s'arrête, on documente le blocage dans CLAUDE.md §8, et on décide (simplifier ou demander de l'aide) plutôt que de s'enliser.

Légende statut : `[ ]` à faire · `[~]` en cours · `[x]` fait

---

## M0 — Bootstrap & infrastructure
**Estimation : 2-3 jours** · Statut : `[~]` (reste : accréditation PhysioNet + push pour CI verte)

**Objectif** : un environnement reproductible et les démarches d'accès lancées.

**Tâches**
- [x] Accès **Stanford AIMI** obtenu (CheXpert Plus + MURA ; CheXpert standalone non proposé séparément) — clé API dans `AIMI_API_KEY.txt` (gitignored), exposée via `AIMI_API_KEY`
- [ ] Lancer l'**accréditation PhysioNet** + formation CITI (VinDr, MIMIC) — *priorité absolue, délai long* ⟵ **action utilisateur**
- [x] Init repo (`git init`, branche `main`), `pyproject.toml` pinné (uv), venv Python 3.11, `.gitignore` (data/ + checkpoints + secret API)
- [x] CUDA + PyTorch vérifiés sur la RTX 4080 Super (torch 2.6.0+cu124, AMP forward OK, sm_89, 16 Go)
- [x] Hydra (`configs/`) + W&B offline en place
- [x] Squelette `src/radgap/` + `tests/` (4 tests verts) + CI GitHub Actions (lint ruff + pytest)

**Definition of Done**
- [x] `pytest` passe (4/4) — CI verte *à confirmer après le 1er push GitHub*
- [x] `scripts/check_env.py` affiche GPU, versions, et fait un forward AMP de démo
- [~] Demandes d'accès **envoyées** : Stanford ✅ ; PhysioNet à lancer

**❌ Hors-scope** : tout code modèle ou données. On ne fait QUE l'infra.

---

## M1 — Acquisition & harmonisation des données
**Estimation : 1 semaine** (partiellement gated par les accès) · Statut : `[ ]`

**Objectif** : un *manifest unifié* (CSV/Parquet) reliant chaque image à ses labels harmonisés, pour tous les datasets.

**Tâches**
- [ ] `scripts/download_*.py` par dataset (CheXpert Plus, MURA, NIH, PadChest, VinDr) — voir SKILL `medical-imaging-data`
- [ ] Loaders standardisés renvoyant `(image_path, labels_vector, split, demographics)`
- [ ] **Table de correspondance des labels** canonique (le cœur du croisement) → sous-ensemble de pathologies communes figé
- [ ] Politique des labels incertains CheXpert (U-Zeros / U-Ones) figée et documentée
- [ ] Splits train/val/test par patient (jamais de fuite patient entre splits)
- [ ] Manifest unifié + checks d'intégrité (`scripts/validate_manifests.py`)

**Definition of Done**
- Un seul fichier `data/manifests/unified.parquet` charge tous les datasets disponibles
- Test automatique : 0 fuite patient inter-split, 0 chemin image cassé
- Les pathologies communes inter-datasets sont listées dans le manifest et dans CLAUDE.md

**❌ Hors-scope** : preprocessing image (M3), aucun entraînement.

---

## M2 — Baselines supervisées
**Estimation : 3-4 jours** · Statut : `[ ]`

**Objectif** : reproduire une baseline historique pour avoir un point de comparaison crédible.

**Tâches**
- [ ] Charger un DenseNet-121 CheXpert via `torchxrayvision`
- [ ] Évaluer son AUROC par pathologie sur le test CheXpert (reproduire l'ordre de grandeur publié)
- [ ] (option) Fine-tuner un DenseNet-121 ImageNet sur CheXpert comme seconde baseline
- [ ] Brancher le pipeline d'éval (SKILL `medical-ml-evaluation`) : AUROC + IC bootstrap

**Definition of Done**
- Table "Baseline supervisée — AUROC par pathologie (± IC 95%)" générée automatiquement
- Les chiffres sont dans l'ordre de grandeur de la littérature CheXpert (sanity check passé)

**❌ Hors-scope** : foundation models (M3), généralisation OOD (M4).

---

## M3 — Embeddings foundation models + linear probes (in-distribution)
**Estimation : 1 semaine** · Statut : `[ ]`

**Objectif** : la performance in-distribution de chaque backbone gelé sur CheXpert.

**Tâches**
- [ ] Wrappers backbones unifiés (RAD-DINO, BiomedCLIP, DINOv2-ImageNet, Medical MAE) — SKILL `foundation-model-adaptation`
- [ ] `scripts/extract_embeddings.py` : extrait et **cache sur disque** les embeddings CheXpert pour chaque backbone
- [ ] Entraîner une tête (linear probe + MLP léger) par backbone
- [ ] Évaluer AUROC in-distribution (test CheXpert) pour tous les backbones

**Definition of Done**
- Embeddings cachés réutilisables (`experiments/embeddings/<backbone>/`)
- Table comparative "AUROC in-distribution par backbone (± IC 95%)"
- Test DeLong : différences significatives ou non entre backbones documentées

**❌ Hors-scope** : LoRA (M7), OOD (M4), fairness (M6).

---

## M4 — Étude de généralisation cross-dataset (C1) ★ RÉSULTAT CŒUR
**Estimation : 1 semaine** · Statut : `[ ]`

**Objectif** : quantifier l'effondrement des perfs hors distribution — la contribution centrale.

**Tâches**
- [ ] Évaluer chaque tête (entraînée sur CheXpert) sur NIH, PadChest, VinDr **sans réentraînement**
- [ ] Restreindre aux pathologies communes (table M1)
- [ ] Calculer le **generalization gap** : Δ AUROC = AUROC(CheXpert test) − AUROC(dataset externe)
- [ ] Comparer le gap entre foundation models médicaux et baselines (hypothèse : les FMs médicaux généralisent mieux)

**Definition of Done**
- Figure "Generalization gap par backbone et par dataset cible"
- Table Δ AUROC avec IC bootstrap
- Une phrase de conclusion claire (ex. "RAD-DINO perd X points hors distribution vs Y pour la baseline")

**❌ Hors-scope** : méthodes pour *réduire* le gap (idée parking : test-time adaptation). On *mesure*, on ne corrige pas encore.

---

## M5 — Courbes d'efficacité-label (C2)
**Estimation : 4-5 jours** · Statut : `[ ]`

**Objectif** : montrer combien de labels chaque backbone "économise".

**Tâches**
- [ ] Sous-échantillonner le train CheXpert à {1%, 10%, 100%} (stratifié, seed fixé, n répétitions)
- [ ] Réentraîner les têtes sur chaque fraction, par backbone
- [ ] Tracer AUROC = f(fraction de labels) par backbone

**Definition of Done**
- Figure "Label efficiency" avec barres d'erreur (n seeds)
- Conclusion quantifiée (ex. "RAD-DINO @1% ≈ baseline @10%")

**❌ Hors-scope** : nouvelles architectures, augmentation de données avancée.

---

## M6 — Audit d'équité (C3)
**Estimation : 1 semaine** · Statut : `[ ]`

**Objectif** : auditer les biais démographiques — rigueur clinique, fort impact en soutenance.

**Tâches**
- [ ] Joindre la démographie (CheXpert Plus + labels de race CheXpert) au manifest
- [ ] Pour la config gagnante : calculer AUROC et TPR par sous-groupe (sexe, tranches d'âge, race auto-déclarée) à seuil fixé
- [ ] Calculer les **écarts** (max−min) d'AUROC/TPR entre sous-groupes
- [ ] Méthodo défendable : seuil global vs seuils par groupe documentés

**Definition of Done**
- Table "Performance par sous-groupe + écarts"
- Discussion honnête des limites (taille des sous-groupes, intersectionnalité)

**❌ Hors-scope** : méthodes de *dé-biaisage* (idée parking). On audite, on ne corrige pas dans le MVP.

---

## M7 — (Stretch) Transfert cross-région + LoRA
**Estimation : 1 semaine** · Statut : `[ ]`

**Objectif** : pousser l'idée de "croisement" et tester le fine-tuning paramétrique.

**Tâches**
- [ ] Évaluer un encodeur thoracique (RAD-DINO) en transfert sur MURA (os) — linear probe
- [ ] LoRA fine-tuning de la config gagnante, comparer à la tête gelée
- [ ] Mesurer le gain LoRA vs coût (paramètres, temps)

**Definition of Done**
- Table "Transfert thorax → os" + "Gelé vs LoRA"
- Verdict : LoRA vaut-il le coût sur ce setup ?

**❌ Hors-scope** : full fine-tuning multi-backbone, IRM 3D.

---

## M8 — Rédaction, repo public, soutenance
**Estimation : 1 semaine** · Statut : `[ ]`

**Objectif** : transformer les résultats en livrable présentable et reproductible.

**Tâches**
- [ ] `README.md` complet (objectif, install, repro en N commandes, résultats clés)
- [ ] Toutes les tables/figures régénérables via `scripts/` (reproductibilité)
- [ ] Brouillon de short paper (intro, méthodo, résultats C1-C3, limites) — template type workshop
- [ ] Slides UGA (10-12 slides : problème → méthode → 3 résultats → limites → ouverture)
- [ ] Carte modèle + carte données (model card / datasheet) pour l'éthique

**Definition of Done**
- Un nouvel utilisateur peut cloner, suivre le README, et reproduire une figure clé
- Slides prêtes, paper en brouillon relu

**❌ Hors-scope** : soumission à une conférence (post-soutenance éventuellement).

---

## Registre des risques

| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| Accréditation PhysioNet lente | Bloque C1 partiellement | Élevée | Lancée à M0 ; NIH (Kaggle) comme test OOD de repli immédiat |
| Volume CheXpert Plus / stockage | Bloque M1 | Moyenne | Privilégier la version downsampled (PNG basse-résolution) pour itérer ; DICOM haute-résolution réservé aux résultats finaux |
| Harmonisation labels ambiguë | Fragilise C1 | Moyenne | Figer le sous-ensemble commun + documenter chaque choix |
| Scope creep (TDAH/FOMO) | Projet jamais fini | Élevée | Idées parking + garde-fous par jalon + time-box |
| Sur-interprétation fairness | Critique méthodo en soutenance | Moyenne | Limites explicites, pas de claim causal |

---

## Chemin critique

```
M0 (lance accès) ──► M1 ──► M3 ──► M4 ★ ──► M5 ──► M6 ──► M8
        │                    ▲
        └─ accréditation ────┘ (parallèle, ne bloque pas M2/M3)
   M2 (baseline) peut tourner en parallèle de M1 dès que CheXpert est dispo
```

**MVP minimal soutenable** = M0 → M1 → M3 → M4 (généralisation) → M8. C1 seul fait déjà un POC défendable. C2/C3 renforcent. C4 est bonus.
