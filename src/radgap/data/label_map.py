"""Harmonisation des labels cross-dataset — l'artefact central de RadGap (C1).

Le cœur d'un benchmark cross-dataset n'est pas de charger des pixels, c'est de rendre
les labels comparables. On définit un **vocabulaire canonique** de pathologies, puis on
mappe chaque dataset dessus. Les métriques cross-dataset (C1) ne sont calculées QUE sur
le sous-ensemble de pathologies partagé entre source et cible.

Toute correspondance non triviale (ex. NIH `Infiltration` -> `Lung Opacity`) est
documentée ici et dans CLAUDE.md §8 — les relecteurs académiques poseront la question.

Voir le SKILL `.claude/skills/medical-imaging-data`.
"""

from __future__ import annotations

# --- Vocabulaire canonique (sous-ensemble bien défini et représenté sur les datasets CXR) ---
CANONICAL: list[str] = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
    "Pneumothorax",
    "Pneumonia",
    "Lung Opacity",
    "Fracture",
    "No Finding",
]


def canonical_column(name: str) -> str:
    """`"Pleural Effusion"` -> `"label_pleural_effusion"` (nom de colonne du manifest)."""
    return "label_" + name.lower().replace(" ", "_")


# Colonnes de labels canoniques telles qu'elles apparaissent dans le manifest unifié.
LABEL_COLUMNS: list[str] = [canonical_column(p) for p in CANONICAL]


# --- Table de correspondance : label_brut_dataset -> label_canonique (None = non mappable) ---
# Plusieurs labels bruts peuvent pointer vers le même canonique : on agrège alors en OR
# (max en ignorant les NaN) au moment de l'harmonisation (cf. harmonize_labels).
LABEL_MAP: dict[str, dict[str, str | None]] = {
    # CheXpert / CheXpert Plus partagent les 14 mêmes observations.
    "chexpert_plus": {
        "Cardiomegaly": "Cardiomegaly",
        "Edema": "Edema",
        "Consolidation": "Consolidation",
        "Atelectasis": "Atelectasis",
        "Pleural Effusion": "Pleural Effusion",
        "Pneumothorax": "Pneumothorax",
        "Pneumonia": "Pneumonia",
        "Lung Opacity": "Lung Opacity",
        "Fracture": "Fracture",
        "No Finding": "No Finding",
        # Enlarged Cardiomediastinum, Lung Lesion, Pleural Other, Support Devices -> hors canonique
    },
    "nih_cxr14": {
        "Cardiomegaly": "Cardiomegaly",
        "Edema": "Edema",
        "Consolidation": "Consolidation",
        "Atelectasis": "Atelectasis",
        "Effusion": "Pleural Effusion",  # nom différent
        "Pneumothorax": "Pneumothorax",
        "Pneumonia": "Pneumonia",
        "Infiltration": "Lung Opacity",  # APPROXIMATION documentée
        "No Finding": "No Finding",
        # Mass, Nodule, Emphysema, Fibrosis, Pleural_Thickening, Hernia -> drop
    },
    # PadChest : vocabulaire hiérarchique large (labels standardisés). Mappings explicites.
    "padchest": {
        "cardiomegaly": "Cardiomegaly",
        "pulmonary edema": "Edema",
        "consolidation": "Consolidation",
        "atelectasis": "Atelectasis",
        "pleural effusion": "Pleural Effusion",
        "pneumothorax": "Pneumothorax",
        "pneumonia": "Pneumonia",
        "infiltrates": "Lung Opacity",  # APPROXIMATION
        "alveolar pattern": "Lung Opacity",  # APPROXIMATION (multi -> Lung Opacity, agrégé en OR)
        "rib fracture": "Fracture",
        "fracture": "Fracture",
        "normal": "No Finding",
    },
    # VinDr-CXR : 22 findings locaux + 6 diagnostics globaux.
    "vindr_cxr": {
        "Cardiomegaly": "Cardiomegaly",
        "Consolidation": "Consolidation",
        "Atelectasis": "Atelectasis",
        "Pleural effusion": "Pleural Effusion",
        "Pneumothorax": "Pneumothorax",
        "Pneumonia": "Pneumonia",
        "Lung Opacity": "Lung Opacity",
        "Infiltration": "Lung Opacity",  # APPROXIMATION (agrégé en OR avec Lung Opacity)
        "Rib fracture": "Fracture",
        "No finding": "No Finding",
        # VinDr ne fournit pas d'"Edema" explicite -> non mappé
    },
}


def canonical_pathologies(dataset_a: str, dataset_b: str) -> list[str]:
    """Pathologies partagées (après mapping) entre deux datasets — le sous-ensemble d'éval C1.

    Conserve l'ordre canonique pour des tables reproductibles.
    """
    a = {v for v in LABEL_MAP[dataset_a].values() if v is not None}
    b = {v for v in LABEL_MAP[dataset_b].values() if v is not None}
    return [p for p in CANONICAL if p in a and p in b]
