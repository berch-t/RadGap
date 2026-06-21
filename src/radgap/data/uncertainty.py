"""Politique des labels incertains de CheXpert — figée à M1, jamais changée en douce.

CheXpert encode chaque observation dans {1.0, 0.0, -1.0 (incertain), NaN (vide)}.
On choisit une politique par pathologie, on la documente, on la stocke en config.

Défaut défendable (plusieurs papiers CheXpert) : U-Ones pour `Atelectasis` et `Edema`,
U-Zeros ailleurs. Voir SKILL `medical-imaging-data`.
"""

from __future__ import annotations

import math

# Politique par défaut par pathologie (override possible via config Hydra).
DEFAULT_UNCERTAINTY_POLICY: dict[str, str] = {
    "Atelectasis": "u_ones",
    "Edema": "u_ones",
}
GLOBAL_DEFAULT = "u_zeros"  # pour toute pathologie non listée ci-dessus


def apply_uncertainty_policy(value: float, policy: str = "u_zeros") -> float:
    """Mappe une valeur CheXpert vers {0.0, 1.0, NaN} selon la politique.

    - `u_zeros` : incertain -> 0.0
    - `u_ones`  : incertain -> 1.0
    - `ignore`  : incertain -> NaN (exclu de la loss et des métriques)
    """
    if value == -1.0:
        if policy == "u_zeros":
            return 0.0
        if policy == "u_ones":
            return 1.0
        if policy == "ignore":
            return math.nan
        raise ValueError(f"politique inconnue : {policy!r}")
    return value


def resolve_policy(pathology: str, override: str | None = None) -> str:
    """Politique effective pour une pathologie (override global > défaut par pathologie)."""
    if override is not None:
        return override
    return DEFAULT_UNCERTAINTY_POLICY.get(pathology, GLOBAL_DEFAULT)
