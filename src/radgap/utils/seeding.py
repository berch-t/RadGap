"""Seeding déterministe pour des runs reproductibles (cf. SKILL reproducible-ml-research)."""

from __future__ import annotations

import os
import random


def set_determinism(seed: int = 42) -> None:
    """Fixe toutes les sources d'aléa (python, numpy, torch, cudnn).

    Pour les courbes d'efficacité-label et tout sous-échantillonnage, lancer
    plusieurs seeds et reporter moyenne ± std : un seul seed est une anecdote.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
