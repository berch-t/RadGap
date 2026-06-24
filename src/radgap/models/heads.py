"""Têtes légères pour le probing multi-label sur embeddings gelés (M3).

Multi-label CXR -> sorties sigmoïde + BCE **masquée** (les pathologies non annotées sont NaN
et exclues de la loss — non négociable en cross-dataset, cf. SKILL `foundation-model-adaptation`).
"""

from __future__ import annotations

import torch
from torch import nn


class Head(nn.Module):
    """Linear probe (hidden=0) ou MLP à une couche cachée."""

    def __init__(self, in_dim: int, n_labels: int, hidden: int = 0, p_drop: float = 0.1):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden, n_labels),
            )
        else:
            self.net = nn.Linear(in_dim, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits en ignorant les cibles NaN (non annotées)."""
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return logits.sum() * 0.0  # batch sans label : loss nulle différentiable
    return nn.functional.binary_cross_entropy_with_logits(
        logits[mask], targets[mask].float()
    )
