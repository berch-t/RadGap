"""Wrapper unifié des backbones gelés : une interface `.embed()` -> embedding (B, D) poolé.

Protocole RadGap (cf. SKILL `foundation-model-adaptation`) : backbones **gelés**, on extrait
les embeddings une fois, on les cache, puis on entraîne des têtes légères. Le wrapper met
tous les backbones sur un pied d'égalité (même protocole, même type de sortie).

Backbones supportés ici (embeddings vision) :
  - rad-dino / rad-dino-maira-2 / dinov2 : ViT HuggingFace, embedding = token CLS.
  - biomedclip : via open_clip, embedding = `encode_image`.
DenseNet-121 (torchxrayvision) est une baseline supervisée à logits (M2), gérée à part.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# Identifiants HF par défaut (surchargés par la config Hydra `backbone.hf_id`).
_HF_DEFAULTS: dict[str, str] = {
    "rad-dino": "microsoft/rad-dino",
    "rad-dino-maira-2": "microsoft/rad-dino-maira-2",
    "dinov2": "facebook/dinov2-base",
}
_OPEN_CLIP_DEFAULTS: dict[str, str] = {
    "biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
}


class Backbone:
    """Encodeur d'images gelé. `.transform(pil_image) -> tensor` ; `.embed(batch) -> (B, D)`."""

    def __init__(self, name: str, hf_id: str | None = None, device: str = "cuda"):
        self.name = name
        self.device = device
        self.transform: Callable
        self.model, self.transform, self.dim, self._kind = self._load(name, hf_id)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _load(self, name: str, hf_id: str | None):
        if name in _HF_DEFAULTS or (hf_id and not hf_id.startswith("hf-hub:")):
            from transformers import AutoImageProcessor, AutoModel

            model_id = hf_id or _HF_DEFAULTS[name]
            model = AutoModel.from_pretrained(model_id)
            proc = AutoImageProcessor.from_pretrained(model_id)

            def transform(img):  # PIL -> pixel_values (C, H, W)
                return proc(img.convert("RGB"), return_tensors="pt")["pixel_values"][0]

            return model, transform, int(model.config.hidden_size), "hf_vit"

        if name in _OPEN_CLIP_DEFAULTS or (hf_id and hf_id.startswith("hf-hub:")):
            import open_clip

            model_id = hf_id or _OPEN_CLIP_DEFAULTS[name]
            model, _, preprocess = open_clip.create_model_and_transforms(model_id)

            def transform(img):
                return preprocess(img.convert("RGB"))

            dim = int(getattr(model.visual, "output_dim", 512))
            return model, transform, dim, "open_clip"

        raise ValueError(f"backbone inconnu : {name!r}")

    @torch.no_grad()
    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embedding poolé (B, D) pour un batch de pixel_values déjà transformés."""
        pixel_values = pixel_values.to(self.device)
        if self._kind == "hf_vit":
            out = self.model(pixel_values)
            return out.last_hidden_state[:, 0]  # token CLS
        if self._kind == "open_clip":
            return self.model.encode_image(pixel_values)
        raise RuntimeError(f"kind inattendu : {self._kind}")
