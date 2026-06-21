---
name: foundation-model-adaptation
description: >
  Load and adapt medical vision foundation models (RAD-DINO, BiomedCLIP, DINOv2, Medical MAE,
  TorchXRayVision DenseNet) for downstream classification. Covers a unified backbone wrapper,
  frozen-feature embedding extraction with on-disk caching, linear-probe and MLP-head training,
  parameter-efficient LoRA fine-tuning, and the mixed-precision / gradient-checkpointing tricks
  needed to fit a 16GB GPU (RTX 4080 Super). Use whenever the task involves loading a pretrained
  vision backbone, extracting embeddings, training a classifier head, or fine-tuning efficiently.
---

# Foundation Model Adaptation (on a 16 GB GPU)

## When to use this skill
Loading any pretrained backbone, extracting features, training heads, or LoRA fine-tuning.

## Core strategy: freeze, extract once, train heads fast
For RAD-DINO, fine-tuning is usually unnecessary — a classifier on the CLS token performs well.
The efficient and *comparable-across-backbones* protocol:
1. Freeze the backbone.
2. Extract embeddings once, cache to disk.
3. Train cheap heads (minutes, even on CPU).
4. LoRA-fine-tune only the single winning configuration.

This keeps every backbone on equal footing (same protocol) and makes iteration TDAH-friendly.

## Unified backbone wrapper
One interface for every backbone, returning a fixed-dim embedding:

```python
# src/radgap/models/backbones.py
import torch
from transformers import AutoModel, AutoImageProcessor

class Backbone:
    """Uniform wrapper: .embed(pixel_values) -> (B, D) pooled embedding."""
    def __init__(self, name: str, device="cuda"):
        self.name = name
        self.device = device
        self.model, self.processor, self.dim = self._load(name)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _load(self, name):
        if name == "rad-dino":
            m = AutoModel.from_pretrained("microsoft/rad-dino")
            p = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
            return m, p, 768
        if name == "dinov2":
            m = AutoModel.from_pretrained("facebook/dinov2-base")
            p = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            return m, p, 768
        if name == "biomedclip":
            # via open_clip; expose .encode_image and the shipped preprocess
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
            return model, preprocess, 512
        raise ValueError(f"unknown backbone {name}")

    @torch.no_grad()
    def embed(self, pixel_values):
        pixel_values = pixel_values.to(self.device)
        if self.name in {"rad-dino", "dinov2"}:
            out = self.model(pixel_values)
            # CLS / pooled token; for DINOv2-family use last_hidden_state[:, 0]
            cls = out.last_hidden_state[:, 0]
            return cls
        if self.name == "biomedclip":
            return self.model.encode_image(pixel_values)
```
> Note the embedding dims differ (768 vs 512). Heads are sized per backbone — store `dim` in the manifest of cached embeddings.

## Model zoo cheat-sheet
| Backbone | Load via | Dim | Notes |
|---|---|---|---|
| RAD-DINO | `transformers.AutoModel("microsoft/rad-dino")` | 768 | CXR-specialized; primary FM. Also `microsoft/rad-dino-maira-2` (more data) |
| DINOv2-ImageNet | `transformers.AutoModel("facebook/dinov2-base")` | 768 | Natural-image baseline — the "is medical pretraining worth it?" control |
| BiomedCLIP | `open_clip` hf-hub | 512 | Vision-language baseline; enables zero-shot extension later |
| Medical MAE | GitHub `lambert-x/medical_mae` | 768 | Alternative medical SSL baseline |
| DenseNet-121 (CheXpert) | `torchxrayvision` | n/a | Supervised historical baseline; outputs pathology logits directly |

## Embedding extraction with on-disk caching
```python
# scripts/extract_embeddings.py (sketch)
import torch, numpy as np
from pathlib import Path

def extract_and_cache(backbone, dataloader, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    embs, labels, ids = [], [], []
    autocast = torch.autocast("cuda", dtype=torch.float16)
    for pixel_values, y, image_id in dataloader:
        with autocast:
            e = backbone.embed(pixel_values).float().cpu()
        embs.append(e); labels.append(y); ids.extend(image_id)
    np.save(out_dir / "embeddings.npy", torch.cat(embs).numpy())
    np.save(out_dir / "labels.npy", torch.cat(labels).numpy())
    (out_dir / "ids.txt").write_text("\n".join(ids))
```
Cache path convention: `experiments/embeddings/<backbone>/<dataset>/<split>/`. Re-running
training never re-touches the GPU once embeddings exist.

## Linear probe & MLP head (multi-label)
Multi-label CXR -> sigmoid outputs + `BCEWithLogitsLoss`, with masking for unlabeled entries.
```python
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, in_dim, n_labels, hidden=0, p_drop=0.1):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(p_drop),
                nn.Linear(hidden, n_labels))
        else:
            self.net = nn.Linear(in_dim, n_labels)  # pure linear probe
    def forward(self, x): return self.net(x)

def masked_bce(logits, targets):
    # targets may contain NaN for "not labeled" -> exclude from loss
    mask = ~torch.isnan(targets)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits[mask], targets[mask].float())
    return loss
```
> The mask is non-negotiable for cross-dataset work: different datasets label different
> pathologies, so most targets are partially NaN. Masking keeps the loss honest.

## LoRA fine-tuning (winning config only)
```python
from peft import LoraConfig, get_peft_model

lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["query", "value"],   # attention proj for ViT
    bias="none",
)
peft_model = get_peft_model(backbone.model, lora_cfg)  # unfreeze only LoRA params
peft_model.print_trainable_parameters()                # typically <1% of weights
```
Run LoRA end-to-end (backbone + head) only for the configuration that already won under
the frozen protocol. Compare AUROC gain vs added cost.

## 16 GB GPU survival kit
- **AMP**: wrap forward/backward in `torch.autocast("cuda", dtype=torch.float16)` + `GradScaler`.
- **Gradient checkpointing** (fine-tuning only): `model.gradient_checkpointing_enable()`.
- **Batch size**: frozen extraction can use large batches (no grad); end-to-end fine-tuning
  start at 32, drop if OOM. Use gradient accumulation to simulate larger batches.
- **Pin the data root**, keep `num_workers` ~ 4-8, `pin_memory=True`.
- **Empty cache** between backbones: `torch.cuda.empty_cache()`.
- Feasible: ViT-B frozen extraction, all heads, LoRA. Not feasible: pretraining from scratch.

## Anti-patterns
- Re-extracting embeddings every epoch when the backbone is frozen (wasteful).
- Forgetting `model.eval()` / `requires_grad_(False)` on a frozen backbone (BatchNorm/dropout drift, accidental grads).
- One head dimension assumed for all backbones (RAD-DINO 768 vs BiomedCLIP 512).
- Treating multi-label as multi-class (softmax) — pathologies co-occur; use sigmoid + BCE.
- LoRA-tuning every backbone (combinatorial blowup, no extra insight).
