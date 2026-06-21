---
name: radiology-preprocessing
description: >
  Preprocess chest and musculoskeletal radiographs for deep-learning backbones: DICOM
  reading and windowing, photometric interpretation handling, intensity normalization,
  CLAHE, resizing to each backbone's expected input, per-backbone normalization stats,
  and clinically-valid data augmentation. Use whenever the task involves turning raw
  X-ray images (DICOM/PNG/JPG) into model-ready tensors, choosing augmentations, or
  matching a foundation model's preprocessing. Flags the laterality/flip pitfall that
  silently breaks medical models.
---

# Radiology Preprocessing & Clinically-Valid Augmentation

## When to use this skill
Whenever raw radiographs become model inputs: DICOM decoding, normalization, resizing, and
augmentation. Medical imaging has domain-specific traps that generic CV pipelines get wrong.

## Critical principle: match the backbone's training preprocessing
A frozen foundation model expects inputs distributed like its pretraining data. Mismatched
preprocessing quietly destroys performance. For RAD-DINO, BiomedCLIP, and DINOv2, fetch the
official processor instead of reinventing it:

```python
from transformers import AutoImageProcessor
processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
inputs = processor(images=pil_image, return_tensors="pt")  # handles resize + normalize
```
Only build a manual pipeline for backbones without a published processor (e.g. some GitHub
checkpoints), and then replicate their stated preprocessing exactly.

## DICOM handling (full CheXpert, MIMIC, VinDr)
```python
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

def read_dicom(path):
    ds = pydicom.dcmread(path)
    arr = apply_voi_lut(ds.pixel_array, ds)          # apply window/level if present
    # MONOCHROME1 means higher pixel value = darker; invert to MONOCHROME2 convention
    if ds.get("PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)  # -> [0, 1]
    return arr
```
> The `MONOCHROME1` inversion is a classic silent bug: forget it and a fraction of your
> images are contrast-inverted, tanking AUROC for no obvious reason.

## Intensity normalization & contrast
- Always min-max (or percentile clip 1-99%) to `[0,1]` before anything else.
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) often helps CXR contrast, but
  it is a *design choice*, not a default — ablate it, and apply it consistently to train AND
  eval. Never CLAHE the train set only.

```python
import cv2
def clahe(img01, clip=2.0, grid=(8, 8)):
    img8 = (img01 * 255).astype("uint8")
    out = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(img8)
    return out.astype("float32") / 255.0
```

## Grayscale -> 3 channels
Most ImageNet/medical ViTs expect 3-channel input. Replicate the single channel:
```python
tensor_3c = img01[None].repeat(3, axis=0)  # (3, H, W)
```
Do **not** apply a color map (jet/viridis) — it injects artificial color structure.

## Resizing
- Resize to the backbone's native resolution (RAD-DINO/DINOv2: 224 with patch 14; BiomedCLIP: 224 patch 16).
- Preserve aspect ratio with padding when feasible; if center-cropping, verify the lungs/region
  of interest are not cut off (apical pneumothorax lives at the top of the image).
- Use the same interpolation for train and eval.

## Per-backbone normalization stats
| Backbone | Mean/Std source |
|---|---|
| RAD-DINO | Use its `AutoImageProcessor` (do not hardcode) |
| DINOv2-ImageNet | ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]` |
| BiomedCLIP | Use the `open_clip` preprocess transform shipped with the checkpoint |
| Medical MAE / custom | Replicate the repo's stated normalization |

## Clinically-valid augmentation
Augmentation must not change the clinical meaning of the image.

**Safe for CXR**
- Small rotations (+/- 5-10 deg)
- Mild translation / scale (+/- 5-10%)
- Brightness/contrast jitter (small)
- Random resized crop with a conservative scale range (e.g. 0.8-1.0)
- Gaussian noise (mild)

**Dangerous / forbidden**
- ❌ **Horizontal flip** on chest X-rays: it swaps left/right and breaks laterality (the heart is
  on the patient's left; situs and unilateral findings become wrong). Disable it for CXR unless
  you deliberately study laterality-invariance.
- ❌ Vertical flip (anatomically meaningless).
- ❌ Aggressive elastic deformation / cutout over the region of interest (can erase the pathology).
- ❌ Heavy color jitter / hue shifts (grayscale domain).

```python
# torchvision-style, CXR-safe
import torchvision.transforms.v2 as T
train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
    T.RandomRotation(degrees=7),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    # NO RandomHorizontalFlip for chest X-ray
])
eval_tf = T.Compose([T.Resize(224, antialias=True), T.CenterCrop(224)])
```
> MURA (bones) tolerates horizontal flip (left/right limbs are roughly symmetric for
> abnormality detection) — but make the flip policy a per-dataset config flag, not a global default.

## When you cache embeddings, freeze the eval transform
If you precompute frozen-backbone embeddings (recommended), augmentation is applied **before**
extraction only if you re-extract per epoch. For the standard "extract once, train head" path,
extract embeddings with the deterministic `eval_tf` so they are stable and comparable across
backbones. Apply augmentation only in the (rarer) end-to-end fine-tuning path.

## Anti-patterns
- Different preprocessing for train vs eval (CLAHE on one only, different resize, etc.).
- Forgetting the MONOCHROME1 inversion.
- Horizontal-flipping chest X-rays.
- Normalizing with ImageNet stats for a backbone that shipped its own processor.
- Applying color maps to grayscale radiographs.
