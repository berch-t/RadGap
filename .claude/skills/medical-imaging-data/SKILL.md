---
name: medical-imaging-data
description: >
  Acquire, organize, and harmonize public radiology datasets (Stanford AIMI CheXpert /
  CheXpert Plus / MURA, plus external test sets NIH ChestX-ray14, PadChest, VinDr-CXR,
  MIMIC-CXR) for a cross-dataset benchmark. Use whenever the task involves downloading a
  medical imaging dataset, writing a dataset loader, building a unified manifest, mapping
  pathology labels across datasets, splitting by patient, or handling data-use agreements
  and licensing for radiology data. Covers the label-harmonization table that is the core
  of any cross-dataset study.
---

# Medical Imaging Data: Acquisition & Cross-Dataset Harmonization

## When to use this skill
Any time you acquire, load, split, or harmonize a radiology dataset, or build the unified
manifest that ties images to labels across datasets. The hard part of a cross-dataset
benchmark is **not** loading pixels — it is making labels comparable. This skill owns that.

## Golden rules (read first)
1. **Never commit raw data.** Stanford datasets are under a Research Use Agreement
   (non-commercial, no redistribution). `data/` must be in `.gitignore`. Commit download
   *scripts*, not data.
2. **Split by patient, never by image.** A patient with multiple studies must live in a
   single split. Image-level splits leak and inflate metrics.
3. **Freeze the uncertain-label policy once.** CheXpert labels are `{1, 0, -1 (uncertain), blank}`.
   Pick a policy (U-Zeros or U-Ones) per pathology, document it, never silently change it.
4. **PhysioNet datasets need credentialing.** VinDr-CXR and MIMIC-CXR require an approved
   PhysioNet account + CITI training. Lead time is days — request on day 1.
5. **Evaluate cross-dataset only on the shared label subset.** Do not invent label mappings
   to inflate coverage.

## Dataset inventory & access

| Dataset | Modality | Access point | Notes |
|---|---|---|---|
| CheXpert | Chest X-ray | Stanford AIMI portal | `small` (~11 GB, ~390x320) for iteration; `full` (~440 GB, DICOM) for final runs |
| CheXpert Plus | CXR + reports + demographics | Stanford AIMI | Adds free-text reports, patient demographics, DICOM — used for fairness (C3) |
| MURA | Musculoskeletal X-ray | Stanford AIMI | Binary normal/abnormal per study, 7 body parts — used for cross-region transfer (C4) |
| NIH ChestX-ray14 | Chest X-ray | Kaggle (`nih-chest-xrays/data`) | No credentialing — the fast OOD test-set of first resort |
| PadChest | Chest X-ray | BIMCV (Valencia) | Spanish hospital population, large label vocabulary |
| VinDr-CXR | Chest X-ray | PhysioNet | Vietnamese population, radiologist bounding boxes |
| MIMIC-CXR | CXR + reports | PhysioNet | Largest paired CXR+text; needed only for the VLM extension |

## Recommended on-disk layout
```
data/
├── raw/
│   ├── chexpert/            # as downloaded
│   ├── chexpert_plus/
│   ├── mura/
│   ├── nih_cxr14/
│   ├── padchest/
│   └── vindr_cxr/
├── processed/               # optional resized caches
└── manifests/
    ├── chexpert.parquet
    ├── nih_cxr14.parquet
    ├── ...
    └── unified.parquet      # the single source of truth for training/eval
```

## The unified manifest schema
Every row = one image. This is the contract the rest of the codebase depends on.

| Column | Type | Description |
|---|---|---|
| `image_path` | str | Absolute or repo-relative path to the image file |
| `dataset` | str | `chexpert` / `nih_cxr14` / `padchest` / `vindr_cxr` / `mura` |
| `patient_id` | str | Stable per-patient id (prefix with dataset to avoid collisions) |
| `split` | str | `train` / `val` / `test` (assigned by patient) |
| `view` | str | `PA` / `AP` / `LATERAL` / `unknown` |
| `label_<canonical>` | int | One column per canonical pathology: 1 / 0 / NaN (not labeled) |
| `sex` | str | `M` / `F` / `unknown` (for C3) |
| `age` | float | Years, or NaN |
| `race` | str | Self-reported, CheXpert only (for C3) — handle with care |

## Canonical label harmonization (the core artifact)
Define a single canonical pathology vocabulary, then map each dataset onto it. Evaluate
cross-dataset metrics **only** on pathologies present in both source and target.

Recommended canonical subset (well-represented and consistently defined across CXR datasets):
`Cardiomegaly`, `Edema`, `Consolidation`, `Atelectasis`, `Pleural Effusion`, `Pneumothorax`,
`Pneumonia`, `Lung Opacity`, `Fracture`, `No Finding`.

Encode the mapping as a versioned dictionary, not scattered `if` statements:

```python
# src/radgap/data/label_map.py
CANONICAL = [
    "Cardiomegaly", "Edema", "Consolidation", "Atelectasis",
    "Pleural Effusion", "Pneumothorax", "Pneumonia", "Lung Opacity",
    "Fracture", "No Finding",
]

# dataset_label -> canonical_label (None = drop / not mappable)
LABEL_MAP = {
    "chexpert": {
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
        # Enlarged Cardiomediastinum, Lung Lesion, Pleural Other, Support Devices -> not in canonical subset
    },
    "nih_cxr14": {
        "Cardiomegaly": "Cardiomegaly",
        "Edema": "Edema",
        "Consolidation": "Consolidation",
        "Atelectasis": "Atelectasis",
        "Effusion": "Pleural Effusion",          # name differs
        "Pneumothorax": "Pneumothorax",
        "Pneumonia": "Pneumonia",
        "Infiltration": "Lung Opacity",           # approximate, document the assumption
        "No Finding": "No Finding",
        # Mass, Nodule, Emphysema, Fibrosis, Pleural_Thickening, Hernia -> drop
    },
    # padchest / vindr_cxr: add explicit mappings; both use different vocabularies.
}

def canonical_pathologies(dataset_a: str, dataset_b: str) -> list[str]:
    """Pathologies shared (after mapping) between two datasets — the eval subset."""
    a = set(LABEL_MAP[dataset_a].values())
    b = set(LABEL_MAP[dataset_b].values())
    return [p for p in CANONICAL if p in a and p in b]
```

> Document every non-trivial mapping (e.g. NIH `Infiltration` -> `Lung Opacity`) in CLAUDE.md §8.
> Reviewers at a university *will* ask about these choices.

## Uncertain-label policy (CheXpert)
```python
def apply_uncertainty_policy(value, pathology, policy="u_zeros"):
    # value in {1.0, 0.0, -1.0 (uncertain), NaN (blank)}
    if value == -1.0:
        if policy == "u_zeros":
            return 0.0
        if policy == "u_ones":
            return 1.0
        if policy == "ignore":
            return float("nan")  # excluded from loss/metric for that label
    return value
```
A common, defensible default: U-Ones for `Atelectasis` and `Edema`, U-Zeros elsewhere
(matches several CheXpert papers). Freeze the choice in config, not in code.

## Patient-level splitting
```python
from sklearn.model_selection import GroupShuffleSplit

def split_by_patient(df, test_size=0.1, val_size=0.1, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss.split(df, groups=df["patient_id"]))
    trainval = df.iloc[trainval_idx]
    rel_val = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    train_idx, val_idx = next(gss2.split(trainval, groups=trainval["patient_id"]))
    df.loc[trainval.iloc[train_idx].index, "split"] = "train"
    df.loc[trainval.iloc[val_idx].index, "split"]   = "val"
    df.loc[df.iloc[test_idx].index, "split"]        = "test"
    return df
```

## Validation checks (make these a pytest)
- Zero `patient_id` appears in more than one split.
- Zero broken `image_path`.
- Every external dataset shares >= 4 canonical pathologies with CheXpert (else C1 is too thin).
- Label columns contain only `{0, 1, NaN}` after the uncertainty policy.

## Anti-patterns
- Splitting before harmonizing labels (you will re-split).
- Hard-coding label names in loaders instead of using `LABEL_MAP`.
- Treating NaN (not labeled) as 0 (negative) — that silently corrupts AUROC. Mask it instead.
- Storing absolute machine-specific paths in the committed manifest (use a configurable data root).
