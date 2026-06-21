---
name: reproducible-ml-research
description: >
  Engineer an ML research project to be reproducible and publication-ready: repository
  structure, Hydra/OmegaConf configuration, deterministic seeding, experiment tracking with
  Weights & Biases, environment pinning, automatic regeneration of result tables and figures,
  model cards / datasheets, and a publishable-repo checklist. Use whenever the task involves
  setting up project scaffolding, configuration, experiment logging, ensuring runs are
  reproducible, or preparing a repo for public release / academic presentation.
---

# Reproducible ML Research

## When to use this skill
Project setup, configuration, seeding, experiment tracking, and preparing the repo for public
release and a university defense. A POC that cannot be reproduced is not science.

## Repository structure (single source of truth)
```
radgap/
├── CLAUDE.md PLAN.md README.md
├── pyproject.toml          # pinned deps; the only dependency source of truth
├── .gitignore              # data/, experiments/embeddings/, *.ckpt, wandb/
├── configs/                # Hydra configs; nothing hard-coded in code
│   ├── config.yaml
│   ├── dataset/{chexpert,nih,padchest,vindr}.yaml
│   ├── backbone/{rad_dino,dinov2,biomedclip,densenet}.yaml
│   └── experiment/{in_dist,gen_gap,label_eff,fairness}.yaml
├── src/radgap/             # importable package (data, preprocessing, models, eval, utils)
├── scripts/                # thin CLIs that call src/ (download, extract, train, evaluate)
├── experiments/            # outputs: embeddings (gitignored), results CSVs, figures
├── notebooks/              # EDA only; no business logic
└── tests/                  # pytest; CI runs these
```
Rule: **logic lives in `src/`, configuration in `configs/`, entry points in `scripts/`,
outputs in `experiments/`.** Notebooks are scratch, never imported.

## Configuration with Hydra
Every knob (dataset path, backbone, learning rate, seed, uncertainty policy) is a config value.
```yaml
# configs/config.yaml
defaults:
  - dataset: chexpert
  - backbone: rad_dino
  - experiment: in_dist
  - _self_

seed: 42
data_root: ${oc.env:RADGAP_DATA_ROOT}   # machine-specific path via env var, never committed
output_dir: experiments/${experiment.name}/${backbone.name}/${dataset.name}
uncertainty_policy: u_zeros
```
```python
# scripts/train.py
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    set_determinism(cfg.seed)
    ...  # cfg fully describes the run; log it verbatim
```
Benefits: every run is described by its resolved config; sweeps are a one-liner; no
magic constants buried in code.

## Determinism & seeding
```python
import os, random, numpy as np, torch

def set_determinism(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```
For label-efficiency curves and any subsampling, run **multiple seeds** and report mean +/-
std (the error bars). A single seed is an anecdote.

## Experiment tracking (Weights & Biases)
```python
import wandb
wandb.init(project="radgap", config=dict(cfg), mode="offline")  # offline OK, sync later
wandb.log({"val/macro_auroc": macro_auroc, "epoch": epoch})
wandb.summary["test/macro_auroc"] = test_auroc
```
Log: full config, per-epoch val metrics, final test metrics, and the git commit hash. Tag runs
by milestone (M3/M4/...). Even in offline mode you get a queryable run history — essential when
the experiment matrix grows (backbones x datasets x label fractions).

## Environment pinning
```toml
# pyproject.toml (excerpt) — pin exact versions for reproducibility
[project]
name = "radgap"
requires-python = ">=3.11"
dependencies = [
  "torch==2.*", "torchvision==0.*", "transformers==4.*",
  "open_clip_torch", "torchxrayvision", "peft",
  "pydicom", "opencv-python-headless", "pillow",
  "pandas", "pyarrow", "scikit-learn", "scipy", "statsmodels",
  "hydra-core", "omegaconf", "wandb",
  "matplotlib", "seaborn", "pytorch-grad-cam",
]
```
Pin to exact versions before the final results runs. Record CUDA/driver versions in the README.
Capture an exact lock (`uv pip freeze` or `pip freeze > requirements.lock`) for the archived run.

## Reproducibility: results regenerate from scratch
- Each result table is written to `experiments/results/*.csv` by a script.
- Each figure is rendered from a CSV by `scripts/make_figures.py` — no manual plotting.
- A top-level `make all` (or a documented sequence) runs: download -> manifest -> extract ->
  train -> evaluate -> figures. A reader reproduces a key figure in a handful of commands.

## Provenance & ethics artifacts (for public release)
- **Model card**: intended use (research only), training data, known biases (e.g. RAD-DINO
  trained on three countries), limitations, license (MSRLA / research use).
- **Datasheet**: which datasets, their licenses (Stanford Research Use Agreement, PhysioNet
  credentialing), what is and is not redistributed (no raw data in repo).
- **LICENSE** for *your code* (MIT/Apache-2.0) — but state clearly that data and model weights
  carry their own non-commercial research licenses.

## Publishable-repo checklist
- [ ] `README.md`: one-paragraph pitch, install, "reproduce a key result in N commands", results table
- [ ] No data or heavy checkpoints committed (verify `.gitignore` + `git ls-files | wc -l` sanity)
- [ ] Every figure regenerable from a versioned CSV
- [ ] Seeds fixed; multi-seed error bars where subsampling is involved
- [ ] Config-driven runs (no hard-coded paths/constants); data root via env var
- [ ] CI: lint + pytest green
- [ ] Model card + datasheet + LICENSE present
- [ ] Git commit hash logged with each experiment
- [ ] Limitations section written honestly (generalization scope, fairness caveats)

## Anti-patterns
- Hard-coded paths and hyperparameters scattered in scripts.
- Plotting figures by hand in a notebook (not regenerable).
- One seed reported as if definitive.
- Committing data or 1 GB checkpoints (license violation + bloated repo).
- "It works on my machine" — no pinned environment, no recorded CUDA version.
- Results that cannot be traced back to a config + commit.
