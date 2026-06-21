---
name: medical-ml-evaluation
description: >
  Rigorously evaluate medical image classifiers: per-pathology and macro/micro AUROC,
  bootstrap 95% confidence intervals, DeLong test for comparing AUROCs, calibration (ECE,
  reliability diagrams), operating-point and threshold selection, the cross-dataset
  generalization-gap protocol, subgroup/fairness metrics (AUROC and TPR gaps), and Grad-CAM /
  attention saliency for ViT backbones. Use whenever the task involves measuring model
  performance, comparing models statistically, auditing fairness, or producing publication-grade
  result tables and figures for a radiology study.
---

# Medical ML Evaluation (publication-grade)

## When to use this skill
Any time you measure performance, compare models, audit fairness, or build result tables/figures.
A serious POC lives or dies on evaluation rigor — point estimates without confidence intervals
will get torn apart in a university defense.

## Non-negotiables
1. **Every number gets a confidence interval** (bootstrap). No bare AUROC.
2. **Freeze the protocol before looking at results.** Decide metrics, thresholds, and subsets
   up front. No post-hoc metric shopping (p-hacking).
3. **Compare models with a statistical test** (DeLong), not eyeballed AUROC differences.
4. **Multi-label = per-pathology metrics first**, then aggregate. Never only report a single
   averaged number that hides per-disease failures.
5. **Mask unlabeled entries** (NaN) out of every metric — never count them as negatives.

## Primary metric: per-pathology AUROC + bootstrap CI
```python
import numpy as np
from sklearn.metrics import roc_auc_score

def auroc_per_label(y_true, y_score):
    """y_true, y_score: (N, L) arrays; NaN in y_true = unlabeled -> skipped."""
    out = {}
    for j in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, j])
        yt, ys = y_true[mask, j], y_score[mask, j]
        if len(np.unique(yt)) < 2:        # need both classes present
            out[j] = np.nan
        else:
            out[j] = roc_auc_score(yt, ys)
    return out

def bootstrap_ci(y_true, y_score, metric_fn, n=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y_true)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, N, N)            # resample with replacement
        stats.append(metric_fn(y_true[idx], y_score[idx]))
    lo, hi = np.nanpercentile(stats, [100*alpha/2, 100*(1-alpha/2)])
    return float(np.nanmean(stats)), float(lo), float(hi)
```
Report as `AUROC = 0.87 (95% CI 0.85-0.89)`. Aggregate with **macro-AUROC** (mean over
pathologies) as the headline, and show the per-pathology breakdown in a table.

## Comparing two models: DeLong test
Use DeLong for paired AUROC comparison on the *same* test set (correlated predictions).
```python
# use a maintained implementation, e.g. from a vetted gist or `delong` utilities
from radgap.eval.delong import delong_roc_test  # implement once, test once
p_value = delong_roc_test(y_true_binary, scores_model_a, scores_model_b)
```
> Without DeLong, "RAD-DINO beats BiomedCLIP by 0.4 AUROC" is meaningless — it may be noise.

## Calibration (clinically important)
A model can rank well (good AUROC) yet output miscalibrated probabilities. Report Expected
Calibration Error and a reliability diagram per pathology.
```python
def expected_calibration_error(y_true, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum() == 0:
            continue
        conf, acc = p[m].mean(), y_true[m].mean()
        ece += (m.sum() / len(p)) * abs(conf - acc)
    return ece
```

## Threshold / operating point
For TPR-based fairness and any sensitivity/specificity claim, fix a threshold policy:
- Choose per-pathology thresholds on the **validation** set (e.g. Youden's J or a fixed
  target specificity), then apply unchanged to test and to all subgroups.
- Document the policy; never tune thresholds on the test set.

## Cross-dataset generalization protocol (contribution C1)
```
For each backbone:
  train head on CheXpert train
  for target in {CheXpert test (in-dist), NIH, PadChest, VinDr (OOD)}:
      restrict to canonical_pathologies(CheXpert, target)
      compute macro-AUROC + bootstrap CI
  gap = AUROC(CheXpert test) - AUROC(target)      # the generalization gap
report gap per backbone per target, with CIs
```
Key comparison: do medical foundation models (RAD-DINO) show a *smaller* gap than the
ImageNet/supervised baselines? That is the scientific question of C1.

## Fairness / subgroup audit (contribution C3)
For the winning config, at the fixed threshold:
- Compute AUROC and TPR (sensitivity) per subgroup: sex, age bins, self-reported race.
- Report the **gap** = max(subgroup metric) - min(subgroup metric).
- Show subgroup sample sizes (small groups -> wide CIs -> interpret cautiously).
```python
def subgroup_report(df, y_true, y_score, group_col, threshold):
    rows = []
    for g, sub in df.groupby(group_col):
        idx = sub.index.to_numpy()
        auc, lo, hi = bootstrap_ci(y_true[idx], y_score[idx],
                                   lambda a, b: roc_auc_score(a, b))
        tpr = ((y_score[idx] >= threshold)[y_true[idx] == 1]).mean()
        rows.append({"group": g, "n": len(idx), "auroc": auc,
                     "ci": (lo, hi), "tpr": tpr})
    return rows
```
> Ethics/methodology: report gaps descriptively, never make causal claims, state limitations
> (group sizes, intersectionality not modeled). This honesty is what makes the audit credible.

## Interpretability: Grad-CAM / attention for ViT
Clinicians (and examiners) want to see *where* the model looks. For DINOv2-family ViTs use
attention rollout or Grad-CAM on the last block; overlay on the radiograph.
```python
# pytorch-grad-cam supports ViT with a reshape_transform for token grids
from pytorch_grad_cam import GradCAM
# target the final transformer block; provide reshape_transform to map tokens -> HxW
```
Sanity check: heatmaps for `Cardiomegaly` should concentrate on the cardiac silhouette,
`Pleural Effusion` at the costophrenic angles. Off-anatomy attention is a red flag (shortcut
learning — see the known "model reads the scanner/laterality marker" failure mode).

## Result artifacts (auto-generated, versioned)
- `experiments/results/auroc_in_distribution.csv`
- `experiments/results/generalization_gap.csv`
- `experiments/results/label_efficiency.csv`
- `experiments/results/fairness_<group>.csv`
- `experiments/figures/*.png` (gap bars, label-efficiency curves, reliability diagrams, Grad-CAM panels)
Every figure regenerable from a CSV by a single script — reproducibility requirement.

## Anti-patterns
- Reporting AUROC without CIs.
- Tuning thresholds on the test set.
- A single averaged metric hiding per-pathology collapse.
- Counting NaN (unlabeled) as negatives.
- Comparing models without DeLong.
- Fairness claims with no sample sizes / no stated limitations.
- Grad-CAM screenshots cherry-picked without negative examples.
