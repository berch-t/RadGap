"""Couche données : harmonisation des labels cross-dataset, splits, manifest unifié (M1)."""

from radgap.data.harmonize import (
    apply_uncertainty_to_frame,
    harmonize_labels,
)
from radgap.data.label_map import (
    CANONICAL,
    LABEL_COLUMNS,
    LABEL_MAP,
    canonical_column,
    canonical_pathologies,
)
from radgap.data.loaders import (
    expand_nih_findings,
    load_chexpert_plus,
    load_mura,
    load_nih,
)
from radgap.data.manifest import (
    build_unified_manifest,
    build_validate_save,
    load_manifest,
    save_manifest,
)
from radgap.data.schema import (
    METADATA_COLUMNS,
    label_columns,
    validate_manifest,
)
from radgap.data.splits import split_by_patient
from radgap.data.uncertainty import (
    apply_uncertainty_policy,
    resolve_policy,
)

__all__ = [
    "CANONICAL",
    "LABEL_COLUMNS",
    "LABEL_MAP",
    "METADATA_COLUMNS",
    "apply_uncertainty_policy",
    "apply_uncertainty_to_frame",
    "build_unified_manifest",
    "build_validate_save",
    "canonical_column",
    "canonical_pathologies",
    "expand_nih_findings",
    "harmonize_labels",
    "label_columns",
    "load_chexpert_plus",
    "load_manifest",
    "load_mura",
    "load_nih",
    "resolve_policy",
    "save_manifest",
    "split_by_patient",
    "validate_manifest",
]
