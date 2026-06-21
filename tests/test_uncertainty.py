"""Tests de la politique des labels incertains CheXpert."""

import math

import pandas as pd

from radgap.data import apply_uncertainty_policy, apply_uncertainty_to_frame, resolve_policy


def test_apply_policy_values():
    assert apply_uncertainty_policy(-1.0, "u_zeros") == 0.0
    assert apply_uncertainty_policy(-1.0, "u_ones") == 1.0
    assert math.isnan(apply_uncertainty_policy(-1.0, "ignore"))
    # valeurs certaines inchangées
    assert apply_uncertainty_policy(1.0, "u_zeros") == 1.0
    assert apply_uncertainty_policy(0.0, "u_ones") == 0.0


def test_resolve_policy_per_pathology_default():
    assert resolve_policy("Atelectasis") == "u_ones"
    assert resolve_policy("Edema") == "u_ones"
    assert resolve_policy("Cardiomegaly") == "u_zeros"
    # override global gagne
    assert resolve_policy("Atelectasis", override="ignore") == "ignore"


def test_apply_to_frame_per_pathology():
    df = pd.DataFrame({"Atelectasis": [-1.0], "Cardiomegaly": [-1.0]})
    out = apply_uncertainty_to_frame(df, ["Atelectasis", "Cardiomegaly"])
    assert out["Atelectasis"].iloc[0] == 1.0  # u_ones par défaut
    assert out["Cardiomegaly"].iloc[0] == 0.0  # u_zeros par défaut


def test_apply_to_frame_global_override():
    df = pd.DataFrame({"Atelectasis": [-1.0]})
    out = apply_uncertainty_to_frame(df, ["Atelectasis"], global_policy="u_zeros")
    assert out["Atelectasis"].iloc[0] == 0.0
