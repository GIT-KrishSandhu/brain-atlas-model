import os
import pandas as pd
import pytest


def test_split_integrity():
    split_csv = "experiments/splits/topaneu_v1.csv"
    assert os.path.exists(split_csv), f"Split file not found: {split_csv}"

    df = pd.read_csv(split_csv)
    # Total scans
    assert len(df) == 415, f"Expected 415 cases, got {len(df)}"

    # Total patients
    unique_patients = df["patient_id"].unique()
    assert len(unique_patients) == 408, f"Expected 408 patients, got {len(unique_patients)}"

    # Check partitions exist
    splits = set(df["split"].unique())
    assert splits == {"train", "val", "test"}, f"Unexpected splits: {splits}"

    # Check zero leakage
    train_pids = set(df[df["split"] == "train"]["patient_id"])
    val_pids = set(df[df["split"] == "val"]["patient_id"])
    test_pids = set(df[df["split"] == "test"]["patient_id"])

    assert len(train_pids.intersection(val_pids)) == 0, "Patient leakage: train & val overlap!"
    assert len(train_pids.intersection(test_pids)) == 0, "Patient leakage: train & test overlap!"
    assert len(val_pids.intersection(test_pids)) == 0, "Patient leakage: val & test overlap!"


def test_split_class_balance():
    split_csv = "experiments/splits/topaneu_v1.csv"
    df = pd.read_csv(split_csv)

    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        prev = sub["presence"].mean()
        # Prevalence should be roughly 72-74%
        assert 0.70 <= prev <= 0.76, f"Prevalence in {s} out of expected bounds: {prev:.3f}"
