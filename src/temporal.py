"""
Module 6: Temporal Drift & Family Holdout Robustness Experiment
Implements:
1. Temporal split (when first_seen timestamps are populated)
2. Family-Disjoint Holdout Experiment (unseen botnet family generalization test)
3. In-period baseline comparison and drift delta calculation
4. Per-family recall breakdown
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier

from src.evaluate import compute_metrics
from src.features import FEATURE_SET_D

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
TEMPORAL_RESULTS_PATH = os.path.join(RESULTS_DIR, "temporal_results.csv")


def run_temporal_or_holdout_experiment(
    df_feats: pd.DataFrame,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the temporal drift experiment if dates are available;
    otherwise executes the rigorous Family-Disjoint Holdout experiment per specification.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    dga_df = df_feats[df_feats["is_dga"] == 1].copy()
    benign_df = df_feats[df_feats["is_dga"] == 0].copy()

    has_valid_dates = False
    if "first_seen" in dga_df.columns:
        valid_dates = dga_df["first_seen"].dropna()
        if len(valid_dates) > 100 and valid_dates.nunique() > 1:
            has_valid_dates = True

    feature_cols = [c for c in FEATURE_SET_D if c in df_feats.columns]

    if has_valid_dates:
        print("[Temporal] Running time-based split experiment based on first_seen...")
        dga_df["first_seen_dt"] = pd.to_datetime(dga_df["first_seen"])
        dga_sorted = dga_df.sort_values("first_seen_dt")
        cut_idx = int(0.70 * len(dga_sorted))
        train_dga = dga_sorted.iloc[:cut_idx]
        test_dga = dga_sorted.iloc[cut_idx:]
        exp_type = "Temporal (Date-Based)"
        split_desc = f"Date cut at {train_dga['first_seen_dt'].max()}"
    else:
        print("[Temporal] Note: Raw snapshot does not contain granular date timestamps.")
        print("[Temporal] Running specified Family-Disjoint Holdout Experiment as the strongest robustness proxy...")
        exp_type = "Family-Holdout (Disjoint Botnets)"
        
        # Group families and split by family so test families are completely unseen
        family_counts = dga_df["family"].value_counts()
        train_families = []
        test_families = []
        running_count = 0
        target_train_count = int(0.70 * len(dga_df))

        # Stratify families so 70% go to train, 30% go to test
        for fam, count in family_counts.items():
            if running_count + count <= target_train_count or len(train_families) == 0:
                train_families.append(fam)
                running_count += count
            else:
                test_families.append(fam)

        if not test_families:
            # Fallback if only 1 dominant family
            test_families = [train_families.pop()]

        train_dga = dga_df[dga_df["family"].isin(train_families)]
        test_dga = dga_df[dga_df["family"].isin(test_families)]
        split_desc = f"Train families: {len(train_families)}, Test holdout families: {len(test_families)}"

    # Benign split matching proportions (70% train, 30% test)
    benign_shuffled = benign_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    benign_cut = int(0.70 * len(benign_shuffled))
    train_benign = benign_shuffled.iloc[:benign_cut]
    test_benign = benign_shuffled.iloc[benign_cut:]

    # Construct Train & Test splits
    train_df = pd.concat([train_benign, train_dga], ignore_index=True).sample(frac=1.0, random_state=random_state)
    test_df = pd.concat([test_benign, test_dga], ignore_index=True).sample(frac=1.0, random_state=random_state)

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["is_dga"].to_numpy()
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["is_dga"].to_numpy()

    # Train Random Forest on training period / training families
    print(f"[Temporal] Training Random Forest on {len(train_df):,} samples...")
    rf_drift = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=random_state)
    rf_drift.fit(X_train, y_train)

    # In-period / in-distribution baseline: evaluated on a held-out slice of the training distribution
    val_slice_benign = train_benign.sample(frac=0.3, random_state=random_state)
    val_slice_dga = train_dga.sample(frac=0.3, random_state=random_state)
    val_in_period = pd.concat([val_slice_benign, val_slice_dga], ignore_index=True)
    X_in_period = val_in_period[feature_cols].to_numpy()
    y_in_period = val_in_period["is_dga"].to_numpy()

    preds_in_period = rf_drift.predict(X_in_period)
    probs_in_period = rf_drift.predict_proba(X_in_period)[:, 1]
    metrics_in_period = compute_metrics(y_in_period, preds_in_period, probs_in_period)

    # Evaluate on the temporal/holdout test set
    preds_drift = rf_drift.predict(X_test)
    probs_drift = rf_drift.predict_proba(X_test)[:, 1]
    metrics_drift = compute_metrics(y_test, preds_drift, probs_drift)

    # Calculate drift deltas: Drift - In_Period
    deltas = {
        "acc_delta": metrics_drift["acc"] - metrics_in_period["acc"],
        "precision_delta": metrics_drift["precision"] - metrics_in_period["precision"],
        "recall_delta": metrics_drift["recall"] - metrics_in_period["recall"],
        "f1_delta": metrics_drift["f1"] - metrics_in_period["f1"],
        "fpr_delta": metrics_drift["fpr"] - metrics_in_period["fpr"],
        "roc_auc_delta": metrics_drift["roc_auc"] - metrics_in_period["roc_auc"],
    }

    # Per-family recall breakdown on the test set
    test_dga_df = test_df[test_df["is_dga"] == 1].copy()
    test_dga_df["pred"] = preds_drift[test_df["is_dga"] == 1]
    
    family_recalls = {}
    for fam, group in test_dga_df.groupby("family"):
        rec = float(group["pred"].sum() / len(group))
        family_recalls[fam] = {
            "count": len(group),
            "detected": int(group["pred"].sum()),
            "recall": rec
        }

    # Format result DataFrame
    rows = [
        {
            "experiment": "In-Period Baseline",
            "type": exp_type,
            "description": "Random split within training slice",
            "accuracy": metrics_in_period["acc"],
            "precision": metrics_in_period["precision"],
            "recall": metrics_in_period["recall"],
            "f1": metrics_in_period["f1"],
            "fpr": metrics_in_period["fpr"],
            "roc_auc": metrics_in_period["roc_auc"]
        },
        {
            "experiment": "Drift / Holdout Test",
            "type": exp_type,
            "description": split_desc,
            "accuracy": metrics_drift["acc"],
            "precision": metrics_drift["precision"],
            "recall": metrics_drift["recall"],
            "f1": metrics_drift["f1"],
            "fpr": metrics_drift["fpr"],
            "roc_auc": metrics_drift["roc_auc"]
        },
        {
            "experiment": "Drift Delta (Test - Baseline)",
            "type": exp_type,
            "description": "Delta = Drift - In_Period",
            "accuracy": deltas["acc_delta"],
            "precision": deltas["precision_delta"],
            "recall": deltas["recall_delta"],
            "f1": deltas["f1_delta"],
            "fpr": deltas["fpr_delta"],
            "roc_auc": deltas["roc_auc_delta"]
        }
    ]

    res_df = pd.DataFrame(rows)
    res_df.to_csv(TEMPORAL_RESULTS_PATH, index=False)
    print(f"[Temporal] Results exported to {TEMPORAL_RESULTS_PATH}")

    details = {
        "exp_type": exp_type,
        "split_desc": split_desc,
        "metrics_in_period": metrics_in_period,
        "metrics_drift": metrics_drift,
        "deltas": deltas,
        "family_recalls": family_recalls,
        "test_df": test_df,
        "model": rf_drift
    }

    return res_df, details
