"""
Module 5: Evaluation, Benchmarking & Statistical Analysis
Computes classification metrics, latency benchmarks, memory footprint,
computational cost, and Mann-Whitney U statistical significance tests.
"""

import os
import time
import tracemalloc
import psutil
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Callable, Optional
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from src.features import (
    extract_features,
    FEATURE_SET_A,
    FEATURE_SET_B,
    FEATURE_SET_C,
    FEATURE_SET_D
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
METRICS_CSV_PATH = os.path.join(RESULTS_DIR, "metrics.csv")
STAT_TESTS_CSV_PATH = os.path.join(RESULTS_DIR, "statistical_tests.csv")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Computes Accuracy, Precision, Recall, F1, FPR, ROC-AUC, and Confusion Matrix.
    Positive class = 1 (DGA).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    fpr = fp / max(fp + tn, 1)

    auc = 0.5
    if y_prob is not None:
        try:
            if y_prob.ndim == 2:
                prob1 = y_prob[:, 1]
            else:
                prob1 = y_prob
            auc = roc_auc_score(y_true, prob1)
        except Exception:
            auc = 0.5

    return {
        "acc": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "fpr": float(fpr),
        "roc_auc": float(auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def benchmark_end_to_end_latency(
    pipeline_fn: Callable[[str], int],
    domains: List[str],
    batch_fn: Optional[Callable[[List[str]], Any]] = None,
    runs: int = 5,
    sample_size: int = 10_000
) -> Dict[str, float]:
    """
    Measures end-to-end detection path: domain string -> feature extraction -> prediction.
    Tests individual per-domain latency (microseconds) and batched throughput.
    """
    test_domains = domains[:sample_size] if len(domains) >= sample_size else domains
    
    # 1. Warm-up
    for d in test_domains[:100]:
        _ = pipeline_fn(d)

    latencies_us = []
    run_averages = []

    for _ in range(runs):
        t0 = time.perf_counter()
        for d in test_domains:
            _ = pipeline_fn(d)
        total_time = time.perf_counter() - t0
        avg_us = (total_time / len(test_domains)) * 1e6
        run_averages.append(avg_us)

    # Individual timing run for p95 calculation
    t_samples = []
    for d in test_domains[:min(2000, len(test_domains))]:
        t0 = time.perf_counter()
        _ = pipeline_fn(d)
        t_samples.append((time.perf_counter() - t0) * 1e6)

    mean_latency = float(np.mean(run_averages))
    p95_latency = float(np.percentile(t_samples, 95))
    throughput = float(1e6 / mean_latency) if mean_latency > 0 else 0.0

    batched_throughput = throughput
    if batch_fn is not None:
        try:
            _ = batch_fn(test_domains[:100])
            b_times = []
            for _ in range(runs):
                bt0 = time.perf_counter()
                _ = batch_fn(test_domains)
                b_times.append(time.perf_counter() - bt0)
            avg_b_time = float(np.mean(b_times))
            batched_throughput = float(len(test_domains) / max(avg_b_time, 1e-6))
        except Exception:
            batched_throughput = throughput

    return {
        "mean_latency_us": mean_latency,
        "p95_latency_us": p95_latency,
        "throughput_domains_per_sec": throughput,
        "batched_throughput_domains_per_sec": batched_throughput,
    }


def measure_peak_memory_and_size(
    model_obj: Any,
    predict_fn: Callable[[Any], Any],
    sample_X: Any,
    model_file_path: str = None
) -> Dict[str, float]:
    """
    Measures model disk file size (KB) and peak inference memory (MB) using tracemalloc.
    """
    # File size
    file_size_kb = 0.0
    if model_file_path and os.path.exists(model_file_path):
        file_size_kb = os.path.getsize(model_file_path) / 1024.0

    # Tracemalloc peak memory
    tracemalloc.start()
    _ = predict_fn(sample_X)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024.0 * 1024.0)

    # Estimate number of nodes / parameters
    num_params = 0
    if hasattr(model_obj, "tree_"):
        num_params = model_obj.tree_.node_count
    elif hasattr(model_obj, "estimators_"):
        num_params = sum(est.tree_.node_count for est in model_obj.estimators_)
    elif hasattr(model_obj, "coef_"):
        num_params = model_obj.coef_.size + (model_obj.intercept_.size if hasattr(model_obj, "intercept_") else 0)

    return {
        "model_size_kb": float(file_size_kb),
        "peak_inference_memory_mb": float(peak_mb),
        "parameters_or_nodes": int(num_params),
    }


def perform_statistical_analysis(df_feats: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Executes Mann-Whitney U test between DGA and benign domains for every required feature.
    Calculates U-statistic, p-value, and Rank Biserial Correlation effect size r.
    r = 1 - (2U / (n1 * n2))  [-1.0 to 1.0]
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dga_group = df_feats[df_feats["is_dga"] == 1]
    benign_group = df_feats[df_feats["is_dga"] == 0]

    n1 = len(dga_group)
    n2 = len(benign_group)

    results = []
    for col in feature_cols:
        if col not in df_feats.columns:
            continue
        vals1 = dga_group[col].to_numpy(dtype=float)
        vals2 = benign_group[col].to_numpy(dtype=float)

        # Mann-Whitney U test
        res = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
        u_stat = float(res.statistic)
        p_val = float(res.pvalue)

        # Rank-biserial correlation effect size r
        effect_size_r = 1.0 - (2.0 * u_stat / (n1 * n2))

        # KS test (Kolmogorov-Smirnov)
        ks_res = stats.ks_2samp(vals1, vals2)

        results.append({
            "feature": col,
            "benign_mean": float(np.mean(vals2)),
            "benign_std": float(np.std(vals2)),
            "dga_mean": float(np.mean(vals1)),
            "dga_std": float(np.std(vals1)),
            "mann_whitney_u": u_stat,
            "p_value": p_val,
            "effect_size_r": float(effect_size_r),
            "ks_statistic": float(ks_res.statistic),
            "ks_pvalue": float(ks_res.pvalue),
            "statistically_significant": bool(p_val < 0.001)
        })

    stat_df = pd.DataFrame(results)
    stat_df.to_csv(STAT_TESTS_CSV_PATH, index=False)
    print(f"[Evaluate] Saved statistical tests to {STAT_TESTS_CSV_PATH}")
    return stat_df
