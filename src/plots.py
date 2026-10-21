"""
Module 7: Plotting and Visualization for DGA Detection
Generates all eight required figures plus threshold sweep curves.
Saved under results/figures/ with clear labels, legends, titles, and units.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "figures")


def set_plot_style():
    """Configure clean academic plot style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.autolayout": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })


def plot_feature_histograms(df_feats: pd.DataFrame, out_path: str = None):
    """Figure 1: Feature histograms: DGA vs benign for key discriminative features."""
    set_plot_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = out_path or os.path.join(FIGURES_DIR, "01_feature_histograms.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    key_features = [
        ("entropy", "Shannon Entropy (bits)", (1.5, 5.0)),
        ("length", "SLD Length (chars)", (3, 40)),
        ("digit_ratio", "Digit Ratio", (0.0, 0.6)),
        ("max_consonant_run", "Max Consonant Run Length", (0, 15))
    ]

    benign = df_feats[df_feats["is_dga"] == 0]
    dga = df_feats[df_feats["is_dga"] == 1]

    for idx, (col, xlabel, xlim) in enumerate(key_features):
        ax = axes[idx]
        b_vals = benign[col].dropna()
        d_vals = dga[col].dropna()

        ax.hist(b_vals, bins=35, density=True, alpha=0.6, color="#2b5c8f", label="Benign (Tranco)")
        ax.hist(d_vals, bins=35, density=True, alpha=0.6, color="#d95f02", label="DGA (Bambenek)")
        ax.set_title(f"Distribution of {col.replace('_', ' ').title()}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability Density")
        ax.set_xlim(xlim)
        ax.grid(True)
        ax.legend()

    plt.suptitle("Figure 1: Lexical & Statistical Feature Histograms (DGA vs Benign)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved Figure 1 to {out_path}")


def plot_correlation_heatmap(df_feats: pd.DataFrame, out_path: str = None):
    """Figure 2: Correlation heatmap for scalar features."""
    set_plot_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = out_path or os.path.join(FIGURES_DIR, "02_correlation_heatmap.png")

    cols = [
        "length", "entropy", "digit_ratio", "unique_char_ratio",
        "vowel_ratio", "consonant_ratio", "vowel_consonant_ratio",
        "repeated_char_freq", "max_consonant_run", "special_char_ratio"
    ]
    sub_df = df_feats[[c for c in cols if c in df_feats.columns]]
    corr = sub_df.corr().to_numpy()

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)

    ax.set_xticks(range(len(sub_df.columns)))
    ax.set_yticks(range(len(sub_df.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in sub_df.columns], rotation=45, ha="right")
    ax.set_yticklabels([c.replace("_", " ") for c in sub_df.columns])

    # Annotate numeric values
    for i in range(len(sub_df.columns)):
        for j in range(len(sub_df.columns)):
            val = corr[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation Coefficient")
    ax.set_title("Figure 2: Pearson Correlation Heatmap across Scalar Features")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved Figure 2 to {out_path}")


def plot_roc_curves(roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]], out_path: str = None):
    """Figure 3: ROC curves for all applicable detection methods."""
    set_plot_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = out_path or os.path.join(FIGURES_DIR, "03_roc_curves.png")

    plt.figure(figsize=(8, 6))
    colors = {
        "Entropy-only": "#7570b3",
        "Rule-based": "#e7298a",
        "Logistic Regression": "#1b9e77",
        "Decision Tree": "#e6ab02",
        "Random Forest": "#d95f02"
    }

    for name, (fpr, tpr, auc) in roc_data.items():
        color = colors.get(name, None)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", lw=2, color=color)

    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.500)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate / Recall (TPR)")
    plt.title("Figure 3: Receiver Operating Characteristic (ROC) Curves")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.grid(True)
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved Figure 3 to {out_path}")


def plot_confusion_matrices(cm_data: Dict[str, np.ndarray], out_path: str = None):
    """Figure 4: Confusion matrices for all detectors."""
    set_plot_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = out_path or os.path.join(FIGURES_DIR, "04_confusion_matrices.png")

    methods = list(cm_data.keys())
    n = len(methods)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, name in enumerate(methods):
        ax = axes[idx]
        cm = cm_data[name]  # [[TN, FP], [FN, TP]]
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign (0)", "DGA (1)"])
        ax.set_yticklabels(["Benign (0)", "DGA (1)"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        labels = [["TN", "FP"], ["FN", "TP"]]
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                tag = labels[i][j]
                color = "white" if val > thresh else "black"
                ax.text(j, i, f"{tag}\n{val:,}", ha="center", va="center", color=color, fontsize=10)

    # Hide any unused subplots
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Figure 4: Confusion Matrices per Detection Method", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved Figure 4 to {out_path}")


def plot_f1_fpr_barchart(metrics_df: pd.DataFrame, out_path: str = None):
    """Figure 5: Bar chart of F1 and FPR per method."""
    set_plot_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = out_path or os.path.join(FIGURES_DIR, "05_f1_fpr_comparison.png")

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    methods = metrics_df["Method"].tolist()
    x = np.arange(len(methods))
    width = 0.35

    f1_vals = metrics_df["F1"].to_numpy()
    fpr_vals = metrics_df["FPR"].to_numpy()

    bars1 = ax1.bar(x - width/2, f1_vals, width, label="F1 Score", color="#2b5c8f", alpha=0.85)
    bars2 = ax1.bar(x + width/2, fpr_vals, width, label="False Positive Rate (FPR)", color="#d95f02", alpha=0.85)

    ax1.set_ylabel("Metric Value (0.0 - 1.0)")
    ax1.set_title("Figure 5: Detection F1 Score vs False Positive Rate (FPR) across Methods")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha="right")
    ax1.set_ylim([0.0, 1.1])
    ax1.grid(True, axis="y")
    ax1.legend(loc="upper right")

    # Add data values on bars
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    for bar in bars2:
        h = bar.get_height()
        ax1.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved Figure 5 to {out_path}")


def plot_detection_time_vs_f1(metrics_df: pd.DataFrame, out_path: str = None):
    """Figure 6: Detection time vs F1 scatter plot (performance/cost trade-off)."""
    set_plot_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = out_path or os.path.join(FIGURES_DIR, "06_detection_time_vs_f1.png")

    fig, ax = plt.subplots(figsize=(8, 6))

    times = metrics_df["Time (us)"].to_numpy()
    f1s = metrics_df["F1"].to_numpy()
    methods = metrics_df["Method"].tolist()

    scatter = ax.scatter(times, f1s, s=160, c="#1b9e77", edgecolors="black", zorder=5)

    for t, f, m in zip(times, f1s, methods):
        ax.annotate(f" {m}", (t, f), textcoords="offset points", xytext=(6, 4), fontsize=9, fontweight="bold")

    ax.set_xlabel("Latency per Domain (microseconds, $\\mu$s)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Figure 6: Detection Latency vs F1 Performance Trade-off")
    ax.grid(True)

    # Highlight Pareto frontier region
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved Figure 6 to {out_path}")


def plot_rf_feature_importance(importances: pd.Series, top_k: int = 15, out_path: str = None):
    """Figure 7: Random Forest feature importances."""
    set_plot_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = out_path or os.path.join(FIGURES_DIR, "07_rf_feature_importance.png")

    top_feats = importances.head(top_k).iloc[::-1]

    plt.figure(figsize=(9, 6.5))
    y_pos = np.arange(len(top_feats))
    plt.barh(y_pos, top_feats.values, color="#2ca02c", alpha=0.85, edgecolor="black")
    plt.yticks(y_pos, [n.replace("_", " ") for n in top_feats.index])
    plt.xlabel("Gini Feature Importance")
    plt.title(f"Figure 7: Top {top_k} Most Discriminative Features in Random Forest")
    plt.grid(True, axis="x")

    for i, v in enumerate(top_feats.values):
        plt.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved Figure 7 to {out_path}")


def plot_temporal_drift_comparison(temporal_df: pd.DataFrame, out_path: str = None):
    """Figure 8: In-period vs temporal-drift / holdout comparison."""
    set_plot_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = out_path or os.path.join(FIGURES_DIR, "08_temporal_drift_comparison.png")

    sub = temporal_df[temporal_df["experiment"].isin(["In-Period Baseline", "Drift / Holdout Test"])]
    if len(sub) < 2:
        return

    metrics_list = ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc"]
    names = ["Accuracy", "Precision", "Recall", "F1", "FPR", "ROC-AUC"]

    base_vals = sub[sub["experiment"] == "In-Period Baseline"][metrics_list].iloc[0].values
    drift_vals = sub[sub["experiment"] == "Drift / Holdout Test"][metrics_list].iloc[0].values

    x = np.arange(len(metrics_list))
    width = 0.35

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width/2, base_vals, width, label="In-Period / In-Distribution Baseline", color="#1f77b4", alpha=0.85)
    plt.bar(x + width/2, drift_vals, width, label="Holdout / Unseen Botnet Evaluation", color="#d62728", alpha=0.85)

    plt.xticks(x, names)
    plt.ylabel("Metric Score (0.0 - 1.0)")
    plt.title("Figure 8: In-Period Baseline vs Robustness Holdout Generalization Comparison")
    plt.ylim([0.0, 1.1])
    plt.grid(True, axis="y")
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved Figure 8 to {out_path}")


def plot_threshold_sweeps(sweep_entropy: pd.DataFrame, sweep_rf: pd.DataFrame, out_path: str = None):
    """Figure 9: FPR-vs-Recall Threshold Sweep Curves for Entropy & Random Forest."""
    set_plot_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = out_path or os.path.join(FIGURES_DIR, "09_threshold_sweeps.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Ax1: FPR vs Recall trade-off curve
    ax1.plot(sweep_entropy["fpr"], sweep_entropy["recall"], label="Entropy Detector", color="#7570b3", lw=2)
    ax1.plot(sweep_rf["fpr"], sweep_rf["recall"], label="Random Forest Prob", color="#d95f02", lw=2)
    ax1.set_xlabel("False Positive Rate (FPR)")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.set_title("FPR vs Recall Trade-off Frontier")
    ax1.set_xlim([0.0, 0.4])
    ax1.set_ylim([0.0, 1.02])
    ax1.grid(True)
    ax1.legend()

    # Ax2: F1 vs Threshold parameter
    ax2.plot(sweep_entropy["threshold"], sweep_entropy["f1"], label="Entropy Detector (Thresh)", color="#7570b3", lw=2)
    ax2.set_xlabel("Entropy Threshold (bits)")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Entropy Threshold Sensitivity Curve")
    ax2.grid(True)
    ax2.legend()

    plt.suptitle("Figure 9: Operating Threshold Sweeps & Error Sensitivity", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved Figure 9 to {out_path}")
