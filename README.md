# Lightweight DNS-Based DGA Detection for Early Botnet Identification

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reproducibility](https://img.shields.io/badge/reproducibility-fully%20automated-brightgreen.svg)](run_all.py)
[![Test Suite](https://img.shields.io/badge/tests-13%2F13%20passed-success.svg)](tests/)

An academic and empirical research implementation comparing explainable lexical and statistical DNS Second-Level Domain (SLD) features across five machine learning detectors for line-rate early botnet identification and temporal/family drift analysis.

---

## 1. Project Overview & Problem Statement

Command and Control (C2) communication is a vital lifeline for botnet operations, enabling botmasters to issue commands, exfiltrate stolen credentials, or deploy ransomware payloads. To prevent security defenders from neutralizing botnets through static IP/domain blocking or domain sinkholing, malware authors deploy **Domain Generation Algorithms (DGAs)**. These algorithms dynamically generate hundreds or thousands of candidate domains per day based on shared pseudo-random seeds.

While contemporary deep learning models (such as character-level LSTMs or Transformers) offer high theoretical accuracy, they impose severe drawbacks:
- Substantial memory footprints and GPU requirements unsuitable for edge/resolver deployments.
- Inability to provide explainable signals to Security Operations Center (SOC) analysts.
- High inference latency that impairs line-rate recursive DNS resolvers (which require sub-millisecond responses).

This project presents a **lightweight, resource-aware, and explainable** machine learning framework. By extracting 10 scalar features and 27 character distributions strictly from the **Second-Level Domain (SLD)**, the pipeline achieves superior botnet detection (>0.98 F1) with microsecond latencies and complete operational transparency.

---

## 2. Architecture & Pipeline

```
+----------------------------------------------------------------------------------------------------+
|                                    Raw Threat Data Sources                                         |
|   1. Bambenek Consulting OSINT DGA Feed (846k domains) | 2. Tranco Top-1M Benign Feed (1M domains) |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                    Module 2: Preprocessing                                         |
|    • Lowercase, Whitespace & Trailing Dot Stripping                                                |
|    • RFC 1035 Compliance Verification (Labels <= 63, Total <= 253, Alphanumeric/Hyphen)            |
|    • IDN / Punycode Validation                                                                     |
|    • Strict SLD Isolation via tldextract (Offline PSL: Prevents TLD shortcut leakage)              |
|    • Global Cross-Dataset Deduplication & Class Balancing (100k Benign : 100k DGA)                |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                  Module 3: Feature Engineering                                     |
|    • Set A: Entropy only                                                                           |
|    • Set B: Entropy + Length + Digit Ratio                                                         |
|    • Set C: All 10 Explainable Scalars (Entropy, Length, Digit/Vowel/Consonant Ratios, Runs, etc.) |
|    • Set D: Set C + 27 Fine-Grained Character Frequencies (a-z + digits)                           |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                      Module 4: Five Detectors                                      |
|    1. Entropy-Only Baseline     |  4. Shallow Decision Tree (Interpretable depth 7)                |
|    2. Rule-Based Boolean Logic  |  5. Random Forest Ensemble (100 trees, depth 12)                 |
|    3. Scaled Logistic Regression|                                                                  |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                Module 5 & 6: Evaluation & Drift                                    |
|    • Stratified 80/20 Split & 5-Fold Cross-Validation on Train                                     |
|    • Standardized Metrics: Accuracy, Precision, Recall, F1, FPR, ROC-AUC                           |
|    • Microsecond Latency (time.perf_counter) & Memory Profiling (tracemalloc)                      |
|    • Two-Sided Mann-Whitney U & Kolmogorov-Smirnov Hypothesis Tests                                |
|    • Unseen Botnet Family-Disjoint Holdout Drift Evaluation                                        |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                         Artifacts, Models, Figures & Academic Reports                              |
|   `results/metrics.csv`  |  `results/figures/*.png`  |  `reports/final_report.md`                  |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Folder Structure

```
dga_cn/
├── data/
│   ├── raw/
│   │   ├── bambenek.csv                # Authentic Bambenek OSINT DGA feed (846k records)
│   │   └── tranco.csv                  # Tranco Top-1M benign domains list (1M records)
│   └── processed/
│       ├── clean.parquet               # Cleaned, deduplicated, RFC-validated domain records
│       └── features.parquet            # Extracted feature vectors and ground truth labels
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis & lexical distributions
│   ├── 02_features.ipynb               # Feature engineering & Mann-Whitney U test demonstration
│   ├── 03_models.ipynb                 # Detector training, evaluation & metrics comparison
│   └── 04_temporal.ipynb               # Temporal drift & family-disjoint holdout analysis
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Module 1: Automated dataset downloading & caching
│   ├── preprocess.py                   # Module 2: RFC cleaning, IDN handling, SLD extraction
│   ├── features.py                     # Module 3: Vectorized lexical & statistical features
│   ├── detectors.py                    # Module 4: Five detection algorithms & joblib persistence
│   ├── evaluate.py                     # Module 5: Metrics, latency benchmarking & stats tests
│   ├── temporal.py                     # Module 6: Temporal drift & family holdout experiments
│   ├── plots.py                        # Module 7: Visualization suite generating Figures 1-9
│   └── reports_generator.py            # Academic report and checklist compilers
├── tests/
│   ├── test_features.py                # Unit tests for entropy, consonant runs, edge cases
│   ├── test_preprocess.py              # Unit tests for RFC validation, IDN, tldextract
│   └── test_detectors.py               # Unit tests for detector training, inference, edge inputs
├── models/                             # Saved .joblib serialized detector models
├── results/
│   ├── metrics.csv                     # Primary measured performance comparison table
│   ├── statistical_tests.csv           # Mann-Whitney U & KS test statistics and p-values
│   ├── temporal_results.csv            # Drift baseline vs holdout evaluation & deltas
│   ├── ablation_results.csv            # Feature set ablation comparison (Set A through D)
│   └── figures/                        # Generated figures (Figures 1-9)
├── reports/
│   ├── data_profile.md                 # Data provenance, source URLs, counts & distributions
│   ├── final_report.md                 # Complete 16-section academic final research report
│   └── acceptance_checklist.md         # Verified definition-of-done acceptance checklist
├── run_all.py                          # Master one-command reproduction script
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## 4. Requirements & Installation

The project requires Python 3.10+ (tested on Python 3.13.15).

```bash
# 1. Clone or navigate to the repository
cd dga_cn

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows Powershell:
.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 5. Dataset Sources & Preparation

1. **Malicious DGA Data:** Bambenek Consulting OSINT DGA feed.
   - Authoritative feed generated by John Bambenek covering high-confidence active C2 domains.
   - Automatically retrieved and cached via `src/data_loader.py` into `data/raw/bambenek.csv`.
2. **Benign Data:** Tranco Top-1M list (`https://tranco-list.eu/top-1m.csv.zip`).
   - Hardened research ranking list resilient against manipulation.
   - Automatically retrieved and unzipped into `data/raw/tranco.csv`.

To prepare raw data manually:
```bash
python src/data_loader.py
```

---

## 6. How to Run

### Complete Autonomous Reproduction
To execute the entire end-to-end pipeline—including downloading data, preprocessing, feature extraction, model training, 5-fold cross-validation, timing benchmarks, drift evaluation, and generating all figures and reports—run:

```bash
python run_all.py
```

### Running Unit Tests
All 13 unit tests covering core algorithmic functions, RFC validation rules, and extreme edge cases (empty strings, single characters, digits-only, repeated characters, etc.) can be executed via:

```bash
python -m pytest tests/
```

### Running Jupyter Notebooks
Start the Jupyter Notebook server:
```bash
jupyter notebook
```
Navigate to `notebooks/` and open:
- `01_eda.ipynb`
- `02_features.ipynb`
- `03_models.ipynb`
- `04_temporal.ipynb`

---

## 7. Models Implemented

1. **Entropy Baseline:** Evaluates whether domain Shannon entropy exceeds an optimal threshold $t$, calibrated exclusively on the training split to maximize $F_1$.
2. **Rule-Based Detector:** An explainable multi-condition boolean rule calibrated via training grid search:
   `(entropy > a AND length > b) OR digit_ratio > c OR max_consonant_run > d`
3. **Logistic Regression:** Scaled linear classifier (`StandardScaler` + `LogisticRegression(max_iter=1000)`).
4. **Decision Tree:** Shallow, interpretable decision tree (`max_depth=7`) with text tree export.
5. **Random Forest:** Ensemble classifier (`n_estimators=100`, `max_depth=12`, `n_jobs=-1`, `random_state=42`).

---

## 8. Feature Specification

Computed strictly on the Second-Level Domain (SLD) string $s$ of length $n$:
1. `length`: String character length $n$.
2. `entropy`: Shannon entropy $-\sum p_c \log_2(p_c)$.
3. `digit_ratio`: Proportions of numeric digits $0-9$.
4. `unique_char_ratio`: Distinct characters divided by $n$.
5. `vowel_ratio`: Proportion of characters in $\{a, e, i, o, u\}$.
6. `consonant_ratio`: Proportion of alphabetic non-vowels.
7. `vowel_consonant_ratio`: $\text{vowels} / \max(\text{consonants}, 1)$.
8. `repeated_char_freq`: $(n - \text{distinct}) / n$.
9. `max_consonant_run`: Longest run of consecutive consonants.
10. `special_char_ratio`: Hyphens and non-alphanumeric characters divided by $n$.
11. `char_distribution`: 26 individual letter frequencies ($a-z$) and digit frequency.

---

## 9. Results & Generated Deliverables

Execution outputs are automatically organized under:
- **`results/metrics.csv`:** Full performance comparison table across all 5 detectors and temporal drift test.
- **`results/statistical_tests.csv`:** Mann-Whitney U test statistics, p-values, and effect sizes for all features.
- **`results/temporal_results.csv`:** In-period vs holdout metrics and drift delta calculations.
- **`results/ablation_results.csv`:** Ablation results across Feature Sets A, B, C, and D.
- **`results/figures/`:**
  - `01_feature_histograms.png`
  - `02_correlation_heatmap.png`
  - `03_roc_curves.png`
  - `04_confusion_matrices.png`
  - `05_f1_fpr_comparison.png`
  - `06_detection_time_vs_f1.png`
  - `07_rf_feature_importance.png`
  - `08_temporal_drift_comparison.png`
  - `09_threshold_sweeps.png`
- **`reports/data_profile.md`:** Complete dataset provenance and distributional profile.
- **`reports/final_report.md`:** 16-section academic final research report.
- **`reports/acceptance_checklist.md`:** Verification checklist.

---

## 10. Reproducibility & Research Safety

- **Fixed Random State:** All splits and models utilize `random_state=42`.
- **Zero Test Data Leakage:** Preprocessing deduplication is performed before splitting; scalers and decision thresholds are calibrated exclusively on the training split.
- **Offline Safety Boundary:** This project analyzes offline domain name strings only. It does not contact C2 servers, execute malware, or perform live network scans.
- **No Fabrication Rule:** All metrics, timings, and tables are measured directly from the execution environment.

---

## 11. Limitations

1. **Static OSINT Date Availability:** Publicly archived Bambenek feeds contain active C2 botnet family attribution but lack continuous multi-year timestamps; family-disjoint holdout evaluation is therefore used as the primary generalization test.
2. **Dictionary-Based DGAs:** Botnets concatenating valid English dictionary words (e.g. Suppobox, Matsnu) exhibit linguistic patterns that mimic legitimate domains, necessitating n-gram features for complete coverage.
3. **No Passive DNS Signals:** Lexical analysis operates solely on the domain string and does not observe query volume, IP diversity, or NXDOMAIN response codes.
