"""
Script to create the 4 required Jupyter notebooks using nbformat.
"""

import os
import nbformat as nbf

os.makedirs("notebooks", exist_ok=True)

# 1. 01_eda.ipynb
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell(
        "# Notebook 01: Exploratory Data Analysis (EDA)\n"
        "**Project:** Lightweight DNS-Based DGA Detection for Early Botnet Identification\n\n"
        "Profiles the cleaned domain datasets, evaluates class balances, and visualizes lexical distributions."
    ),
    nbf.v4.new_code_cell(
        "import os, sys\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "# Add project root to sys.path\n"
        "sys.path.insert(0, os.path.abspath('..'))\n"
        "from src.preprocess import CLEAN_PARQUET_PATH\n\n"
        "df = pd.read_parquet(os.path.join('..', CLEAN_PARQUET_PATH))\n"
        "print(f'Total domains: {len(df):,}')\n"
        "df.head(5)"
    ),
    nbf.v4.new_code_cell(
        "# Class Distribution\n"
        "class_dist = df['is_dga'].value_counts()\n"
        "print('Class counts:\\n', class_dist)\n"
        "plt.figure(figsize=(6, 3.5))\n"
        "plt.bar(['Benign (0)', 'DGA (1)'], class_dist.values, color=['#2b5c8f', '#d95f02'])\n"
        "plt.title('Dataset Class Distribution')\n"
        "plt.ylabel('Count')\n"
        "plt.show()"
    ),
    nbf.v4.new_code_cell(
        "# Top DGA Malware Families\n"
        "dga_families = df[df['is_dga'] == 1]['family'].value_counts().head(15)\n"
        "print('Top DGA Families:\\n', dga_families)\n"
        "plt.figure(figsize=(8, 4))\n"
        "dga_families.plot(kind='barh', color='#2ca02c')\n"
        "plt.gca().invert_yaxis()\n"
        "plt.title('Top 15 Botnet DGA Families')\n"
        "plt.xlabel('Domain Count')\n"
        "plt.show()"
    )
]
with open("notebooks/01_eda.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb1, f)

# 2. 02_features.ipynb
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell(
        "# Notebook 02: Feature Engineering & Statistical Significance\n\n"
        "Extracts explainable lexical features on Second-Level Domains (SLDs) and computes Mann-Whitney U hypothesis tests."
    ),
    nbf.v4.new_code_cell(
        "import os, sys\n"
        "import pandas as pd\n\n"
        "sys.path.insert(0, os.path.abspath('..'))\n"
        "from src.features import extract_features, FEATURE_SET_C\n"
        "from src.evaluate import STAT_TESTS_CSV_PATH\n"
        "from src.preprocess import FEATURES_PARQUET_PATH\n\n"
        "# Test feature extraction on known example\n"
        "demo = extract_features('google', include_distribution=False)\n"
        "for k, v in demo.items():\n"
        "    print(f'  {k:<24}: {v}')"
    ),
    nbf.v4.new_code_cell(
        "# Load features dataset\n"
        "df_feats = pd.read_parquet(os.path.join('..', FEATURES_PARQUET_PATH))\n"
        "print(f'Features DataFrame Shape: {df_feats.shape}')\n"
        "df_feats[FEATURE_SET_C].head(5)"
    ),
    nbf.v4.new_code_cell(
        "# Statistical hypothesis tests\n"
        "stats_df = pd.read_csv(os.path.join('..', STAT_TESTS_CSV_PATH))\n"
        "stats_df[['feature', 'benign_mean', 'dga_mean', 'mann_whitney_u', 'p_value', 'effect_size_r', 'statistically_significant']]"
    )
]
with open("notebooks/02_features.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb2, f)

# 3. 03_models.ipynb
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    nbf.v4.new_markdown_cell(
        "# Notebook 03: Machine Learning Detectors Comparison\n\n"
        "Trains and evaluates the five detectors: Entropy-only, Rule-based, Logistic Regression, Decision Tree, and Random Forest."
    ),
    nbf.v4.new_code_cell(
        "import os, sys\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "sys.path.insert(0, os.path.abspath('..'))\n"
        "from src.evaluate import METRICS_CSV_PATH\n\n"
        "metrics_df = pd.read_csv(os.path.join('..', METRICS_CSV_PATH))\n"
        "metrics_df"
    ),
    nbf.v4.new_code_cell(
        "# Compare F1 Score and False Positive Rate\n"
        "plt.figure(figsize=(9, 4.5))\n"
        "plt.bar(metrics_df['Method'], metrics_df['F1'], color='#1f77b4', alpha=0.8, label='F1')\n"
        "plt.title('Detector Performance Comparison (F1 Score)')\n"
        "plt.xticks(rotation=20, ha='right')\n"
        "plt.ylabel('F1 Score')\n"
        "plt.grid(True, axis='y')\n"
        "plt.show()"
    )
]
with open("notebooks/03_models.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb3, f)

# 4. 04_temporal.ipynb
nb4 = nbf.v4.new_notebook()
nb4.cells = [
    nbf.v4.new_markdown_cell(
        "# Notebook 04: Temporal Drift & Family Holdout Robustness\n\n"
        "Evaluates model degradation and generalization across held-out, completely unseen botnet families."
    ),
    nbf.v4.new_code_cell(
        "import os, sys\n"
        "import pandas as pd\n\n"
        "sys.path.insert(0, os.path.abspath('..'))\n"
        "from src.temporal import TEMPORAL_RESULTS_PATH\n\n"
        "temp_df = pd.read_csv(os.path.join('..', TEMPORAL_RESULTS_PATH))\n"
        "temp_df"
    ),
    nbf.v4.new_code_cell(
        "# Display Drift Deltas\n"
        "deltas = temp_df[temp_df['experiment'] == 'Drift Delta (Test - Baseline)']\n"
        "print('Observed Drift Delta:\\n', deltas[['accuracy', 'precision', 'recall', 'f1', 'fpr']])"
    )
]
with open("notebooks/04_temporal.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb4, f)

print("All 4 Jupyter notebooks successfully generated!")
