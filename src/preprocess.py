"""
Module 2: Preprocessing and Data Cleaning for DGA Detection
Implements RFC compliance checks, IDN/punycode handling, SLD extraction via tldextract,
deduplication, metadata tracking, and balancing.
"""

import os
import re
import idna
import tldextract
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from src.data_loader import load_raw_datasets, RAW_DIR
from src.features import extract_features, FEATURE_SET_D

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
CLEAN_PARQUET_PATH = os.path.join(PROCESSED_DIR, "clean.parquet")
FEATURES_PARQUET_PATH = os.path.join(PROCESSED_DIR, "features.parquet")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
DATA_PROFILE_PATH = os.path.join(REPORTS_DIR, "data_profile.md")

# Strict hostname regex: alphanumeric and hyphens in labels separated by dots
HOSTNAME_LABEL_REGEX = re.compile(r'^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$')


def validate_and_extract_sld(raw_domain: str, extractor: tldextract.TLDExtract) -> Tuple[bool, str, str, str, str]:
    """
    Validates a raw domain string and extracts (subdomain, sld, suffix).
    Returns (is_valid, reason_or_cleaned_domain, subdomain, sld, suffix).
    """
    if not isinstance(raw_domain, str):
        return False, "non_string", "", "", ""
    
    # 1. Lowercase and strip whitespace
    d = raw_domain.strip().lower()
    
    # 2. Strip trailing dot
    d = d.rstrip('.')
    
    if not d:
        return False, "empty", "", "", ""
    
    # 3. Reject total length over 253 chars
    if len(d) > 253:
        return False, "length_over_253", "", "", ""

    # 4. Handle IDN / Punycode
    if "xn--" in d or any(ord(c) > 127 for c in d):
        try:
            # Decode punycode / IDN to check validity
            d_decoded = idna.decode(d)
            # Re-encode to ascii punycode
            d = idna.encode(d_decoded).decode('ascii').lower()
        except Exception:
            return False, "invalid_idn_punycode", "", "", ""

    # 5. Check labels
    labels = d.split('.')
    if len(labels) < 2:
        return False, "single_label", "", "", ""
        
    for label in labels:
        if len(label) == 0:
            return False, "empty_label", "", "", ""
        if len(label) > 63:
            return False, "label_over_63", "", "", ""
        # Check allowed characters in label
        if not re.match(r'^[a-z0-9-_]+$', label):
            return False, "illegal_characters", "", "", ""

    # 6. Extract using tldextract (offline)
    extracted = extractor(d)
    sld = extracted.domain
    suffix = extracted.suffix
    subdomain = extracted.subdomain

    # SLD must be non-empty and valid
    if not sld or len(sld) > 63:
        return False, "invalid_sld", "", "", ""
    
    if not suffix:
        return False, "missing_suffix", "", "", ""

    return True, d, subdomain, sld, suffix


def clean_dataframe(df: pd.DataFrame, is_dga: bool, extractor: tldextract.TLDExtract) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean and validate raw DataFrame with optimized throughput."""
    initial_count = len(df)
    stats = {
        "initial_count": initial_count,
        "valid_count": 0,
        "invalid_count": 0,
        "invalid_reasons": {}
    }

    raw_domains = df["domain"].astype(str).values
    families = df["family"].values if "family" in df.columns else [("unknown" if is_dga else "benign")] * initial_count
    first_seens = df["first_seen"].values if "first_seen" in df.columns else [None] * initial_count
    ranks = df["rank"].values if "rank" in df.columns else [None] * initial_count

    records = []
    for raw_dom, fam, f_seen, rnk in zip(raw_domains, families, first_seens, ranks):
        valid, val, sub, sld, suf = validate_and_extract_sld(raw_dom, extractor)
        if not valid:
            stats["invalid_count"] += 1
            stats["invalid_reasons"][val] = stats["invalid_reasons"].get(val, 0) + 1
            continue

        stats["valid_count"] += 1
        records.append({
            "raw_domain": val,
            "sld": sld,
            "subdomain": sub,
            "suffix": suf,
            "is_dga": 1 if is_dga else 0,
            "family": fam if is_dga else "benign",
            "first_seen": f_seen if is_dga else None,
            "rank": rnk if not is_dga else None
        })

    cleaned_df = pd.DataFrame(records)
    return cleaned_df, stats


def preprocess_data(
    sample_size_per_class: int = 100_000,
    random_state: int = 42,
    force_recompute: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes end-to-end preprocessing pipeline:
    Loads raw data, cleans, extracts SLD, deduplicates, balances, extracts features,
    and saves clean.parquet & features.parquet.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if not force_recompute and os.path.exists(CLEAN_PARQUET_PATH) and os.path.exists(FEATURES_PARQUET_PATH):
        print(f"[Preprocess] Loading existing processed parquets from {PROCESSED_DIR}...")
        df_clean = pd.read_parquet(CLEAN_PARQUET_PATH)
        df_features = pd.read_parquet(FEATURES_PARQUET_PATH)
        return df_clean, df_features, {}

    print("[Preprocess] Starting raw data preprocessing...")
    df_bambenek_raw, df_tranco_raw, prov_info = load_raw_datasets()

    extractor = tldextract.TLDExtract(suffix_list_urls=None)

    print(f"[Preprocess] Cleaning Bambenek dataset ({len(df_bambenek_raw):,} rows)...")
    clean_dga, dga_stats = clean_dataframe(df_bambenek_raw, is_dga=True, extractor=extractor)

    print(f"[Preprocess] Cleaning Tranco dataset ({len(df_tranco_raw):,} rows)...")
    clean_benign, benign_stats = clean_dataframe(df_tranco_raw, is_dga=False, extractor=extractor)

    # Deduplication
    print("[Preprocess] Deduplicating records...")
    dga_initial = len(clean_dga)
    clean_dga = clean_dga.drop_duplicates(subset=["sld"]).copy()
    dga_dups_removed = dga_initial - len(clean_dga)

    benign_initial = len(clean_benign)
    clean_benign = clean_benign.drop_duplicates(subset=["sld"]).copy()
    benign_dups_removed = benign_initial - len(clean_benign)

    # Check cross-class collision (e.g. if a DGA domain happens to collide with benign SLD)
    common_slds = set(clean_dga["sld"]).intersection(set(clean_benign["sld"]))
    if common_slds:
        print(f"[Preprocess] Found {len(common_slds)} SLD overlap between DGA and benign. Removing overlaps to avoid label ambiguity.")
        clean_dga = clean_dga[~clean_dga["sld"].isin(common_slds)].copy()
        clean_benign = clean_benign[~clean_benign["sld"].isin(common_slds)].copy()

    # Sampling for balanced experiment
    target_dga_count = min(sample_size_per_class, len(clean_dga))
    target_benign_count = min(sample_size_per_class, len(clean_benign))
    actual_sample_count = min(target_dga_count, target_benign_count)

    print(f"[Preprocess] Sampling {actual_sample_count:,} domains per class (seed={random_state})...")
    sampled_dga = clean_dga.sample(n=actual_sample_count, random_state=random_state)
    sampled_benign = clean_benign.sample(n=actual_sample_count, random_state=random_state)

    df_clean = pd.concat([sampled_benign, sampled_dga], ignore_index=True)
    # Shuffle
    df_clean = df_clean.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print(f"[Preprocess] Saving clean dataset to {CLEAN_PARQUET_PATH} ({len(df_clean):,} records)...")
    df_clean.to_parquet(CLEAN_PARQUET_PATH, index=False)

    # Feature extraction
    print(f"[Preprocess] Extracting feature vectors for {len(df_clean):,} domains...")
    feature_rows = []
    slds = df_clean["sld"].tolist()
    for s in slds:
        feature_rows.append(extract_features(s, include_distribution=True))

    df_feats = pd.DataFrame(feature_rows)
    # Merge metadata & labels
    df_feats["is_dga"] = df_clean["is_dga"].values
    df_feats["family"] = df_clean["family"].values
    df_feats["sld"] = df_clean["sld"].values
    df_feats["raw_domain"] = df_clean["raw_domain"].values
    df_feats["first_seen"] = df_clean["first_seen"].values

    print(f"[Preprocess] Saving features dataset to {FEATURES_PARQUET_PATH}...")
    df_feats.to_parquet(FEATURES_PARQUET_PATH, index=False)

    # Profile statistics
    profile_info = {
        "prov_info": prov_info,
        "dga_stats": dga_stats,
        "benign_stats": benign_stats,
        "dga_dups_removed": dga_dups_removed,
        "benign_dups_removed": benign_dups_removed,
        "overlap_removed": len(common_slds),
        "total_clean_dga_available": len(clean_dga),
        "total_clean_benign_available": len(clean_benign),
        "balanced_sample_size": actual_sample_count,
        "df_clean": df_clean,
        "df_feats": df_feats
    }

    # Generate reports/data_profile.md
    generate_data_profile_report(profile_info)

    return df_clean, df_feats, profile_info


def generate_data_profile_report(info: Dict[str, Any]):
    """Generates comprehensive reports/data_profile.md document."""
    df_clean = info["df_clean"]
    df_feats = info["df_feats"]
    dga_feats = df_feats[df_feats["is_dga"] == 1]
    benign_feats = df_feats[df_feats["is_dga"] == 0]

    family_counts = df_clean[df_clean["is_dga"] == 1]["family"].value_counts().head(20)

    report_content = f"""# Data Profile & Provenance Report
**Project:** Lightweight DNS-Based DGA Detection for Early Botnet Identification  
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  

## 1. Data Sources & Provenance

### Malicious DGA Dataset (Bambenek Feed)
- **Source:** John Bambenek, Bambenek Consulting OSINT Feed (Research Archive Mirror)
- **Reference URL:** `{info['prov_info'].get('bambenek', {}).get('url', 'N/A')}`
- **Download Timestamp:** `{info['prov_info'].get('bambenek', {}).get('download_date', 'N/A')}`
- **Raw File Size:** `{info['prov_info'].get('bambenek', {}).get('file_size_bytes', 0):,} bytes`
- **Initial Raw Records:** `{info['dga_stats']['initial_count']:,}`
- **Valid Records Extracted:** `{info['dga_stats']['valid_count']:,}`
- **Duplicates Removed:** `{info['dga_dups_removed']:,}`
- **Clean Available DGA Domains:** `{info['total_clean_dga_available']:,}`

### Benign Dataset (Tranco Top-1M)
- **Source:** Tranco List (Legitimate Top Ranked Global Domains)
- **Reference URL:** `{info['prov_info'].get('tranco', {}).get('url', 'https://tranco-list.eu/top-1m.csv.zip')}`
- **Download Timestamp:** `{info['prov_info'].get('tranco', {}).get('download_date', 'N/A')}`
- **Raw File Size:** `{info['prov_info'].get('tranco', {}).get('file_size_bytes', 0):,} bytes`
- **Initial Raw Records:** `{info['benign_stats']['initial_count']:,}`
- **Valid Records Extracted:** `{info['benign_stats']['valid_count']:,}`
- **Duplicates Removed:** `{info['benign_dups_removed']:,}`
- **Clean Available Benign Domains:** `{info['total_clean_benign_available']:,}`

---

## 2. Experimental Dataset Composition

- **Balanced Experiment Size:** `{info['balanced_sample_size'] * 2:,}` domains total
  - **Benign (Class 0):** `{info['balanced_sample_size']:,}` (50.0%)
  - **DGA (Class 1):** `{info['balanced_sample_size']:,}` (50.0%)
- **Random Sampling Seed:** `42`
- **Cross-Class Overlap Excluded:** `{info['overlap_removed']:,}` ambiguous domains

---

## 3. Top Malicious Families in Balanced Sample

| Rank | Malware Family | Domain Count | Percentage |
|:---:|:---|:---:|:---:|
"""
    for rank, (fam, count) in enumerate(family_counts.items(), 1):
        pct = (count / len(dga_feats)) * 100
        report_content += f"| {rank} | `{fam}` | {count:,} | {pct:.2f}% |\n"

    report_content += f"""
---

## 4. Second-Level Domain (SLD) Lexical Distribution

| Metric | Benign Domains | DGA Domains | Overall |
|:---|:---:|:---:|:---:|
| **SLD Length (Mean ± Std)** | {benign_feats['length'].mean():.2f} ± {benign_feats['length'].std():.2f} | {dga_feats['length'].mean():.2f} ± {dga_feats['length'].std():.2f} | {df_feats['length'].mean():.2f} ± {df_feats['length'].std():.2f} |
| **SLD Length (Median [IQR])** | {benign_feats['length'].median():.1f} [{benign_feats['length'].quantile(0.25):.1f} - {benign_feats['length'].quantile(0.75):.1f}] | {dga_feats['length'].median():.1f} [{dga_feats['length'].quantile(0.25):.1f} - {dga_feats['length'].quantile(0.75):.1f}] | {df_feats['length'].median():.1f} |
| **Entropy (Mean ± Std)** | {benign_feats['entropy'].mean():.3f} ± {benign_feats['entropy'].std():.3f} | {dga_feats['entropy'].mean():.3f} ± {dga_feats['entropy'].std():.3f} | {df_feats['entropy'].mean():.3f} ± {df_feats['entropy'].std():.3f} |
| **Digit Ratio (Mean ± Std)** | {benign_feats['digit_ratio'].mean():.4f} ± {benign_feats['digit_ratio'].std():.4f} | {dga_feats['digit_ratio'].mean():.4f} ± {dga_feats['digit_ratio'].std():.4f} | {df_feats['digit_ratio'].mean():.4f} ± {df_feats['digit_ratio'].std():.4f} |
| **Max Consonant Run (Mean ± Std)** | {benign_feats['max_consonant_run'].mean():.2f} ± {benign_feats['max_consonant_run'].std():.2f} | {dga_feats['max_consonant_run'].mean():.2f} ± {dga_feats['max_consonant_run'].std():.2f} | {df_feats['max_consonant_run'].mean():.2f} ± {df_feats['max_consonant_run'].std():.2f} |

---

## 5. Temporal Range & Limitation Analysis

- **Bambenek Date Availability:** The public snapshot contains malware family attribution labels for `{len(family_counts):,}` active botnet families. Timestamps (`first_seen`) are largely unpopulated in this snapshot.
- **Methodological Handling:** Per Section 17 & 18 of the specification, because granular continuous time series are unavailable in static feeds, the primary drift proxy is executed via a **Family-Disjoint Holdout Evaluation** (training on disjoint botnet families and evaluating on unseen families), which provides the strongest empirical measure of DGA structural drift.
"""

    with open(DATA_PROFILE_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[Preprocess] Written data profile to {DATA_PROFILE_PATH}")


if __name__ == "__main__":
    df_c, df_f, stats = preprocess_data(sample_size_per_class=100_000, force_recompute=True)
    print("Clean Data Shape:", df_c.shape)
    print("Features Data Shape:", df_f.shape)
