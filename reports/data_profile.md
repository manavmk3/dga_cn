# Data Profile & Provenance Report
**Project:** Lightweight DNS-Based DGA Detection for Early Botnet Identification  
**Generated:** 2026-10-07 00:37:07  

## 1. Data Sources & Provenance

### Malicious DGA Dataset (Bambenek Feed)
- **Source:** John Bambenek, Bambenek Consulting OSINT Feed (Research Archive Mirror)
- **Reference URL:** `https://raw.githubusercontent.com/hmaccelerate/DGA_Detection/master/data/bambenek_dga_domain.csv`
- **Download Timestamp:** `2026-10-07 00:36:40`
- **Raw File Size:** `28,440,466 bytes`
- **Initial Raw Records:** `846,616`
- **Valid Records Extracted:** `845,871`
- **Duplicates Removed:** `48,605`
- **Clean Available DGA Domains:** `797,221`

### Benign Dataset (Tranco Top-1M)
- **Source:** Tranco List (Legitimate Top Ranked Global Domains)
- **Reference URL:** `https://tranco-list.eu/top-1m.csv.zip`
- **Download Timestamp:** `2026-10-07 00:36:40`
- **Raw File Size:** `22,606,787 bytes`
- **Initial Raw Records:** `1,000,000`
- **Valid Records Extracted:** `999,985`
- **Duplicates Removed:** `125,227`
- **Clean Available Benign Domains:** `874,713`

---

## 2. Experimental Dataset Composition

- **Balanced Experiment Size:** `200,000` domains total
  - **Benign (Class 0):** `100,000` (50.0%)
  - **DGA (Class 1):** `100,000` (50.0%)
- **Random Sampling Seed:** `42`
- **Cross-Class Overlap Excluded:** `45` ambiguous domains

---

## 3. Top Malicious Families in Balanced Sample

| Rank | Malware Family | Domain Count | Percentage |
|:---:|:---|:---:|:---:|
| 1 | `banjori` | 55,181 | 55.18% |
| 2 | `tinba` | 8,253 | 8.25% |
| 3 | `Post` | 8,197 | 8.20% |
| 4 | `ramnit` | 5,988 | 5.99% |
| 5 | `qakbot` | 5,019 | 5.02% |
| 6 | `necurs` | 4,015 | 4.01% |
| 7 | `murofet` | 1,784 | 1.78% |
| 8 | `pykspa` | 1,754 | 1.75% |
| 9 | `simda` | 1,733 | 1.73% |
| 10 | `ranbyus` | 1,724 | 1.72% |
| 11 | `shiotob/urlzone/bebloh` | 1,569 | 1.57% |
| 12 | `dyre` | 980 | 0.98% |
| 13 | `Cryptolocker` | 774 | 0.77% |
| 14 | `nymaim` | 742 | 0.74% |
| 15 | `locky` | 516 | 0.52% |
| 16 | `shifu` | 293 | 0.29% |
| 17 | `P2P` | 264 | 0.26% |
| 18 | `ramdo` | 229 | 0.23% |
| 19 | `pushdo` | 202 | 0.20% |
| 20 | `suppobox` | 124 | 0.12% |

---

## 4. Second-Level Domain (SLD) Lexical Distribution

| Metric | Benign Domains | DGA Domains | Overall |
|:---|:---:|:---:|:---:|
| **SLD Length (Mean ± Std)** | 9.98 ± 4.19 | 17.21 ± 5.73 | 13.60 ± 6.18 |
| **SLD Length (Median [IQR])** | 9.0 [7.0 - 12.0] | 17.0 [12.0 - 21.0] | 12.0 |
| **Entropy (Mean ± Std)** | 2.790 ± 0.499 | 3.439 ± 0.403 | 3.114 ± 0.558 |
| **Digit Ratio (Mean ± Std)** | 0.0409 ± 0.1323 | 0.0367 ± 0.1134 | 0.0388 ± 0.1232 |
| **Max Consonant Run (Mean ± Std)** | 2.49 ± 1.09 | 4.90 ± 2.46 | 3.70 ± 2.25 |

---

## 5. Temporal Range & Limitation Analysis

- **Bambenek Date Availability:** The public snapshot contains malware family attribution labels for `20` active botnet families. Timestamps (`first_seen`) are largely unpopulated in this snapshot.
- **Methodological Handling:** Per Section 17 & 18 of the specification, because granular continuous time series are unavailable in static feeds, the primary drift proxy is executed via a **Family-Disjoint Holdout Evaluation** (training on disjoint botnet families and evaluating on unseen families), which provides the strongest empirical measure of DGA structural drift.
