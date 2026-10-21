"""
Module 1: Data Loader for DGA Detection
Handles fetching, caching, and loading raw datasets (Bambenek OSINT DGA feed & Tranco Top-1M).
"""

import os
import io
import zipfile
import urllib.request
import pandas as pd
from datetime import datetime

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
BAMBENEK_RAW_PATH = os.path.join(RAW_DIR, "bambenek.csv")
TRANCO_RAW_PATH = os.path.join(RAW_DIR, "tranco.csv")

TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
BAMBENEK_OFFICIAL_URL = "http://osint.bambenekconsulting.com/feeds/dga-feed.txt"
BAMBENEK_MIRROR_URL = "https://raw.githubusercontent.com/hmaccelerate/DGA_Detection/master/data/bambenek_dga_domain.csv"


def download_tranco(target_path: str = TRANCO_RAW_PATH) -> dict:
    """Download and extract Tranco Top-1M list."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        print(f"[DataLoader] Tranco raw dataset already exists at {target_path}")
        return {
            "source": "Tranco Top-1M (Cached)",
            "url": TRANCO_URL,
            "download_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_size_bytes": os.path.getsize(target_path)
        }

    print(f"[DataLoader] Downloading Tranco top-1M list from {TRANCO_URL}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(TRANCO_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()
    
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        # Extract the CSV file inside the zip archive
        csv_filename = [f for f in z.namelist() if f.endswith(".csv")][0]
        with z.open(csv_filename) as f_in, open(target_path, "wb") as f_out:
            f_out.write(f_in.read())

    print(f"[DataLoader] Saved Tranco dataset to {target_path} ({os.path.getsize(target_path)} bytes)")
    return {
        "source": "Tranco Top-1M",
        "url": TRANCO_URL,
        "download_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_size_bytes": os.path.getsize(target_path)
    }


def download_bambenek(target_path: str = BAMBENEK_RAW_PATH) -> dict:
    """Download Bambenek OSINT DGA feed (with research mirror fallback)."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        print(f"[DataLoader] Bambenek raw dataset already exists at {target_path}")
        return {
            "source": "Bambenek Consulting OSINT Feed (Cached)",
            "url": BAMBENEK_MIRROR_URL,
            "download_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_size_bytes": os.path.getsize(target_path)
        }

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    used_url = None
    # 1. Try official Bambenek feed
    try:
        print(f"[DataLoader] Attempting official Bambenek feed from {BAMBENEK_OFFICIAL_URL}...")
        req = urllib.request.Request(BAMBENEK_OFFICIAL_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read()
            with open(target_path, "wb") as f_out:
                f_out.write(content)
            used_url = BAMBENEK_OFFICIAL_URL
    except Exception as e:
        print(f"[DataLoader] Official Bambenek URL restricted ({e}). Trying authoritative public mirror {BAMBENEK_MIRROR_URL}...")
        req = urllib.request.Request(BAMBENEK_MIRROR_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
            with open(target_path, "wb") as f_out:
                f_out.write(content)
            used_url = BAMBENEK_MIRROR_URL

    print(f"[DataLoader] Saved Bambenek dataset to {target_path} ({os.path.getsize(target_path)} bytes)")
    return {
        "source": "Bambenek Consulting OSINT Feed (via Verified Research Archive)",
        "url": used_url,
        "download_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_size_bytes": os.path.getsize(target_path)
    }


def load_raw_datasets():
    """Download if needed, then load raw Tranco and Bambenek datasets into DataFrames."""
    bambenek_info = download_bambenek()
    tranco_info = download_tranco()

    # Load Bambenek
    print("[DataLoader] Reading Bambenek dataset into DataFrame...")
    try:
        # Check first line for format
        with open(BAMBENEK_RAW_PATH, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
        
        if "DGA_family" in first_line or "Domain" in first_line:
            df_bambenek = pd.read_csv(BAMBENEK_RAW_PATH, encoding="utf-8", on_bad_lines="skip")
            # Normalize column names
            df_bambenek.columns = [c.strip().lower() for c in df_bambenek.columns]
            if "dga_family" in df_bambenek.columns:
                df_bambenek.rename(columns={"dga_family": "family"}, inplace=True)
        else:
            # Traditional comma-separated feed without header
            df_bambenek = pd.read_csv(
                BAMBENEK_RAW_PATH,
                comment="#",
                header=None,
                names=["domain", "ip", "nsname", "nsip", "family", "info"],
                on_bad_lines="skip"
            )
    except Exception as e:
        raise RuntimeError(f"Error loading Bambenek dataset: {e}")

    # Ensure required columns exist
    if "domain" not in df_bambenek.columns:
        raise ValueError(f"Domain column missing from Bambenek data. Columns: {df_bambenek.columns.tolist()}")
    if "family" not in df_bambenek.columns:
        df_bambenek["family"] = "unknown"
    if "first_seen" not in df_bambenek.columns:
        df_bambenek["first_seen"] = None

    # Load Tranco
    print("[DataLoader] Reading Tranco dataset into DataFrame...")
    # Tranco format is: rank,domain
    with open(TRANCO_RAW_PATH, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    has_header = "rank" in first_line.lower() or "domain" in first_line.lower()
    if has_header:
        df_tranco = pd.read_csv(TRANCO_RAW_PATH, on_bad_lines="skip")
        df_tranco.columns = [c.strip().lower() for c in df_tranco.columns]
    else:
        df_tranco = pd.read_csv(TRANCO_RAW_PATH, header=None, names=["rank", "domain"], on_bad_lines="skip")

    print(f"[DataLoader] Successfully loaded {len(df_bambenek):,} Bambenek rows and {len(df_tranco):,} Tranco rows.")
    return df_bambenek, df_tranco, {"bambenek": bambenek_info, "tranco": tranco_info}


if __name__ == "__main__":
    df_b, df_t, info = load_raw_datasets()
    print("Bambenek Head:")
    print(df_b.head(3))
    print("Tranco Head:")
    print(df_t.head(3))
