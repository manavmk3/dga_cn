"""
Module 3: Feature Engineering for DGA Detection
Computes explainable lexical and statistical features on the Second-Level Domain (SLD).
"""

import math
import re
from collections import Counter
from typing import Dict, Any, List, Optional
import numpy as np
import tldextract

_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=None)


def extract_sld(domain_or_sld: str) -> str:
    """
    Extracts the Second-Level Domain (SLD) string from any raw domain name or bare SLD string.
    Per the PRD specification (Rule: Compute features only on the SLD), TLD suffixes
    and subdomains are stripped so the model evaluates only the registered core label.
    """
    s = (domain_or_sld or "").strip().lower().rstrip(".")
    if not s:
        return ""
    if "." in s:
        extracted = _EXTRACTOR(s)
        if extracted.domain:
            return extracted.domain
        parts = s.split(".")
        return parts[-2] if len(parts) >= 2 else parts[0]
    return s


VOWELS = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")
ALPHABET = [chr(c) for c in range(ord('a'), ord('z') + 1)]
CONSONANT_RUN_REGEX = re.compile(r'[bcdfghjklmnpqrstvwxyz]+')


def shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of string s (in bits)."""
    n = len(s)
    if n == 0:
        return 0.0
    counts = Counter(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def max_consonant_run(s: str) -> int:
    """Find the length of the longest consecutive run of consonants."""
    if not s:
        return 0
    runs = CONSONANT_RUN_REGEX.findall(s.lower())
    return max((len(r) for r in runs), default=0)


def extract_scalar_features(sld: str) -> Dict[str, float]:
    """
    Extract the 10 required scalar features from an SLD string.
    Handles empty strings safely without producing NaN or inf.
    """
    s = (sld or "").lower()
    n = len(s)
    if n == 0:
        return {
            "length": 0,
            "entropy": 0.0,
            "digit_ratio": 0.0,
            "unique_char_ratio": 0.0,
            "vowel_ratio": 0.0,
            "consonant_ratio": 0.0,
            "vowel_consonant_ratio": 0.0,
            "repeated_char_freq": 0.0,
            "max_consonant_run": 0,
            "special_char_ratio": 0.0,
        }

    distinct_chars = len(set(s))
    vowels_count = sum(ch in VOWELS for ch in s)
    digits_count = sum(ch.isdigit() for ch in s)
    special_count = sum(not ch.isalnum() for ch in s)
    consonants_count = sum(ch in CONSONANTS for ch in s)

    return {
        "length": n,
        "entropy": float(shannon_entropy(s)),
        "digit_ratio": float(digits_count / n),
        "unique_char_ratio": float(distinct_chars / n),
        "vowel_ratio": float(vowels_count / n),
        "consonant_ratio": float(consonants_count / n),
        "vowel_consonant_ratio": float(vowels_count / max(consonants_count, 1)),
        "repeated_char_freq": float((n - distinct_chars) / n),
        "max_consonant_run": int(max_consonant_run(s)),
        "special_char_ratio": float(special_count / n),
    }


def extract_char_distribution(sld: str) -> Dict[str, float]:
    """
    Extract 26 letter frequencies and 1 digit frequency from an SLD string.
    """
    s = (sld or "").lower()
    n = len(s)
    if n == 0:
        res = {f"freq_{ch}": 0.0 for ch in ALPHABET}
        res["freq_digit"] = 0.0
        return res

    counts = Counter(s)
    res = {f"freq_{ch}": float(counts.get(ch, 0) / n) for ch in ALPHABET}
    digits_count = sum(ch.isdigit() for ch in s)
    res["freq_digit"] = float(digits_count / n)
    return res


def extract_features(domain_or_sld: str, include_distribution: bool = True) -> Dict[str, Any]:
    """
    Primary feature extraction interface.
    Accepts any domain string (e.g. 'google.com', 'x8912kdmn901.biz') or bare SLD ('google'),
    extracts the Second-Level Domain (SLD), and returns the feature dictionary.
    """
    s = extract_sld(domain_or_sld)
    feats = extract_scalar_features(s)
    if include_distribution:
        feats.update(extract_char_distribution(s))
    return feats


# Feature Ablation Sets
FEATURE_SET_A = ["entropy"]
FEATURE_SET_B = ["entropy", "length", "digit_ratio"]
FEATURE_SET_C = [
    "length",
    "entropy",
    "digit_ratio",
    "unique_char_ratio",
    "vowel_ratio",
    "consonant_ratio",
    "vowel_consonant_ratio",
    "repeated_char_freq",
    "max_consonant_run",
    "special_char_ratio",
]
CHAR_DIST_COLS = [f"freq_{ch}" for ch in ALPHABET] + ["freq_digit"]
FEATURE_SET_D = FEATURE_SET_C + CHAR_DIST_COLS


def extract_feature_vector(domain_or_sld: str, include_distribution: bool = True) -> np.ndarray:
    """
    Fast extraction of feature values directly into a 1D numpy float64 array.
    Avoids DataFrame construction overhead for high-throughput per-domain inference.
    """
    feats = extract_features(domain_or_sld, include_distribution=include_distribution)
    cols = FEATURE_SET_D if include_distribution else FEATURE_SET_C
    return np.array([feats[c] for c in cols], dtype=float)
