"""
Unit tests for Feature Extraction module (tests/test_features.py)
Covers edge cases: empty string, 1-char, digits only, repeated chars, normal, long, specials.
"""

import math
import pytest
from src.features import (
    shannon_entropy,
    max_consonant_run,
    extract_features,
    extract_scalar_features,
    FEATURE_SET_A,
    FEATURE_SET_B,
    FEATURE_SET_C,
    FEATURE_SET_D
)


def test_shannon_entropy_edge_cases():
    # Empty string
    assert shannon_entropy("") == 0.0
    # Single character
    assert shannon_entropy("a") == 0.0
    # All identical characters
    assert shannon_entropy("aaaaaaa") == 0.0
    # Two equiprobable characters (1 bit)
    assert math.isclose(shannon_entropy("ab"), 1.0, rel_tol=1e-5)
    # Digits only
    assert shannon_entropy("12345678") > 2.0


def test_max_consonant_run():
    # Empty string
    assert max_consonant_run("") == 0
    # Vowels only
    assert max_consonant_run("aeiou") == 0
    # Single consonant
    assert max_consonant_run("b") == 1
    # Multiple consonant runs
    assert max_consonant_run("strng") == 5
    assert max_consonant_run("abcdfgehjk") == 5
    # Digits and symbols don't count as consonants
    assert max_consonant_run("b-c-d") == 1


def test_feature_extraction_normal_domain():
    feats = extract_features("google", include_distribution=True)
    assert feats["length"] == 6
    assert feats["entropy"] > 0.0
    assert feats["digit_ratio"] == 0.0
    assert feats["vowel_ratio"] == 3 / 6
    assert feats["consonant_ratio"] == 3 / 6
    assert feats["repeated_char_freq"] == 2 / 6  # 'o' and 'g' repeated: 6 - 4 distinct = 2
    assert "freq_g" in feats
    assert feats["freq_g"] == 2 / 6


def test_feature_extraction_edge_cases():
    # 1. Empty string
    f_empty = extract_features("", include_distribution=True)
    assert f_empty["length"] == 0
    assert f_empty["entropy"] == 0.0
    assert not any(math.isnan(v) or math.isinf(v) for v in f_empty.values())

    # 2. One character domain
    f_one = extract_features("a", include_distribution=True)
    assert f_one["length"] == 1
    assert f_one["entropy"] == 0.0
    assert f_one["vowel_ratio"] == 1.0

    # 3. Digits only
    f_digits = extract_features("1234567890", include_distribution=True)
    assert f_digits["digit_ratio"] == 1.0
    assert f_digits["vowel_ratio"] == 0.0
    assert f_digits["consonant_ratio"] == 0.0

    # 4. Repeated characters
    f_rep = extract_features("zzzzzzzz", include_distribution=True)
    assert f_rep["repeated_char_freq"] == (8 - 1) / 8
    assert f_rep["unique_char_ratio"] == 1 / 8

    # 5. Long domain (60 chars)
    long_sld = "a" * 60
    f_long = extract_features(long_sld, include_distribution=True)
    assert f_long["length"] == 60

    # 6. Special characters (hyphens)
    f_spec = extract_features("my-cool-site-123", include_distribution=True)
    assert f_spec["special_char_ratio"] > 0.0


def test_feature_sets_consistency():
    f = extract_features("samplebotnetdga123")
    for col in FEATURE_SET_A:
        assert col in f
    for col in FEATURE_SET_B:
        assert col in f
    for col in FEATURE_SET_C:
        assert col in f
    for col in FEATURE_SET_D:
        assert col in f


if __name__ == "__main__":
    test_shannon_entropy_edge_cases()
    test_max_consonant_run()
    test_feature_extraction_normal_domain()
    test_feature_extraction_edge_cases()
    test_feature_sets_consistency()
    print("All feature tests passed successfully!")
