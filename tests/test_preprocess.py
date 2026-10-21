"""
Unit tests for Preprocessing module (tests/test_preprocess.py)
Validates RFC rules, tldextract splitting, IDN handling, rejection filters, and metadata preservation.
"""

import tldextract
from src.preprocess import validate_and_extract_sld


def test_preprocess_normal_domains():
    extractor = tldextract.TLDExtract(suffix_list_urls=None)
    
    # Standard benign domain
    valid, d, sub, sld, suf = validate_and_extract_sld("www.google.com.", extractor)
    assert valid is True
    assert d == "www.google.com"
    assert sld == "google"
    assert suf == "com"
    assert sub == "www"

    # Multi-level TLD
    valid, d, sub, sld, suf = validate_and_extract_sld("bbc.co.uk", extractor)
    assert valid is True
    assert sld == "bbc"
    assert suf == "co.uk"


def test_preprocess_edge_cases_and_rejections():
    extractor = tldextract.TLDExtract(suffix_list_urls=None)

    # Empty string
    valid, reason, _, _, _ = validate_and_extract_sld("", extractor)
    assert valid is False

    # Single label (no TLD)
    valid, reason, _, _, _ = validate_and_extract_sld("localhost", extractor)
    assert valid is False

    # Label exceeding 63 characters
    long_label = "a" * 64 + ".com"
    valid, reason, _, _, _ = validate_and_extract_sld(long_label, extractor)
    assert valid is False
    assert reason == "label_over_63"

    # Total domain length exceeding 253 chars
    huge_domain = "a." * 130 + "com"
    valid, reason, _, _, _ = validate_and_extract_sld(huge_domain, extractor)
    assert valid is False

    # Illegal characters
    valid, reason, _, _, _ = validate_and_extract_sld("bad$domain*.com", extractor)
    assert valid is False
    assert reason == "illegal_characters"


def test_punycode_handling():
    extractor = tldextract.TLDExtract(suffix_list_urls=None)
    # Valid punycode for münchen.de -> xn--mnchen-3ya.de
    valid, d, sub, sld, suf = validate_and_extract_sld("xn--mnchen-3ya.de", extractor)
    assert valid is True
    assert suf == "de"


if __name__ == "__main__":
    test_preprocess_normal_domains()
    test_preprocess_edge_cases_and_rejections()
    test_punycode_handling()
    print("All preprocess tests passed successfully!")
