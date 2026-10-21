"""
Unit tests for Detectors module (tests/test_detectors.py)
Validates all five detectors on synthetic/toy data for correct predictions,
probability outputs, and edge cases.
"""

import numpy as np
import pandas as pd
from src.detectors import (
    EntropyDetector,
    RuleBasedDetector,
    LogisticRegressionDetector,
    DecisionTreeDetector,
    RandomForestDetector
)


def create_toy_data():
    np.random.seed(42)
    # 50 benign samples: low entropy, short length, no digits
    benign_rows = []
    for _ in range(50):
        benign_rows.append({
            "length": np.random.randint(4, 9),
            "entropy": np.random.uniform(1.5, 2.8),
            "digit_ratio": 0.0,
            "unique_char_ratio": np.random.uniform(0.4, 0.7),
            "vowel_ratio": 0.4,
            "consonant_ratio": 0.6,
            "vowel_consonant_ratio": 0.66,
            "repeated_char_freq": 0.2,
            "max_consonant_run": np.random.randint(1, 3),
            "special_char_ratio": 0.0,
            "is_dga": 0
        })

    # 50 DGA samples: high entropy, long length, high consonant run
    dga_rows = []
    for _ in range(50):
        dga_rows.append({
            "length": np.random.randint(15, 25),
            "entropy": np.random.uniform(3.4, 4.5),
            "digit_ratio": np.random.uniform(0.1, 0.4),
            "unique_char_ratio": np.random.uniform(0.7, 0.95),
            "vowel_ratio": 0.2,
            "consonant_ratio": 0.8,
            "vowel_consonant_ratio": 0.25,
            "repeated_char_freq": 0.05,
            "max_consonant_run": np.random.randint(5, 9),
            "special_char_ratio": 0.0,
            "is_dga": 1
        })

    df = pd.DataFrame(benign_rows + dga_rows)
    cols = [c for c in df.columns if c != "is_dga"]
    X = df[cols]
    y = df["is_dga"].to_numpy()
    return X, y


def test_entropy_detector():
    X, y = create_toy_data()
    det = EntropyDetector()
    det.fit(X, y)
    assert det.fitted_ is True
    assert 2.0 <= det.threshold <= 4.0

    preds = det.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})

    probs = det.predict_proba(X)
    assert probs.shape == (len(y), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_rule_based_detector():
    X, y = create_toy_data()
    det = RuleBasedDetector()
    det.fit(X, y)
    assert det.fitted_ is True
    assert len(det.rule_string_) > 0

    preds = det.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})

    probs = det.predict_proba(X)
    assert probs.shape == (len(y), 2)


def test_logistic_regression_detector():
    X, y = create_toy_data()
    det = LogisticRegressionDetector()
    det.fit(X, y)

    preds = det.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})

    probs = det.predict_proba(X)
    assert probs.shape == (len(y), 2)


def test_decision_tree_detector():
    X, y = create_toy_data()
    det = DecisionTreeDetector(max_depth=5)
    det.fit(X, y)

    preds = det.predict(X)
    assert len(preds) == len(y)
    tree_text = det.export_tree_text()
    assert len(tree_text) > 0


def test_random_forest_detector():
    X, y = create_toy_data()
    det = RandomForestDetector(n_estimators=10, max_depth=5)
    det.fit(X, y)

    preds = det.predict(X)
    assert len(preds) == len(y)

    importances = det.get_feature_importances()
    assert len(importances) == X.shape[1]
    assert np.isclose(importances.sum(), 1.0)


if __name__ == "__main__":
    test_entropy_detector()
    test_rule_based_detector()
    test_logistic_regression_detector()
    test_decision_tree_detector()
    test_random_forest_detector()
    print("All detector tests passed successfully!")
