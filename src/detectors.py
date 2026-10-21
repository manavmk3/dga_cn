"""
Module 4: Detectors for DGA Detection
Implements five detection methods:
1. Entropy-only detector (threshold-tuned on train)
2. Rule-based detector (explainable human-readable boolean rule)
3. Logistic Regression (with StandardScaler)
4. Decision Tree (shallow, interpretable depth 5-10)
5. Random Forest (100 estimators, max_depth=12)
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class EntropyDetector(BaseEstimator, ClassifierMixin):
    """
    Entropy-only detector.
    Predicts DGA (1) if Shannon entropy > threshold t.
    Threshold t is selected exclusively on training data to maximize F1 or achieve target FPR.
    """
    def __init__(self, threshold: float = 3.5):
        self.threshold = threshold
        self.fitted_ = False

    def fit(self, X, y, target_fpr: Optional[float] = None):
        """
        Tunes threshold on training data.
        X can be a DataFrame or 2D/1D array with entropy values.
        """
        vals = self._extract_entropy(X)
        y_true = np.asarray(y, dtype=int)
        y_pos = (y_true == 1)
        y_neg = (y_true == 0)
        total_pos = np.count_nonzero(y_pos)
        total_neg = np.count_nonzero(y_neg)

        # Candidate thresholds between min and max entropy
        candidates = np.linspace(vals.min(), vals.max(), 200)
        best_t = candidates[0]

        if target_fpr is not None:
            # Pick threshold achieving FPR <= target_fpr while maximizing recall
            best_recall = -1.0
            for t in candidates:
                preds = vals > t
                fp = np.count_nonzero(preds & y_neg)
                fpr = fp / max(total_neg, 1)
                if fpr <= target_fpr:
                    rec = np.count_nonzero(preds & y_pos) / max(total_pos, 1)
                    if rec > best_recall:
                        best_recall = rec
                        best_t = t
        else:
            # Maximize F1 score
            best_f1 = -1.0
            for t in candidates:
                preds = vals > t
                tp = np.count_nonzero(preds & y_pos)
                fp = np.count_nonzero(preds & y_neg)
                fn = total_pos - tp
                denom = 2 * tp + fp + fn
                f1 = (2.0 * tp / denom) if denom > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t

        self.threshold = float(best_t)
        self.fitted_ = True
        return self

    def _extract_entropy(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if "entropy" in X.columns:
                return X["entropy"].to_numpy(dtype=float)
            return X.iloc[:, 0].to_numpy(dtype=float)
        elif isinstance(X, np.ndarray):
            if X.ndim == 2:
                return X[:, 0].astype(float)
            return X.astype(float)
        elif isinstance(X, list):
            return np.asarray(X, dtype=float)
        raise ValueError(f"Unsupported format for entropy extraction: {type(X)}")

    def predict(self, X) -> np.ndarray:
        vals = self._extract_entropy(X)
        return (vals > self.threshold).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        """Sigmoid approximation centered at threshold for ROC calculations."""
        vals = self._extract_entropy(X)
        p1 = 1.0 / (1.0 + np.exp(-3.0 * (vals - self.threshold)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


class RuleBasedDetector(BaseEstimator, ClassifierMixin):
    """
    Explainable Rule-based detector.
    Rule: (entropy > a AND length > b) OR digit_ratio > c OR max_consonant_run > d
    Thresholds tuned using grid search exclusively on training data.
    """
    def __init__(self, a: float = 3.5, b: float = 12.0, c: float = 0.2, d: float = 4.0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.rule_string_ = ""
        self.fitted_ = False

    def fit(self, X, y):
        """Grid search over training data candidate quantiles."""
        if isinstance(X, pd.DataFrame):
            ent_arr = X["entropy"].to_numpy(dtype=float)
            len_arr = X["length"].to_numpy(dtype=float)
            dig_arr = X["digit_ratio"].to_numpy(dtype=float)
            mcr_arr = X["max_consonant_run"].to_numpy(dtype=float)
        else:
            X_mat = np.asarray(X, dtype=float)
            len_arr = X_mat[:, 0]
            ent_arr = X_mat[:, 1]
            dig_arr = X_mat[:, 2]
            mcr_arr = X_mat[:, 8]
        y_true = np.asarray(y, dtype=int)
        y_pos = (y_true == 1)
        y_neg = (y_true == 0)

        best_f1 = -1.0
        best_params = (self.a, self.b, self.c, self.d)

        # Quantile candidates based on data
        a_grid = np.percentile(ent_arr, [50, 65, 75, 85])
        b_grid = np.percentile(len_arr, [40, 60, 75])
        c_grid = [0.1, 0.2, 0.3]
        d_grid = [3.0, 4.0, 5.0]

        for a_cand in a_grid:
            cond_a = ent_arr > a_cand
            for b_cand in b_grid:
                cond1 = cond_a & (len_arr > b_cand)
                for c_cand in c_grid:
                    cond12 = cond1 | (dig_arr > c_cand)
                    for d_cand in d_grid:
                        preds = cond12 | (mcr_arr > d_cand)
                        tp = np.count_nonzero(preds & y_pos)
                        fp = np.count_nonzero(preds & y_neg)
                        fn = np.count_nonzero((~preds) & y_pos)
                        denom = (2 * tp + fp + fn)
                        f1 = (2.0 * tp / denom) if denom > 0 else 0.0

                        if f1 > best_f1:
                            best_f1 = f1
                            best_params = (a_cand, b_cand, c_cand, d_cand)

        self.a, self.b, self.c, self.d = [float(p) for p in best_params]
        self.rule_string_ = (
            f"(entropy > {self.a:.2f} AND length > {self.b:.1f}) "
            f"OR digit_ratio > {self.c:.2f} "
            f"OR max_consonant_run > {self.d:.1f}"
        )
        self.fitted_ = True
        return self

    def _ensure_df(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        cols = ["length", "entropy", "digit_ratio", "unique_char_ratio",
                "vowel_ratio", "consonant_ratio", "vowel_consonant_ratio",
                "repeated_char_freq", "max_consonant_run", "special_char_ratio"]
        return pd.DataFrame(X, columns=cols[:X.shape[1]])

    def predict(self, X) -> np.ndarray:
        if isinstance(X, np.ndarray):
            cond1 = (X[:, 1] > self.a) & (X[:, 0] > self.b)
            cond2 = X[:, 2] > self.c
            cond3 = X[:, 8] > self.d
            return (cond1 | cond2 | cond3).astype(int)
        df = self._ensure_df(X)
        cond1 = (df["entropy"] > self.a) & (df["length"] > self.b)
        cond2 = df["digit_ratio"] > self.c
        cond3 = df["max_consonant_run"] > self.d
        return (cond1 | cond2 | cond3).astype(int).to_numpy()

    def predict_proba(self, X) -> np.ndarray:
        """Heuristic score matching conditions matched."""
        if isinstance(X, np.ndarray):
            c1 = (X[:, 1] > self.a) & (X[:, 0] > self.b)
            c2 = X[:, 2] > self.c
            c3 = X[:, 8] > self.d
            score = (c1.astype(float) * 0.4 + c2.astype(float) * 0.3 + c3.astype(float) * 0.3)
            p0 = 1.0 - score
            return np.column_stack([p0, score])
        df = self._ensure_df(X)
        c1 = (df["entropy"] > self.a) & (df["length"] > self.b)
        c2 = df["digit_ratio"] > self.c
        c3 = df["max_consonant_run"] > self.d
        score = (c1.astype(float) * 0.4 + c2.astype(float) * 0.3 + c3.astype(float) * 0.3).to_numpy()
        p0 = 1.0 - score
        return np.column_stack([p0, score])


class LogisticRegressionDetector:
    """Logistic Regression with StandardScaler fit strictly on train split."""
    def __init__(self, max_iter: int = 1000, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        self.feature_names = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)
        
        X_scaled = self.scaler.fit_transform(X_arr)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_scaled = self.scaler.transform(X_arr)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_scaled = self.scaler.transform(X_arr)
        return self.model.predict_proba(X_scaled)


class DecisionTreeDetector:
    """Shallow, interpretable Decision Tree (max_depth 5-10)."""
    def __init__(self, max_depth: int = 7, random_state: int = 42):
        self.max_depth = max_depth
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.feature_names = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)
        self.model.fit(X_arr, y)
        return self

    def predict(self, X):
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model.predict(X_arr)

    def predict_proba(self, X):
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model.predict_proba(X_arr)

    def export_tree_text(self, max_depth_display: int = 4) -> str:
        names = self.feature_names if self.feature_names else None
        return export_text(self.model, feature_names=names, max_depth=max_depth_display)


class RandomForestDetector:
    """Random Forest Classifier (100 estimators, max_depth=12)."""
    def __init__(self, n_estimators: int = 100, max_depth: int = 12, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state
        )
        self.feature_names = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)
        self.model.fit(X_arr, y)
        self._tree_objects = [e.tree_ for e in self.model.estimators_]
        return self

    def predict(self, X):
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.shape[0] == 1:
            if not hasattr(self, "_tree_objects") or not self._tree_objects:
                self._tree_objects = [e.tree_ for e in self.model.estimators_]
            X_f32 = np.ascontiguousarray(X_arr, dtype=np.float32)
            votes = sum(t.predict(X_f32)[0, 1] > t.predict(X_f32)[0, 0] for t in self._tree_objects)
            return np.array([1 if votes > (len(self._tree_objects) / 2.0) else 0])
        return self.model.predict(X_arr)

    def predict_proba(self, X):
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.shape[0] == 1:
            if not hasattr(self, "_tree_objects") or not self._tree_objects:
                self._tree_objects = [e.tree_ for e in self.model.estimators_]
            X_f32 = np.ascontiguousarray(X_arr, dtype=np.float32)
            p1 = float(np.mean([t.predict(X_f32)[0, 1] / max(np.sum(t.predict(X_f32)[0]), 1e-6) for t in self._tree_objects]))
            return np.array([[1.0 - p1, p1]])
        return self.model.predict_proba(X_arr)

    def get_feature_importances(self) -> pd.Series:
        if not self.feature_names:
            names = [f"feat_{i}" for i in range(len(self.model.feature_importances_))]
        else:
            names = self.feature_names
        return pd.Series(self.model.feature_importances_, index=names).sort_values(ascending=False)


def save_detector(detector_obj, filename: str):
    """Save model to models/ folder."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, filename)
    joblib.dump(detector_obj, out_path)
    print(f"[Detectors] Saved model to {out_path}")
    return out_path


def load_detector(filename: str):
    """Load model from models/ folder."""
    in_path = os.path.join(MODELS_DIR, filename)
    return joblib.load(in_path)
