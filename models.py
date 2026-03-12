"""
models.py
---------
Binary classification models for identifying profitable FTR opportunities.

Problem framing
---------------
Given features for (EID, MONTH, PEAKID) at decision time (7th of month M),
predict whether Profit(o) = PR_o - C_o > 0 for month M+1.

Class imbalance (~5-17% positive rate) is handled via:
  - class_weight / scale_pos_weight parameters in each model
  - Threshold tuning (predict_proba > threshold, not default 0.5)
  - Optional SMOTE oversampling (toggle via use_smote flag)

Models
------
1. LogisticRegression  — fast baseline, interpretable coefficients
2. LightGBM            — primary model, handles large sparse data efficiently
3. XGBoost             — secondary model, strong alternative

Anti-leakage contract
---------------------
This module is stateless — it receives pre-filtered feature matrices.
The caller (train.py / main.py) is responsible for ensuring no future
data leaks into the feature matrix passed here.
"""

import numpy as np
import pandas as pd
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score,
    classification_report
)

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Feature columns used by all models
# (must match feature_builder.py output — excludes ID and target columns)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Monthly simulation features
    "sm_ACTIVATIONLEVEL_mean",
    "sm_PSM_mean",
    "sm_PSM_scenario_std",
    "sm_WINDIMPACT_mean",
    "sm_SOLARIMPACT_mean",
    "sm_HYDROIMPACT_mean",
    "sm_NONRENEWBALIMPACT_mean",
    "sm_EXTERNALIMPACT_mean",
    "sm_LOADIMPACT_mean",
    "sm_TRANSMISSIONOUTAGEIMPACT_mean",
    # Daily simulation features
    "sd_ACTIVATIONLEVEL_mean",
    "sd_PSD_mean",
    "sd_PSD_scenario_std",
    "sd_WINDIMPACT_mean",
    "sd_SOLARIMPACT_mean",
    "sd_HYDROIMPACT_mean",
    # Historical features
    "hist_profit_rate_6m",
    "hist_mean_profit_6m",
    "hist_mean_pr_6m",
    "hist_mean_c_6m",
    "hist_months_active_6m",
    # Peak type as binary feature
    "PEAKID",
]


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Return only the feature columns that actually exist in df."""
    return [c for c in FEATURE_COLS if c in df.columns]


# ---------------------------------------------------------------------------
# 1. Logistic Regression Baseline
# ---------------------------------------------------------------------------

def build_logistic_regression(class_weight: str = "balanced") -> Pipeline:
    """
    Logistic Regression baseline with StandardScaler.

    'balanced' class_weight automatically adjusts weights inversely
    proportional to class frequencies — handles imbalance without SMOTE.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        ))
    ])


# ---------------------------------------------------------------------------
# 2. LightGBM
# ---------------------------------------------------------------------------

def build_lightgbm(pos_weight: float = 5.0) -> "lgb.LGBMClassifier":
    """
    LightGBM binary classifier.

    pos_weight: weight applied to positive class.
                Set to (n_negative / n_positive) for balanced training.
                Higher = more recall, fewer missed profitable trades.
                Lower = more precision, fewer false signals.
    """
    if not LGBM_AVAILABLE:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    return lgb.LGBMClassifier(
        objective="binary",
        metric="average_precision",   # better than AUC for imbalanced data
        scale_pos_weight=pos_weight,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# 3. XGBoost
# ---------------------------------------------------------------------------

def build_xgboost(pos_weight: float = 5.0) -> "xgb.XGBClassifier":
    """
    XGBoost binary classifier.

    scale_pos_weight equivalent to pos_weight in LightGBM.
    """
    if not XGB_AVAILABLE:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",          # area under precision-recall curve
        scale_pos_weight=pos_weight,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        use_label_encoder=False,
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_pos_weight(y: pd.Series) -> float:
    """Compute scale_pos_weight = n_negative / n_positive."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def apply_smote(X: np.ndarray, y: np.ndarray,
                random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to balance classes.
    Only used if imblearn is installed and explicitly requested.
    Falls back to no-op if SMOTE unavailable.
    """
    if not SMOTE_AVAILABLE:
        print("[WARN] imbalanced-learn not installed — skipping SMOTE.")
        return X, y

    sm = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"  SMOTE: {len(y):,} → {len(y_res):,} samples "
          f"(+{len(y_res)-len(y):,} synthetic positives)")
    return X_res, y_res


def tune_threshold(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    thresholds: Optional[list[float]] = None,
    metric: str = "f1",
) -> float:
    """
    Find the probability threshold that maximizes F1 (or precision/recall)
    on a validation set.

    Default 0.5 is rarely optimal for imbalanced problems.
    Lower threshold → more positives predicted → higher recall, lower precision.
    Higher threshold → fewer positives → higher precision, lower recall.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.05).tolist()

    proba = model.predict_proba(X_val)[:, 1]
    best_threshold, best_score = 0.5, 0.0

    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_val, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_val, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_val, y_pred, zero_division=0)
        else:
            score = f1_score(y_val, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"  Best threshold: {best_threshold:.2f} → {metric}={best_score:.4f}")
    return best_threshold


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    label: str = "Evaluation",
) -> dict:
    """Full evaluation report for a trained model."""
    proba  = model.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    metrics = {
        "label":      label,
        "threshold":  threshold,
        "f1":         f1_score(y, y_pred, zero_division=0),
        "precision":  precision_score(y, y_pred, zero_division=0),
        "recall":     recall_score(y, y_pred, zero_division=0),
        "roc_auc":    roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0.0,
        "pr_auc":     average_precision_score(y, proba) if len(np.unique(y)) > 1 else 0.0,
        "n_positive_pred": int(y_pred.sum()),
        "n_positive_true": int(y.sum()),
    }

    print(f"\n  [{label}]")
    print(f"  Threshold:    {threshold:.2f}")
    print(f"  F1:           {metrics['f1']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:       {metrics['pr_auc']:.4f}")
    print(f"  Predicted +ve:{metrics['n_positive_pred']:,} / {len(y):,}")
    print(f"\n{classification_report(y, y_pred, zero_division=0)}")

    return metrics
