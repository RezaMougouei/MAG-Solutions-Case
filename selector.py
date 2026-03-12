"""
selector.py
-----------
Scores each (EID, PEAKID) candidate and selects between 10 and 100
opportunities per month (total ON + OFF combined).

Strategy: Rule-based scoring with weighted signals.
---------------------------------------------------------------------------
We do NOT use a supervised ML model in this first version because:
  * Training requires realized M+1 labels which are NOT available at cutoff.
  * A rule-based approach grounded in domain logic is fully interpretable
    and avoids any risk of accidental leakage.

Scoring logic (all signals are available at cutoff):
  1. Simulated price signal   (PSM mean across scenarios)       → high = good
  2. Activation level         (ACTIVATIONLEVEL mean)            → high = good
  3. Historical profit rate   (fraction of past months profit.) → high = good
  4. Historical mean profit   (magnitude of past profits)       → high = good
  5. Scenario agreement bonus (low scenario std → more certain) → low std = good

Each signal is min-max normalized over the candidate pool, then combined
with configurable weights into a composite score in [0, 1].
Top-K candidates are selected subject to the 10-100 constraint.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Default weights (tune via backtesting)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "sm_PSM_mean":              0.35,   # forward simulated price → main signal
    "sm_ACTIVATIONLEVEL_mean":  0.25,   # opportunity intensity
    "hist_profit_rate":         0.20,   # historical precision
    "hist_mean_profit":         0.15,   # historical magnitude
    "scenario_agreement":       0.05,   # certainty bonus (inverted std)
}

# How many opportunities to select per month (total ON + OFF)
DEFAULT_TARGET_K = 50   # midpoint; tune via cross-validation
MIN_K = 10
MAX_K = 100


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _minmax_normalize(series: pd.Series) -> pd.Series:
    """Scale a series to [0, 1]; returns 0.5 if all values identical."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _build_score(feature_matrix: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Compute composite score for each row in feature_matrix.
    Missing columns are treated as contributing 0 (neutral).
    """
    score = pd.Series(0.0, index=feature_matrix.index)
    total_weight = 0.0

    # --- simulated price signal ---------------------------------------------
    col = "sm_PSM_mean"
    if col in feature_matrix.columns:
        score += weights.get(col, 0) * _minmax_normalize(feature_matrix[col])
        total_weight += weights.get(col, 0)

    # --- activation level signal ---------------------------------------------
    col = "sm_ACTIVATIONLEVEL_mean"
    if col in feature_matrix.columns:
        score += weights.get(col, 0) * _minmax_normalize(feature_matrix[col])
        total_weight += weights.get(col, 0)

    # --- historical profit rate ----------------------------------------------
    # Column name varies with lookback window; pick whichever is present
    hist_rate_cols = [c for c in feature_matrix.columns if c.startswith("hist_profit_rate")]
    if hist_rate_cols:
        avg_hist_rate = feature_matrix[hist_rate_cols].mean(axis=1)
        score += weights.get("hist_profit_rate", 0) * _minmax_normalize(avg_hist_rate)
        total_weight += weights.get("hist_profit_rate", 0)

    # --- historical mean profit ----------------------------------------------
    hist_profit_cols = [c for c in feature_matrix.columns if c.startswith("hist_mean_profit")]
    if hist_profit_cols:
        avg_hist_profit = feature_matrix[hist_profit_cols].mean(axis=1)
        # Clip negatives to 0 — we only reward historically positive elements
        avg_hist_profit = avg_hist_profit.clip(lower=0)
        score += weights.get("hist_mean_profit", 0) * _minmax_normalize(avg_hist_profit)
        total_weight += weights.get("hist_mean_profit", 0)

    # --- scenario agreement (inverted std) -----------------------------------
    std_col = "sm_PSM_scenario_std"
    if std_col in feature_matrix.columns:
        inverted_std = -feature_matrix[std_col]   # lower std → higher agreement
        score += weights.get("scenario_agreement", 0) * _minmax_normalize(inverted_std)
        total_weight += weights.get("scenario_agreement", 0)

    # Normalise by total weight actually used (graceful degradation)
    if total_weight > 0:
        score = score / total_weight

    return score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_opportunities(
    feature_matrix: pd.DataFrame,
    target_month: str,
    target_k: int = DEFAULT_TARGET_K,
    weights: Optional[dict] = None,
    score_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Select between MIN_K and MAX_K opportunities for target_month.

    Parameters
    ----------
    feature_matrix : output of feature_builder.build_feature_matrix()
    target_month   : YYYY-MM string (M+1)
    target_k       : desired number of selections (clamped to [10, 100])
    weights        : override DEFAULT_WEIGHTS
    score_threshold: minimum composite score to be considered (default 0 = no filter)

    Returns
    -------
    DataFrame with columns [TARGET_MONTH, PEAK_TYPE, EID]
    matching the required output format.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    k = max(MIN_K, min(MAX_K, target_k))

    fm = feature_matrix.copy()

    # Compute composite score
    fm["_score"] = _build_score(fm, weights)

    # Apply optional threshold
    fm = fm[fm["_score"] >= score_threshold]

    # Sort by score descending, take top-k
    fm = fm.sort_values("_score", ascending=False).head(k)

    # Ensure minimum selection even if few candidates survive threshold
    if len(fm) < MIN_K:
        # Relax threshold and take from original pool
        fallback = feature_matrix.copy()
        fallback["_score"] = _build_score(fallback, weights)
        fallback = fallback.sort_values("_score", ascending=False).head(MIN_K)
        fm = fallback

    # Format output
    output = fm[["EID", "PEAKID"]].copy()
    output["TARGET_MONTH"] = target_month
    output["PEAK_TYPE"] = output["PEAKID"].map({0: "OFF", 1: "ON"})
    output = output[["TARGET_MONTH", "PEAK_TYPE", "EID"]].drop_duplicates()

    return output.reset_index(drop=True)
