"""
feature_builder.py
------------------
Builds the feature matrix used for opportunity scoring / selection.

For each (EID, TARGET_MONTH, PEAKID) candidate at decision time (cutoff = 7th
of month M, selecting for M+1), we aggregate signals from:

  1. Monthly simulations for M+1  (forward-looking, legally available at cutoff)
  2. Daily simulations for days 1-7 of M  (available at cutoff)
  3. Historical profitability features from months prior to M+1

Anti-leakage contract
---------------------
* sim_monthly_df  must contain ONLY rows where MONTH <= M+1
                  (caller responsibility — see main.py)
* sim_daily_df    must contain ONLY rows where DATETIME <= 8th-of-M 00:00:00
                  (caller responsibility)
* hist_profit_df  must contain ONLY rows where MONTH <= M  (i.e., not M+1)
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

SIM_IMPACT_COLS = [
    "ACTIVATIONLEVEL",
    "WINDIMPACT",
    "SOLARIMPACT",
    "HYDROIMPACT",
    "NONRENEWBALIMPACT",
    "EXTERNALIMPACT",
    "LOADIMPACT",
    "TRANSMISSIONOUTAGEIMPACT",
]


def _agg_sim(
    sim_df: pd.DataFrame,
    price_col: str,   # "PSM" or "PSD"
    prefix: str,
) -> pd.DataFrame:
    """
    Aggregate simulation signals to (EID, MONTH, PEAKID) level.

    For each numeric column we compute mean across scenarios and hours.
    We also compute scenario disagreement (std of per-scenario means) as a
    measure of uncertainty — high disagreement → weaker signal.
    """
    agg_cols = SIM_IMPACT_COLS + [price_col]
    available = [c for c in agg_cols if c in sim_df.columns]

    # Mean across hours AND scenarios → one row per (EID, MONTH, PEAKID)
    mean_features = (
        sim_df
        .groupby(["EID", "MONTH", "PEAKID"])[available]
        .mean()
        .reset_index()
        .rename(columns={c: f"{prefix}_{c}_mean" for c in available})
    )

    # Scenario-level means first, then std across scenarios
    if "SCENARIOID" in sim_df.columns:
        per_scenario = (
            sim_df
            .groupby(["EID", "MONTH", "PEAKID", "SCENARIOID"])[[price_col]]
            .mean()
            .reset_index()
        )
        scenario_std = (
            per_scenario
            .groupby(["EID", "MONTH", "PEAKID"])[[price_col]]
            .std(ddof=0)
            .reset_index()
            .rename(columns={price_col: f"{prefix}_{price_col}_scenario_std"})
        )
        mean_features = mean_features.merge(
            scenario_std, on=["EID", "MONTH", "PEAKID"], how="left"
        )

    return mean_features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_sim_monthly_features(
    sim_monthly_df: pd.DataFrame,
    target_month: str,
) -> pd.DataFrame:
    """
    Extract features from MONTHLY simulations for a specific TARGET_MONTH.

    Returns one row per (EID, PEAKID) with aggregated simulation statistics.
    """
    sub = sim_monthly_df[sim_monthly_df["MONTH"] == target_month].copy()
    if sub.empty:
        return pd.DataFrame(columns=["EID", "PEAKID"])

    feats = _agg_sim(sub, price_col="PSM", prefix="sm")
    feats = feats.drop(columns=["MONTH"])
    return feats


def build_sim_daily_features(
    sim_daily_df: pd.DataFrame,
    decision_month: str,    # month M (not target M+1)
) -> pd.DataFrame:
    """
    Extract features from DAILY simulations available at cutoff (days 1-7 of M).
    These are short-term refinements and capture recent grid state.

    Returns one row per (EID, PEAKID) aggregated over the available days.
    """
    sub = sim_daily_df[sim_daily_df["MONTH"] == decision_month].copy()
    if sub.empty:
        return pd.DataFrame(columns=["EID", "PEAKID"])

    feats = _agg_sim(sub, price_col="PSD", prefix="sd")
    feats = feats.drop(columns=["MONTH"])
    return feats


def build_historical_features(
    hist_profit_df: pd.DataFrame,
    decision_month: str,    # month M — history is everything strictly before M+1
    lookback_months: int = 6,
) -> pd.DataFrame:
    """
    Build historical performance features for each (EID, PEAKID).

    Features:
      - hist_profit_rate_<N>m : fraction of months profitable over last N months
      - hist_mean_profit_<N>m : average Profit(o) over last N months
      - hist_mean_pr_<N>m     : average PR over last N months
      - hist_mean_c_<N>m      : average C over last N months

    decision_month (M) is the cutoff month. We use data up to and including M.
    """
    all_months = sorted(hist_profit_df["MONTH"].unique())

    # Select at most `lookback_months` months ending at decision_month
    cutoff_idx = next(
        (i for i, m in enumerate(all_months) if m == decision_month), len(all_months) - 1
    )
    window = all_months[max(0, cutoff_idx - lookback_months + 1): cutoff_idx + 1]

    sub = hist_profit_df[hist_profit_df["MONTH"].isin(window)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["EID", "PEAKID"])

    agg = (
        sub
        .groupby(["EID", "PEAKID"])
        .agg(
            hist_profit_rate=("IS_PROFITABLE", "mean"),
            hist_mean_profit=("PROFIT", "mean"),
            hist_mean_pr=("PR", "mean"),
            hist_mean_c=("C", "mean"),
            hist_months_active=("MONTH", "count"),
        )
        .reset_index()
    )
    agg.columns = [
        "EID", "PEAKID",
        f"hist_profit_rate_{lookback_months}m",
        f"hist_mean_profit_{lookback_months}m",
        f"hist_mean_pr_{lookback_months}m",
        f"hist_mean_c_{lookback_months}m",
        f"hist_months_active_{lookback_months}m",
    ]
    return agg


def build_feature_matrix(
    sim_monthly_df: pd.DataFrame,
    sim_daily_df: pd.DataFrame,
    hist_profit_df: pd.DataFrame,
    target_month: str,   # M+1
    decision_month: str, # M
    lookback_months: int = 6,
) -> pd.DataFrame:
    """
    Combine all feature sources into a single feature matrix for scoring.

    Returns: DataFrame indexed by (EID, PEAKID) with all features.
             NaN values are filled with 0 (consistent with implicit-zero rule).
    """
    sm_feats = build_sim_monthly_features(sim_monthly_df, target_month)
    sd_feats = build_sim_daily_features(sim_daily_df, decision_month)
    hist_feats = build_historical_features(hist_profit_df, decision_month, lookback_months)

    # Start from the union of (EID, PEAKID) seen in monthly sims for target month
    base = sm_feats[["EID", "PEAKID"]].drop_duplicates()

    matrix = base.copy()
    for feats in [sm_feats, sd_feats, hist_feats]:
        if not feats.empty and "EID" in feats.columns:
            matrix = matrix.merge(feats, on=["EID", "PEAKID"], how="left")

    # Implicit zero for missing feature values
    numeric_cols = matrix.select_dtypes(include="number").columns
    matrix[numeric_cols] = matrix[numeric_cols].fillna(0.0)

    matrix["TARGET_MONTH"] = target_month

    return matrix
