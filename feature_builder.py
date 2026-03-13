"""
feature_builder.py
------------------
Builds feature matrix for each (EID, TARGET_MONTH, PEAKID) candidate.

Updated to use int32 MONTH encoding (YYYYMM) throughout.
Uses inplace=True where possible to avoid intermediate copies (Fix 5).
"""

import pandas as pd
import numpy as np
from data_loader import month_str_to_int


SIM_IMPACT_COLS = [
    "ACTIVATIONLEVEL",
    "WINDIMPACT", "SOLARIMPACT", "HYDROIMPACT",
    "NONRENEWBALIMPACT", "EXTERNALIMPACT",
    "LOADIMPACT", "TRANSMISSIONOUTAGEIMPACT",
]


def _to_month_int(month) -> int:
    """Accept either 'YYYY-MM' string or int YYYYMM."""
    if isinstance(month, str):
        return month_str_to_int(month)
    return int(month)


def _agg_sim(
    sim_df: pd.DataFrame,
    price_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Aggregate simulation signals to (EID, MONTH, PEAKID) level."""
    agg_cols  = [c for c in SIM_IMPACT_COLS + [price_col] if c in sim_df.columns]

    mean_features = (
        sim_df
        .groupby(["EID", "PEAKID"])[agg_cols]
        .mean()
        .reset_index()
        .rename(columns={c: f"{prefix}_{c}_mean" for c in agg_cols})
    )

    # Scenario disagreement — std across per-scenario means
    if "SCENARIOID" in sim_df.columns and price_col in sim_df.columns:
        per_scenario = (
            sim_df
            .groupby(["EID", "PEAKID", "SCENARIOID"])[[price_col]]
            .mean()
            .reset_index()
        )
        scenario_std = (
            per_scenario
            .groupby(["EID", "PEAKID"])[[price_col]]
            .std(ddof=0)
            .reset_index()
            .rename(columns={price_col: f"{prefix}_{price_col}_scenario_std"})
        )
        del per_scenario
        mean_features = mean_features.merge(
            scenario_std, on=["EID", "PEAKID"], how="left"
        )
        del scenario_std

    return mean_features

def diagnostic_feature_test(df, feat_cols):
    # Check for zero variance
    variances = df[feat_cols].var()
    useless_const = variances[variances == 0].index.tolist()
    
    # Check for high correlation
    corr_matrix = df[feat_cols].corr().abs()
    # Logic to find pairs with > 0.95 correlation
    
    return useless_const

def build_sim_monthly_features(
    sim_monthly_df: pd.DataFrame,
    target_month_int: int,
) -> pd.DataFrame:
    sub = sim_monthly_df[sim_monthly_df["MONTH"] == target_month_int].copy()
    if sub.empty:
        return pd.DataFrame(columns=["EID", "PEAKID"])
    feats = _agg_sim(sub, price_col="PSM", prefix="sm")
    del sub
    return feats


def build_sim_daily_features(
    sim_daily_df: pd.DataFrame,
    decision_month_int: int,
) -> pd.DataFrame:
    sub = sim_daily_df[sim_daily_df["MONTH"] == decision_month_int].copy()
    if sub.empty:
        return pd.DataFrame(columns=["EID", "PEAKID"])
    feats = _agg_sim(sub, price_col="PSD", prefix="sd")
    del sub
    return feats


def build_historical_features(
    hist_profit_df: pd.DataFrame,
    decision_month_int: int,
    lookback_months: int = 6,
) -> pd.DataFrame:
    """Historical performance features using int32 MONTH."""
    all_months = sorted(hist_profit_df["MONTH"].unique())

    cutoff_idx = next(
        (i for i, m in enumerate(all_months) if m == decision_month_int),
        len(all_months) - 1
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
    del sub

    agg.columns = [
        "EID", "PEAKID",
        f"hist_profit_rate_{lookback_months}m",
        f"hist_mean_profit_{lookback_months}m",
        f"hist_mean_pr_{lookback_months}m",
        f"hist_mean_c_{lookback_months}m",
        f"hist_months_active_{lookback_months}m",
    ]

    # Fix 5: inplace dtype conversion — no intermediate copy
    for col in agg.select_dtypes(include="float64").columns:
        agg[col] = agg[col].astype("float32")

    return agg


def build_feature_matrix(
    sim_monthly_df: pd.DataFrame,
    sim_daily_df: pd.DataFrame,
    hist_profit_df: pd.DataFrame,
    target_month: str,
    decision_month: str,
    lookback_months: int = 6,
) -> pd.DataFrame:
    """
    Combine all feature sources into a single matrix.
    Accepts string months, converts to int32 internally.
    """
    target_month_int   = _to_month_int(target_month)
    decision_month_int = _to_month_int(decision_month)

    sm_feats   = build_sim_monthly_features(sim_monthly_df, target_month_int)
    sd_feats   = build_sim_daily_features(sim_daily_df, decision_month_int)
    hist_feats = build_historical_features(
        hist_profit_df, decision_month_int, lookback_months
    )

    if sm_feats.empty:
        return pd.DataFrame()

    # Normalize EID to string for consistent merging
    sm_feats["EID"] = sm_feats["EID"].astype(str)

    matrix = sm_feats.copy()
    del sm_feats

    for feats in [sd_feats, hist_feats]:
        if not feats.empty and "EID" in feats.columns:
            feats["EID"] = feats["EID"].astype(str)
            matrix = matrix.merge(feats, on=["EID", "PEAKID"], how="left")
        del feats

    # Fix 5: fillna inplace on numeric columns only
    num_cols = matrix.select_dtypes(include="number").columns
    matrix[num_cols] = matrix[num_cols].fillna(0.0)

    matrix["TARGET_MONTH"] = target_month

    return matrix
