"""
profitability.py
----------------
Computes Profit(o) = PR_o - C_o for every (EID, MONTH, PEAKID) opportunity.

Updated to work with int32 MONTH encoding (YYYYMM) from the optimized
data_loader. String MONTH is also accepted for backwards compatibility.
"""

import pandas as pd
from data_loader import month_str_to_int, month_int_to_str


def _ensure_int_month(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize MONTH to int32 regardless of input format."""
    if "MONTH" in df.columns and df["MONTH"].dtype == object:
        df = df.copy()
        df["MONTH"] = df["MONTH"].str.replace("-", "").astype("int32")
    return df


def compute_monthly_pr(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly realized prices → monthly PR per (EID, MONTH, PEAKID)."""
    prices_df = _ensure_int_month(prices_df)
    pr = (
        prices_df
        .groupby(["EID", "MONTH", "PEAKID"], as_index=False)["PRICEREALIZED"]
        .sum()
        .rename(columns={"PRICEREALIZED": "PR"})
    )
    pr["PR"] = pr["PR"].astype("float32")
    return pr


def compute_profitability(
    prices_df: pd.DataFrame,
    costs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Profit(o) = PR_o - C_o.

    Returns DataFrame with columns:
        EID, MONTH (int32), PEAKID, PR, C, PROFIT, IS_PROFITABLE
    """
    prices_df = _ensure_int_month(prices_df)
    costs_df  = _ensure_int_month(costs_df)

    pr_df = compute_monthly_pr(prices_df)
    del prices_df

    merged = pd.merge(
        pr_df,
        costs_df[["EID", "MONTH", "PEAKID", "C"]],
        on=["EID", "MONTH", "PEAKID"],
        how="outer",
    )
    del pr_df, costs_df

    merged["PR"]            = merged["PR"].fillna(0.0).astype("float32")
    merged["C"]             = merged["C"].fillna(0.0).astype("float32")
    merged["PROFIT"]        = (merged["PR"] - merged["C"]).astype("float32")
    merged["IS_PROFITABLE"] = merged["PROFIT"] > 0
    merged["MONTH"]         = merged["MONTH"].astype("int32")

    return merged.reset_index(drop=True)


def profitability_summary(profit_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        profit_df
        .groupby(["MONTH", "PEAKID"])
        .agg(
            total_opportunities=("EID", "count"),
            profitable=("IS_PROFITABLE", "sum"),
        )
        .assign(profit_rate=lambda x: x["profitable"] / x["total_opportunities"])
        .reset_index()
    )
    return summary
