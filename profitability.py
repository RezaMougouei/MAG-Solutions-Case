"""
profitability.py
----------------
Step 4 from the brief: compute historical profitability for every
(EID, MONTH, PEAKID) triplet using realized prices and exposure costs.

This module produces the TARGET VARIABLE used to:
  a) Validate our selections in backtesting.
  b) Train / calibrate any supervised learning approach.

Key rules enforced here
-----------------------
* Implicit zero: missing prices → 0, missing costs → 0.
* Profitability condition: PR_o - C_o > 0
* Anti-leakage: caller must only pass data for months ≤ M (not M+1).
"""

import pandas as pd


def compute_monthly_pr(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly realized prices → monthly PR per (EID, MONTH, PEAKID).

    Parameters
    ----------
    prices_df : DataFrame with columns [EID, MONTH, PEAKID, PRICEREALIZED]
                (MONTH already derived as YYYY-MM string)

    Returns
    -------
    DataFrame with columns [EID, MONTH, PEAKID, PR]
    """
    pr = (
        prices_df
        .groupby(["EID", "MONTH", "PEAKID"], as_index=False)["PRICEREALIZED"]
        .sum()
        .rename(columns={"PRICEREALIZED": "PR"})
    )
    return pr


def compute_profitability(
    prices_df: pd.DataFrame,
    costs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Profit(o) = PR_o - C_o for every observable opportunity.

    Parameters
    ----------
    prices_df : output of load_prices() — hourly granularity
    costs_df  : output of load_costs()  — monthly granularity

    Returns
    -------
    DataFrame with columns:
        EID, MONTH, PEAKID, PR, C, PROFIT, IS_PROFITABLE (bool)

    Notes
    -----
    * Only opportunities that appear in AT LEAST ONE of the two datasets
      are included (outer join). All-zero rows (never in either dataset)
      are excluded — they are trivially PR=0, C=0, Profit=0, not profitable.
    * This matches the "sparsified / implicit zero" convention.
    """
    # --- Step A: aggregate hourly prices to monthly PR -----------------------
    pr_df = compute_monthly_pr(prices_df)

    # --- Step B: merge with costs (outer join to capture implicit zeros) ------
    merged = pd.merge(
        pr_df,
        costs_df[["EID", "MONTH", "PEAKID", "C"]],
        on=["EID", "MONTH", "PEAKID"],
        how="outer",
    )
    merged["PR"] = merged["PR"].fillna(0.0)
    merged["C"] = merged["C"].fillna(0.0)

    # --- Step C: compute profit and label ------------------------------------
    merged["PROFIT"] = merged["PR"] - merged["C"]
    merged["IS_PROFITABLE"] = merged["PROFIT"] > 0

    return merged.reset_index(drop=True)


def profitability_summary(profit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Print and return a high-level summary of historical profitability rates
    broken down by MONTH and PEAKID.

    Useful for understanding the base rate (~<5% profitable per the case brief).
    """
    summary = (
        profit_df
        .groupby(["MONTH", "PEAKID"])
        .agg(
            total_opportunities=("EID", "count"),
            profitable=("IS_PROFITABLE", "sum"),
        )
        .assign(
            profit_rate=lambda x: x["profitable"] / x["total_opportunities"]
        )
        .reset_index()
    )
    return summary
