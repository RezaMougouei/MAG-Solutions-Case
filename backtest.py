"""
backtest.py
-----------
Offline backtesting script to evaluate the selector's historical performance
on the 2020-2023 development data WITHOUT any leakage.

Computes per-month:
  - Precision, Recall, F1 (same formula as the competition)
  - Total net profit
  - Number of true / false positives

Usage:
    python backtest.py --start-month 2020-06 --end-month 2023-12

We start at 2020-06 to allow a 6-month lookback history from 2020-01.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_costs, load_prices, load_sim_monthly, load_sim_daily
from profitability import compute_profitability, profitability_summary
from feature_builder import build_feature_matrix
from selector import select_opportunities
from main import month_range, prev_month, years_needed


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_f1(selected_df: pd.DataFrame, truth_df: pd.DataFrame) -> dict:
    """
    Compute Precision, Recall, F1 for a single (month, peakid) slice,
    or averaged across all months.

    Parameters
    ----------
    selected_df : DataFrame [TARGET_MONTH, PEAK_TYPE, EID]
    truth_df    : DataFrame [MONTH, PEAKID, EID, IS_PROFITABLE]
    """
    # Convert truth to same format
    truth_df = truth_df.copy()
    truth_df["PEAK_TYPE"] = truth_df["PEAKID"].map({0: "OFF", 1: "ON"})
    truth_df = truth_df.rename(columns={"MONTH": "TARGET_MONTH"})

    profitable_set = set(
        zip(
            truth_df.loc[truth_df["IS_PROFITABLE"], "TARGET_MONTH"],
            truth_df.loc[truth_df["IS_PROFITABLE"], "PEAK_TYPE"],
            truth_df.loc[truth_df["IS_PROFITABLE"], "EID"],
        )
    )

    selected_set = set(
        zip(selected_df["TARGET_MONTH"], selected_df["PEAK_TYPE"], selected_df["EID"])
    )

    tp = len(selected_set & profitable_set)
    fp = len(selected_set - profitable_set)
    fn = len(profitable_set - selected_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"TP": tp, "FP": fp, "FN": fn,
            "Precision": precision, "Recall": recall, "F1": f1}


def compute_net_profit(
    selected_df: pd.DataFrame,
    profit_df: pd.DataFrame,
) -> float:
    """Total net profit of all selected opportunities."""
    profit_df = profit_df.copy()
    profit_df["PEAK_TYPE"] = profit_df["PEAKID"].map({0: "OFF", 1: "ON"})
    profit_df = profit_df.rename(columns={"MONTH": "TARGET_MONTH"})

    merged = selected_df.merge(
        profit_df[["TARGET_MONTH", "PEAK_TYPE", "EID", "PROFIT"]],
        on=["TARGET_MONTH", "PEAK_TYPE", "EID"],
        how="left",
    )
    merged["PROFIT"] = merged["PROFIT"].fillna(0.0)
    return merged["PROFIT"].sum()


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(start_month: str, end_month: str, target_k: int = 50):
    target_months = month_range(start_month, end_month)
    decision_months = [prev_month(tm) for tm in target_months]
    all_years = years_needed(target_months + decision_months)
    lookback_years = sorted(set(all_years + [min(all_years) - 1]))

    print(f"[INFO] Loading data for years {lookback_years}...")
    costs_df  = load_costs()
    prices_df = load_prices()
    sim_monthly_df = load_sim_monthly(years=lookback_years)
    sim_daily_df   = load_sim_daily(years=lookback_years)

    # Full historical profitability — used both as training signal AND as labels
    hist_profit_df = compute_profitability(prices_df, costs_df)

    # Print base rate
    print("\n[INFO] Historical profitability base rates:")
    summary = profitability_summary(hist_profit_df)
    print(summary.groupby("PEAKID")["profit_rate"].describe().to_string())
    print()

    all_selections = []
    results = []

    for target_month, decision_month in zip(target_months, decision_months):
        # Anti-leakage filtering (same as main.py)
        sm_allowed = sim_monthly_df[sim_monthly_df["MONTH"] <= target_month]
        cutoff_dt  = pd.Timestamp(f"{decision_month}-08 00:00:00")
        sd_allowed = sim_daily_df[sim_daily_df["DATETIME"] <= cutoff_dt]
        hist_allowed = hist_profit_df[hist_profit_df["MONTH"] <= decision_month]

        feature_matrix = build_feature_matrix(
            sm_allowed, sd_allowed, hist_allowed,
            target_month=target_month,
            decision_month=decision_month,
            lookback_months=6,
        )
        if feature_matrix.empty:
            continue

        selections = select_opportunities(feature_matrix, target_month, target_k)
        all_selections.append(selections)

        # Evaluate against ground truth for this target month
        truth = hist_profit_df[hist_profit_df["MONTH"] == target_month]
        if truth.empty:
            continue

        metrics = compute_f1(selections, truth)
        profit  = compute_net_profit(selections, truth)

        results.append({
            "MONTH": target_month,
            "N_selected": len(selections),
            **metrics,
            "Net_Profit": profit,
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("[WARN] No results to report.")
        return results_df

    print("=" * 65)
    print("BACKTEST RESULTS")
    print("=" * 65)
    print(results_df[["MONTH", "N_selected", "Precision", "Recall",
                       "F1", "Net_Profit"]].to_string(index=False))
    print("-" * 65)
    print(f"Mean F1:          {results_df['F1'].mean():.4f}")
    print(f"Mean Precision:   {results_df['Precision'].mean():.4f}")
    print(f"Mean Recall:      {results_df['Recall'].mean():.4f}")
    print(f"Total Net Profit: {results_df['Net_Profit'].sum():,.2f}")
    print("=" * 65)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the opportunity selector")
    parser.add_argument("--start-month", default="2020-06")
    parser.add_argument("--end-month",   default="2023-12")
    parser.add_argument("--target-k",    type=int, default=50)
    args = parser.parse_args()

    run_backtest(args.start_month, args.end_month, args.target_k)
