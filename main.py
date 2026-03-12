"""
main.py
-------
Entry point for the MAG Energy Solutions Data Challenge.

Usage:
    python main.py --start-month 2020-01 --end-month 2023-12

The script processes every month in [start_month, end_month] and writes
opportunities.csv at the project root.

Anti-leakage is enforced here: for each TARGET_MONTH (M+1) we only load
data available on the 7th day of the preceding month M.
"""

import argparse
import sys
from pathlib import Path
from dateutil.relativedelta import relativedelta
import pandas as pd
from tqdm import tqdm

# Make src importable when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_costs, load_prices, load_sim_monthly, load_sim_daily
from profitability import compute_profitability
from feature_builder import build_feature_matrix
from selector import select_opportunities


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def month_range(start: str, end: str) -> list[str]:
    """Return list of YYYY-MM strings from start to end inclusive."""
    from pandas import Period
    s = Period(start, "M")
    e = Period(end, "M")
    months = []
    cur = s
    while cur <= e:
        months.append(str(cur))
        cur += 1
    return months


def prev_month(month_str: str) -> str:
    """Return the month before month_str as YYYY-MM string."""
    p = pd.Period(month_str, "M") - 1
    return str(p)


def years_needed(months: list[str]) -> list[int]:
    """Extract unique years from a list of YYYY-MM strings."""
    return sorted(set(int(m[:4]) for m in months))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(start_month: str, end_month: str, target_k: int = 50) -> pd.DataFrame:
    """
    Run the full pipeline for all months in [start_month, end_month].

    Returns the complete opportunities DataFrame.
    """
    target_months = month_range(start_month, end_month)

    # Determine the decision months (M) = one month before each target month
    decision_months = [prev_month(tm) for tm in target_months]

    # All years we need to load (targets + decisions + lookback)
    all_months_needed = target_months + decision_months
    all_years = years_needed(all_months_needed)
    # Add previous year for lookback history
    lookback_years = sorted(set(all_years + [min(all_years) - 1]))

    print(f"[INFO] Loading data for years: {lookback_years}")

    # -----------------------------------------------------------------------
    # Load full datasets once (year-filtered to keep memory reasonable)
    # -----------------------------------------------------------------------
    print("[INFO] Loading costs...")
    costs_df  = load_costs()

    print("[INFO] Loading realized prices...")
    prices_df = load_prices()

    print("[INFO] Loading monthly simulations...")
    sim_monthly_df = load_sim_monthly(years=all_years)

    print("[INFO] Loading daily simulations...")
    sim_daily_df = load_sim_daily(years=all_years)

    # -----------------------------------------------------------------------
    # Pre-compute historical profitability (the "answer key" for past months)
    # -----------------------------------------------------------------------
    print("[INFO] Computing historical profitability labels...")
    hist_profit_df = compute_profitability(prices_df, costs_df)

    # -----------------------------------------------------------------------
    # Loop over each target month
    # -----------------------------------------------------------------------
    all_selections = []

    for target_month, decision_month in tqdm(
        zip(target_months, decision_months),
        total=len(target_months),
        desc="Processing months",
    ):
        # --- Anti-leakage filtering ------------------------------------------
        # Monthly sims: allowed up to M+1 (the target month itself is available
        #               because simulations for M+1 are produced before 7th of M)
        sm_allowed = sim_monthly_df[sim_monthly_df["MONTH"] <= target_month]

        # Daily sims: only days 1-7 of M (DATETIME <= 8th of M at 00:00:00)
        cutoff_dt = pd.Timestamp(f"{decision_month}-08 00:00:00")
        sd_allowed = sim_daily_df[sim_daily_df["DATETIME"] <= cutoff_dt]

        # Historical profitability: only months up to and including M (not M+1)
        hist_allowed = hist_profit_df[hist_profit_df["MONTH"] <= decision_month]

        # --- Build feature matrix --------------------------------------------
        feature_matrix = build_feature_matrix(
            sim_monthly_df=sm_allowed,
            sim_daily_df=sd_allowed,
            hist_profit_df=hist_allowed,
            target_month=target_month,
            decision_month=decision_month,
            lookback_months=6,
        )

        if feature_matrix.empty:
            print(f"[WARN] No features available for {target_month}, skipping.")
            continue

        # --- Select opportunities --------------------------------------------
        selections = select_opportunities(
            feature_matrix=feature_matrix,
            target_month=target_month,
            target_k=target_k,
        )
        all_selections.append(selections)

    if not all_selections:
        print("[ERROR] No selections generated for any month.")
        return pd.DataFrame(columns=["TARGET_MONTH", "PEAK_TYPE", "EID"])

    result = pd.concat(all_selections, ignore_index=True)

    # Deduplicate (robustness rule)
    result = result.drop_duplicates(subset=["TARGET_MONTH", "PEAK_TYPE", "EID"])

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MAG Energy Solutions — FTR Opportunity Selector"
    )
    parser.add_argument(
        "--start-month",
        required=True,
        help="Start month in YYYY-MM format (e.g. 2020-01)",
    )
    parser.add_argument(
        "--end-month",
        required=True,
        help="End month in YYYY-MM format (e.g. 2023-12)",
    )
    parser.add_argument(
        "--target-k",
        type=int,
        default=50,
        help="Target number of selections per month (10-100, default: 50)",
    )
    parser.add_argument(
        "--output",
        default="opportunities.csv",
        help="Output CSV path (default: opportunities.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"[INFO] Running for {args.start_month} → {args.end_month}")
    print(f"[INFO] Target K = {args.target_k} opportunities/month")

    opportunities = run(
        start_month=args.start_month,
        end_month=args.end_month,
        target_k=args.target_k,
    )

    output_path = Path(args.output)
    opportunities.to_csv(output_path, index=False)

    print(f"\n[DONE] {len(opportunities)} total opportunities written to {output_path}")
    print(opportunities.groupby("TARGET_MONTH").size().describe())
