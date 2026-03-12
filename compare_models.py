"""
compare_models.py
-----------------
Runs Logistic Regression, LightGBM, and XGBoost on identical walk-forward
folds and prints a side-by-side comparison table.

Key optimization: full dataset built ONCE and shared across all three models.

Usage:
    python compare_models.py --start-month 2020-01 --end-month 2022-12
"""

import argparse
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, average_precision_score
)
from tqdm import tqdm

from data_loader import load_costs, load_prices, load_sim_monthly, load_sim_daily
from profitability import compute_profitability
from models import (
    build_logistic_regression, build_lightgbm, build_xgboost,
    get_available_features, compute_pos_weight, tune_threshold,
)
from train import build_full_dataset, select_from_model
from main import month_range, prev_month, years_needed


def _lgbm_ok():
    try:
        import lightgbm; return True
    except ImportError:
        return False


def _xgb_ok():
    try:
        import xgboost; return True
    except ImportError:
        return False


def run_comparison(
    start_month: str,
    end_month: str,
    min_train_months: int = 6,
    target_k: int = 50,
):
    all_months = month_range(start_month, end_month)
    all_years  = years_needed(all_months + [prev_month(m) for m in all_months])

    # ------------------------------------------------------------------
    # Load data once — shared across ALL models
    # ------------------------------------------------------------------
    print("[INFO] Loading data...")
    costs_df  = load_costs()
    prices_df = load_prices()
    hist_profit_df = compute_profitability(prices_df, costs_df)
    del prices_df
    gc.collect()

    sim_monthly_df = load_sim_monthly(years=all_years)
    sim_daily_df   = load_sim_daily(years=all_years)

    # ------------------------------------------------------------------
    # Build full dataset ONCE — shared across ALL models
    # ------------------------------------------------------------------
    print("[INFO] Building full dataset...")
    full_df = build_full_dataset(
        all_months, sim_monthly_df, sim_daily_df, hist_profit_df
    )
    del sim_monthly_df, sim_daily_df
    gc.collect()

    if full_df.empty:
        print("[ERROR] Empty dataset.")
        return

    feat_cols = get_available_features(full_df)
    print(f"[INFO] {len(feat_cols)} features | "
          f"{full_df['TARGET_MONTH'].nunique()} months\n")

    # ------------------------------------------------------------------
    # Walk-forward evaluation — all models on same folds
    # ------------------------------------------------------------------
    model_names = ["logistic"]
    if _lgbm_ok(): model_names.append("lightgbm")
    if _xgb_ok():  model_names.append("xgboost")

    summary = {name: [] for name in model_names}

    for i in tqdm(range(min_train_months, len(all_months)), desc="Folds"):
        train_months = all_months[:i]
        val_month    = all_months[i]

        train_df = full_df[full_df["TARGET_MONTH"].isin(set(train_months))]
        val_df   = full_df[full_df["TARGET_MONTH"] == val_month]

        if train_df.empty or train_df["IS_PROFITABLE"].sum() < 5:
            continue
        if val_df.empty or val_df["IS_PROFITABLE"].nunique() < 2:
            continue

        X_train = train_df[feat_cols].fillna(0).values.astype(np.float32)
        y_train = train_df["IS_PROFITABLE"].values.astype(int)
        X_val   = val_df[feat_cols].fillna(0).values.astype(np.float32)
        y_val   = val_df["IS_PROFITABLE"].values.astype(int)

        pos_w = compute_pos_weight(pd.Series(y_train))

        models_to_run = {
            "logistic": build_logistic_regression(),
            "lightgbm": build_lightgbm(pos_weight=pos_w) if _lgbm_ok() else None,
            "xgboost":  build_xgboost(pos_weight=pos_w)  if _xgb_ok()  else None,
        }

        for name in model_names:
            model = models_to_run.get(name)
            if model is None:
                continue
            try:
                model.fit(X_train, y_train)
                threshold = tune_threshold(model, X_val, y_val, metric="f1")
                proba     = model.predict_proba(X_val)[:, 1]
                y_pred    = (proba >= threshold).astype(int)

                # Net profit
                val_features = val_df[feat_cols + ["EID", "PEAKID"]].copy()
                selections   = select_from_model(
                    model, val_features, val_month, threshold, feat_cols, target_k
                )
                sel_set = set(zip(selections["PEAK_TYPE"], selections["EID"]))
                truth   = hist_profit_df[hist_profit_df["MONTH"] == val_month].copy()
                truth["PEAK_TYPE"] = truth["PEAKID"].map({0: "OFF", 1: "ON"})
                truth["EID"]       = truth["EID"].astype(str)
                matched = truth[
                    truth.apply(lambda r: (r["PEAK_TYPE"], str(r["EID"])) in sel_set, axis=1)
                ]
                net_profit = matched["PROFIT"].sum() if not matched.empty else 0.0

                summary[name].append({
                    "MONTH":      val_month,
                    "f1":         f1_score(y_val, y_pred, zero_division=0),
                    "precision":  precision_score(y_val, y_pred, zero_division=0),
                    "recall":     recall_score(y_val, y_pred, zero_division=0),
                    "pr_auc":     average_precision_score(y_val, proba),
                    "net_profit": net_profit,
                })
            except Exception as e:
                print(f"  [WARN] {name} failed on {val_month}: {e}")

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 75)
    print(f"{'Model':<12} {'Mean F1':>9} {'Precision':>10} {'Recall':>8} "
          f"{'PR-AUC':>8} {'Net Profit':>12}")
    print("-" * 75)

    for name in model_names:
        results = summary[name]
        if not results:
            continue
        df = pd.DataFrame(results)
        print(f"{name:<12} "
              f"{df['f1'].mean():>9.4f} "
              f"{df['precision'].mean():>10.4f} "
              f"{df['recall'].mean():>8.4f} "
              f"{df['pr_auc'].mean():>8.4f} "
              f"{df['net_profit'].sum():>12.2f}")

    print("=" * 75)
    print("\nMetric guide:")
    print("  F1        — harmonic mean of precision & recall (primary metric)")
    print("  Precision — of selected opportunities, fraction that are profitable")
    print("  Recall    — of all profitable opportunities, fraction we caught")
    print("  PR-AUC    — best metric for imbalanced problems (higher = better)")
    print("  Net Profit— total economic value generated by selections")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all models side by side")
    parser.add_argument("--start-month",      required=True)
    parser.add_argument("--end-month",        required=True)
    parser.add_argument("--min-train-months", type=int, default=6)
    parser.add_argument("--target-k",         type=int, default=50)
    args = parser.parse_args()

    run_comparison(
        args.start_month, args.end_month,
        args.min_train_months, args.target_k
    )
