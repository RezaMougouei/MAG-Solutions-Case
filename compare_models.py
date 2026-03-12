"""
compare_models.py
-----------------
Runs Logistic Regression, LightGBM, and XGBoost side by side and
produces a comparison table of F1, Precision, Recall, PR-AUC, and Net Profit.

Usage:
    python compare_models.py --start-month 2020-01 --end-month 2022-12
"""

import argparse
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loader import load_costs, load_prices, load_sim_monthly, load_sim_daily
from profitability import compute_profitability
from feature_builder import build_feature_matrix
from models import (
    build_logistic_regression, build_lightgbm, build_xgboost,
    get_available_features, compute_pos_weight,
    tune_threshold, evaluate_model
)
from train import build_training_dataset, select_from_model
from main import month_range, prev_month, years_needed
from backtest import compute_net_profit


def run_comparison(start_month: str, end_month: str,
                   min_train_months: int = 6, target_k: int = 50):

    all_months = month_range(start_month, end_month)
    all_years  = years_needed(all_months + [prev_month(m) for m in all_months])

    # ------------------------------------------------------------------
    # Load data once — shared across all models
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
    # Model configs
    # ------------------------------------------------------------------
    model_configs = {
        "logistic":  build_logistic_regression,
        "lightgbm":  None,   # built dynamically with pos_weight
        "xgboost":   None,
    }

    summary = {name: [] for name in model_configs}

    # ------------------------------------------------------------------
    # Walk-forward evaluation for each model
    # ------------------------------------------------------------------
    for i in tqdm(range(min_train_months, len(all_months)), desc="Folds"):
        train_months = all_months[:i]
        val_month    = all_months[i]

        train_df = build_training_dataset(
            train_months, sim_monthly_df, sim_daily_df, hist_profit_df
        )
        if train_df.empty or train_df["IS_PROFITABLE"].sum() < 5:
            continue

        feat_cols = get_available_features(train_df)
        X_train   = train_df[feat_cols].fillna(0).values.astype(np.float32)
        y_train   = train_df["IS_PROFITABLE"].values.astype(int)
        pos_w     = compute_pos_weight(pd.Series(y_train))

        val_df = build_training_dataset(
            [val_month], sim_monthly_df, sim_daily_df, hist_profit_df
        )
        if val_df.empty or len(np.unique(val_df["IS_PROFITABLE"])) < 2:
            continue

        X_val = val_df[feat_cols].fillna(0).values.astype(np.float32)
        y_val = val_df["IS_PROFITABLE"].values.astype(int)

        # Evaluate each model
        models_to_run = {
            "logistic": build_logistic_regression(),
            "lightgbm": build_lightgbm(pos_weight=pos_w) if _lgbm_ok() else None,
            "xgboost":  build_xgboost(pos_weight=pos_w)  if _xgb_ok()  else None,
        }

        for name, model in models_to_run.items():
            if model is None:
                continue
            try:
                model.fit(X_train, y_train)
                threshold = tune_threshold(model, X_val, y_val, metric="f1")
                proba     = model.predict_proba(X_val)[:, 1]
                y_pred    = (proba >= threshold).astype(int)

                from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
                val_fm    = val_df[feat_cols + ["EID", "PEAKID"]].copy()
                selections = select_from_model(model, val_fm, val_month, threshold, target_k)
                net_profit = compute_net_profit(selections, hist_profit_df[hist_profit_df["MONTH"] == val_month].rename(columns={"MONTH": "TARGET_MONTH"}))

                summary[name].append({
                    "MONTH":      val_month,
                    "f1":         f1_score(y_val, y_pred, zero_division=0),
                    "precision":  precision_score(y_val, y_pred, zero_division=0),
                    "recall":     recall_score(y_val, y_pred, zero_division=0),
                    "pr_auc":     average_precision_score(y_val, proba),
                    "net_profit": net_profit,
                    "threshold":  threshold,
                })
            except Exception as e:
                print(f"  [WARN] {name} failed on {val_month}: {e}")

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 75)
    print(f"{'Model':<12} {'Mean F1':>9} {'Mean Prec':>10} {'Mean Rec':>9} "
          f"{'PR-AUC':>8} {'Net Profit':>12}")
    print("-" * 75)

    for name, results in summary.items():
        if not results:
            continue
        df = pd.DataFrame(results)
        print(f"{name:<12} "
              f"{df['f1'].mean():>9.4f} "
              f"{df['precision'].mean():>10.4f} "
              f"{df['recall'].mean():>9.4f} "
              f"{df['pr_auc'].mean():>8.4f} "
              f"{df['net_profit'].sum():>12.2f}")

    print("=" * 75)
    print("\nInterpretation:")
    print("  F1        — balance of precision and recall (higher = better)")
    print("  Precision — fraction of selected that are profitable (quality)")
    print("  Recall    — fraction of all profitable that we caught (coverage)")
    print("  PR-AUC    — area under precision-recall curve (best for imbalanced)")
    print("  Net Profit— total $ value generated by selections")

    return summary


def _lgbm_ok():
    try:
        import lightgbm
        return True
    except ImportError:
        return False


def _xgb_ok():
    try:
        import xgboost
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all three models")
    parser.add_argument("--start-month",      required=True)
    parser.add_argument("--end-month",        required=True)
    parser.add_argument("--min-train-months", type=int, default=6)
    parser.add_argument("--target-k",         type=int, default=50)
    args = parser.parse_args()

    run_comparison(
        args.start_month, args.end_month,
        args.min_train_months, args.target_k
    )
