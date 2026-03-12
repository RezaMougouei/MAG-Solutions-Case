"""
train.py
--------
Optimized training pipeline — O(n) instead of O(n²).

Key fix: build the full labeled feature matrix ONCE across all months,
then slice train/val windows from it in memory. No redundant recomputation.

Walk-Forward Cross-Validation (Time Series Split)
--------------------------------------------------
For each fold i:
  - Train on rows where MONTH in all_months[:i]
  - Validate on rows where MONTH == all_months[i]
  - Anti-leakage enforced at feature level inside build_feature_matrix.

Usage:
    python train.py --start-month 2020-01 --end-month 2023-12 --model lightgbm
    python train.py --start-month 2020-01 --end-month 2023-12 --model logistic
    python train.py --start-month 2020-01 --end-month 2023-12 --model xgboost --smote
"""

import argparse
import gc
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loader import load_costs, load_prices, load_sim_monthly, load_sim_daily
from profitability import compute_profitability
from feature_builder import build_feature_matrix
from models import (
    build_logistic_regression, build_lightgbm, build_xgboost,
    get_available_features, compute_pos_weight,
    apply_smote, tune_threshold, evaluate_model,
)
from main import month_range, prev_month, years_needed

MIN_K = 10
MAX_K = 100


# ---------------------------------------------------------------------------
# Build full dataset ONCE — O(n)
# ---------------------------------------------------------------------------

def build_full_dataset(
    all_months: list[str],
    sim_monthly_df: pd.DataFrame,
    sim_daily_df: pd.DataFrame,
    hist_profit_df: pd.DataFrame,
    lookback_months: int = 6,
) -> pd.DataFrame:
    """
    Build labeled feature matrix for ALL months in a single pass.
    Each row = one (EID, TARGET_MONTH, PEAKID) with IS_PROFITABLE label.

    Anti-leakage is enforced inside build_feature_matrix per month —
    each month only sees data available at its decision time.
    """
    frames = []

    for target_month in tqdm(all_months, desc="Building dataset"):
        decision_month = prev_month(target_month)
        cutoff_dt      = pd.Timestamp(f"{decision_month}-08 00:00:00")

        sm_allowed   = sim_monthly_df[sim_monthly_df["MONTH"] <= target_month]
        sd_allowed   = sim_daily_df[sim_daily_df["DATETIME"] <= cutoff_dt]
        hist_allowed = hist_profit_df[hist_profit_df["MONTH"] <= decision_month]

        fm = build_feature_matrix(
            sm_allowed, sd_allowed, hist_allowed,
            target_month=target_month,
            decision_month=decision_month,
            lookback_months=lookback_months,
        )
        if fm.empty:
            continue

        # Attach ground truth
        truth = hist_profit_df[
            hist_profit_df["MONTH"] == target_month
        ][["EID", "PEAKID", "IS_PROFITABLE", "PROFIT"]].copy()

        fm["EID"]    = fm["EID"].astype(str)
        truth["EID"] = truth["EID"].astype(str)

        fm = fm.merge(truth, on=["EID", "PEAKID"], how="left")
        fm["IS_PROFITABLE"] = fm["IS_PROFITABLE"].fillna(False).astype(int)
        fm["PROFIT"]        = fm["PROFIT"].fillna(0.0)
        fm["TARGET_MONTH"]  = target_month
        frames.append(fm)

    if not frames:
        return pd.DataFrame()

    full_df = pd.concat(frames, ignore_index=True)
    pos_rate = full_df["IS_PROFITABLE"].mean() * 100
    print(f"\n[INFO] Full dataset: {full_df.shape[0]:,} rows | "
          f"{pos_rate:.2f}% profitable | "
          f"{full_df['TARGET_MONTH'].nunique()} months")
    return full_df


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_from_model(
    model,
    feature_matrix: pd.DataFrame,
    target_month: str,
    threshold: float,
    feat_cols: list[str],
    target_k: int = 50,
) -> pd.DataFrame:
    """Select 10-100 opportunities ranked by predicted probability."""
    X     = feature_matrix[feat_cols].fillna(0).values.astype(np.float32)
    proba = model.predict_proba(X)[:, 1]

    fm = feature_matrix.copy()
    fm["_proba"] = proba
    fm = fm.sort_values("_proba", ascending=False)

    above    = fm[fm["_proba"] >= threshold]
    selected = above.head(MAX_K) if len(above) >= MIN_K else fm.head(MIN_K)

    selected = selected.copy()
    selected["TARGET_MONTH"] = target_month
    selected["PEAK_TYPE"]    = selected["PEAKID"].map({0: "OFF", 1: "ON"})
    return selected[["TARGET_MONTH", "PEAK_TYPE", "EID"]].drop_duplicates()


# ---------------------------------------------------------------------------
# Walk-forward — O(n) slicing from pre-built dataset
# ---------------------------------------------------------------------------

def run_training(
    start_month: str,
    end_month: str,
    model_type: str = "lightgbm",
    use_smote: bool = False,
    target_k: int = 50,
    min_train_months: int = 6,
    lookback_months: int = 6,
    save_model: bool = True,
) -> tuple:

    all_months = month_range(start_month, end_month)
    all_years  = years_needed(all_months + [prev_month(m) for m in all_months])

    # ------------------------------------------------------------------
    # Load data once
    # ------------------------------------------------------------------
    print("[INFO] Loading costs and prices...")
    costs_df  = load_costs()
    prices_df = load_prices()

    print("[INFO] Computing historical profitability...")
    hist_profit_df = compute_profitability(prices_df, costs_df)
    del prices_df
    gc.collect()

    print(f"[INFO] Loading sim_monthly for {all_years}...")
    sim_monthly_df = load_sim_monthly(years=all_years)

    print(f"[INFO] Loading sim_daily for {all_years}...")
    sim_daily_df = load_sim_daily(years=all_years)

    # ------------------------------------------------------------------
    # Build full dataset ONCE
    # ------------------------------------------------------------------
    t0      = time.perf_counter()
    full_df = build_full_dataset(
        all_months, sim_monthly_df, sim_daily_df,
        hist_profit_df, lookback_months
    )
    print(f"[INFO] Dataset built in {time.perf_counter()-t0:.1f}s")

    # Free sim data — not needed anymore
    del sim_monthly_df, sim_daily_df
    gc.collect()

    if full_df.empty:
        print("[ERROR] Empty dataset.")
        return None, 0.3, pd.DataFrame()

    feat_cols = get_available_features(full_df)
    print(f"[INFO] {len(feat_cols)} features: {feat_cols}\n")

    # ------------------------------------------------------------------
    # Walk-forward loop — just array slicing, no recomputation
    # ------------------------------------------------------------------
    n_folds = len(all_months) - min_train_months
    print(f"[INFO] Walk-forward ({model_type}) | {n_folds} folds\n")

    all_val_results = []
    all_selections  = []
    final_model     = None
    final_threshold = 0.3

    for i in tqdm(range(min_train_months, len(all_months)), desc="Walk-forward"):
        train_months = all_months[:i]
        val_month    = all_months[i]

        # Pure in-memory slice — O(1)
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

        if use_smote:
            X_train, y_train = apply_smote(X_train, y_train)

        # Build and train model
        if model_type == "logistic":
            model = build_logistic_regression()
        elif model_type == "lightgbm":
            model = build_lightgbm(pos_weight=pos_w)
        elif model_type == "xgboost":
            model = build_xgboost(pos_weight=pos_w)
        else:
            raise ValueError(f"Unknown model: {model_type}")

        model.fit(X_train, y_train)

        # Tune decision threshold on validation set
        threshold = tune_threshold(model, X_val, y_val, metric="f1")

        # Evaluate
        metrics = evaluate_model(
            model, X_val, y_val,
            threshold=threshold, label=val_month
        )
        all_val_results.append({
            **metrics, "MONTH": val_month, "n_train": len(y_train)
        })

        # Select opportunities for this month
        val_features = val_df[feat_cols + ["EID", "PEAKID"]].copy()
        selections   = select_from_model(
            model, val_features, val_month, threshold, feat_cols, target_k
        )
        all_selections.append(selections)

        final_model     = model
        final_threshold = threshold

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    if not all_val_results:
        print("[ERROR] No validation results.")
        return final_model, final_threshold, pd.DataFrame()

    results_df = pd.DataFrame(all_val_results)
    print("\n" + "=" * 70)
    print(f"WALK-FORWARD RESULTS — {model_type.upper()}")
    print("=" * 70)
    cols = ["MONTH", "n_train", "f1", "precision", "recall", "pr_auc"]
    print(results_df[[c for c in cols if c in results_df]].to_string(index=False))
    print("-" * 70)
    print(f"Mean F1:        {results_df['f1'].mean():.4f}")
    print(f"Mean Precision: {results_df['precision'].mean():.4f}")
    print(f"Mean Recall:    {results_df['recall'].mean():.4f}")
    print(f"Mean PR-AUC:    {results_df['pr_auc'].mean():.4f}")
    print("=" * 70)

    # Save model
    if save_model and final_model is not None:
        model_path = Path(f"model_{model_type}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model":      final_model,
                "threshold":  final_threshold,
                "feat_cols":  feat_cols,
                "model_type": model_type,
            }, f)
        print(f"\n[INFO] Model saved → {model_path}")

    selections_df = (
        pd.concat(all_selections, ignore_index=True)
        if all_selections else pd.DataFrame()
    )
    return final_model, final_threshold, selections_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FTR classifier")
    parser.add_argument("--start-month",      required=True)
    parser.add_argument("--end-month",        required=True)
    parser.add_argument("--model",            default="lightgbm",
                        choices=["logistic", "lightgbm", "xgboost"])
    parser.add_argument("--smote",            action="store_true")
    parser.add_argument("--target-k",         type=int, default=50)
    parser.add_argument("--min-train-months", type=int, default=6)
    parser.add_argument("--output",           default="opportunities.csv")
    args = parser.parse_args()

    model, threshold, selections = run_training(
        start_month=args.start_month,
        end_month=args.end_month,
        model_type=args.model,
        use_smote=args.smote,
        target_k=args.target_k,
        min_train_months=args.min_train_months,
    )

    if selections is not None and not selections.empty:
        out = Path(args.output)
        selections.to_csv(out, index=False)
        print(f"\n[DONE] {len(selections)} opportunities → {out}")
        print(selections.groupby("TARGET_MONTH").size().describe())
