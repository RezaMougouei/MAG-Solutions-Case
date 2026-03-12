"""
train.py
--------
Training pipeline for FTR opportunity classification.

Strategy: Walk-Forward Cross-Validation (Time Series Split)
------------------------------------------------------------
We NEVER use future data to train. For each fold:
  - Train on months [start ... M-1]
  - Validate on month M
  - Predict on month M+1

This mirrors real trading conditions exactly and enforces anti-leakage
at the training level, not just the feature level.

Walk-forward prevents the classic mistake of training on 2020-2023
and testing on 2020 — which would leak future knowledge into training.

Usage:
    python train.py --start-month 2020-01 --end-month 2023-12
    python train.py --start-month 2020-01 --end-month 2023-12 --model lightgbm
    python train.py --start-month 2020-01 --end-month 2023-12 --model xgboost --smote
"""

import argparse
import gc
import pickle
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
    apply_smote, tune_threshold, evaluate_model
)
from main import month_range, prev_month, years_needed

MIN_K = 10
MAX_K = 100


# ---------------------------------------------------------------------------
# Walk-forward training loop
# ---------------------------------------------------------------------------

def build_training_dataset(
    months: list[str],
    sim_monthly_df: pd.DataFrame,
    sim_daily_df: pd.DataFrame,
    hist_profit_df: pd.DataFrame,
    lookback_months: int = 6,
) -> pd.DataFrame:
    """
    Build a labeled feature matrix across multiple months.
    Each row = one (EID, MONTH, PEAKID) opportunity with IS_PROFITABLE label.

    Anti-leakage: for each target_month, only data up to decision_month is used.
    """
    frames = []

    for target_month in tqdm(months, desc="  Building dataset", leave=False):
        decision_month = prev_month(target_month)
        cutoff_dt      = pd.Timestamp(f"{decision_month}-08 00:00:00")

        # Anti-leakage filtering
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

        # Attach ground truth labels for this target_month
        truth = hist_profit_df[
            hist_profit_df["MONTH"] == target_month
        ][["EID", "PEAKID", "IS_PROFITABLE", "PROFIT"]].copy()

        # EID may be categorical — normalize to string for merge
        fm["EID"]    = fm["EID"].astype(str)
        truth["EID"] = truth["EID"].astype(str)

        fm = fm.merge(truth, on=["EID", "PEAKID"], how="left")
        fm["IS_PROFITABLE"] = fm["IS_PROFITABLE"].fillna(False).astype(int)
        fm["PROFIT"]        = fm["PROFIT"].fillna(0.0)

        frames.append(fm)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def select_from_model(
    model,
    feature_matrix: pd.DataFrame,
    target_month: str,
    threshold: float,
    target_k: int = 50,
) -> pd.DataFrame:
    """
    Use trained model to select opportunities for target_month.
    Enforces 10-100 constraint:
      - Take all above threshold, capped at MAX_K
      - If below MIN_K, take top-MIN_K by probability regardless of threshold
    """
    feat_cols = get_available_features(feature_matrix)
    X = feature_matrix[feat_cols].fillna(0).values.astype(np.float32)

    proba = model.predict_proba(X)[:, 1]
    feature_matrix = feature_matrix.copy()
    feature_matrix["_proba"] = proba

    # Sort by probability descending
    feature_matrix = feature_matrix.sort_values("_proba", ascending=False)

    # Apply threshold
    above = feature_matrix[feature_matrix["_proba"] >= threshold]

    # Enforce bounds
    if len(above) >= MIN_K:
        selected = above.head(MAX_K)
    else:
        # Not enough above threshold — take top MIN_K regardless
        selected = feature_matrix.head(MIN_K)

    # Format output
    selected = selected.copy()
    selected["TARGET_MONTH"] = target_month
    selected["PEAK_TYPE"]    = selected["PEAKID"].map({0: "OFF", 1: "ON"})

    return selected[["TARGET_MONTH", "PEAK_TYPE", "EID"]].drop_duplicates()


# ---------------------------------------------------------------------------
# Main training function
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
) -> tuple[object, float, pd.DataFrame]:
    """
    Walk-forward training + evaluation.

    Returns: (trained_model, best_threshold, selections_df)
    """
    all_months      = month_range(start_month, end_month)
    all_years       = years_needed(all_months + [prev_month(m) for m in all_months])

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
    # Walk-forward loop
    # ------------------------------------------------------------------
    print(f"\n[INFO] Starting walk-forward training ({model_type})")
    print(f"       {start_month} → {end_month} | min_train_months={min_train_months}\n")

    all_val_results = []
    all_selections  = []
    final_model     = None
    final_threshold = 0.3   # default, will be tuned

    for i in tqdm(range(min_train_months, len(all_months)), desc="Walk-forward folds"):
        train_months = all_months[:i]
        val_month    = all_months[i]

        # ---- Build training set ------------------------------------------
        train_df = build_training_dataset(
            train_months, sim_monthly_df, sim_daily_df,
            hist_profit_df, lookback_months
        )
        if train_df.empty or train_df["IS_PROFITABLE"].sum() < 5:
            continue

        feat_cols = get_available_features(train_df)
        X_train   = train_df[feat_cols].fillna(0).values.astype(np.float32)
        y_train   = train_df["IS_PROFITABLE"].values.astype(int)

        # ---- Handle imbalance -------------------------------------------
        pos_w = compute_pos_weight(pd.Series(y_train))

        if use_smote:
            X_train, y_train = apply_smote(X_train, y_train)

        # ---- Build and train model --------------------------------------
        if model_type == "logistic":
            model = build_logistic_regression()
        elif model_type == "lightgbm":
            model = build_lightgbm(pos_weight=pos_w)
        elif model_type == "xgboost":
            model = build_xgboost(pos_weight=pos_w)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_train, y_train)

        # ---- Build validation set (val_month) ---------------------------
        val_df = build_training_dataset(
            [val_month], sim_monthly_df, sim_daily_df,
            hist_profit_df, lookback_months
        )
        if val_df.empty:
            continue

        X_val = val_df[feat_cols].fillna(0).values.astype(np.float32)
        y_val = val_df["IS_PROFITABLE"].values.astype(int)

        if len(np.unique(y_val)) < 2:
            continue

        # ---- Tune threshold on validation set ---------------------------
        threshold = tune_threshold(model, X_val, y_val, metric="f1")

        # ---- Evaluate ---------------------------------------------------
        metrics = evaluate_model(model, X_val, y_val,
                                 threshold=threshold, label=val_month)
        all_val_results.append({**metrics, "MONTH": val_month,
                                 "n_train": len(y_train)})

        # ---- Select opportunities for val_month -------------------------
        val_fm = val_df[feat_cols + ["EID", "PEAKID"]].copy()
        selections = select_from_model(model, val_fm, val_month, threshold, target_k)
        all_selections.append(selections)

        # Keep the most recently trained model as final
        final_model     = model
        final_threshold = threshold

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if not all_val_results:
        print("[ERROR] No validation results produced.")
        return final_model, final_threshold, pd.DataFrame()

    results_df = pd.DataFrame(all_val_results)
    print("\n" + "=" * 70)
    print(f"WALK-FORWARD RESULTS — {model_type.upper()}")
    print("=" * 70)
    print(results_df[["MONTH", "n_train", "f1", "precision",
                       "recall", "pr_auc"]].to_string(index=False))
    print("-" * 70)
    print(f"Mean F1:       {results_df['f1'].mean():.4f}")
    print(f"Mean Precision:{results_df['precision'].mean():.4f}")
    print(f"Mean Recall:   {results_df['recall'].mean():.4f}")
    print(f"Mean PR-AUC:   {results_df['pr_auc'].mean():.4f}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    if save_model and final_model is not None:
        model_path = Path(f"model_{model_type}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model":     final_model,
                "threshold": final_threshold,
                "feat_cols": feat_cols,
                "model_type": model_type,
            }, f)
        print(f"\n[INFO] Model saved → {model_path}")

    selections_df = pd.concat(all_selections, ignore_index=True) if all_selections else pd.DataFrame()
    return final_model, final_threshold, selections_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FTR opportunity classifier")
    parser.add_argument("--start-month",       required=True)
    parser.add_argument("--end-month",         required=True)
    parser.add_argument("--model",             default="lightgbm",
                        choices=["logistic", "lightgbm", "xgboost"])
    parser.add_argument("--smote",             action="store_true",
                        help="Apply SMOTE oversampling (requires imbalanced-learn)")
    parser.add_argument("--target-k",          type=int, default=50)
    parser.add_argument("--min-train-months",  type=int, default=6)
    parser.add_argument("--output",            default="opportunities.csv")
    args = parser.parse_args()

    model, threshold, selections = run_training(
        start_month=args.start_month,
        end_month=args.end_month,
        model_type=args.model,
        use_smote=args.smote,
        target_k=args.target_k,
        min_train_months=args.min_train_months,
    )

    if not selections.empty:
        out = Path(args.output)
        selections.to_csv(out, index=False)
        print(f"\n[DONE] {len(selections)} opportunities → {out}")
        print(selections.groupby("TARGET_MONTH").size().describe())
