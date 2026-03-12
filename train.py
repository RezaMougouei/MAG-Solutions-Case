"""
train.py
--------
Memory-efficient training pipeline.

Core strategy: process ONE YEAR of sim data at a time.
- Load sim_monthly for year Y (~2-3 GB)
- Build features for all months in year Y
- Discard sim data
- Move to year Y+1

This keeps peak memory to ~3-4 GB instead of 10+ GB.

Usage:
    python train.py --start-month 2020-01 --end-month 2023-12 --model lightgbm
    python train.py --start-month 2020-01 --end-month 2023-12 --model logistic
    python train.py --start-month 2020-01 --end-month 2023-12 --model xgboost
"""

import argparse
import gc
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import (
    load_costs, load_prices,
    load_sim_monthly_year, load_sim_daily_year,
)
from profitability import compute_profitability
from feature_builder import build_feature_matrix
from models import (
    build_logistic_regression, build_lightgbm, build_xgboost,
    get_available_features, compute_pos_weight,
    apply_smote, tune_threshold, evaluate_model,
)
from main import month_range, prev_month, years_needed
from progress import StageTimer, PipelineProgress, DatasetBuildProgress, get_memory_mb

MIN_K = 10
MAX_K = 100


# ---------------------------------------------------------------------------
# Build dataset YEAR BY YEAR — never more than ~3GB in memory at once
# ---------------------------------------------------------------------------

def build_full_dataset_by_year(
    all_months: list[str],
    hist_profit_df: pd.DataFrame,
    lookback_months: int = 6,
) -> pd.DataFrame:
    """
    Build full labeled feature matrix processing ONE year of sim data at a time.

    For each year:
      1. Load sim_monthly_year  (~2-3 GB)
      2. Load sim_daily_year    (~1-2 GB)
      3. Build features for all months in that year
      4. Delete sim data → back to baseline memory
      5. Move to next year
    """
    # Group months by their decision year
    # e.g. target_month=2021-03 → decision_month=2021-02 → decision_year=2021
    year_to_months: dict[int, list[str]] = {}
    for tm in all_months:
        dm   = prev_month(tm)
        year = int(dm[:4])
        year_to_months.setdefault(year, []).append(tm)

    all_years = sorted(year_to_months.keys())
    total_months = len(all_months)
    progress = DatasetBuildProgress(total_months=total_months)
    all_frames = []

    for year in all_years:
        months_this_year = year_to_months[year]

        print(f"\n  ── Year {year} "
              f"({len(months_this_year)} months | mem={get_memory_mb():.0f} MB) ──")

        # Load sim data for this year only
        with StageTimer(f"  sim_monthly {year}"):
            sim_monthly_df = load_sim_monthly_year(year)

        with StageTimer(f"  sim_daily {year}"):
            sim_daily_df = load_sim_daily_year(year)

        if sim_monthly_df.empty:
            print(f"  [WARN] No sim_monthly data for {year}, skipping.")
            del sim_monthly_df, sim_daily_df
            gc.collect()
            continue

        # Build features for each month in this year
        for target_month in months_this_year:
            decision_month = prev_month(target_month)
            cutoff_dt      = pd.Timestamp(f"{decision_month}-08 00:00:00")

            sm_allowed   = sim_monthly_df[sim_monthly_df["MONTH"] <= target_month]
            sd_allowed   = sim_daily_df[sim_daily_df["DATETIME"] <= cutoff_dt] \
                           if not sim_daily_df.empty else pd.DataFrame()
            hist_allowed = hist_profit_df[hist_profit_df["MONTH"] <= decision_month]

            fm = build_feature_matrix(
                sm_allowed, sd_allowed, hist_allowed,
                target_month=target_month,
                decision_month=decision_month,
                lookback_months=lookback_months,
            )

            if fm.empty:
                progress.update(target_month, 0, 0)
                continue

            # Attach ground truth labels
            truth = hist_profit_df[
                hist_profit_df["MONTH"] == target_month
            ][["EID", "PEAKID", "IS_PROFITABLE", "PROFIT"]].copy()

            fm["EID"]    = fm["EID"].astype(str)
            truth["EID"] = truth["EID"].astype(str)

            fm = fm.merge(truth, on=["EID", "PEAKID"], how="left")
            fm["IS_PROFITABLE"] = fm["IS_PROFITABLE"].fillna(False).astype(int)
            fm["PROFIT"]        = fm["PROFIT"].fillna(0.0)
            fm["TARGET_MONTH"]  = target_month

            progress.update(target_month, len(fm), int(fm["IS_PROFITABLE"].sum()))
            all_frames.append(fm)

        # Discard sim data immediately — back to baseline memory
        del sim_monthly_df, sim_daily_df
        gc.collect()
        print(f"  Sim data freed | mem={get_memory_mb():.0f} MB")

    if not all_frames:
        return pd.DataFrame()

    print(f"\n  Concatenating {len(all_frames)} monthly frames...",
          end=" ", flush=True)
    t0      = time.perf_counter()
    full_df = pd.concat(all_frames, ignore_index=True)
    print(f"{time.perf_counter()-t0:.1f}s")

    pos_rate = full_df["IS_PROFITABLE"].mean() * 100
    print(f"  Dataset: {full_df.shape[0]:,} rows | "
          f"{pos_rate:.2f}% profitable | "
          f"{full_df['TARGET_MONTH'].nunique()} months | "
          f"mem={get_memory_mb():.0f} MB")

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
# Main training pipeline
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
    n_folds    = len(all_months) - min_train_months

    print(f"\n{'='*60}")
    print(f"  MAG Energy — Training Pipeline")
    print(f"  Model:  {model_type.upper()}")
    print(f"  Range:  {start_month} → {end_month}")
    print(f"  Months: {len(all_months)} | Folds: {n_folds}")
    print(f"  Memory: {get_memory_mb():.0f} MB at start")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Stage 1: Load costs + prices (small files, keep in memory)
    # ------------------------------------------------------------------
    with StageTimer("Loading costs & prices"):
        costs_df  = load_costs()
        prices_df = load_prices()

    with StageTimer("Computing historical profitability"):
        hist_profit_df = compute_profitability(prices_df, costs_df)
        del prices_df
        gc.collect()

    # ------------------------------------------------------------------
    # Stage 2: Build dataset year by year
    # ------------------------------------------------------------------
    with StageTimer(f"Building feature dataset ({len(all_months)} months)"):
        full_df = build_full_dataset_by_year(
            all_months, hist_profit_df, lookback_months
        )

    if full_df.empty:
        print("[ERROR] Empty dataset — check data paths.")
        return None, 0.3, pd.DataFrame()

    feat_cols = get_available_features(full_df)
    print(f"\n  Features ({len(feat_cols)}): {feat_cols}")

    # ------------------------------------------------------------------
    # Stage 3: Walk-forward training — pure in-memory slicing
    # ------------------------------------------------------------------
    pipeline       = PipelineProgress(total_folds=n_folds, model_type=model_type)
    all_selections = []
    final_model    = None
    final_threshold = 0.3

    print(f"\n{'─'*60}")
    print(f"  ▶  Walk-forward training — {n_folds} folds")
    print(f"{'─'*60}")

    for i in range(min_train_months, len(all_months)):
        train_months = all_months[:i]
        val_month    = all_months[i]
        fold_idx     = i - min_train_months + 1

        pipeline.start_fold(fold_idx, len(train_months), val_month)

        train_df = full_df[full_df["TARGET_MONTH"].isin(set(train_months))]
        val_df   = full_df[full_df["TARGET_MONTH"] == val_month]

        if train_df.empty or train_df["IS_PROFITABLE"].sum() < 5:
            print(f"       ⚠ Skipped — not enough positive samples")
            continue
        if val_df.empty or val_df["IS_PROFITABLE"].nunique() < 2:
            print(f"       ⚠ Skipped — single class in validation")
            continue

        X_train = train_df[feat_cols].fillna(0).values.astype(np.float32)
        y_train = train_df["IS_PROFITABLE"].values.astype(int)
        X_val   = val_df[feat_cols].fillna(0).values.astype(np.float32)
        y_val   = val_df["IS_PROFITABLE"].values.astype(int)

        pos_w = compute_pos_weight(pd.Series(y_train))

        if use_smote:
            X_train, y_train = apply_smote(X_train, y_train)

        if model_type == "logistic":
            model = build_logistic_regression()
        elif model_type == "lightgbm":
            model = build_lightgbm(pos_weight=pos_w)
        elif model_type == "xgboost":
            model = build_xgboost(pos_weight=pos_w)
        else:
            raise ValueError(f"Unknown model: {model_type}")

        model.fit(X_train, y_train)
        threshold = tune_threshold(model, X_val, y_val, metric="f1")
        metrics   = evaluate_model(
            model, X_val, y_val, threshold=threshold, label=val_month
        )
        pipeline.end_fold(metrics)

        val_features = val_df[feat_cols + ["EID", "PEAKID"]].copy()
        selections   = select_from_model(
            model, val_features, val_month, threshold, feat_cols, target_k
        )
        all_selections.append(selections)

        final_model     = model
        final_threshold = threshold

    pipeline.print_final_summary()

    if save_model and final_model is not None:
        model_path = Path(f"model_{model_type}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "model":      final_model,
                "threshold":  final_threshold,
                "feat_cols":  feat_cols,
                "model_type": model_type,
            }, f)
        print(f"\n  Model saved → {model_path}")

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
