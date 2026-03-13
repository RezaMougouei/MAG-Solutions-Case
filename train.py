"""
train.py
--------
Memory-efficient training pipeline implementing all 5 fixes.

Fix 4: Instead of accumulating all yearly DataFrames in memory and
concatenating at the end, each year's feature matrix is written to a
temporary Parquet file on disk. The final dataset is assembled by
reading these small temp files — peak memory = one year at a time.

Usage:
    python train.py --start-month 2020-02 --end-month 2023-12 --model lightgbm
    python train.py --start-month 2020-02 --end-month 2023-12 --model logistic
    python train.py --start-month 2020-02 --end-month 2023-12 --model xgboost
"""

import argparse
import gc
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import (
    load_costs, load_prices,
    load_sim_monthly_year, load_sim_daily_year,
    month_str_to_int,
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
# Fix 4: Build dataset year by year, write to temp Parquet files
# ---------------------------------------------------------------------------

def build_full_dataset_by_year(
    all_months: list[str],
    hist_profit_df: pd.DataFrame,
    lookback_months: int = 6,
    temp_dir: Path = None,
) -> Path:
    """
    Build labeled feature matrix one year at a time.
    Each year's result is written to a temp Parquet file immediately,
    then the DataFrame is freed. Returns path to combined Parquet file.

    Peak memory = one year of sim data + one year of features (not all years).
    """
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp())
    temp_dir.mkdir(parents=True, exist_ok=True)

    year_to_months: dict[int, list[str]] = {}
    for tm in all_months:
        year = int(prev_month(tm)[:4])
        year_to_months.setdefault(year, []).append(tm)

    all_years  = sorted(year_to_months.keys())
    progress   = DatasetBuildProgress(total_months=len(all_months))
    temp_files = []

    for year in all_years:
        months_this_year = year_to_months[year]
        print(f"\n  ── Year {year} "
              f"({len(months_this_year)} months | mem={get_memory_mb():.0f} MB) ──")

        with StageTimer(f"  sim_monthly {year}"):
            sim_monthly_df = load_sim_monthly_year(year)

        with StageTimer(f"  sim_daily {year}"):
            sim_daily_df = load_sim_daily_year(year)

        if sim_monthly_df.empty:
            print(f"  [WARN] No sim_monthly for {year}, skipping.")
            del sim_monthly_df, sim_daily_df
            gc.collect()
            continue

        year_frames = []

        for target_month in months_this_year:
            decision_month    = prev_month(target_month)
            decision_month_int = month_str_to_int(decision_month)
            target_month_int   = month_str_to_int(target_month)
            cutoff_dt          = pd.Timestamp(f"{decision_month}-08 00:00:00")

            sm_allowed = sim_monthly_df[
                sim_monthly_df["MONTH"] <= target_month_int
            ]
            sd_allowed = sim_daily_df[
                sim_daily_df["DATETIME"] <= cutoff_dt
            ] if not sim_daily_df.empty else pd.DataFrame()
            hist_allowed = hist_profit_df[
                hist_profit_df["MONTH"] <= decision_month_int
            ]

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
                hist_profit_df["MONTH"] == target_month_int
            ][["EID", "PEAKID", "IS_PROFITABLE", "PROFIT"]].copy()

            fm["EID"]    = fm["EID"].astype(str)
            truth["EID"] = truth["EID"].astype(str)

            fm = fm.merge(truth, on=["EID", "PEAKID"], how="left")
            fm["IS_PROFITABLE"] = fm["IS_PROFITABLE"].fillna(False).astype("int8")
            fm["PROFIT"]        = fm["PROFIT"].fillna(0.0).astype("float32")
            fm["TARGET_MONTH"]  = target_month

            del truth, sm_allowed, sd_allowed, hist_allowed
            progress.update(target_month, len(fm), int(fm["IS_PROFITABLE"].sum()))
            year_frames.append(fm)
            del fm

        # Fix 4: write this year to disk immediately, free memory
        if year_frames:
            year_df   = pd.concat(year_frames, ignore_index=True)
            del year_frames
            gc.collect()

            temp_path = temp_dir / f"features_{year}.parquet"
            year_df.to_parquet(temp_path, index=False)
            temp_files.append(temp_path)
            print(f"  Written {len(year_df):,} rows → {temp_path.name}")
            del year_df
            gc.collect()

        del sim_monthly_df, sim_daily_df
        gc.collect()
        print(f"  Sim data freed | mem={get_memory_mb():.0f} MB")

    if not temp_files:
        return None

    # Combine temp files into one final Parquet
    print(f"\n  Merging {len(temp_files)} temp files...", end=" ", flush=True)
    t0     = time.perf_counter()
    frames = [pd.read_parquet(f) for f in temp_files]
    full_df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    final_path = temp_dir / "full_dataset.parquet"
    full_df.to_parquet(final_path, index=False)
    print(f"{time.perf_counter()-t0:.1f}s | {len(full_df):,} rows")

    # Clean up yearly temp files
    for f in temp_files:
        f.unlink()

    pos_rate = full_df["IS_PROFITABLE"].mean() * 100
    print(f"  Dataset: {full_df.shape[0]:,} rows | "
          f"{pos_rate:.2f}% profitable | "
          f"{full_df['TARGET_MONTH'].nunique()} months | "
          f"mem={get_memory_mb():.0f} MB")

    del full_df
    gc.collect()
    return final_path


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
    del X

    fm = feature_matrix.copy()
    fm["_proba"] = proba
    fm.sort_values("_proba", ascending=False, inplace=True)

    above    = fm[fm["_proba"] >= threshold]
    selected = above.head(MAX_K) if len(above) >= MIN_K else fm.head(MIN_K)

    selected = selected.copy()
    selected["TARGET_MONTH"] = target_month
    selected["PEAK_TYPE"]    = selected["PEAKID"].map({0: "OFF", 1: "ON"})
    return selected[["TARGET_MONTH", "PEAK_TYPE", "EID"]].drop_duplicates()


# ---------------------------------------------------------------------------
# Walk-forward training
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
    # Stage 1: Costs + prices — delete costs_df after profitability (Fix 5)
    # ------------------------------------------------------------------
    with StageTimer("Loading costs & prices"):
        costs_df  = load_costs()
        prices_df = load_prices()

    with StageTimer("Computing historical profitability"):
        hist_profit_df = compute_profitability(prices_df, costs_df)
        # Fix 5: delete both immediately
        del prices_df, costs_df
        gc.collect()

    # ------------------------------------------------------------------
    # Stage 2: Build dataset — disk-backed, one year at a time (Fix 4)
    # ------------------------------------------------------------------
    temp_dir = Path(tempfile.mkdtemp(prefix="mag_features_"))
    print(f"\n  Temp dir: {temp_dir}")

    with StageTimer(f"Building feature dataset ({len(all_months)} months)"):
        dataset_path = build_full_dataset_by_year(
            all_months, hist_profit_df, lookback_months, temp_dir
        )

    # Free hist_profit_df — no longer needed after dataset is built
    del hist_profit_df
    gc.collect()

    if dataset_path is None:
        print("[ERROR] Empty dataset.")
        return None, 0.3, pd.DataFrame()

    # Load final dataset from disk
    print(f"\n  Loading final dataset from {dataset_path.name}...",
          end=" ", flush=True)
    t0      = time.perf_counter()
    full_df = pd.read_parquet(dataset_path)
    print(f"{time.perf_counter()-t0:.1f}s | mem={get_memory_mb():.0f} MB")

    feat_cols = get_available_features(full_df)
    print(f"  Features ({len(feat_cols)}): {feat_cols}")

    # ------------------------------------------------------------------
    # Stage 3: Walk-forward — O(1) slicing, aggressive cleanup per fold
    # ------------------------------------------------------------------
    pipeline        = PipelineProgress(total_folds=n_folds, model_type=model_type)
    all_selections  = []
    final_model     = None
    final_threshold = 0.3

    print(f"\n{'─'*60}")
    print(f"  ▶  Walk-forward training — {n_folds} folds")
    print(f"{'─'*60}")

    for i in range(min_train_months, len(all_months)):
        val_month = all_months[i]
        fold_idx  = i - min_train_months + 1

        gc.collect()
        pipeline.start_fold(fold_idx, i, val_month)

        # < comparison avoids large set allocation
        train_df = full_df[full_df["TARGET_MONTH"] < val_month]
        val_df   = full_df[full_df["TARGET_MONTH"] == val_month]

        if train_df.empty or train_df["IS_PROFITABLE"].sum() < 5:
            print(f"       ⚠ Skipped — not enough positive samples")
            del train_df, val_df
            continue
        if val_df.empty or val_df["IS_PROFITABLE"].nunique() < 2:
            print(f"       ⚠ Skipped — single class in validation")
            del train_df, val_df
            continue

        # Convert to numpy then free dataframes immediately
        X_train = train_df[feat_cols].fillna(0).values.astype(np.float32)
        y_train = train_df["IS_PROFITABLE"].values.astype(int)
        X_val   = val_df[feat_cols].fillna(0).values.astype(np.float32)
        y_val   = val_df["IS_PROFITABLE"].values.astype(int)
        val_features = val_df[feat_cols + ["EID", "PEAKID"]].copy()

        del train_df
        gc.collect()

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

        # Free training arrays immediately after fit
        del X_train, y_train
        gc.collect()

        threshold = tune_threshold(model, X_val, y_val, metric="f1")
        metrics   = evaluate_model(
            model, X_val, y_val, threshold=threshold, label=val_month
        )

        del X_val, y_val, val_df
        gc.collect()

        pipeline.end_fold(metrics)

        selections = select_from_model(
            model, val_features, val_month, threshold, feat_cols, target_k
        )
        del val_features
        all_selections.append(selections)

        final_model     = model
        final_threshold = threshold

    # Clean up temp files
    try:
        dataset_path.unlink()
        temp_dir.rmdir()
    except Exception:
        pass

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
