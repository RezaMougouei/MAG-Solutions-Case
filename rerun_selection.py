"""
rerun_selection.py
------------------
Reruns opportunity selection using the already-saved model pickle.
Use this when training completed but selection crashed.

The saved model pickle contains the exact feat_cols it was trained on,
so feature mismatch errors cannot occur.

Usage:
    python rerun_selection.py --model logistic
    python rerun_selection.py --model lightgbm
"""

import argparse
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from progress import StageTimer, get_memory_mb

MIN_K = 10
MAX_K = 100


def select_from_model(
    model,
    val_df: pd.DataFrame,
    target_month: str,
    threshold: float,
    feat_cols: list[str],
) -> pd.DataFrame:
    # Use ONLY the feat_cols the model was trained on
    X     = val_df[feat_cols].fillna(0).values.astype(np.float32)
    proba = model.predict_proba(X)[:, 1]
    del X

    fm           = val_df[["EID", "PEAKID"]].copy()
    fm["_proba"] = proba
    fm.sort_values("_proba", ascending=False, inplace=True)

    above    = fm[fm["_proba"] >= threshold]
    selected = above.head(MAX_K) if len(above) >= MIN_K else fm.head(MIN_K)

    selected = selected.copy()
    selected["TARGET_MONTH"] = target_month
    selected["PEAK_TYPE"]    = selected["PEAKID"].map({0: "OFF", 1: "ON"})
    return selected[["TARGET_MONTH", "PEAK_TYPE", "EID"]].drop_duplicates()


def main(model_type: str, output: str, min_train_months: int):
    model_path = Path(f"model_{model_type}.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model at {model_path}")

    print(f"\n{'='*60}")
    print(f"  Rerunning selection from saved model: {model_path}")
    print(f"  Memory: {get_memory_mb():.0f} MB")
    print(f"{'='*60}")

    with StageTimer("Loading model"):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        model     = bundle["model"]
        threshold = bundle["threshold"]
        feat_cols = bundle["feat_cols"]
        print(f"    threshold={threshold:.3f} | features={len(feat_cols)}")

    with StageTimer("Loading feature matrix"):
        full_df = pd.read_parquet("/workspace/output/feature_matrix.parquet")
        full_df["TARGET_MONTH"] = pd.to_datetime(full_df["TARGET_MONTH"])
        print(f"    Shape: {full_df.shape}")

    all_months = sorted(full_df["TARGET_MONTH"].unique())
    val_months = all_months[min_train_months:]

    print(f"\n  Running selection on {len(val_months)} validation months...")
    all_selections = []

    for val_month in val_months:
        val_df = full_df[full_df["TARGET_MONTH"] == val_month]
        if val_df.empty:
            continue

        # Check all feat_cols exist — fill missing with 0
        missing = [c for c in feat_cols if c not in val_df.columns]
        if missing:
            print(f"  [WARN] {str(val_month)[:7]}: {len(missing)} missing cols, filling 0")
            for c in missing:
                val_df = val_df.copy()
                val_df[c] = 0.0

        sel = select_from_model(
            model, val_df, str(val_month)[:7], threshold, feat_cols
        )
        print(f"  {str(val_month)[:7]}: {len(sel)} selected")
        all_selections.append(sel)

    if not all_selections:
        print("[ERROR] No selections generated.")
        return

    result = pd.concat(all_selections, ignore_index=True)
    result.to_csv(output, index=False)
    print(f"\n[DONE] {len(result)} total opportunities → {output}")
    print(result.groupby("TARGET_MONTH").size().describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logistic",
                        choices=["logistic", "lightgbm", "xgboost"])
    parser.add_argument("--output", default="opportunities.csv")
    parser.add_argument("--min-train-months", type=int, default=6)
    args = parser.parse_args()

    main(args.model, args.output, args.min_train_months)
