"""
data_loader.py
--------------
Handles loading of all four data sources (costs, prices, sim_monthly, sim_daily)
from Parquet files with year-by-year chunking to avoid memory overload.

Anti-leakage is enforced at the call site (main.py / feature_builder.py),
NOT here — this module is a pure I/O layer.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_ROOT  = Path("/workspace/data")
COSTS_FILE       = DATA_ROOT / "costs.parquet"
PRICES_FILE      = DATA_ROOT / "prices.parquet"


def _parquet_files_for_years(folder: Path, years: list[int]) -> list[Path]:
    """Return sorted Parquet files whose name contains one of the requested years."""
    files = []
    for f in sorted(folder.glob("*.parquet")):
        for y in years:
            if str(y) in f.stem:
                files.append(f)
                break
    # Fallback: if naming convention differs, just return all files in folder
    if not files:
        files = sorted(folder.glob("*.parquet"))
    return files


def _load_parquet_folder(
    folder: Path,
    years: Optional[list[int]] = None,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load one or more Parquet files from a folder, optionally filtered by year."""
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")

    all_files = sorted(folder.glob("*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No Parquet files found in: {folder}")

    if years is not None:
        target_files = _parquet_files_for_years(folder, years)
        if not target_files:
            target_files = all_files  # graceful fallback
    else:
        target_files = all_files

    frames = []
    for f in target_files:
        df = pd.read_parquet(f, columns=columns)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_costs() -> pd.DataFrame:
    """
    Load monthly exposure cost data.

    Columns: EID, MONTH (str YYYY-MM), PEAKID (0/1), C (float)
    Missing combos → cost = 0 (implicit zero rule).
    """
    df = pd.read_parquet(COSTS_FILE)
    df["MONTH"] = df["MONTH"].astype(str)
    df["PEAKID"] = df["PEAKID"].astype(int)
    df["C"] = df["C"].astype(float)
    return df


def load_prices() -> pd.DataFrame:
    """
    Load realized hourly price data.

    Columns: EID, DATETIME (datetime64), PEAKID (0/1), PRICEREALIZED (float)
    Missing combos → price = 0 (implicit zero rule).
    """
    df = pd.read_parquet(PRICES_FILE)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["PEAKID"] = df["PEAKID"].astype(int)
    df["PRICEREALIZED"] = df["PRICEREALIZED"].astype(float)
    # Derive MONTH string for easy grouping
    df["MONTH"] = df["DATETIME"].dt.to_period("M").astype(str)
    return df


def load_sim_monthly(years: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Load monthly simulation data (3 scenarios).

    Columns: SCENARIOID, EID, DATETIME, PEAKID, ACTIVATIONLEVEL,
             WINDIMPACT, SOLARIMPACT, HYDROIMPACT, NONRENEWBALIMPACT,
             EXTERNALIMPACT, LOADIMPACT, TRANSMISSIONOUTAGEIMPACT, PSM
    """
    df = _load_parquet_folder(DATA_ROOT / "sim_monthly", years=years)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["MONTH"] = df["DATETIME"].dt.to_period("M").astype(str)
    df["PEAKID"] = df["PEAKID"].astype(int)
    return df


def load_sim_daily(years: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Load daily simulation data (3 scenarios).

    Columns: SCENARIOID, EID, DATETIME, PEAKID, ACTIVATIONLEVEL,
             WINDIMPACT, SOLARIMPACT, HYDROIMPACT, NONRENEWBALIMPACT,
             EXTERNALIMPACT, LOADIMPACT, TRANSMISSIONOUTAGEIMPACT, PSD
    """
    df = _load_parquet_folder(DATA_ROOT / "sim_daily", years=years)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["MONTH"] = df["DATETIME"].dt.to_period("M").astype(str)
    df["PEAKID"] = df["PEAKID"].astype(int)
    return df
