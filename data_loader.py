"""
data_loader.py
--------------
Optimized I/O layer.
Key design: load sim data ONE YEAR AT A TIME to avoid 10GB+ memory spikes.
"""

import time
from pathlib import Path
from typing import Optional
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT   = Path("/workspace/data")
COSTS_FILE  = DATA_ROOT / "costs.parquet"
PRICES_FILE = DATA_ROOT / "prices.parquet"

SIM_MONTHLY_COLS = [
    "SCENARIOID", "EID", "DATETIME", "PEAKID",
    "ACTIVATIONLEVEL",
    "WINDIMPACT", "SOLARIMPACT", "HYDROIMPACT",
    "NONRENEWBALIMPACT", "EXTERNALIMPACT",
    "LOADIMPACT", "TRANSMISSIONOUTAGEIMPACT",
    "PSM"
]
SIM_DAILY_COLS = [
    "SCENARIOID", "EID", "DATETIME", "PEAKID",
    "ACTIVATIONLEVEL",
    "WINDIMPACT", "SOLARIMPACT", "HYDROIMPACT",
    "NONRENEWBALIMPACT", "EXTERNALIMPACT",
    "LOADIMPACT", "TRANSMISSIONOUTAGEIMPACT",
    "PSD"
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")
    for col in ["EID", "MONTH", "SCENARIOID"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _find_file_for_year(folder: Path, year: int) -> Optional[Path]:
    """Find the parquet file for a specific year in a folder."""
    matches = [f for f in sorted(folder.glob("*.parquet")) if str(year) in f.stem]
    return matches[0] if matches else None


def _load_single_file(path: Path, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Load one parquet file with progress output."""
    size = _mb(path)
    print(f"    Reading {path.name} ({size:.0f} MB)...", end=" ", flush=True)
    t0 = time.perf_counter()
    df = pd.read_parquet(path, columns=columns)
    print(f"{len(df):,} rows | {time.perf_counter()-t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_costs() -> pd.DataFrame:
    print(f"    Reading costs.parquet ({_mb(COSTS_FILE):.0f} MB)...",
          end=" ", flush=True)
    t0 = time.perf_counter()
    df = pd.read_parquet(COSTS_FILE)
    print(f"{len(df):,} rows | {time.perf_counter()-t0:.1f}s")
    df["MONTH"]  = df["MONTH"].astype(str)
    df["PEAKID"] = df["PEAKID"].astype(int)
    df["C"]      = df["C"].astype("float32")
    df["EID"]    = df["EID"].astype("category")
    return df


def load_prices() -> pd.DataFrame:
    print(f"    Reading prices.parquet ({_mb(PRICES_FILE):.0f} MB)...",
          end=" ", flush=True)
    t0 = time.perf_counter()
    df = pd.read_parquet(PRICES_FILE)
    print(f"{len(df):,} rows | {time.perf_counter()-t0:.1f}s")
    df["DATETIME"]      = pd.to_datetime(df["DATETIME"])
    df["PEAKID"]        = df["PEAKID"].astype(int)
    df["PRICEREALIZED"] = df["PRICEREALIZED"].astype("float32")
    df["MONTH"]         = df["DATETIME"].dt.to_period("M").astype(str)
    df["EID"]           = df["EID"].astype("category")
    return df


def load_sim_monthly_year(year: int) -> pd.DataFrame:
    """Load sim_monthly for a SINGLE year only."""
    path = _find_file_for_year(DATA_ROOT / "sim_monthly", year)
    if path is None:
        print(f"    [WARN] No sim_monthly file found for year {year}")
        return pd.DataFrame()
    df = _load_single_file(path, columns=SIM_MONTHLY_COLS)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["MONTH"]    = df["DATETIME"].dt.to_period("M").astype(str)
    df["PEAKID"]   = df["PEAKID"].astype(int)
    df = _optimize_dtypes(df)
    return df


def load_sim_daily_year(year: int) -> pd.DataFrame:
    """Load sim_daily for a SINGLE year only."""
    path = _find_file_for_year(DATA_ROOT / "sim_daily", year)
    if path is None:
        print(f"    [WARN] No sim_daily file found for year {year}")
        return pd.DataFrame()
    df = _load_single_file(path, columns=SIM_DAILY_COLS)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["MONTH"]    = df["DATETIME"].dt.to_period("M").astype(str)
    df["PEAKID"]   = df["PEAKID"].astype(int)
    df = _optimize_dtypes(df)
    return df


def load_sim_monthly(years: Optional[list[int]] = None) -> pd.DataFrame:
    """Load sim_monthly for multiple years — kept for compatibility."""
    folder = DATA_ROOT / "sim_monthly"
    all_files = sorted(folder.glob("*.parquet"))
    if years is not None:
        files = [f for f in all_files if any(str(y) in f.stem for y in years)]
        if not files:
            files = all_files
    else:
        files = all_files

    frames = []
    for f in files:
        df = _load_single_file(f, columns=SIM_MONTHLY_COLS)
        df["DATETIME"] = pd.to_datetime(df["DATETIME"])
        df["MONTH"]    = df["DATETIME"].dt.to_period("M").astype(str)
        df["PEAKID"]   = df["PEAKID"].astype(int)
        df = _optimize_dtypes(df)
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_sim_daily(years: Optional[list[int]] = None) -> pd.DataFrame:
    """Load sim_daily for multiple years — kept for compatibility."""
    folder = DATA_ROOT / "sim_daily"
    all_files = sorted(folder.glob("*.parquet"))
    if years is not None:
        files = [f for f in all_files if any(str(y) in f.stem for y in years)]
        if not files:
            files = all_files
    else:
        files = all_files

    frames = []
    for f in files:
        df = _load_single_file(f, columns=SIM_DAILY_COLS)
        df["DATETIME"] = pd.to_datetime(df["DATETIME"])
        df["MONTH"]    = df["DATETIME"].dt.to_period("M").astype(str)
        df["PEAKID"]   = df["PEAKID"].astype(int)
        df = _optimize_dtypes(df)
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
