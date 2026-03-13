"""
data_loader.py
--------------
Optimized I/O layer implementing all 5 memory fixes:

1. Predicate pushdown via pyarrow.dataset — reads only matching row groups,
   never loads full 10GB file into RAM before filtering.
2. Column-specific loading — callers specify exactly which columns they need.
3. MONTH stored as int32 (YYYYMM) instead of string — 4x less memory,
   faster comparisons.
4. EID and SCENARIOID categorized at load time.
5. Aggressive variable deletion after each step.
"""

import time
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow as pa

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

# Minimal columns needed just for profitability computation
PRICES_PROFIT_COLS = ["EID", "DATETIME", "PEAKID", "PRICEREALIZED"]
COSTS_PROFIT_COLS  = ["EID", "MONTH", "PEAKID", "C"]

# ---------------------------------------------------------------------------
# MONTH encoding helpers
# Fix 3: use int32 YYYYMM instead of strings
# ---------------------------------------------------------------------------

def month_str_to_int(month_str: str) -> int:
    """'2022-06' → 202206"""
    return int(month_str.replace("-", ""))


def month_int_to_str(month_int: int) -> str:
    """202206 → '2022-06'"""
    s = str(month_int)
    return f"{s[:4]}-{s[4:]}"


def datetime_to_month_int(dt_series: pd.Series) -> pd.Series:
    """Convert datetime64 series → int32 YYYYMM."""
    return (dt_series.dt.year * 100 + dt_series.dt.month).astype("int32")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def _optimize_sim_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast floats to float32, categorize EID and SCENARIOID.
    MONTH kept as int32 — do NOT categorize (used for ordered comparison).
    """
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")
    for col in ["EID", "SCENARIOID"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# In data_loader.py, replace _read_parquet_with_pushdown with this:

def _read_parquet_with_pushdown(
    path: Path,
    columns: Optional[list[str]] = None,
    filters: Optional[list] = None,
) -> pd.DataFrame:
    """
    High-performance Parquet reader using Predicate Pushdown.
    
    This function filters 10GB+ files at the scan level to prevent 
    MemoryErrors and ensures type-safety between Python strings 
    and Parquet timestamp[ns] columns.
    """
    # 1. Calculate file size for progress logging
    size_mb = path.stat().st_size / 1024 / 1024
    filter_info = f" | filters={filters}" if filters else ""
    print(f"    {path.name} ({size_mb:.0f} MB){filter_info}...", end=" ", flush=True)
    
    t0 = time.perf_counter()

    if filters is not None:
        # 2. Type Alignment: Convert filter strings to pd.Timestamp
        # This prevents 'ArrowNotImplementedError' during the kernel execution.
        processed_filters = []
        for col, op, val in filters:
            if col in ["DATETIME", "MONTH"] and isinstance(val, str):
                val = pd.Timestamp(val)
            processed_filters.append((col, op, val))

        # 3. Construct PyArrow Compute Expression
        # This resolves 'TypeError: Cannot convert list to pyarrow._compute.Expression'
        pa_expression = None
        for col, op, val in processed_filters:
            field = pc.field(col)
            if op == "==":
                cond = (field == val)
            elif op == ">=":
                cond = (field >= val)
            elif op == "<=":
                cond = (field <= val)
            elif op == ">":
                cond = (field > val)
            elif op == "<":
                cond = (field < val)
            elif op == "!=":
                cond = (field != val)
            else:
                raise ValueError(f"Unsupported operator: {op}")
            
            pa_expression = cond if pa_expression is None else (pa_expression & cond)

        # 4. Execute Scanner with Predicate Pushdown
        # Only relevant Row Groups are read into memory.
        dataset = ds.dataset(str(path), format="parquet")
        scanner = dataset.scanner(columns=columns, filter=pa_expression)
        table = scanner.to_table()
        
        # 5. Final conversion and memory cleanup
        df = table.to_pandas()
        del table
    else:
        # Standard load if no filters are required
        df = pd.read_parquet(path, columns=columns)

    elapsed = time.perf_counter() - t0
    print(f"{len(df):,} rows | {elapsed:.1f}s")
    return df
   
def _build_pa_filter(filters: list):
    """
    Convert simple filter list to pyarrow expression.
    Supports: [("col", "=", val), ("col", "in", [v1,v2]), ("col", ">=", val)]
    """
    exprs = []
    for col, op, val in filters:
        field = ds.field(col)
        if op == "=":
            exprs.append(field == val)
        elif op == "in":
            exprs.append(field.isin(val))
        elif op == ">=":
            exprs.append(field >= val)
        elif op == "<=":
            exprs.append(field <= val)
        elif op == ">":
            exprs.append(field > val)
        elif op == "<":
            exprs.append(field < val)

    if not exprs:
        return None
    result = exprs[0]
    for e in exprs[1:]:
        result = result & e
    return result


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_costs(columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Fix 2: accepts columns argument — pass COSTS_PROFIT_COLS for profitability
    computation to avoid loading unnecessary columns.
    """
    cols = columns or COSTS_PROFIT_COLS
    df   = _read_parquet_with_pushdown(COSTS_FILE, columns=cols)
    df["PEAKID"] = df["PEAKID"].astype("int8")
    df["C"]      = df["C"].astype("float32")
    df["EID"]    = df["EID"].astype("category")
    # Fix 3: MONTH as int32
    if "MONTH" in df.columns:
        df["MONTH"] = df["MONTH"].astype(str).str.replace("-", "").astype("int32")
    return df


def load_prices(columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Fix 2: accepts columns argument — pass PRICES_PROFIT_COLS for profitability.
    """
    cols = columns or PRICES_PROFIT_COLS
    df   = _read_parquet_with_pushdown(PRICES_FILE, columns=cols)
    df["DATETIME"]      = pd.to_datetime(df["DATETIME"])
    df["PEAKID"]        = df["PEAKID"].astype("int8")
    df["PRICEREALIZED"] = df["PRICEREALIZED"].astype("float32")
    df["EID"]           = df["EID"].astype("category")
    # Fix 3: MONTH as int32
    df["MONTH"] = datetime_to_month_int(df["DATETIME"])
    return df


def load_sim_monthly_year(year: int) -> pd.DataFrame:
    path = _find_year_file(DATA_ROOT / "sim_monthly", year)
    if path is None:
        print(f"    [WARN] No sim_monthly file for {year}")
        return pd.DataFrame()
    filters = [("DATETIME", ">=", f"{year}-01-01"), ("DATETIME", "<=", f"{year}-12-31")]
    df = _read_parquet_with_pushdown(path, columns=SIM_MONTHLY_COLS, filters=filters)

    # Predicate pushdown: only rows where DATETIME year == year
    year_start = f"{year}-01-01"
    year_end   = f"{year}-12-31"
    filters    = [("DATETIME", ">=", year_start), ("DATETIME", "<=", year_end)]

    df = _read_parquet_with_pushdown(path, columns=SIM_MONTHLY_COLS, filters=filters)
    if df.empty:
        return df

    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["PEAKID"]   = df["PEAKID"].astype("int8")
    # Fix 3: int32 month
    df["MONTH"]    = datetime_to_month_int(df["DATETIME"])
    df = _optimize_sim_dtypes(df)
    return df


def load_sim_daily_year(year: int) -> pd.DataFrame:
    """Same optimizations as load_sim_monthly_year."""
    path = _find_year_file(DATA_ROOT / "sim_daily", year)
    if path is None:
        print(f"    [WARN] No sim_daily file for {year}")
        return pd.DataFrame()

    year_start = f"{year}-01-01"
    year_end   = f"{year}-12-31"
    filters    = [("DATETIME", ">=", year_start), ("DATETIME", "<=", year_end)]

    df = _read_parquet_with_pushdown(path, columns=SIM_DAILY_COLS, filters=filters)
    if df.empty:
        return df

    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["PEAKID"]   = df["PEAKID"].astype("int8")
    df["MONTH"]    = datetime_to_month_int(df["DATETIME"])
    df = _optimize_sim_dtypes(df)
    return df


def load_sim_monthly(years: Optional[list[int]] = None) -> pd.DataFrame:
    """Multi-year loader — kept for compatibility with main.py/backtest.py."""
    folder = DATA_ROOT / "sim_monthly"
    files  = _get_year_files(folder, years)
    frames = []
    for f in files:
        year = _year_from_filename(f)
        df   = load_sim_monthly_year(year)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_sim_daily(years: Optional[list[int]] = None) -> pd.DataFrame:
    """Multi-year loader — kept for compatibility."""
    folder = DATA_ROOT / "sim_daily"
    files  = _get_year_files(folder, years)
    frames = []
    for f in files:
        year = _year_from_filename(f)
        df   = load_sim_daily_year(year)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _find_year_file(folder: Path, year: int) -> Optional[Path]:
    matches = [f for f in sorted(folder.glob("*.parquet"))
               if str(year) in f.stem]
    return matches[0] if matches else None


def _get_year_files(folder: Path, years: Optional[list[int]]) -> list[Path]:
    all_files = sorted(folder.glob("*.parquet"))
    if years is None:
        return all_files
    filtered = [f for f in all_files if any(str(y) in f.stem for y in years)]
    return filtered if filtered else all_files


def _year_from_filename(path: Path) -> int:
    """Extract year from filename like sim_monthly_2022.parquet → 2022."""
    for part in path.stem.split("_"):
        if part.isdigit() and len(part) == 4:
            return int(part)
    raise ValueError(f"Cannot extract year from filename: {path.name}")
