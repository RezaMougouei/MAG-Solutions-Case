"""
progress.py
-----------
Rich progress tracking for the MAG Energy pipeline.
Shows stage progress, elapsed time, ETA, and live memory usage.
"""

import gc
import time
import psutil
import os
from datetime import timedelta
from typing import Optional


def get_memory_mb() -> float:
    """Current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def format_time(seconds: float) -> str:
    """Format seconds into human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return str(timedelta(seconds=int(seconds)))[2:]  # MM:SS
    else:
        return str(timedelta(seconds=int(seconds)))       # HH:MM:SS


class StageTimer:
    """Tracks a single named pipeline stage with elapsed time and memory."""

    def __init__(self, name: str, total: Optional[int] = None):
        self.name    = name
        self.total   = total
        self.start   = None
        self.mem_start = None

    def __enter__(self):
        self.start     = time.perf_counter()
        self.mem_start = get_memory_mb()
        label = f"[{self.total} items]" if self.total else ""
        print(f"\n{'─'*60}")
        print(f"  ▶  {self.name} {label}")
        print(f"{'─'*60}")
        return self

    def __exit__(self, *args):
        elapsed  = time.perf_counter() - self.start
        mem_now  = get_memory_mb()
        mem_diff = mem_now - self.mem_start
        sign     = "+" if mem_diff >= 0 else ""
        print(f"  ✓  {self.name} done")
        print(f"     Time:   {format_time(elapsed)}")
        print(f"     Memory: {mem_now:.0f} MB ({sign}{mem_diff:.0f} MB)")


class PipelineProgress:
    """
    Full pipeline progress tracker.
    Shows live updates for each fold including metrics and ETA.
    """

    def __init__(self, total_folds: int, model_type: str):
        self.total_folds  = total_folds
        self.model_type   = model_type
        self.fold_times   = []
        self.fold_results = []
        self.pipeline_start = time.perf_counter()

    def start_fold(self, fold_idx: int, train_month_count: int, val_month: str):
        self._fold_start     = time.perf_counter()
        self._fold_idx       = fold_idx
        self._val_month      = val_month
        self._train_count    = train_month_count

        pct = fold_idx / self.total_folds * 100
        mem = get_memory_mb()

        # ETA based on average fold time so far
        if self.fold_times:
            avg   = sum(self.fold_times) / len(self.fold_times)
            remaining = avg * (self.total_folds - fold_idx)
            eta   = f"ETA {format_time(remaining)}"
        else:
            eta = "ETA --:--"

        print(f"\n  Fold {fold_idx:>3}/{self.total_folds} "
              f"[{pct:5.1f}%] | "
              f"val={val_month} | "
              f"train={train_month_count}mo | "
              f"mem={mem:.0f}MB | {eta}")

    def end_fold(self, metrics: dict):
        elapsed = time.perf_counter() - self._fold_start
        self.fold_times.append(elapsed)
        self.fold_results.append({**metrics, "MONTH": self._val_month})

        f1   = metrics.get("f1", 0)
        prec = metrics.get("precision", 0)
        rec  = metrics.get("recall", 0)
        prauc = metrics.get("pr_auc", 0)
        n_sel = metrics.get("n_positive_pred", 0)

        # Rolling averages
        avg_f1   = sum(r["f1"]  for r in self.fold_results) / len(self.fold_results)
        avg_prauc = sum(r.get("pr_auc", 0) for r in self.fold_results) / len(self.fold_results)

        print(f"       F1={f1:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | "
              f"PR-AUC={prauc:.3f} | selected={n_sel} | "
              f"[{format_time(elapsed)}] | "
              f"avg_F1={avg_f1:.3f}")

    def print_final_summary(self):
        total_time = time.perf_counter() - self.pipeline_start
        import pandas as pd
        results_df = pd.DataFrame(self.fold_results)

        print("\n" + "=" * 70)
        print(f"  FINAL SUMMARY — {self.model_type.upper()}")
        print(f"  Total time: {format_time(total_time)}")
        print("=" * 70)

        if results_df.empty:
            print("  No results recorded.")
            return

        metrics = ["f1", "precision", "recall", "pr_auc"]
        for m in metrics:
            if m in results_df:
                print(f"  {m:<12}: "
                      f"mean={results_df[m].mean():.4f} | "
                      f"std={results_df[m].std():.4f} | "
                      f"min={results_df[m].min():.4f} | "
                      f"max={results_df[m].max():.4f}")

        print("=" * 70)


class DatasetBuildProgress:
    """Progress tracker for the dataset building phase."""

    def __init__(self, total_months: int):
        self.total   = total_months
        self.current = 0
        self.start   = time.perf_counter()
        self.month_times = []

    def update(self, month: str, n_rows: int, n_profitable: int):
        self.current += 1
        elapsed = time.perf_counter() - self.start

        if self.month_times:
            avg = sum(self.month_times) / len(self.month_times)
            remaining = avg * (self.total - self.current)
            eta = format_time(remaining)
        else:
            eta = "--:--"

        pct = self.current / self.total * 100
        mem = get_memory_mb()
        rate = n_profitable / n_rows * 100 if n_rows > 0 else 0

        print(f"  [{self.current:>3}/{self.total}] {month} | "
              f"{n_rows:>6} rows | "
              f"{rate:4.1f}% profitable | "
              f"mem={mem:.0f}MB | "
              f"ETA {eta} | "
              f"{pct:.0f}%",
              flush=True)

        self.month_times.append(time.perf_counter() - self.start -
                                (sum(self.month_times)))
