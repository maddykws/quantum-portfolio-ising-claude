"""
Utility helpers: data download, quarter generation, plotting.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# ---------------------------------------------------------------------------
# Quarter generation
# ---------------------------------------------------------------------------

def get_quarter_ends(start_year: int = 2010,
                     end_year:   int = 2024) -> list:
    """
    Return a list of quarter-end dates (Mar, Jun, Sep, Dec) as strings.

    Covers start_year Q1 through end_year Q4.
    """
    dates = []
    for year in range(start_year, end_year + 1):
        for month, day in [(3, 31), (6, 30), (9, 30), (12, 31)]:
            dates.append(f"{year}-{month:02d}-{day:02d}")
    return dates


def quarter_window(quarter_end: str, hold_months: int = 3) -> tuple:
    """
    Return (hold_start, hold_end) strings for a given quarter_end date.

    The hold period starts the day after quarter_end.
    """
    end   = pd.Timestamp(quarter_end)
    start = end + pd.Timedelta(days=1)
    hend  = start + pd.DateOffset(months=hold_months) - pd.Timedelta(days=1)
    return start.strftime("%Y-%m-%d"), hend.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_returns(tickers: list,
                     start:   str = "2005-01-01",
                     end:     str = "2025-01-01",
                     cache_path: str | None = None) -> pd.DataFrame:
    """
    Download adjusted close prices and compute daily returns.

    Optionally caches to CSV so repeated runs skip the download.
    Drops columns with <80% data coverage and rows with any NaN.
    """
    if cache_path and Path(cache_path).exists():
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        raw    = yf.download(tickers, start=start, end=end,
                             auto_adjust=True, progress=False)
        prices = raw["Close"] if "Close" in raw.columns else raw
        if cache_path:
            prices.to_csv(cache_path)

    min_rows = int(len(prices) * 0.80)
    prices   = prices.dropna(axis=1, thresh=min_rows).dropna()
    returns  = prices.pct_change().dropna()
    return returns


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_results(results: list, path: str) -> None:
    """Save list of window result dicts to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path: str) -> list:
    """Load results from JSON."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sampling efficiency
# ---------------------------------------------------------------------------

def sampling_efficiency(counts: dict, optimal_bits: str) -> float:
    """
    Estimate shots needed vs random search to find optimal bitstring.

    Returns shots_to_optimal / (1 / random_probability).
    """
    total = sum(counts.values())
    if total == 0 or optimal_bits not in counts:
        return 1.0
    opt_prob    = counts[optimal_bits] / total
    n_bits      = len(optimal_bits)
    random_prob = 1 / (2 ** n_bits)
    if random_prob == 0:
        return 1.0
    shots_to_optimal = 1 / opt_prob if opt_prob > 0 else total
    shots_random     = 1 / random_prob
    return shots_random / shots_to_optimal


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cumulative_returns(results: list, out_path: str = "results/cumulative.png") -> None:
    """
    Plot cumulative returns: QAOA vs SPY vs equal-weight over all windows.
    """
    quarters = [r["quarter"] for r in results]
    qaoa_cr  = np.cumprod([1 + r["qaoa_return"]          for r in results])
    spy_cr   = np.cumprod([1 + r.get("spy_return", 0)    for r in results])
    ew_cr    = np.cumprod([1 + r.get("equal_weight_return", 0) for r in results])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(quarters, qaoa_cr, label="QAOA Portfolio", linewidth=2, color="#76b900")
    ax.plot(quarters, spy_cr,  label="SPY Passive",    linewidth=1.5, linestyle="--", color="#888")
    ax.plot(quarters, ew_cr,   label="Equal Weight",   linewidth=1.5, linestyle=":",  color="#0070c0")

    ax.set_title("Quantum Portfolio vs Benchmarks — 56-Quarter Backtest (2010–2024)",
                 fontsize=13)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Cumulative Return (× initial)")
    ax.legend()
    ax.set_xticks(quarters[::4])
    ax.set_xticklabels(quarters[::4], rotation=45, ha="right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Chart saved → {out_path}")
