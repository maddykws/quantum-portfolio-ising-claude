"""
QUBO formulation for portfolio optimisation.

Maps the stock selection problem to a Quadratic
Unconstrained Binary Optimisation matrix suitable
for QAOA quantum circuits.
"""

import numpy as np
import pandas as pd


def build_qubo(returns_df: pd.DataFrame,
               lam: float = 2.0) -> list:
    """
    Build QUBO matrix for portfolio optimisation.

    Formulation:
        minimise: -sharpe^T @ sharpe + lam * covariance

    Binary variable z_i = 1 if stock i is selected.
    Penalty weight lam balances return vs risk.

    Args:
        returns_df: Daily returns (stocks as columns)
        lam: Covariance penalty weight (default 2.0)

    Returns:
        Flattened QUBO matrix as list for CUDA-Q
    """
    cov    = returns_df.cov().values
    mu     = returns_df.mean().values
    sigma  = np.sqrt(np.diag(cov) + 1e-12)
    sharpe = mu / sigma

    Q = -np.outer(sharpe, sharpe) + lam * cov
    return Q.flatten().tolist()


def get_dynamic_universe(returns_df: pd.DataFrame,
                          quarter_end: str,
                          lookback_years: int = 5,
                          n: int = 25) -> list:
    """
    Select top-N stocks by individual Sharpe ratio
    at each quarterly construction point.

    Uses only data available at quarter_end —
    no lookahead bias.

    Args:
        returns_df: Full returns history
        quarter_end: Date string YYYY-MM-DD
        lookback_years: Years of history to use
        n: Number of stocks to select

    Returns:
        List of ticker symbols (top-N by Sharpe)
    """
    end   = pd.Timestamp(quarter_end)
    start = end - pd.DateOffset(years=lookback_years)
    w     = returns_df[start.strftime("%Y-%m-%d"):
                       end.strftime("%Y-%m-%d")]

    sharpes = {}
    for t in w.columns:
        r = w[t].dropna()
        if len(r) < 100:
            continue
        mu    = r.mean() * 252
        sigma = r.std()  * (252 ** 0.5)
        if sigma < 1e-9:
            continue
        sharpes[t] = mu / sigma

    return sorted(
        sharpes, key=sharpes.get,
        reverse=True)[:n]


def generate_windows(start_year: int = 2010,
                     end_year: int = 2024,
                     lookback_years: int = 5) -> list:
    """
    Generate quarterly portfolio construction windows.
    Matches Infleqtion Q-CHOP methodology exactly.

    Returns:
        List of (start_date, end_date, label) tuples
    """
    windows = []
    for year in range(start_year, end_year):
        for month in [1, 4, 7, 10]:
            end_dt   = pd.Timestamp(
                year=year, month=month, day=1)
            start_dt = end_dt - pd.DateOffset(
                years=lookback_years)
            if end_dt > pd.Timestamp("2025-01-01"):
                continue
            label = f"Q{(month//3)+1}-{year}"
            windows.append((
                start_dt.strftime("%Y-%m-%d"),
                end_dt.strftime("%Y-%m-%d"),
                label
            ))
    return windows
