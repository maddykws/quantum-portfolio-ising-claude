"""
QUBO (Quadratic Unconstrained Binary Optimization)
formulation for portfolio optimisation.
"""

import numpy as np
import pandas as pd


def build_qubo(returns_df: pd.DataFrame, lam: float = 2.0) -> list:
    """
    Build QUBO matrix for portfolio optimisation.

    Formulation: minimise -(Sharpe covariance) + lam * risk
    Binary variable z_i = 1 if stock i is selected.

    Args:
        returns_df: Daily returns DataFrame (stocks as columns)
        lam: Penalty weight for covariance term (default 2.0)

    Returns:
        Flattened QUBO matrix as list (CUDA-Q compatible)
    """
    cov    = returns_df.cov().values
    mu     = returns_df.mean().values
    sigma  = np.sqrt(np.diag(cov) + 1e-12)
    sharpe = mu / sigma
    Q      = -np.outer(sharpe, sharpe) + lam * cov
    return Q.flatten().tolist()


def get_top_n_at_quarter(returns_df: pd.DataFrame,
                          quarter_end: str,
                          lookback_years: int = 5,
                          n: int = 25) -> list:
    """
    Dynamic universe selection — top-N stocks by individual
    Sharpe ratio at each quarterly construction point.

    Uses only data available at quarter_end (no lookahead).
    Matches Infleqtion's published methodology exactly.

    Args:
        returns_df: Full returns history
        quarter_end: Date string 'YYYY-MM-DD'
        lookback_years: Years of history to use
        n: Number of stocks to select

    Returns:
        List of ticker symbols (top-N by Sharpe)
    """
    end   = pd.Timestamp(quarter_end)
    start = end - pd.DateOffset(years=lookback_years)
    w     = returns_df[start.strftime("%Y-%m-%d"):end.strftime("%Y-%m-%d")]

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

    return sorted(sharpes, key=sharpes.get, reverse=True)[:n]
