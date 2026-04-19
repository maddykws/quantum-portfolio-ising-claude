"""
Classical portfolio baselines for honest comparison.

Three baselines of increasing difficulty:
1. SPY passive index (weakest — investor benchmark)
2. Top-N equal weight (naive selection)
3. Top-N optimal weight (best classical approach)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple


def portfolio_sharpe_equal(tickers: list,
                            returns_df: pd.DataFrame,
                            td: int = 252) -> float:
    """Equal-weight portfolio Sharpe ratio."""
    if not tickers:
        return 0.0
    r   = returns_df[tickers]
    mu  = r.mean().values * td
    cov = r.cov().values  * td
    w   = np.ones(len(tickers)) / len(tickers)
    vol = np.sqrt(w @ cov @ w)
    return float((w @ mu) / (vol + 1e-9))


def optimise_weights(tickers: list,
                      returns_df: pd.DataFrame,
                      td: int = 252) -> np.ndarray:
    """
    Classical mean-variance weight optimisation.

    Constraints:
        - Weights sum to 1
        - Each weight between 5% and 40%
    """
    if len(tickers) == 1:
        return np.array([1.0])

    r   = returns_df[tickers]
    mu  = r.mean().values * td
    cov = r.cov().values  * td
    n   = len(tickers)

    def neg_sharpe(w):
        return -(w @ mu) / (
            np.sqrt(w @ cov @ w) + 1e-9)

    res = minimize(
        neg_sharpe,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.05, 0.40)] * n,
        constraints=[{
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1
        }],
        options={"maxiter": 500}
    )
    return res.x if res.success \
           else np.ones(n) / n


def portfolio_sharpe_weighted(
        tickers: list,
        weights: np.ndarray,
        returns_df: pd.DataFrame,
        td: int = 252) -> float:
    """Sharpe ratio with specific weights."""
    if not tickers:
        return 0.0
    r   = returns_df[tickers]
    mu  = r.mean().values * td
    cov = r.cov().values  * td
    w   = np.array(weights)
    vol = np.sqrt(w @ cov @ w)
    return float((w @ mu) / (vol + 1e-9))


def spy_sharpe(spy_returns: pd.Series,
               start: str,
               end: str,
               td: int = 252) -> float:
    """SPY passive index Sharpe ratio."""
    r = spy_returns[start:end]
    if len(r) < 50:
        return 0.0
    return float(
        (r.mean() * td) / (r.std() * td ** 0.5))


def top_n_equal(tickers: list,
                returns_df: pd.DataFrame,
                n: int) -> Tuple[float, list]:
    """
    Top-N by individual Sharpe, equal weight.
    Weak classical baseline.
    """
    ind = {}
    for t in tickers:
        r      = returns_df[t].dropna()
        ind[t] = float(
            (r.mean() * 252) /
            (r.std() * 252 ** 0.5 + 1e-9))
    top = sorted(
        ind, key=ind.get, reverse=True)[:n]
    return portfolio_sharpe_equal(
        top, returns_df), top


def top_n_optimal(tickers: list,
                   returns_df: pd.DataFrame,
                   n: int) -> Tuple[float, list]:
    """
    Top-N by individual Sharpe, optimal weight.
    Hard classical baseline — best classical approach.
    """
    ind = {}
    for t in tickers:
        r      = returns_df[t].dropna()
        ind[t] = float(
            (r.mean() * 252) /
            (r.std() * 252 ** 0.5 + 1e-9))
    top     = sorted(
        ind, key=ind.get, reverse=True)[:n]
    weights = optimise_weights(top, returns_df)
    return portfolio_sharpe_weighted(
        top, weights, returns_df), top
