"""
Classical portfolio baselines for comparison with QAOA results.

Three baselines:
  1. SPY passive buy-and-hold
  2. Top-N equal weight
  3. Top-N optimal weight (mean-variance, scipy)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def spy_cumulative_return(returns_df: pd.DataFrame,
                           start: str,
                           end: str) -> float:
    """Cumulative return of SPY between two dates."""
    r = returns_df.loc[start:end, "SPY"].dropna()
    return float((1 + r).prod() - 1)


def equal_weight_return(returns_df: pd.DataFrame,
                         tickers: list,
                         start: str,
                         end: str) -> float:
    """Equal-weight portfolio cumulative return."""
    cols = [t for t in tickers if t in returns_df.columns]
    r = returns_df.loc[start:end, cols].dropna(how="all")
    port = r.mean(axis=1)
    return float((1 + port).prod() - 1)


def optimal_weight_return(returns_df: pd.DataFrame,
                           tickers: list,
                           start: str,
                           end: str) -> float:
    """
    Max-Sharpe (mean-variance) portfolio using in-sample data.

    Note: this baseline has lookahead bias (uses returns from the
    test window itself). It serves as an upper bound, not a fair
    comparison.
    """
    cols = [t for t in tickers if t in returns_df.columns]
    r = returns_df.loc[start:end, cols].dropna(how="all")
    if r.empty or len(cols) < 2:
        return 0.0

    mu  = r.mean().values * 252
    cov = r.cov().values  * 252
    n   = len(cols)

    def neg_sharpe(w):
        w  = np.array(w)
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w + 1e-12)
        return -(ret / vol)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = [(0, 1)] * n
    x0          = np.ones(n) / n

    res = minimize(neg_sharpe, x0, method="SLSQP",
                   bounds=bounds, constraints=constraints)
    if not res.success:
        return equal_weight_return(returns_df, tickers, start, end)

    w_opt = res.x
    port  = r.values @ w_opt
    return float((1 + port).prod() - 1)


def qaoa_portfolio_return(returns_df: pd.DataFrame,
                           bitstring: str,
                           tickers: list,
                           start: str,
                           end: str) -> float:
    """
    Cumulative return of the QAOA-selected portfolio.

    Stocks are selected where bitstring[i] == '1'.
    """
    selected = [tickers[i] for i, b in enumerate(bitstring)
                if b == "1" and i < len(tickers) and tickers[i] in returns_df.columns]
    if not selected:
        return 0.0
    r    = returns_df.loc[start:end, selected].dropna(how="all")
    port = r.mean(axis=1)
    return float((1 + port).prod() - 1)


def summarise_window(returns_df: pd.DataFrame,
                      tickers: list,
                      bitstring: str,
                      start: str,
                      end: str,
                      include_spy: bool = True) -> dict:
    """Return all baseline metrics for a single quarter window."""
    result = {
        "qaoa":          qaoa_portfolio_return(returns_df, bitstring, tickers, start, end),
        "equal_weight":  equal_weight_return(returns_df, tickers, start, end),
        "optimal_weight": optimal_weight_return(returns_df, tickers, start, end),
    }
    if include_spy and "SPY" in returns_df.columns:
        result["spy"] = spy_cumulative_return(returns_df, start, end)
    return result
