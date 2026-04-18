"""
Claude AI investment memo narrator.

Uses Anthropic Claude to generate an investment committee memo
from QAOA results, calibration report, and baseline metrics.
"""

import anthropic


MEMO_SYSTEM = """You are a quantitative portfolio manager writing for an investment committee.
Your memos are data-driven, precise, and concise. You highlight quantum advantage
where real, and are honest about limitations. Never fabricate numbers."""


def generate_investment_memo(
    tickers: list,
    bitstring: str,
    quarter: str,
    qaoa_return: float,
    spy_return: float,
    equal_weight_return: float,
    calibration_report: str,
    energy: float,
    params: list,
    api_key: str,
    model: str = "claude-opus-4-7",
) -> str:
    """
    Generate an investment committee memo for a single quarterly window.

    Args:
        tickers:             Stock universe for this quarter
        bitstring:           QAOA best bitstring (1 = selected)
        quarter:             Quarter end date string e.g. '2023-03-31'
        qaoa_return:         QAOA portfolio quarterly return (decimal)
        spy_return:          SPY return over same period (decimal)
        equal_weight_return: Equal-weight top-N return (decimal)
        calibration_report:  Ising calibration text
        energy:              Best QAOA energy
        params:              Optimised γ/β parameters
        api_key:             Anthropic API key
        model:               Claude model ID

    Returns:
        Formatted investment committee memo string
    """
    selected = [tickers[i] for i, b in enumerate(bitstring)
                if b == "1" and i < len(tickers)]

    user_prompt = f"""
Quarter: {quarter}
Universe ({len(tickers)} stocks): {', '.join(tickers)}
QAOA selected ({len(selected)} stocks): {', '.join(selected) if selected else 'None'}

Performance:
  QAOA portfolio:     {qaoa_return:+.2%}
  SPY (passive):      {spy_return:+.2%}
  Equal-weight top-N: {equal_weight_return:+.2%}
  vs SPY alpha:       {qaoa_return - spy_return:+.2%}

QAOA diagnostics:
  Best energy:  {energy:.4f}
  Params (γ,β): {[round(p, 3) for p in params]}

Ising Calibration Summary:
{calibration_report}

Write a professional 3-paragraph investment committee memo that:
1. States the recommended portfolio and quantum optimisation rationale
2. Compares performance vs benchmarks with honest commentary
3. Notes any calibration concerns and whether to act on this quarter's signal
"""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=600,
        system=MEMO_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


def generate_annual_summary(
    quarterly_results: list,
    api_key: str,
    model: str = "claude-opus-4-7",
) -> str:
    """
    Generate an annual portfolio review memo from a list of quarterly dicts.

    Each dict should have keys: quarter, qaoa_return, spy_return,
    equal_weight_return, selected_tickers.
    """
    if not quarterly_results:
        return "No quarterly results available."

    rows = "\n".join(
        f"  {r['quarter']}: QAOA {r['qaoa_return']:+.2%}  "
        f"SPY {r['spy_return']:+.2%}  "
        f"EW {r['equal_weight_return']:+.2%}  "
        f"α {r['qaoa_return'] - r['spy_return']:+.2%}"
        for r in quarterly_results
    )

    wins = sum(1 for r in quarterly_results
               if r["qaoa_return"] > r["spy_return"])
    total = len(quarterly_results)

    user_prompt = f"""
Annual performance summary across {total} quarters:
{rows}

Win rate vs SPY: {wins}/{total} ({wins/total:.0%})

Write a 2-paragraph annual review memo suitable for an investment committee,
covering: aggregate alpha, consistency, and recommendation for the next year.
"""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=400,
        system=MEMO_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text
