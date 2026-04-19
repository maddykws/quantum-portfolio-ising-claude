"""
Claude AI narration layer.

Generates plain-English investment committee memos
combining quantum portfolio results with Ising
Calibration circuit quality assessments.

QCalEval finding: Ising scores 3.9/5 on circuit
quality. Claude scores 3.9/5 on financial insight.
Combined achieves highest completeness score.
Neither alone is sufficient.
"""

import os
import anthropic


def generate_investment_memo(
    portfolio: list,
    q_sharpe: float,
    spy_sharpe: float,
    q_vs_spy: float,
    window: str,
    ann_returns: dict,
    ising_report: str,
    shots_needed: int,
    eff_gain: float,
    model: str = "claude-haiku-4-5-20251001"
) -> str:
    """
    Generate investment committee memo.

    Combines quantum portfolio result with Ising
    Calibration circuit quality assessment into
    actionable plain-English narration.

    Args:
        portfolio: Selected ticker symbols
        q_sharpe: Quantum portfolio Sharpe ratio
        spy_sharpe: SPY passive Sharpe ratio
        q_vs_spy: Percentage improvement vs SPY
        window: Time window label (e.g. Q1-2018)
        ann_returns: {ticker: annualised_return_%}
        ising_report: Ising Calibration text
        shots_needed: Measurements to near-optimal
        eff_gain: Sampling efficiency vs random
        model: Claude model to use

    Returns:
        Investment committee memo as plain text
    """
    client = anthropic.Anthropic(
        api_key=os.environ.get(
            "ANTHROPIC_API_KEY"))

    ret_str = "\n".join(
        f"  {t}: {r:.1f}% annualised"
        for t, r in ann_returns.items())

    prompt = f"""
QUANTUM PORTFOLIO RESULT
Window:     {window}
Portfolio:  {", ".join(portfolio)}

PERFORMANCE
Quantum Sharpe:  {q_sharpe:.4f}
SPY Sharpe:      {spy_sharpe:.4f}
Improvement:     {q_vs_spy:+.1f}% vs SPY

SAMPLING EFFICIENCY
Shots to near-optimal: {shots_needed}
vs random search:      {eff_gain:.0f}x faster

INDIVIDUAL RETURNS
{ret_str}

ISING CALIBRATION REPORT
{ising_report}
"""

    response = client.messages.create(
        model=model,
        max_tokens=600,
        system=(
            "Senior quantitative analyst. "
            "Write an investment committee memo "
            "from a quantum portfolio result. "
            "Address: why these stocks were "
            "selected, what the Sharpe improvement "
            "means in practical terms, what the "
            "calibration report implies for "
            "reliability, and the sampling "
            "efficiency finding. "
            "Plain English. Cite specific numbers. "
            "4 paragraphs maximum."
        ),
        messages=[{
            "role":    "user",
            "content": prompt
        }]
    )

    return response.content[0].text
