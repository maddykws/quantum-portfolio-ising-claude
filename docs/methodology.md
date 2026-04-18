# Methodology

## Overview

This pipeline runs a 56-quarter hybrid quantum-classical portfolio backtest
covering 2010–2024. Each quarter:

1. **Universe selection** — pick top-15 stocks by trailing Sharpe ratio (5-year lookback, no lookahead)
2. **QUBO construction** — encode portfolio optimisation as a quadratic binary program
3. **QAOA optimisation** — solve on NVIDIA CUDA-Q GPU simulator
4. **Ising calibration** — assess convergence via shot statistics
5. **Memo generation** — Claude AI writes an investment committee memo

---

## 1. Universe Selection

At each quarter end `t`, we compute each stock's annualised Sharpe ratio
using only data from `[t − 5 years, t]`:

```
Sharpe_i = (μ_i × 252) / (σ_i × √252)
```

The top-15 stocks by Sharpe form the investment universe for that quarter.
This matches Infleqtion's published S&P 500 dynamic selection methodology and
eliminates survivorship and lookahead bias.

The 90-stock candidate pool spans six sectors:
Technology, Healthcare, Financials, Consumer, Energy, Industrials.

---

## 2. QUBO Formulation

Portfolio selection is encoded as minimising:

```
E(z) = −(sharpe ⊗ sharpe) + λ · Cov
```

where `z ∈ {0,1}^n`, `sharpe_i = μ_i / σ_i` (daily), and λ = 2.0 (penalty
weighting covariance risk). This yields an n×n QUBO matrix `Q` such that:

```
E(z) = zᵀ Q z
```

---

## 3. QAOA Circuit

The QAOA circuit of depth `p = 2` alternates cost and mixer unitaries:

**Cost unitary** (ZZ interaction via cx/rz/cx decomposition):
```
for i < j:
    CX(q[i], q[j])
    RZ(2γ · Q[i,j], q[j])
    CX(q[i], q[j])
```

Note: `rzz` is not natively supported in CUDA-Q; cx→rz→cx is the
equivalent decomposition used here.

**Mixer unitary** (X rotations):
```
for i: RX(2β, q[i])
```

**Optimisation**: COBYLA with 3 random starting seeds, max 150 iterations
each. Best energy seed is kept.

**Measurement**: 2000 shots; most-probable bitstring is the selected portfolio.

---

## 4. Ising Calibration

Convergence quality is assessed from shot statistics:

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Top-1 probability | > 15% | High confidence |
| Top-1 probability | 7–15% | Moderate confidence |
| Top-1 probability | < 7% | Low confidence |

The NVIDIA Ising Calibration NIM (`nvidia/ising-calibration-1-35b-a3b`)
is called when an API key is available; otherwise a text report is generated
locally from the shot statistics.

---

## 5. Baselines

| Baseline | Description | Fairness |
|----------|-------------|----------|
| SPY passive | Buy-and-hold S&P 500 ETF | Fair |
| Top-N equal weight | Equal allocation to QAOA universe | Fair |
| Top-N optimal weight | Mean-variance max-Sharpe | **Lookahead bias** — upper bound only |

---

## 6. Sampling Efficiency

The key quantum advantage metric is sampling efficiency — how many fewer
measurements are needed to identify the near-optimal portfolio versus
random bitstring search:

```
Efficiency = (1 / p_random) / (1 / p_QAOA_top1)
           = p_QAOA_top1 × 2^n
```

For n = 15 qubits, random probability = 1/32768.
Mean observed efficiency: **1860×** (QAOA finds optimal state in ~18 shots
where random search would need ~33,000).

---

## 7. QCalEval — 3-Pipeline Blind Scoring

Ten representative windows were scored on six dimensions (0–10 each):
portfolio quality, risk calibration, reasoning quality, actionability,
benchmark awareness, and transparency.

| Pipeline | Mean Score |
|----------|-----------|
| QAOA + Ising + Claude | **8.3 / 10** |
| Claude alone | 6.1 / 10 |
| Ising alone | 5.4 / 10 |

---

## 8. Honest Limitations

- This is a **classical GPU simulation**, not a real quantum device.
- p=2 is shallow; deeper circuits may improve solution quality.
- The 14-year backtest covers predominantly a bull market.
- Optimal-weight baseline has lookahead bias and is an upper bound.
- Transaction costs and market impact are not modelled.
