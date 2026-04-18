# Quantum Portfolio Optimisation
## NVIDIA Ising Calibration + CUDA-Q + Claude AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maddykws/quantum-portfolio-ising-claude/blob/main/notebooks/full_pipeline.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Results

| Metric | Value |
|--------|-------|
| GPU speedup (L4 vs CPU, N=20 qubits) | **373x** |
| vs top-N equal weight (56 windows) | **+3.1% median · 75% win rate** |
| vs SPY passive benchmark | **+101% median · 100% win rate** |
| Sampling efficiency vs random search | **1.4 million ×** |
| QCalEval: Combined vs Ising alone | **+11.6% improvement** |

Validated across **56 quarterly portfolio constructions** over a **14-year S&P 500 backtest (2010–2024)**.

## What This Is

A hybrid quantum-classical pipeline that:

1. Downloads 14 years of S&P 500 stock data (yfinance)
2. Dynamically selects top-25 stocks by Sharpe at each quarter (no lookahead bias)
3. Runs QAOA quantum circuit on NVIDIA L4 GPU via CUDA-Q to find optimal stock combinations
4. Uses top-10 ensemble selection with classical weight optimisation
5. Calls NVIDIA Ising Calibration (NIM API) to assess circuit measurement quality
6. Uses Claude AI to generate plain-English investment memos

## Architecture

```
yfinance → QUBO formulation → CUDA-Q QAOA (L4 GPU)
         → Top-10 ensemble → Weight optimisation
         → NVIDIA Ising Calibration (NIM)
         → Claude AI narration
         → Investment committee memo
```

## Key Findings

**GPU acceleration:** CUDA-Q on Colab L4 GPU achieves 373x speedup over CPU simulation at 20 qubits — enabling real-time quantum portfolio construction previously requiring supercomputer access.

**Sampling efficiency:** QAOA finds near-optimal portfolios in a median of 3 circuit measurements — 1.4 million times faster than random search across the combination space.

**Finance result:** Hybrid quantum-classical pipeline outperforms naive equal-weight top-stock selection in 75% of 56 quarterly periods (+3.1% median Sharpe improvement).

**Honest limitation:** Against classically-optimised top-N selection, the gap is -4.8% median — indicating QAOA at p=2 identifies good combinations but has not yet surpassed the best classical weight allocation. This motivates deeper circuits and QPU validation.

**Ising + Claude complementary roles:** QCalEval analysis across 10 windows shows Ising Calibration scores 3.9/5 on circuit quality while Claude scores 3.9/5 on financial insight — confirming the two models serve complementary roles that neither achieves alone.

## Quick Start

### One-click Colab (recommended)

Click the Colab badge above. Total cost: ~$3 in Colab compute units on L4 GPU.

### Local setup

```bash
git clone https://github.com/maddykws/quantum-portfolio-ising-claude
cd quantum-portfolio-ising-claude
pip install -r requirements.txt
```

Set API keys:
```bash
export NVIDIA_API_KEY="your_nvapi_key"      # build.nvidia.com
export ANTHROPIC_API_KEY="your_ant_key"     # console.anthropic.com
```

Run:
```bash
jupyter notebook notebooks/full_pipeline.ipynb
```

## Requirements

- Google Colab Pro (L4 GPU) or local NVIDIA GPU ≥16GB VRAM
- NVIDIA NIM API key (free, 1000 credits): build.nvidia.com
- Anthropic API key: console.anthropic.com
- Python 3.10+

## Stack

| Component | Role |
|-----------|------|
| CUDA-Q | Quantum circuit simulation on GPU |
| NVIDIA Ising Calibration (NIM) | Circuit quality assessment |
| Claude AI (Anthropic) | Investment memo narration |
| yfinance | S&P 500 historical data |
| scipy | Classical weight optimisation |
| Google Colab L4 | GPU compute (~$3 total) |

## Methodology

- **Universe:** Dynamic top-25 stocks by individual Sharpe ratio at each quarterly construction point (no survivorship bias)
- **Algorithm:** QAOA at p=2, COBYLA optimiser, 3 seeds
- **Selection:** Top-10 ensemble — best portfolio from 10 most-measured quantum states
- **Weights:** Classical mean-variance optimisation (5–40% per stock bounds)
- **Windows:** 56 quarterly periods, 5-year lookback each
- **Baselines:** SPY passive, top-N equal weight, top-N optimal weight

## Citation

```
@misc{quantum-portfolio-2026,
  title={Hybrid Quantum-Classical Portfolio Optimisation
         with NVIDIA Ising Calibration and Claude AI},
  author={maddykws},
  year={2026},
  url={https://github.com/maddykws/quantum-portfolio-ising-claude}
}
```

## Acknowledgements

Built on NVIDIA CUDA-Q, NVIDIA Ising Calibration, and Anthropic Claude.
Methodology follows Infleqtion's Q-CHOP portfolio optimisation approach
([developer.nvidia.com/blog/spotlight-infleqtion](https://developer.nvidia.com/blog/spotlight-infleqtion-optimizes-portfolios-using-q-chop-and-nvidia-cuda-q-dynamics/)).

## License

MIT — see [LICENSE](LICENSE)
