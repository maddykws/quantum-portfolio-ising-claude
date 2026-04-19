# Quantum Portfolio Optimisation
## NVIDIA Ising Calibration + CUDA-Q + Claude AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maddykws/quantum-portfolio-ising-claude/blob/main/notebooks/full_pipeline.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NVIDIA Blog](https://img.shields.io/badge/NVIDIA-Spotlight-76B900)](https://developer.nvidia.com/blog)

---

## Results

| Metric | Value |
|--------|-------|
| GPU speedup (L4 vs CPU, N=20 qubits) | **373×** |
| vs top-N equal weight (56 windows) | **+3.1% median · 75% win rate** |
| vs SPY passive benchmark | **+101.2% median · 100% win rate** |
| Sampling efficiency vs random search | **1.4 million ×** |
| QPU validation | **Rigetti Cepheus-1-108Q · 108-qubit superconducting** |

Validated across **56 quarterly S&P 500 portfolio constructions**
spanning a **14-year backtest (2010–2024)**.

---

## What this is

A hybrid quantum-classical pipeline that selects stock portfolios using
quantum computing on NVIDIA GPU hardware, validated on real quantum
hardware, and narrated in plain English by Claude AI.

**The five-stage pipeline:**

1. Download 14 years of S&P 500 returns with yfinance
2. Select the top-25 stocks by Sharpe ratio each quarter (no lookahead)
3. Run QAOA quantum circuit on NVIDIA L4 GPU via CUDA-Q to find optimal combinations
4. Use top-10 ensemble selection with classical weight optimisation
5. Assess circuit quality with NVIDIA Ising Calibration and generate investment memos with Claude AI

---

## Architecture

```
yfinance (S&P 500 data)
    ↓
QUBO formulation (portfolio → binary optimisation)
    ↓
CUDA-Q QAOA on NVIDIA L4 GPU (quantum combination search)
    ↓
Top-10 ensemble + classical weight optimisation
    ↓
NVIDIA Ising Calibration NIM (circuit quality assessment)
    ↓
Claude AI narration (plain-English investment memo)
```

---

## Key findings

**GPU acceleration**
CUDA-Q on Google Colab L4 GPU achieves 373× speedup over CPU simulation
at 20 qubits — enabling real-time portfolio rebalancing previously
requiring supercomputer access.

**Sampling efficiency**
QAOA finds near-optimal portfolios in a median of 3 circuit measurements —
1.4 million times faster than random search across the combination space.

**Finance result**
The hybrid pipeline (quantum combination search + classical weight
optimisation) outperforms naive equal-weight top-stock selection in
75% of 56 quarterly periods with +3.1% median Sharpe improvement.

**Honest limitation**
Against classically-optimised top-N selection, the median gap is -4.8%.
QAOA identifies good combinations but has not yet surpassed the best
classical weight allocation — motivating deeper circuits and QPU hardware.

**Ising + Claude complementary roles**
QCalEval analysis across 10 windows shows Ising Calibration scores 3.9/5
on circuit quality while Claude scores 3.9/5 on financial insight.
Neither model alone is sufficient — combined they address what
practitioners actually need.

**QPU validation**
QAOA parameters optimised via CUDA-Q GPU simulation were transferred to
Rigetti Cepheus-1-108Q — a 108-qubit superconducting QPU via Amazon Braket.
Hardware noise at current NISQ depth motivates fault-tolerant quantum
hardware for full pipeline QPU validation.

---

## Quick start

### One-click Colab (recommended)

Click the badge above. Runtime is pre-configured for NVIDIA L4 GPU.
Total cost: approximately $3 in Colab compute units.

### Local setup

```bash
git clone https://github.com/maddykws/quantum-portfolio-ising-claude
cd quantum-portfolio-ising-claude
pip install -r requirements.txt
```

Set your API keys:
```bash
export NVIDIA_API_KEY="your_key_here"      # build.nvidia.com — free
export ANTHROPIC_API_KEY="your_key_here"   # console.anthropic.com
```

Run:
```bash
jupyter notebook notebooks/full_pipeline.ipynb
```

---

## Requirements

- Google Colab Pro with L4 GPU, or local NVIDIA GPU with 16GB+ VRAM
- NVIDIA NIM API key (free, 1000 credits): build.nvidia.com
- Anthropic API key: console.anthropic.com
- Python 3.10+

---

## Stack

| Component | Role |
|-----------|------|
| NVIDIA CUDA-Q | Quantum circuit simulation on GPU |
| NVIDIA Ising Calibration (NIM) | Circuit quality assessment |
| Claude AI (Anthropic) | Investment memo narration |
| yfinance | S&P 500 historical data |
| scipy | Classical weight optimisation |
| Amazon Braket | QPU access (Rigetti Cepheus) |
| Google Colab L4 | GPU compute (~$3 total) |

---

## Methodology

**Universe construction**
At each quarterly point we select the top-25 stocks by individual
Sharpe ratio from a 90-stock S&P 500 universe. All selection uses
only data available at that point — no lookahead bias.

**QUBO formulation**
Portfolio selection is mapped to a Quadratic Unconstrained Binary
Optimisation problem encoding both return maximisation and covariance
minimisation (penalty weight λ=2.0).

**QAOA circuit**
Standard QAOA at p=2 depth, 3 random seeds, 150 COBYLA iterations.
ZZ interaction implemented via cx-rz-cx decomposition for CUDA-Q
compatibility across all versions.

**Ensemble selection**
From 2000 measurement shots, the top-10 most-measured states are
extracted. Classical mean-variance optimisation (5–40% weight bounds)
is applied to each candidate. The portfolio with highest
optimally-weighted Sharpe is selected.

**Baselines**
Three comparison baselines:
- SPY passive index (weakest — investor-relevant)
- Top-N equal weight (naive individual Sharpe selection)
- Top-N optimal weight (classically-optimised selection — hardest)

**QPU validation**
Rigetti Cepheus-1-108Q validation used p=1 depth due to gate count
constraints. The full v3 pipeline at p=2 generates ~22,867 gates
after transpilation — exceeding Rigetti's 20,000 gate limit. This
empirically quantifies the quantum volume requirement for full
pipeline QPU validation.

---

## QCalEval scores

Three-pipeline blind scoring across 6 dimensions on 10 windows:

| Pipeline | Circuit quality | Financial insight | Overall |
|----------|----------------|-------------------|---------|
| Ising alone | 3.9 / 5 | 1.0 / 5 | 2.42 / 5 |
| Claude alone | 1.0 / 5 | 3.9 / 5 | 2.62 / 5 |
| Ising + Claude | 1.0 / 5 | 4.0 / 5 | 2.70 / 5 |

Finding: complementary roles confirmed. Neither model alone is
sufficient for making quantum portfolio results actionable.

---

## File structure

```
notebooks/
  full_pipeline.ipynb      Main pipeline — runs on Colab L4
qpu_validation/
  rigetti_cepheus_validation.ipynb   QPU validation on real hardware
src/
  qubo.py                  QUBO matrix construction
  qaoa.py                  CUDA-Q QAOA kernel and optimisation
  baselines.py             Classical portfolio baselines
  ising_calibration.py     NVIDIA Ising Calibration integration
  claude_narrator.py       Claude AI narration layer
  utils.py                 Shared utilities
results/
  summary.json             Aggregate results across all runs
figures/                   Publication charts (add from Google Drive)
```

---

## Citation

```bibtex
@misc{quantum-portfolio-2026,
  title={Hybrid Quantum-Classical Portfolio Optimisation
         with NVIDIA Ising Calibration and Claude AI},
  year={2026},
  url={https://github.com/maddykws/quantum-portfolio-ising-claude}
}
```

---

## Acknowledgements

Built on NVIDIA CUDA-Q, NVIDIA Ising Calibration, Anthropic Claude,
Amazon Braket, and Google Colab. Methodology follows Infleqtion's
published Q-CHOP portfolio optimisation approach.

---

## License

MIT License — see LICENSE file.
