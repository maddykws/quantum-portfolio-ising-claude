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
| QPU validation | **Rigetti Cepheus-1-108Q · 108-qubit** |

Validated across **56 quarterly S&P 500 portfolio constructions**
spanning a **14-year backtest (2010–2024)**.

---

## What this is

I built a pipeline that uses quantum computing to select stock portfolios,
runs on NVIDIA GPU hardware, and explains its decisions in plain English
using Claude AI. The whole thing costs about $3 to run on Google Colab.

The idea is straightforward: picking the best 10 stocks from a universe
of 25 candidates means evaluating roughly 3 million possible combinations.
Classical computers do this by checking each combination. Quantum computers
do something more interesting — they encode the problem into a circuit that
naturally concentrates probability on high-quality portfolios, then measure
which ones come up most often.

The surprise was how efficient this is. The quantum circuit finds a
near-optimal portfolio in a median of 3 measurements. Random sampling would
need 1.4 million attempts to match that. That gap — the sampling efficiency
— is the strongest result in this project.

---

## The pipeline

```
Step 1 — Data
Download 14 years of S&P 500 daily returns (yfinance)
Select top-25 stocks by Sharpe ratio each quarter
No lookahead — only data available at that date

Step 2 — Quantum optimisation
Map portfolio selection to QUBO binary optimisation
Run QAOA circuit on NVIDIA L4 GPU via CUDA-Q
2000 measurement shots, 3 random seeds

Step 3 — Ensemble selection
Take the 10 most-measured quantum states
Run classical weight optimisation on each candidate
Pick the portfolio with highest Sharpe ratio

Step 4 — Narration
Send measurement distribution to NVIDIA Ising Calibration
Generate investment memo with Claude AI
Ising handles circuit quality. Claude handles finance.
```

---

## The honest numbers

**What works well:**
- 373× GPU speedup over CPU simulation at 20 qubits
- +3.1% median Sharpe improvement vs naive top-stock selection (75% win rate)
- 100% win rate vs SPY passive index across all 56 quarters
- 1.4 million times more efficient than random portfolio search

**What does not work yet:**
- Against classically-optimised top-N selection the median gap is -4.8%.
  The quantum circuit finds good stock combinations but has not surpassed
  the best classical weight allocation at this circuit depth.
- QPU validation on Rigetti Cepheus showed near-flat measurement distributions.
  Hardware noise at 24-qubit scale prevents meaningful quantum signal on
  current superconducting NISQ hardware. This gap motivates fault-tolerant
  quantum computing.

I am reporting both because hiding the -4.8% result would be dishonest
and because the QPU noise finding is itself interesting — it quantifies
exactly how much better the hardware needs to get.

---

## The Ising + Claude finding

I ran a structured evaluation comparing three pipelines: NVIDIA Ising
Calibration alone, Claude alone, and the two combined. The result was
clean and not what I expected.

Ising scored 3.9/5 on circuit quality assessment. It correctly
identified measurement concentration and convergence quality. It scored
1.0/5 on financial insight because it is not designed for that.

Claude scored 3.9/5 on financial insight and clarity. It translated
quantum results into actionable investment language. It scored 1.0/5
on circuit quality because it had no access to the measurement data.

The combined pipeline scored highest on completeness — the one dimension
that requires both quantum assessment and financial translation.

The finding: neither model alone is sufficient for making quantum results
useful to practitioners. Ising and Claude serve genuinely non-overlapping
roles.

---

## QPU validation

I ran the quantum circuit on real hardware via Amazon Braket:

- **Rigetti Cepheus-1-108Q** (108-qubit superconducting QPU)
- Two runs: 10-qubit p=2 and 24-qubit p=1
- Total cost: $1.44

The result was instructive. GPU simulation shows 15-20% probability
concentration on the best portfolio states. The real QPU showed 0.1%
— essentially random. This is the NISQ noise problem made concrete:
at 24-qubit depth with 0.5% error per two-qubit gate, only about 25%
of circuit executions are noise-free.

The full v3 pipeline at p=2 depth generates roughly 22,000 gates after
transpilation — exceeding Rigetti's 20,000 gate limit. This quantifies
the quantum volume requirement: we need approximately 4-36× more reliable
gate execution depth than current NISQ hardware provides.

---

## Quick start

### One-click Colab (recommended)

Click the badge at the top. The notebook is pre-configured for NVIDIA
L4 GPU. Total cost is approximately $3 in Colab compute units.

You need two API keys:
- **NVIDIA NIM:** free at build.nvidia.com (1000 credits included)
- **Anthropic:** console.anthropic.com

### Local setup

```bash
git clone https://github.com/maddykws/quantum-portfolio-ising-claude
cd quantum-portfolio-ising-claude
pip install -r requirements.txt

export NVIDIA_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"

jupyter notebook notebooks/full_pipeline.ipynb
```

---

## Stack

| Component | What it does |
|-----------|-------------|
| NVIDIA CUDA-Q | Quantum circuit simulation on GPU |
| NVIDIA Ising Calibration | Assesses circuit measurement quality |
| Claude AI (Anthropic) | Writes investment committee memos |
| yfinance | Downloads S&P 500 historical prices |
| scipy | Classical mean-variance weight optimisation |
| Amazon Braket | Access to Rigetti superconducting QPU |
| Google Colab L4 | GPU compute (Ada Lovelace, 22.5GB VRAM) |

---

## Methodology notes

**No survivorship bias.** The stock universe at each quarter uses only
companies that existed and had sufficient trading history at that point.
No future knowledge.

**Dynamic universe.** At each quarterly rebalancing point the top-25
stocks are re-selected based on the most recent 5 years of data.
The same stocks do not appear in every window.

**Fair comparison.** The top-N baseline uses the same stocks and the
same equal-weight assumption as the quantum result when measuring the
+3.1% improvement. The harder baseline (top-N with optimal weights)
shows -4.8% and is also reported.

**Gate decomposition.** The QAOA ZZ interaction is implemented via
cx-rz-cx rather than the rzz gate, making it compatible with all
CUDA-Q versions without modification.

---

## File structure

```
notebooks/
  full_pipeline.ipynb        Main pipeline — runs on Colab L4
qpu_validation/
  rigetti_cepheus_validation.ipynb   Real QPU runs
src/
  qubo.py                    QUBO matrix construction
  qaoa.py                    CUDA-Q kernel and optimisation
  baselines.py               Classical comparison baselines
  ising_calibration.py       NVIDIA Ising Calibration calls
  claude_narrator.py         Claude AI narration layer
  utils.py                   Ensemble selection and utilities
results/
  summary.json               All results in one place
figures/                     Publication charts
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

This project builds on NVIDIA CUDA-Q, NVIDIA Ising Calibration,
Anthropic Claude, Amazon Braket, and Google Colab. The portfolio
optimisation methodology follows Infleqtion's published Q-CHOP
approach. The QCalEval evaluation framework was developed for this
project.

---

## License

MIT — see LICENSE.
