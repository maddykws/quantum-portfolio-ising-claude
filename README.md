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

Step 4 — Assessment and narration
Send QAOA measurement distribution to NVIDIA Ising Calibration
Generate investment memo with Claude AI
Ising assesses circuit output quality. Claude handles financial translation.
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

NVIDIA Ising Calibration launched on April 16, 2026 — four days before this
project was submitted. It is a 35B parameter vision-language model designed
to interpret quantum processor calibration outputs and automate QPU tuning.

I applied it to a different domain: assessing QAOA measurement distribution
quality in portfolio optimisation circuits. This is outside its intended QPU
calibration use case, but the model reads quantum experiment charts regardless
of whether the experiment is a qubit calibration run or a portfolio QAOA circuit.

I then evaluated three pipeline configurations using NVIDIA's QCalEval benchmark
— the world's first evaluation framework for quantum calibration models, released
alongside Ising Calibration. I applied QCalEval's six scoring dimensions to
portfolio circuit outputs instead of QPU calibration outputs:

| Pipeline | Circuit quality | Financial insight | Overall |
|----------|----------------|-------------------|---------|
| Ising Calibration alone | 3.9 / 5 | 1.0 / 5 | 2.42 / 5 |
| Claude alone | 1.0 / 5 | 3.9 / 5 | 2.62 / 5 |
| Ising + Claude combined | — | 4.0 / 5 | 2.70 / 5 |

The finding: in this new application domain, Ising Calibration and Claude AI
serve genuinely complementary roles. Ising reads the circuit output and assesses
measurement quality. Claude translates results into financial language.
Neither alone is sufficient for making quantum portfolio results actionable.

This is an early exploration of Ising Calibration beyond its intended QPU
calibration domain — not a claim about its intended use case.

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
| NVIDIA Ising Calibration (NIM) | Interprets QAOA measurement distribution quality |
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

**Ising Calibration domain note.** NVIDIA Ising Calibration is designed
for QPU calibration — tuning quantum processor noise. This project applies
it to QAOA circuit output assessment in a financial application, which is
outside its original design domain. Results should be interpreted as an
early domain extension experiment rather than a validated production use case.

---

## File structure

```
notebooks/
  full_pipeline.ipynb        Main pipeline — runs on Colab L4
qpu_validation/
  rigetti_cepheus_validation.ipynb   Real QPU runs on Rigetti Cepheus-1-108Q
src/
  qubo.py                    QUBO matrix construction
  qaoa.py                    CUDA-Q kernel and optimisation
  baselines.py               Classical comparison baselines
  ising_calibration.py       NVIDIA Ising Calibration NIM integration
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

Built on NVIDIA CUDA-Q, NVIDIA Ising Calibration, Anthropic Claude,
Amazon Braket, and Google Colab. Portfolio optimisation methodology
follows Infleqtion's published Q-CHOP approach. Pipeline evaluation
uses NVIDIA's QCalEval benchmark framework, applied here to a new
domain beyond its original QPU calibration design.

---

## License

MIT — see LICENSE.
