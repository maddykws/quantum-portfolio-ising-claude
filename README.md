# Quantum Portfolio Optimisation with NVIDIA CUDA-Q
## GPU-Accelerated QAOA · Ising Calibration · Claude AI · Real QPU Validation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maddykws/quantum-portfolio-ising-claude/blob/main/notebooks/full_pipeline.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NVIDIA Blog](https://img.shields.io/badge/NVIDIA-Spotlight-76B900)](https://developer.nvidia.com/blog)

---

## What this does

Selecting the best 10 stocks from 25 candidates means evaluating 3.2 million possible
combinations. Classical greedy approaches — rank by Sharpe, pick the top N — are fast
but miss combinations where stocks with moderate individual returns are exceptional
together due to low correlation.

This project uses QAOA quantum circuits on an NVIDIA L4 GPU to search that combination
space, then selects and weights the best result classically. The full pipeline runs
on Google Colab for approximately $3.

**The GPU is what makes this practical.** At 20 qubits, CUDA-Q on the L4 is 373×
faster than CPU simulation. Without that speedup, the 56-window benchmark would take
over 15 hours per run instead of under 4 minutes.

---

## Numbers

| Metric | Value | How measured |
|--------|-------|-------------|
| GPU speedup | **373×** | Wall-clock: CUDA-Q nvidia vs cpu target, N=20 qubits, identical code |
| vs SPY passive | **+101.2% median · 100% win rate** | 56 quarterly windows, 2010–2024 |
| vs top-N equal weight | **+3.1% median · 75% win rate** | Same stocks, equal weight baseline |
| vs top-N optimal weight | **−4.8% median · 2% win rate** | Hard classical baseline — disclosed |
| Sampling efficiency | **1.4 million ×** | Median 3 shots vs 1.4M random draws |
| QPU validation cost | **$1.44 total** | Two runs, Rigetti Cepheus-1-108Q |

---

## The GPU story

At N=20 qubits the quantum state vector has 2^20 = 1,048,576 complex amplitudes.
Every gate updates all of them. On CPU this is sequential — on GPU it is massively
parallel.

```
N=10 qubits:  CPU ~0.08s   GPU ~0.08s    speedup: ~1×
N=15 qubits:  CPU ~0.8s    GPU ~0.09s    speedup: 9×
N=20 qubits:  CPU ~41s     GPU ~0.11s    speedup: 373×
```

The same CUDA-Q kernel code switches between CPU, GPU, and real QPU with one line:

```python
cudaq.set_target("nvidia")   # L4 GPU — 373× speedup
cudaq.set_target("cpu")      # CPU baseline
cudaq.set_target("quantinuum.h2-1")  # real QPU
```

This portability is the core value of CUDA-Q. Develop on GPU. Validate on QPU.
No code changes required.

---

## Pipeline

```
Stage 1 — Data (no lookahead)
  90-stock S&P 500 universe
  Select top-25 by Sharpe ratio using only prior 5 years of data
  56 quarterly construction points: Q1 2010 → Q4 2023

Stage 2 — QUBO formulation
  minimise: −sharpeᵀ × sharpe + λ × covariance   (λ = 2.0)
  Binary variable: 1 = stock selected, 0 = not selected

Stage 3 — QAOA on NVIDIA L4 GPU via CUDA-Q
  p=2 depth, 3 random seeds, 150 COBYLA iterations each
  200 shots per cost function evaluation
  2000 shots for final measurement
  ZZ interaction: cx-rz-cx decomposition

Stage 4 — Ensemble selection + classical weights
  Extract top-10 most-measured quantum states
  Run scipy SLSQP weight optimisation on each (5–40% bounds)
  Select portfolio with highest optimally-weighted Sharpe

Stage 5 — Assessment and narration
  Send QAOA measurement chart to NVIDIA Ising Calibration
  Generate investment memo with Claude AI
```

---

## Honest results

**The +3.1% result** uses the full v3 configuration: 25-stock universe, top-10
ensemble selection, classical weight optimisation. Earlier runs with 15 stocks
and equal weighting showed −5.7% vs top-N equal weight. Both the expanded
universe and the ensemble+weighting step are necessary.

**The −4.8% result** against classically-optimised top-N selection is disclosed
prominently. QAOA combination search at p=2 is competitive with naive selection
but does not yet consistently beat the best classical approach. Deeper circuits,
better optimisers, and lower-noise QPU hardware are the path forward.

**The 1.4 million × sampling efficiency** is measured, not estimated. The QAOA
circuit finds a near-optimal portfolio (within 1% of best Sharpe) in a median
of 3 shots across 56 windows. Random sampling would need 1.4 million draws
on average.

---

## QPU validation — hybrid workflow, not quantum advantage

We are not claiming quantum advantage over classical methods. We are validating
that a hybrid quantum-classical workflow developed on GPU simulation transfers
to real quantum hardware — which is the practical path in 2026.

**Rigetti Cepheus-1-108Q via Amazon Braket. $1.44 total.**

- Run A: 10 qubits, p=2 — top-state concentration 1.1%. Circuit executes.
- Run B: 24 qubits, p=1 — top-state concentration 0.1%. Noise dominates at this scale.

GPU simulation shows 15–20% top-state concentration. The real QPU shows 0.1%.
That gap is the NISQ noise cost: with ~0.5% error per two-qubit gate and 270 gates,
only ~26% of shots are fully noise-free.

The full v3 pipeline at p=2 generates ~22,000 transpiled gates on Rigetti —
exceeding the 20,000 gate limit. This is a concrete hardware requirement, not
a vague limitation. IonQ trapped ion hardware (~0.2% error rate) would give
58% fidelity at this depth.

---

## Ising Calibration — domain extension experiment

NVIDIA Ising Calibration launched April 16, 2026. It is a 35B parameter
vision-language model designed to automate QPU hardware calibration.

We applied it four days after launch to assess QAOA measurement distributions
in portfolio optimisation circuits — outside its intended QPU calibration domain.
We also evaluated it using NVIDIA's QCalEval benchmark scoring framework,
applied to portfolio circuit outputs.

| Pipeline | Circuit quality | Financial insight | Overall |
|----------|----------------|-------------------|---------|
| Ising Calibration alone | 3.9 / 5 | 1.0 / 5 | 2.42 / 5 |
| Claude AI alone | 1.0 / 5 | 3.9 / 5 | 2.62 / 5 |
| Ising + Claude | 3.9 / 5 | 4.0 / 5 | 2.70 / 5 |

Ising correctly flagged low measurement concentration as poor QAOA convergence.
Claude provided financial translation that Ising does not attempt. Combined they
cover both dimensions. Neither alone is sufficient.

This is an early exploration of Ising Calibration in a new domain — not a claim
about its intended QPU calibration performance.

---

## Quick start

Click the Colab badge above. You need two free API keys:

```bash
# NVIDIA NIM — free at build.nvidia.com (1000 credits)
export NVIDIA_API_KEY="nvapi-..."

# Anthropic — free tier at console.anthropic.com
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or run locally:

```bash
git clone https://github.com/maddykws/quantum-portfolio-ising-claude
cd quantum-portfolio-ising-claude
pip install -r requirements.txt
jupyter notebook notebooks/full_pipeline.ipynb
```

---

## Stack

| Component | Role |
|-----------|------|
| NVIDIA CUDA-Q | Quantum circuit simulation on L4 GPU — the core speedup |
| NVIDIA Ising Calibration (NIM) | Interprets QAOA measurement distribution quality |
| Claude AI (Anthropic) | Investment committee memo narration |
| yfinance | S&P 500 daily returns |
| scipy SLSQP | Classical weight optimisation (5–40% bounds) |
| Amazon Braket | Real QPU access — Rigetti Cepheus-1-108Q |
| Google Colab L4 | Ada Lovelace GPU, 22.5GB VRAM |

---

## Methodology

**Universe construction:** Top-25 stocks by individual Sharpe ratio at each
quarterly point. Uses only data available at that date. 90-stock fixed universe.
No survivorship bias correction — stocks delisted during the test period are excluded.

**QUBO:** Standard portfolio QUBO following Infleqtion Q-CHOP methodology.
Penalty weight λ=2.0 determined empirically.

**QAOA:** p=2 chosen after testing p=1 through p=4. Performance plateaus at p=3
with COBYLA — consistent with barren plateau phenomenon. p=2 is the optimal depth
for this optimiser.

**Baseline comparison:** Three baselines — SPY, top-N equal weight, top-N optimal
weight — all constructed using only data available at the quarterly date.
Top-N size matches the quantum portfolio size window-by-window.

**Ising Calibration note:** Designed for QPU calibration. Applied here to portfolio
QAOA assessment. Domain extension — not validated production use.

---

## File structure

```
notebooks/
  full_pipeline.ipynb              Main pipeline — Colab L4 GPU
qpu_validation/
  rigetti_cepheus_validation.ipynb Real QPU runs — Amazon Braket
src/
  qubo.py                          QUBO matrix construction
  qaoa.py                          CUDA-Q kernel + optimisation
  baselines.py                     Three classical comparison baselines
  ising_calibration.py             Ising Calibration NIM integration
  claude_narrator.py               Claude AI narration layer
  utils.py                         Ensemble selection + sampling efficiency
results/
  summary.json                     All results across all 56 windows
figures/                           Publication charts (add from Google Drive)
```

---

## Citation

```bibtex
@misc{quantum-portfolio-2026,
  title  = {Hybrid Quantum-Classical Portfolio Optimisation
            with NVIDIA Ising Calibration and Claude AI},
  year   = {2026},
  url    = {https://github.com/maddykws/quantum-portfolio-ising-claude},
  note   = {56-window S&P 500 benchmark, NVIDIA L4 GPU + Rigetti Cepheus QPU}
}
```

---

## Acknowledgements

Built on NVIDIA CUDA-Q, NVIDIA Ising Calibration, Anthropic Claude, Amazon Braket,
and Google Colab. Portfolio QUBO methodology follows Infleqtion's published Q-CHOP
approach. QPU evaluation applies NVIDIA's QCalEval scoring framework to a new domain.

MIT License — see LICENSE.
