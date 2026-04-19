# GPU-Accelerated Ising Optimisation for Portfolio Selection
## A CUDA-Q QAOA pipeline benchmarked across 56 quarterly periods

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maddykws/quantum-portfolio-ising-claude/blob/main/notebooks/full_pipeline.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NVIDIA Blog](https://img.shields.io/badge/NVIDIA-Spotlight-76B900)](https://developer.nvidia.com/blog)

---

<img width="1783" height="593" alt="figure_gpu_scaling" src="https://github.com/user-attachments/assets/5970d40a-bd50-4af2-939b-63f6e7e48bad" />
<img width="1633" height="734" alt="figure_qcaleval_3pipeline" src="https://github.com/user-attachments/assets/a6e55f4e-b9a9-4c76-99a0-3eeff57573d6" />
<img width="1934" height="740" alt="figure_topn_disclosure" src="https://github.com/user-attachments/assets/49b6a160-7952-42d7-a36f-84240c818d85" />


## The headline

Portfolio selection maps directly to Ising-form optimisation. The QUBO
formulation for choosing N stocks from K candidates **is** an Ising Hamiltonian —
the same mathematical object QAOA circuits are designed to minimise.

The practical barrier has been speed. Simulating a 25-qubit QAOA circuit on
CPU takes minutes per cost function evaluation, making 56-window backtests
computationally prohibitive.

**NVIDIA CUDA-Q on an L4 GPU drops that to 0.11 seconds — a 373× speedup that
turns an impractical research exercise into a reproducible workflow.** That
speedup is what this project is built around.

---

## Results

| Metric | Value | Measurement method |
|--------|-------|-------------------|
| **GPU speedup** | **373×** | Wall-clock: CUDA-Q nvidia vs cpu target, N=20 qubits, identical code |
| **Sampling efficiency** | **1.4 million ×** | Median 3 shots to near-optimal across 56 windows |
| **vs top-N equal weight** | **+3.1% median · 75% wins** | 42/56 windows, 25-stock dynamic universe |
| **vs SPY passive** | **+101.2% median · 100% wins** | All 56 windows |
| **vs top-N optimal weight** | **−4.8% median · 2% wins** | Hard classical baseline — disclosed |
| **QPU validation** | **Rigetti Cepheus 108Q** | $1.44, Amazon Braket |

---

## Benchmark setup (so the numbers are reproducible)

**Dataset**
- Asset universe: K = 90 fixed S&P 500 stocks
- Selection pool per window: N = 25 (top by individual Sharpe, 5-year lookback)
- Portfolio size: 10 stocks per quarter
- Construction dates: 56 quarterly points, Q1 2010 → Q4 2023
- Data source: yfinance adjusted close prices
- Out-of-sample: following quarter only, no lookahead

**Constraints**
- Long-only, cardinality = 10 stocks, weight bounds 5–40% per stock, fully invested
- Dynamic universe: pool reconstructs each quarter from top-25 by Sharpe

**Classical baselines** (same dynamic universe, same dates)
1. **SPY passive** — buy-and-hold S&P 500 ETF (weakest baseline)
2. **Top-N equal weight** — top stocks by Sharpe, equal weight 1/N (naive classical)
3. **Top-N optimal weight** — top stocks by Sharpe + scipy SLSQP mean-variance
   weight optimisation with 5–40% bounds (best classical approach)

**Quantum method**
CUDA-Q QAOA at p=2, COBYLA optimiser, 3 random seeds × 150 iterations, 2000 final
shots. Top-10 bitstrings extracted as candidate portfolios, classical weight
optimisation applied to each, highest-Sharpe selected.

---

## How the GPU speedup works

QAOA state-vector simulation scales as 2^N. At 25 qubits that is 33 million
complex amplitudes every gate must update. Three GPU mechanisms deliver the 373×:

**1. State vector parallelism**
Each of the 2^20 = 1,048,576 amplitudes at N=20 is updated in parallel across
L4 GPU cores. CPU updates them sequentially.

**2. Batched sampling**
Each COBYLA iteration samples the circuit 200 times. CUDA-Q batches these on
GPU, amortising kernel launch overhead. CPU re-simulates from scratch per shot.

**3. Ising-specific gate fusion**
The ZZ interaction (`cx-rz-cx`) requires three state vector passes on CPU.
On GPU the tensor contraction fuses them into a single parallel operation.

```
N=10 qubits:  CPU 0.08s    GPU 0.08s    speedup: ~1×
N=15 qubits:  CPU 0.8s     GPU 0.09s    speedup: 9×
N=20 qubits:  CPU 41s      GPU 0.11s    speedup: 373×
N=25 qubits:  CPU >30 min  GPU ~0.15s   speedup: ~10,000× (projected)
```

The full 56-window backtest runs in under 4 minutes on Colab L4. The same
workload on CPU would take over 15 hours. The GPU does not just speed things
up — it makes the experiment feasible to run at all.

The same CUDA-Q kernel runs unchanged on A100 or H100 (faster still, pushes
qubit count to N=30+) and on real QPU hardware. One target switch, zero code
changes.

```python
cudaq.set_target("nvidia")      # L4 GPU — 373× speedup
cudaq.set_target("cpu")         # CPU baseline
cudaq.set_target("quantinuum")  # real QPU hardware
```

---

## The portfolio problem as an Ising Hamiltonian

Selecting 10 stocks from 25 candidates means evaluating C(25,10) = 3,268,760
combinations. Classical greedy (rank by Sharpe, pick top N) is fast but misses
combinations where moderate-Sharpe stocks form exceptional portfolios due to
low correlation.

The standard portfolio QUBO:

```
minimise:  -sharpe^T @ sharpe  +  lambda * covariance

where:
  sharpe      = vector of individual stock Sharpe ratios
  covariance  = return covariance matrix
  lambda      = 2.0 (risk penalty weight)
  binary var  = 1 if stock selected, 0 otherwise
```

This is an Ising Hamiltonian. QAOA is designed to minimise Ising Hamiltonians
through quantum interference — probability concentrates on bitstrings with
lower energy (better Sharpe-adjusted portfolios).

---

## Results in detail

**1.4 million × sampling efficiency** is the strongest finding. QAOA finds a
near-optimal portfolio (within 1% of best Sharpe) in a median of 3 shots.
Random sampling from 3.2 million combinations needs 1.4 million draws to match.
Measured directly, not estimated.

**+3.1% vs top-N equal weight** uses the full v3 configuration: 25-stock
universe, top-10 ensemble selection, classical weight optimisation. Earlier
p=1 runs on 15 stocks showed −5.7% — both the expanded universe and the
ensemble+weighting pipeline are necessary.

**−4.8% vs top-N optimal weight** is disclosed prominently. QAOA at p=2 is
competitive with naive classical selection but does not yet consistently beat
the best classical approach. The −4.8% gap is honest and motivates deeper
circuits, better optimisers, and lower-noise QPU hardware.

---

## QPU validation — hybrid workflow, not quantum advantage

This project is not a quantum advantage claim. It is a hybrid workflow
validation — the practical path for quantum in finance in 2026.

QAOA parameters optimised on the NVIDIA L4 GPU were transferred unchanged
to Rigetti Cepheus-1-108Q via Amazon Braket. Total cost: $1.44.

| Run | Qubits | Depth | Top state | Finding |
|-----|--------|-------|-----------|---------|
| A | 10 | p=2 | 1.1% | Circuit executes. GPU parameters transfer. |
| B | 24 | p=1 | 0.1% | Full universe. Noise dominates at this scale. |

GPU simulation shows 15–20% top-state concentration. The real QPU shows 0.1%.
That 5.3 percentage point gap is the empirical NISQ noise cost: with ~0.5%
error per two-qubit gate and 270 gates, only ~26% of shots are noise-free.

The full v3 pipeline at p=2 generates ~22,000 transpiled gates on Rigetti —
exceeding the 20,000 gate limit. IonQ trapped-ion hardware at ~0.2% error rate
would give 58% circuit fidelity at the same depth.

---

## Ising Calibration as circuit quality oracle

NVIDIA Ising Calibration launched April 16, 2026 — a 35B parameter
vision-language model designed to automate QPU hardware calibration. We applied
it four days after launch to a **different domain**: assessing QAOA measurement
distributions in portfolio optimisation circuits.

This is a domain extension experiment — not validated production use. Ising
reads quantum experiment charts regardless of whether the experiment is QPU
calibration or a portfolio QAOA circuit.

An LLM-based orchestration layer combines Ising's circuit quality assessment
with financial context to produce investment committee memos. We evaluated this
three ways using NVIDIA's QCalEval benchmark scoring framework applied to
portfolio circuits:

| Configuration | Circuit quality | Financial insight | Overall |
|--------------|----------------|-------------------|---------|
| Ising Calibration alone | 3.9 / 5 | 1.0 / 5 | 2.42 / 5 |
| LLM orchestration alone | 1.0 / 5 | 3.9 / 5 | 2.62 / 5 |
| Ising + LLM combined | 3.9 / 5 | 4.0 / 5 | 2.70 / 5 |

The finding is structural: Ising reads circuit output and flags quality issues.
The LLM layer translates results into practitioner language. These are
genuinely non-overlapping capabilities, even in the new application domain.

---

## Quick start

Click the Colab badge at the top. You need one free API key:

```bash
export NVIDIA_API_KEY="nvapi-..."    # free at build.nvidia.com
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
| **NVIDIA CUDA-Q** | Quantum circuit simulation on L4 GPU — the core 373× speedup |
| **NVIDIA Ising Calibration (NIM)** | Circuit quality oracle for QAOA measurements |
| **LLM orchestration layer** | Financial context + investment memo generation |
| yfinance | S&P 500 daily returns |
| scipy SLSQP | Classical weight optimisation (5–40% bounds) |
| Amazon Braket | Real QPU access — Rigetti Cepheus-1-108Q |
| Google Colab L4 | Ada Lovelace GPU, 22.5GB VRAM |

---

## Limitations

- **Hard classical baseline:** −4.8% vs top-N optimal weight
- **Barren plateau at p=3** with COBYLA — gradient-based optimisers are next step
- **GPU simulation for finance results** — QPU noise prevents meaningful signal at 24Q scale
- **Ising Calibration domain** — designed for QPU calibration, applied here to portfolio circuits
- **No survivorship bias correction** on the 90-stock universe

## What comes next

- IonQ Forte 1 trapped-ion validation (58% fidelity vs Rigetti's 26% at p=2)
- Adam / SPSA gradient-based optimisers to break the p=3 plateau
- Ising correlation study: do circuit quality scores predict portfolio performance?
- Fault-tolerant QPU via Infleqtion Sqale (Superstaq API)

---

## File structure

```
notebooks/
  full_pipeline.ipynb              Main pipeline — L4 GPU, single notebook
qpu_validation/
  rigetti_cepheus_validation.ipynb Real QPU runs — Amazon Braket
src/
  qubo.py                          Portfolio → Ising Hamiltonian
  qaoa.py                          CUDA-Q kernel + COBYLA optimisation
  baselines.py                     Three classical comparison baselines
  ising_calibration.py             Ising Calibration NIM integration
  claude_narrator.py               LLM orchestration layer
  utils.py                         Ensemble selection + sampling efficiency
results/
  summary.json                     All 56-window results
figures/                           Publication charts
```

---

## Citation

```bibtex
@misc{gpu-accelerated-ising-portfolio-2026,
  title  = {GPU-Accelerated Ising Optimisation for Portfolio Selection:
            A CUDA-Q QAOA Pipeline Benchmarked Across 56 Quarterly Periods},
  year   = {2026},
  url    = {https://github.com/maddykws/quantum-portfolio-ising-claude}
}
```

---

## Acknowledgements

Built on NVIDIA CUDA-Q, NVIDIA Ising Calibration, Amazon Braket, and Google
Colab. Portfolio QUBO methodology follows Infleqtion's published Q-CHOP
approach. Pipeline evaluation applies NVIDIA's QCalEval benchmark scoring
framework to a new domain.

MIT License — see LICENSE.
