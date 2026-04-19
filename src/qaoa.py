"""
QAOA circuit implementation using NVIDIA CUDA-Q.

Implements p-layer QAOA with cx/rz/cx decomposition
for the ZZ interaction term. This decomposition is
compatible with all CUDA-Q versions and targets.
"""

import cudaq
import numpy as np
from scipy.optimize import minimize
from typing import Tuple


@cudaq.kernel
def qaoa_kernel(n: int,
                params: list[float],
                Q: list[float]):
    """
    QAOA kernel for portfolio QUBO optimisation.

    Architecture: p layers of cost + mixer unitaries.
    ZZ interaction via cx-rz-cx decomposition.

    Args:
        n: Number of qubits (= number of stocks)
        params: [gamma_1, beta_1, ..., gamma_p, beta_p]
        Q: Flattened n x n QUBO matrix
    """
    q = cudaq.qvector(n)
    h(q)  # Initial superposition

    p = len(params) // 2

    for layer in range(p):
        gamma = params[2 * layer]
        beta  = params[2 * layer + 1]

        # Cost unitary — ZZ interactions
        for i in range(n):
            for j in range(i + 1, n):
                cx(q[i], q[j])
                rz(2.0 * gamma * Q[i * n + j], q[j])
                cx(q[i], q[j])

        # Mixer unitary — X rotations
        for i in range(n):
            rx(2.0 * beta, q[i])

    mz(q)


def to_counts(result) -> dict:
    """Convert CUDA-Q SampleResult to plain dict."""
    return {k: result.count(k) for k in result}


def optimise_qaoa(Q_flat: list,
                   n_stocks: int,
                   p: int = 2,
                   n_seeds: int = 3,
                   shots_count: int = 2000,
                   maxiter: int = 150) -> Tuple[dict, list, float]:
    """
    Run QAOA optimisation with multiple random seeds.

    Uses COBYLA gradient-free optimisation.
    Returns best result across all seeds.

    Args:
        Q_flat: Flattened QUBO matrix
        n_stocks: Number of stocks (qubits)
        p: QAOA circuit depth
        n_seeds: Number of random starting points
        shots_count: Measurement shots for final sample
        maxiter: COBYLA maximum iterations

    Returns:
        (best_counts, best_params, best_energy)
    """
    n_params    = p * 2
    Q_mat       = np.array(Q_flat).reshape(
        n_stocks, n_stocks)

    def cost(params):
        counts = to_counts(cudaq.sample(
            qaoa_kernel, n_stocks,
            params.tolist(), Q_flat,
            shots_count=200))
        total  = sum(counts.values())
        if total == 0:
            return 0.0
        energy = 0.0
        for bits, cnt in counts.items():
            z = np.array(
                [1 - 2 * int(b) for b in bits])
            energy += cnt * float(z @ Q_mat @ z)
        return energy / total

    best_energy = np.inf
    best_counts = {}
    best_params = []

    for seed in range(n_seeds):
        np.random.seed(seed * 13)
        x0  = np.random.uniform(0, np.pi, n_params)
        res = minimize(
            cost, x0, method="COBYLA",
            options={"maxiter": maxiter,
                     "rhobeg": 0.5})

        cnts = to_counts(cudaq.sample(
            qaoa_kernel, n_stocks,
            res.x.tolist(), Q_flat,
            shots_count=shots_count))

        if res.fun < best_energy:
            best_energy = res.fun
            best_counts = cnts
            best_params = res.x.tolist()

    return best_counts, best_params, best_energy
