"""
QAOA (Quantum Approximate Optimisation Algorithm)
circuit implementation using NVIDIA CUDA-Q.

ZZ interaction uses cx/rz/cx decomposition —
compatible with all CUDA-Q versions.
"""

import cudaq
import numpy as np
from scipy.optimize import minimize


@cudaq.kernel
def qaoa_kernel(n: int, params: list[float], Q: list[float]):
    """
    QAOA kernel for portfolio optimisation QUBO.

    p layers of cost + mixer unitaries.
    ZZ interaction via cx-rz-cx decomposition (RZZ equivalent).

    Args:
        n: Number of qubits (= number of stocks)
        params: [gamma_1, beta_1, ..., gamma_p, beta_p]
        Q: Flattened n×n QUBO matrix
    """
    q = cudaq.qvector(n)
    h(q)
    p = len(params) // 2

    for layer in range(p):
        gamma = params[2 * layer]
        beta  = params[2 * layer + 1]

        for i in range(n):
            for j in range(i + 1, n):
                cx(q[i], q[j])
                rz(2.0 * gamma * Q[i * n + j], q[j])
                cx(q[i], q[j])

        for i in range(n):
            rx(2.0 * beta, q[i])

    mz(q)


def run_qaoa(Q_flat: list,
              n_stocks: int,
              p: int = 2,
              n_seeds: int = 3,
              shots_count: int = 2000,
              maxiter: int = 150) -> tuple:
    """
    Run QAOA with multiple random seeds, return best result.

    Args:
        Q_flat: Flattened QUBO matrix
        n_stocks: Number of stocks (qubits)
        p: QAOA circuit depth
        n_seeds: Number of random starting points
        shots_count: Measurement shots for final sample
        maxiter: COBYLA maximum iterations

    Returns:
        (best_counts dict, best_params, best_energy)
    """
    n_params = p * 2

    def cost(params):
        counts = cudaq.sample(
            qaoa_kernel, n_stocks, params.tolist(), Q_flat, shots_count=200)
        Q_mat  = np.array(Q_flat).reshape(n_stocks, n_stocks)
        total  = sum(counts.values())
        if total == 0:
            return 0.0
        energy = 0.0
        for bits, cnt in counts.items():
            z      = np.array([1 - 2 * int(b) for b in bits])
            energy += cnt * float(z @ Q_mat @ z)
        return energy / total

    best_energy, best_counts, best_params = np.inf, {}, []

    for seed in range(n_seeds):
        np.random.seed(seed * 13)
        x0  = np.random.uniform(0, np.pi, n_params)
        res = minimize(cost, x0, method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 0.5})
        cnts = cudaq.sample(
            qaoa_kernel, n_stocks, res.x.tolist(), Q_flat,
            shots_count=shots_count)
        counts_dict = {b: c for b, c in cnts.items()}

        if res.fun < best_energy:
            best_energy = res.fun
            best_counts = counts_dict
            best_params = res.x.tolist()

    return best_counts, best_params, best_energy
