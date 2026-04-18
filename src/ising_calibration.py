"""
NVIDIA Ising Calibration NIM wrapper.

Generates a text-based calibration report from QAOA shot statistics
when the NIM model endpoint is unavailable, and optionally calls
the NIM API when available.
"""

import json
import numpy as np
import requests


ISING_MODEL = "nvidia/ising-calibration-1-35b-a3b"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _shot_stats(counts: dict) -> dict:
    """Derive basic statistics from a QAOA counts dict."""
    if not counts:
        return {}
    total    = sum(counts.values())
    top5     = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_prob = top5[0][1] / total if top5 else 0.0
    n_unique = len(counts)
    n_bits   = len(top5[0][0]) if top5 else 0

    ones_per_state = [sum(int(b) for b in bits) for bits in counts]
    mean_ones      = float(np.mean(ones_per_state)) if ones_per_state else 0.0

    return {
        "total_shots":       total,
        "unique_bitstrings": n_unique,
        "top5":              [(b, c, round(c / total, 4)) for b, c in top5],
        "top1_probability":  round(top_prob, 4),
        "mean_ones_per_state": round(mean_ones, 3),
        "n_qubits":          n_bits,
    }


def calibration_report_text(counts: dict,
                              params: list,
                              energy: float,
                              tickers: list) -> str:
    """
    Generate a text-based Ising calibration report from shot statistics.

    Used when the NIM API is unavailable or returns null content.
    """
    stats   = _shot_stats(counts)
    top_bit = stats["top5"][0][0] if stats.get("top5") else ""
    selected = [tickers[i] for i, b in enumerate(top_bit)
                if b == "1" and i < len(tickers)] if top_bit else []

    lines = [
        "=== ISING CALIBRATION REPORT ===",
        f"Qubits            : {stats.get('n_qubits', len(tickers))}",
        f"Total shots       : {stats.get('total_shots', 0)}",
        f"Unique bitstrings : {stats.get('unique_bitstrings', 0)}",
        f"Top-1 probability : {stats.get('top1_probability', 0):.2%}",
        f"Mean stocks/state : {stats.get('mean_ones_per_state', 0):.1f}",
        f"Best energy       : {energy:.4f}",
        "",
        "Top-5 bitstrings:",
    ]
    for bits, cnt, prob in stats.get("top5", []):
        sel = [tickers[i] for i, b in enumerate(bits)
               if b == "1" and i < len(tickers)]
        lines.append(f"  {bits}  shots={cnt}  p={prob:.3f}  → {', '.join(sel)}")

    lines += [
        "",
        f"QAOA params (γ, β per layer): {[round(p, 4) for p in params]}",
        "",
        "Calibration assessment:",
    ]

    top1 = stats.get("top1_probability", 0)
    if top1 > 0.15:
        lines.append("  HIGH confidence — top state captures >15% of shots.")
        lines.append("  Ising landscape well-resolved; QAOA converged cleanly.")
    elif top1 > 0.07:
        lines.append("  MODERATE confidence — top state 7–15% of shots.")
        lines.append("  Consider p=3 layers or more COBYLA iterations.")
    else:
        lines.append("  LOW confidence — top state <7% of shots.")
        lines.append("  Increase p-layers or shots_count for better resolution.")

    lines += [
        "",
        f"Recommended portfolio: {', '.join(selected) if selected else 'None selected'}",
        "================================",
    ]
    return "\n".join(lines)


def call_nim_calibration(prompt: str, api_key: str, max_tokens: int = 512) -> str:
    """
    Call the NVIDIA NIM chat endpoint.

    Falls back to a stub response if the model returns null content
    (known issue with reasoning models exceeding token budget).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":      ISING_MODEL,
        "messages":   [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    try:
        resp = requests.post(
            f"{NIM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data    = resp.json()
        content = data["choices"][0]["message"].get("content")
        if content:
            return content
        return "[NIM returned null content — using local calibration report]"
    except Exception as exc:
        return f"[NIM API unavailable: {exc}]"


def get_calibration(counts: dict,
                     params: list,
                     energy: float,
                     tickers: list,
                     api_key: str | None = None) -> str:
    """
    Return calibration report, preferring local text generation.

    If api_key is provided attempts NIM first; falls back to text report.
    """
    local_report = calibration_report_text(counts, params, energy, tickers)

    if not api_key:
        return local_report

    prompt = (
        "You are an Ising machine calibration expert. "
        "Given these QAOA shot statistics, assess convergence quality "
        "and recommend whether to trust the top bitstring as the optimal portfolio.\n\n"
        + local_report
    )
    nim_response = call_nim_calibration(prompt, api_key)
    if nim_response.startswith("[NIM"):
        return local_report
    return nim_response
