"""
NVIDIA Ising Calibration integration.

Calls the Ising Calibration NIM API to assess
QAOA measurement distribution quality.

Model: nvidia/ising-calibration-1-35b-a3b
Access: build.nvidia.com (free, 1000 credits)
"""

import os
import io
import base64
import requests
import matplotlib.pyplot as plt


def run_ising_calibration(counts_dict: dict,
                           label: str,
                           api_key: str = None,
                           max_states: int = 20) -> str:
    """
    Assess QAOA measurement distribution quality
    using NVIDIA Ising Calibration (NIM API).

    Generates a bar chart of the measurement
    distribution and sends it to the Ising model
    for circuit quality assessment.

    Args:
        counts_dict: {bitstring: count} from QAOA
        label: Window label for plot title
        api_key: NVIDIA NIM API key
                 (or set NVIDIA_API_KEY env var)
        max_states: Number of top states to plot

    Returns:
        Calibration assessment string from model
    """
    if api_key is None:
        api_key = os.environ.get("NVIDIA_API_KEY", "")

    if not api_key:
        return "No API key provided."

    if not counts_dict:
        return "No measurement data available."

    # Generate distribution plot
    fig, ax = plt.subplots(figsize=(10, 4))
    top = sorted(counts_dict.items(),
                 key=lambda x: x[1],
                 reverse=True)[:max_states]
    states, freqs = zip(*top)

    ax.bar(range(len(states)), freqs,
           color="#1D9E75")
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(
        [s[:6] + "..." for s in states],
        rotation=45, ha="right", fontsize=7)
    ax.set_title(
        f"QAOA Measurement Distribution — {label}")
    ax.set_ylabel("Count")
    plt.tight_layout()

    buf     = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(
        buf.read()).decode()
    plt.close(fig)

    # Call NIM API
    response = requests.post(
        "https://integrate.api.nvidia.com/v1/"
        "chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json"
        },
        json={
            "model":
                "nvidia/ising-calibration-1-35b-a3b",
            "messages": [{"role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {
                         "url": f"data:image/png;"
                                f"base64,{img_b64}"
                     }},
                    {"type": "text",
                     "text": (
                         "Assess this QAOA measurement "
                         "distribution from a portfolio "
                         "optimisation circuit.\n"
                         "Answer:\n"
                         "1. Is probability mass "
                         "concentrated (good) or "
                         "spread broadly (poor)?\n"
                         "2. Does the distribution "
                         "suggest good QAOA convergence?\n"
                         "3. Rate calibration: "
                         "GOOD, MARGINAL, or POOR.\n"
                         "3 sentences maximum."
                     )}
                ]}],
            "max_tokens": 200
        },
        timeout=60
    )

    if response.status_code != 200:
        return f"API error {response.status_code}"

    return response.json()[
        "choices"][0]["message"]["content"]
