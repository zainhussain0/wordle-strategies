"""Simple plotting utilities for benchmark results.

The original module contained a large amount of exploratory notebook code
executed at import time, which caused failures when ``python -m src.cli
figures`` attempted to import it.  This rewrite keeps only a minimal set
of helpers used by the CLI entry point.
"""

from __future__ import annotations

import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

from .config import get_config

CONFIG = get_config()
SUMMARY_DIR = os.path.join("results", "summary")
PLOTS_DIR = os.path.join("results", "plots")


def plot_avg_guesses_bars(metrics: List[dict]) -> None:
    """Bar chart of average guesses for each solver."""

    df = pd.DataFrame(metrics)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["solver"], df["avg_guesses_success"])
    ax.set_ylabel("Average guesses (success)")
    ax.set_title("Avg guesses by solver")
    path = os.path.join(PLOTS_DIR, "solver_avg_guesses.png")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    print("Saved:", path)


def make_all_figures() -> None:
    """Entry point for ``python -m src.cli figures``.

    Reads the metrics CSV produced by the benchmark run and generates a
    simple bar chart.  This function is intentionally lightweight but
    demonstrates how further analysis could be added.
    """

    metrics_csv = os.path.join(SUMMARY_DIR, f"metrics_{CONFIG['mode']}.csv")
    if not os.path.exists(metrics_csv):
        print(f"No metrics CSV found at {metrics_csv}; run the benchmark first.")
        return
    metrics = pd.read_csv(metrics_csv).to_dict(orient="records")
    plot_avg_guesses_bars(metrics)


__all__ = ["plot_avg_guesses_bars", "make_all_figures"]

