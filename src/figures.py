from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def build_all(mode="smoke", results_dir="results/summary"):
    rd = Path(results_dir)
    metrics = rd / f"metrics_{mode}.csv"
    metrics_ci = rd / f"metrics_with_cis_{mode}.csv"

    if not metrics.exists():
        print(f"[figures] {metrics.name} not found — skipping.")
        return

    df = pd.read_csv(metrics)
    df_ci = pd.read_csv(metrics_ci) if metrics_ci.exists() else None

    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # simple bar chart example
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["solver"], df["avg_guesses"])

    if df_ci is not None:
        errs = [
            df_ci.loc[df_ci["solver"] == s, "avg_guesses_hi"].values[0]
            - df_ci.loc[df_ci["solver"] == s, "avg_guesses"].values[0]
            for s in df["solver"]
        ]
        ax.errorbar(df["solver"], df["avg_guesses"], yerr=errs, fmt="none", ecolor="black")

    ax.set_ylabel("Average guesses")
    ax.set_title("Avg guesses by solver")
    fig.tight_layout()
    fig.savefig(plots_dir / f"solver_bars_{mode}.png", dpi=200)


__all__ = ["build_all"]

