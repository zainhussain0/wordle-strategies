"""Statistical comparison utilities for Wordle solver benchmarks."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import pandas as pd
from scipy import stats


def pairwise_tests(
    games_csv: str | Path,
    metric: str = "guesses",
    test: str = "ttest",
) -> pd.DataFrame:
    """Compute pairwise statistical tests between solvers.

    Parameters
    ----------
    games_csv : str or Path
        Path to a games CSV produced by ``run_benchmark``.
    metric : str, default "guesses"
        Column on which to perform the test (e.g., "guesses" or "success").
    test : {"ttest", "wilcoxon"}, default "ttest"
        Statistical test to use. ``ttest`` performs a paired t-test and
        ``wilcoxon`` performs the Wilcoxon signed-rank test.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``solver1``, ``solver2``, ``test``,
        ``stat`` and ``p_value`` summarising the comparison.
    """
    df = pd.read_csv(games_csv)
    if metric not in df.columns:
        raise ValueError(f"metric '{metric}' not found in {games_csv}")

    # pivot so each solver is a column indexed by target
    pivot = df.pivot(index="target", columns="solver", values=metric).dropna()
    solvers = list(pivot.columns)
    results = []

    for s1, s2 in combinations(solvers, 2):
        x = pivot[s1]
        y = pivot[s2]
        if test == "wilcoxon":
            stat, p = stats.wilcoxon(x, y)
            test_name = "wilcoxon"
        else:
            stat, p = stats.ttest_rel(x, y)
            test_name = "paired_t"
        results.append(
            {
                "solver1": s1,
                "solver2": s2,
                "test": test_name,
                "stat": float(stat),
                "p_value": float(p),
            }
        )
    return pd.DataFrame(results)


def _write_results(df: pd.DataFrame, output: str | Path | None) -> None:
    if output is None:
        print(df.to_csv(index=False))
    else:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)


def main(argv: Iterable[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Pairwise statistical tests for solvers")
    p.add_argument("games_csv", help="Path to games CSV from run_benchmark")
    p.add_argument("--metric", default="guesses")
    p.add_argument("--test", choices=["ttest", "wilcoxon"], default="ttest")
    p.add_argument("--output", help="Optional output CSV path", default=None)
    args = p.parse_args(argv)

    df = pairwise_tests(args.games_csv, metric=args.metric, test=args.test)
    _write_results(df, args.output)


if __name__ == "__main__":
    main()
