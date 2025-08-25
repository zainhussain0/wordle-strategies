from __future__ import annotations

from pathlib import Path
import csv
import random
import numpy as np
import pandas as pd

from .wordle_core import (
    cached_pattern,
    consistent,
    target_words,
    all_valid_words,
)
from .solvers import Solver


# Defaults (runner can overwrite)
RESULTS_DIR = Path("results/summary")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TURNLOG_CSV = RESULTS_DIR / "turnlog.csv"
GAMES_CSV = RESULTS_DIR / "games.csv"    # per-game rows
METRICS_CSV = RESULTS_DIR / "metrics.csv"  # per-solver summary

LOG_TURNS = False  # over-ridden per profile


# --- per-turn logging (safe/no-op in smoke) ---
def append_turn_log(row: dict):
    if not LOG_TURNS:
        return
    TURNLOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = TURNLOG_CSV.exists()
    with open(TURNLOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# --- helper for profile suffixing like games_fast.csv ---
def _with_suffix(base: Path, suffix: str) -> Path:
    if base.suffix != ".csv":
        return base
    return base.with_name(f"{base.stem}_{suffix}.csv")


# --- play a single game ---
def play_game(
    target: str,
    solver: Solver,
    *,
    hard_mode: bool = False,
    allow_probes: bool = True,
    game_id: int | None = None,
):
    candidates = list(target_words)
    history = []
    sequence: list[str] = []
    solver.reset()

    for turn in range(1, 7):
        cand_before = len(candidates)
        guess = solver.guess(
            candidates,
            all_valid_words if allow_probes else candidates,
            history,
            hard_mode,
        )
        sequence.append(guess)
        patt = cached_pattern(guess, target)
        history.append((guess, patt))
        append_turn_log(
            {
                "game_id": game_id,
                "solver": solver.name,
                "turn": turn,
                "target": target,
                "candidates_before": cand_before,
                "guess": guess,
                "pattern": "".join(str(x) for x in patt),
                "success_on_turn": int(guess == target),
            }
        )
        if guess == target:
            return {
                "solver": solver.name,
                "target": target,
                "success": 1,
                "guesses": turn,
                "sequence": " ".join(sequence),
            }
        candidates = [w for w in candidates if consistent(w, guess, patt)]

    return {
        "solver": solver.name,
        "target": target,
        "success": 0,
        "guesses": 6,
        "sequence": " ".join(sequence),
    }


# --- run_benchmark: write per-game + metrics, 2 d.p., profile-aware ---
def run_benchmark(
    solvers,
    *,
    mode: str = "smoke",
    results_dir: Path = RESULTS_DIR,
    log_turns: bool = LOG_TURNS,
    n_targets: int | str | None = None,
    repeats: int = 1,
):
    global RESULTS_DIR, GAMES_CSV, METRICS_CSV, TURNLOG_CSV, LOG_TURNS
    RESULTS_DIR = Path(results_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    LOG_TURNS = bool(log_turns)
    TURNLOG_CSV = RESULTS_DIR / f"turnlog_{mode}.csv"
    GAMES_CSV = _with_suffix(RESULTS_DIR / "games.csv", mode)
    METRICS_CSV = _with_suffix(RESULTS_DIR / "metrics.csv", mode)

    if n_targets in (None, "all"):
        targets = list(target_words)
    else:
        targets = random.sample(list(target_words), k=min(int(n_targets), len(target_words)))

    rows: list[dict] = []
    gid = 0
    for solver in solvers:
        for _ in range(repeats):
            for target in targets:
                gid += 1
                row = play_game(target, solver, game_id=gid)
                rows.append(row)

    games_df = pd.DataFrame(rows)
    games_df.to_csv(GAMES_CSV, index=False, float_format="%.2f")

    agg = (
        games_df.groupby("solver", dropna=False)
        .agg(
            n_games=("success", "size"),
            win_rate=("success", "mean"),
            avg_guesses=("guesses", "mean"),
            fail_rate=("success", lambda s: 1 - s.mean()),
        )
        .reset_index()
    )

    for k in (2, 3, 4, 5, 6):
        col = f"p_solved_{k}"
        sub = games_df.loc[games_df["success"] == 1, ["solver", "guesses"]].copy()
        sub["hit"] = (sub["guesses"] == k).astype(float)
        agg = agg.merge(
            sub.groupby("solver", dropna=False)["hit"].mean().rename(col),
            on="solver",
            how="left",
        )

    agg = agg.fillna(0.0)
    agg.to_csv(METRICS_CSV, index=False, float_format="%.2f")

    return rows, {"games_csv": str(GAMES_CSV), "metrics_csv": str(METRICS_CSV)}


# --- lightweight bootstrap CIs for solver-level metrics ---
def _bootstrap_ci(
    x: np.ndarray,
    stat_fn,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = x.size
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = x[rng.integers(0, n, n)]
        stats[i] = stat_fn(sample)
    lo = np.quantile(stats, (1 - ci) / 2)
    hi = np.quantile(stats, 1 - (1 - ci) / 2)
    return (lo, hi)


def summarize_with_cis(
    input_csv: str,
    output_csv: str | None = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    group_cols=("solver",),
):
    df = pd.read_csv(input_csv)
    if isinstance(group_cols, str):
        group_cols = (group_cols,)

    out = []
    for keys, g in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        win = g["success"].astype(float).to_numpy()
        wr = float(win.mean()) if win.size else np.nan
        wr_lo, wr_hi = _bootstrap_ci(win, np.mean, n_boot=n_boot, ci=ci)

        solved = g.loc[g["success"] == 1, "guesses"].astype(float).to_numpy()
        ag = float(solved.mean()) if solved.size else np.nan
        ag_lo, ag_hi = (
            _bootstrap_ci(solved, np.mean, n_boot=n_boot, ci=ci) if solved.size else (np.nan, np.nan)
        )

        out.append(
            {
                **{c: v for c, v in zip(group_cols, keys)},
                "n_games": int(len(g)),
                "win_rate": wr,
                "win_rate_lo": wr_lo,
                "win_rate_hi": wr_hi,
                "avg_guesses": ag,
                "avg_guesses_lo": ag_lo,
                "avg_guesses_hi": ag_hi,
            }
        )

    out = pd.DataFrame(out).sort_values(list(group_cols)).reset_index(drop=True)

    if output_csv is None:
        inp = Path(input_csv)
        suffix = inp.stem.split("_")[-1] if "_" in inp.stem else "smoke"
        output_csv = inp.parent / f"metrics_with_cis_{suffix}.csv"

    out.to_csv(output_csv, index=False, float_format="%.2f")
    return out


__all__ = [
    "run_benchmark",
    "summarize_with_cis",
    "append_turn_log",
    "play_game",
]

