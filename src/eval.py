"""Game evaluation utilities for Wordle solvers.

This module contains helpers for running solvers against a set of
Wordle targets and collecting aggregate statistics.  The original file
in the repository had a partially copied notebook with duplicated and
unfinished code which resulted in import errors when the CLI attempted
to import ``summarize_with_cis`` from here.  The rewritten module below
provides a minimal but functional implementation.
"""

from __future__ import annotations

import collections
import csv
import itertools
import json
import os
import random
import statistics
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import get_config, config_paths
from .wordle_core import cached_pattern, consistent, target_words, all_valid_words
from .solvers import Solver


def _summary_dir() -> str:
    return str(config_paths()["summary"])


def _turnlog_csv() -> str:
    cfg = get_config()
    return os.path.join(_summary_dir(), f"turnlog_{cfg['mode']}.csv")


@dataclass
class GameResult:
    """Result of a single game of Wordle."""

    solver: str
    target: str
    success: bool
    guesses: int
    sequence: List[str]


def _maybe_init_turnlog() -> None:
    """Create the per-turn log file with a header if logging is enabled."""

    cfg = get_config()
    if not cfg["analysis"]["log_turns"]:
        return
    path = _turnlog_csv()
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "game_id",
                "solver",
                "turn",
                "target",
                "candidates_before",
                "guess",
                "pattern",
                "success_on_turn",
                "is_probe",
                "score_name",
                "score_value",
                "topk",
                "extras",
            ]
        )


def _append_turn_log(game_id, solver, turn, target, cand_before, guess, patt, success_on_turn):
    """Append a row to the turn log if logging is enabled."""

    cfg = get_config()
    if not cfg["analysis"]["log_turns"]:
        return
    _maybe_init_turnlog()
    d = solver.diag() if hasattr(solver, "diag") else {}
    pattern_str = "".join(str(x) for x in patt)
    with open(_turnlog_csv(), "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                game_id,
                solver.name,
                turn,
                target,
                cand_before,
                guess,
                pattern_str,
                int(bool(success_on_turn)),
                int(d.get("is_probe")) if d.get("is_probe") is not None else "",
                d.get("score_name"),
                d.get("score_value"),
                json.dumps(d.get("topk", [])),
                json.dumps(d.get("extras", {})),
            ]
        )


def play_game(target: str, solver: Solver, *, hard_mode: bool = False, allow_probes: bool = True, game_id: int | None = None) -> GameResult:
    """Play a single game of Wordle with ``solver`` against ``target``."""

    candidates = list(target_words)
    history = []
    sequence: List[str] = []
    solver.reset()

    for turn in range(1, 7):
        cand_before = len(candidates)
        guess = solver.guess(candidates, all_valid_words if allow_probes else candidates, history, hard_mode)
        sequence.append(guess)
        patt = cached_pattern(guess, target)
        history.append((guess, patt))
        _append_turn_log(game_id, solver, turn, target, cand_before, guess, patt, guess == target)
        if guess == target:
            return GameResult(solver=solver.name, target=target, success=True, guesses=turn, sequence=sequence)
        candidates = [w for w in candidates if consistent(w, guess, patt)]
    return GameResult(solver=solver.name, target=target, success=False, guesses=6, sequence=sequence)


def run_benchmark(solvers: List[Solver], mode: str | None = None):
    """Run a suite of games for ``solvers``.

    Returns a tuple ``(rows, metrics)`` where ``rows`` is a list of
    :class:`GameResult` and ``metrics`` is a list of summary dictionaries
    written to ``results/summary``.
    """

    cfg = get_config()
    mode = mode or cfg["mode"]
    if mode == "fast":
        subset = random.sample(target_words, k=min(cfg["fast_n_targets"], len(target_words)))
        repeats = cfg["fast_repeats"]
    else:
        subset = list(target_words)
        repeats = cfg["full_repeats"]

    total_games = len(solvers) * repeats * len(subset)
    pbar = tqdm(total=total_games, desc=f"Running {mode} benchmark", ncols=100)

    rows: List[GameResult] = []
    gid = 0
    for solver in solvers:
        for _ in range(repeats):
            for target in subset:
                gid += 1
                res = play_game(
                    target,
                    solver,
                    hard_mode=cfg["hard_mode"],
                    allow_probes=cfg["allow_probes"],
                    game_id=gid,
                )
                rows.append(res)
                pbar.update(1)
    pbar.close()

    # Write per-game CSV
    summary_dir = _summary_dir()
    os.makedirs(summary_dir, exist_ok=True)
    games_csv = os.path.join(summary_dir, f"games_{mode}.csv")
    with open(games_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solver", "target", "success", "guesses", "sequence"])
        for r in rows:
            w.writerow([r.solver, r.target, int(r.success), r.guesses, " ".join(r.sequence)])
    print("Wrote:", games_csv)

    # Aggregate metrics per solver
    metrics = []
    for name, group in itertools.groupby(sorted(rows, key=lambda x: x.solver), key=lambda x: x.solver):
        g = list(group)
        n = len(g)
        wins = sum(r.success for r in g)
        win_rate = 100 * wins / n if n else 0.0
        avg_guesses = statistics.mean([r.guesses for r in g if r.success]) if wins else float("nan")
        dist = collections.Counter(r.guesses for r in g if r.success)
        row = {
            "solver": name,
            "n_games": n,
            "win_rate": win_rate,
            "avg_guesses_success": avg_guesses,
            "fail_rate": 100 - win_rate,
            "p_solved_2": dist.get(2, 0) / n,
            "p_solved_3": dist.get(3, 0) / n,
            "p_solved_4": dist.get(4, 0) / n,
            "p_solved_5": dist.get(5, 0) / n,
            "p_solved_6": dist.get(6, 0) / n,
        }
        metrics.append(row)

    metrics_csv = os.path.join(summary_dir, f"metrics_{mode}.csv")
    with open(metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        w.writeheader()
        w.writerows(metrics)
    print("Wrote:", metrics_csv)

    return rows, metrics


# ---------------------------------------------------------------------------
#  Summary with bootstrap confidence intervals


def _bootstrap_ci(samples: np.ndarray, fn, *, B: int = 2000, alpha: float = 0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(samples)
    stats = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        stats.append(fn(samples[idx]))
    stats.sort()
    lo = stats[int((alpha / 2) * B)]
    hi = stats[int((1 - alpha / 2) * B)]
    return lo, hi


def summarize_with_cis(rows: List[GameResult]) -> pd.DataFrame:
    """Aggregate ``rows`` with bootstrap confidence intervals.

    A CSV is written to ``results/summary`` and the resulting DataFrame is
    returned.  This function mirrors the behaviour of the original
    notebook but is safe to import from the command line interface.
    """

    by_solver: dict[str, List[GameResult]] = collections.defaultdict(list)
    for r in rows:
        by_solver[r.solver].append(r)

    out = []
    for solver, lst in by_solver.items():
        succ = np.array([int(r.success) for r in lst])
        wr = succ.mean() * 100
        wr_lo, wr_hi = _bootstrap_ci(succ, np.mean)
        wr_lo *= 100
        wr_hi *= 100

        gsucc = np.array([r.guesses for r in lst if r.success])
        if len(gsucc):
            avg = gsucc.mean()
            avg_lo, avg_hi = _bootstrap_ci(gsucc, np.mean)
        else:
            avg = float("nan")
            avg_lo = avg_hi = float("nan")

        out.append(
            {
                "solver": solver,
                "win_rate": wr,
                "win_rate_lo": wr_lo,
                "win_rate_hi": wr_hi,
                "avg_guesses": avg,
                "avg_lo": avg_lo,
                "avg_hi": avg_hi,
            }
        )

    df = pd.DataFrame(out)
    cfg = get_config()
    path = os.path.join(_summary_dir(), f"metrics_with_cis_{cfg['mode']}.csv")
    df.to_csv(path, index=False)
    print("Wrote:", path)
    return df


