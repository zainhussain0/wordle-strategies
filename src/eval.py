# %%
import os, csv, json, collections
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

from .config import get_config
from .wordle_core import Pattern, cached_pattern, consistent, target_words, all_valid_words
from .solvers import Solver
CONFIG = get_config()

SUMMARY_DIR = os.path.join("results","summary")  


@dataclass
class GameResult:
    solver: str
    target: str
    success: bool
    guesses: int
    sequence: List[str]

import json, csv

def append_turn_log(game_id, solver, turn, target, cand_before, guess, patt, success_on_turn):
    if not CONFIG["analysis"]["log_turns"]:
        return
    d = solver.diag() if hasattr(solver, "diag") else {}
    pattern_str = "".join(str(x) for x in patt)  # e.g., 20110
    with open(TURNLOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            game_id, solver.name, turn, target,
            cand_before, guess, pattern_str,
            int(bool(success_on_turn)),
            int(d.get("is_probe")) if d.get("is_probe") is not None else "",
            d.get("score_name"), d.get("score_value"),
            json.dumps(d.get("topk", [])),
            json.dumps(d.get("extras", {}))
        ])


def play_game(target: str, solver: Solver, hard_mode=False, allow_probes=True, valid=all_valid_words, game_id=None) -> GameResult:
    candidates = [w for w in target_words]
    history = []
    seq = []
    solver.reset()

    for turn in range(1, 7):
        cand_before = len(candidates)
        g = solver.guess(candidates, all_valid_words if allow_probes else candidates, history, hard_mode)
        seq.append(g)
        patt = cached_pattern(g, target)
        history.append((g, patt))

        # log this turn
        append_turn_log(
            game_id=game_id,
            solver=solver,
            turn=turn,
            target=target,
            cand_before=cand_before,
            guess=g,
            patt=patt,
            success_on_turn=(g == target)
        )

        if g == target:
            return GameResult(solver=solver.name, target=target, success=True, guesses=turn, sequence=seq)

        candidates = [w for w in candidates if consistent(w, g, patt)]

    return GameResult(solver=solver.name, target=target, success=False, guesses=6, sequence=seq)


from tqdm import tqdm

def run_benchmark(solvers: List[Solver], mode=None):
    mode = mode or CONFIG["mode"]
    if mode == "fast":
        subset = random.sample(target_words, k=min(CONFIG["fast_n_targets"], len(target_words)))
        repeats = CONFIG["fast_repeats"]
    else:
        subset = list(target_words)
        repeats = CONFIG["full_repeats"]

    total_games = len(solvers) * repeats * len(subset)
    pbar = tqdm(total=total_games, desc=f"Running {mode} benchmark", ncols=100)

    rows = []
    gid = 0
    for s in solvers:
        for r in range(repeats):
            for t in subset:
                gid += 1
                res = play_game(
                    t, s,
                    hard_mode=CONFIG["hard_mode"],
                    allow_probes=CONFIG["allow_probes"],
                    game_id=gid
                )
                rows.append(res)
                pbar.update(1)
    pbar.close()

    
    return rows, metrics

    # track first guesses separately
    first_guess_tracker = collections.Counter()

    # for each solver
    for s in solvers:
        ...
        for r in range(repeats):
            for t in subset:
                res = play_game(t, s, ...)
                
                rows.append(res)

                # log first guess
                if res["guesses"]:
                    first_guess_tracker[res["guesses"][0]] += 1

        # after loop, dump per-solver summaries
        outdir = Path(CONFIG["analysis"].get("log_dir", "results/summary"))
        outdir.mkdir(parents=True, exist_ok=True)

        # first-guess frequencies
        fg_file = outdir / f"{s.name}_first_guesses.csv"
        with open(fg_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["guess", "count"])
            writer.writerows(first_guess_tracker.most_common())
        print(f"Saved first-guess distribution for {s.name} → {fg_file}")
          


    # Write per-game CSV
    out_csv = os.path.join(SUMMARY_DIR, f"games_{mode}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solver","target","success","guesses","sequence"])
        for r in rows:
            w.writerow([r.solver, r.target, int(r.success), r.guesses, " ".join(r.sequence)])
    print("Wrote:", out_csv)

    # Aggregate metrics
    metrics = []
    for name, group in itertools.groupby(sorted(rows, key=lambda x: x.solver), key=lambda x: x.solver):
        g = list(group)
        n = len(g)
        wins = sum(r.success for r in g)
        fail = n - wins
        win_rate = 100 * wins / n if n else 0.0
        avg_guesses = statistics.mean([r.guesses for r in g if r.success]) if wins else float("nan")

        dist = collections.Counter(r.guesses for r in g if r.success)
        row = {
            "solver": name,
            "n_games": n,
            "win_rate": win_rate,
            "avg_guesses_success": avg_guesses,
            "fail_rate": 100 - win_rate,
            "p_solved_2": dist.get(2,0)/n,
            "p_solved_3": dist.get(3,0)/n,
            "p_solved_4": dist.get(4,0)/n,
            "p_solved_5": dist.get(5,0)/n,
            "p_solved_6": dist.get(6,0)/n,
        }
        metrics.append(row)

    # Write summary CSV
    out_csv2 = os.path.join(SUMMARY_DIR, f"metrics_{mode}.csv")
    with open(out_csv2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        w.writeheader()
        w.writerows(metrics)
    print("Wrote:", out_csv2)
    return rows, metrics

    

  
