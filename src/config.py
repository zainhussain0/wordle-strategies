# %% [markdown]
# # Wordle Solver Evaluation – Modular Framework

# %%
import math, random, json, time, itertools, collections, statistics, os, csv
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Iterable, Optional, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
random.seed(42)
np.random.seed(42)

# Paths
REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results"
SUMMARY_DIR = RESULTS_DIR / "summary"
PLOTS_DIR   = RESULTS_DIR / "plots"
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Config – Fast vs Full evaluation
CONFIG = {
    "mode": "fast",           # "fast" or "full"
    "fast_n_targets": 100,
    "fast_repeats": 10,      # repeats per solver (each repeat samples a new target from the subset)
    "full_repeats": 1,        # one pass per target
    "hard_mode": False,       # if True, legal guesses must satisfy known greens/yellows
    "allow_probes": True,     # if True, entropy/MCTS may guess any valid word (not just candidates)
    "mcts": {
        "rollouts_per_move": 400,   # tune in fast mode; fix for full
        "ucb_c": 1.4
    }
}
print("CONFIG:", json.dumps(CONFIG, indent=2))


# ---- ANALYSIS CONFIG (add at end of your existing CONFIG cell) ----
CONFIG["analysis"] = {
    "log_turns": True,          # log a row per turn for any solver
    "topk": 10,                 # store top-K choices per solver/turn (as JSON)
    "log_dir": "results/summary",
}
os.makedirs(CONFIG["analysis"]["log_dir"], exist_ok=True)

# One file for per-turn logs (all solvers go here)
TURNLOG_CSV = os.path.join(CONFIG["analysis"]["log_dir"], f"turnlog_{CONFIG['mode']}.csv")

# Write header once per run (safe if file exists)
if not os.path.exists(TURNLOG_CSV):
    import csv
    with open(TURNLOG_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "game_id","solver","turn","target",
            "candidates_before","guess","pattern_str",
            "success_on_turn","is_probe",
            "score_name","score_value","topk_json","extras_json"
        ])

def get_config():
    return CONFIG
