# src/config.py
import os, json, yaml
from pathlib import Path
from copy import deepcopy

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results"
SUMMARY_DIR = RESULTS_DIR / "summary"
PLOTS_DIR   = RESULTS_DIR / "plots"
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

CONFIG_DEFAULTS = {
    "mode": "fast",
    "fast_n_targets": 200,
    "fast_repeats": 20,
    "full_repeats": 1,
    "hard_mode": False,
    "allow_probes": True,
    "entropy_max_candidates": 2000,
    "mcts": {"rollouts_per_move": 100, "ucb_c": 1.4},
    "analysis": {"log_turns": True, "topk": 10},
}

CONFIG = deepcopy(CONFIG_DEFAULTS)

def _deep_merge(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return b
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(a.get(k), v) if isinstance(v, dict) else v
    return out

def set_config(new_cfg: dict):
    """Set global CONFIG for the whole run (called by runner)."""
    global CONFIG
    CONFIG = _deep_merge(CONFIG_DEFAULTS, new_cfg or {})
    # ensure output dirs exist
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return CONFIG

def get_config():
    return CONFIG

def config_paths():
    """Convenience for other modules."""
    return {
        "repo": REPO,
        "results": RESULTS_DIR,
        "summary": SUMMARY_DIR,
        "plots": PLOTS_DIR,
    }

