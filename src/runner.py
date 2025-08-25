# src/runner.py
import json
from pathlib import Path
import yaml

from .config import set_config, config_paths
from .eval import run_benchmark, summarize_with_cis
from .solvers import RandomSolver, HeuristicSolver, EntropySolver, MCTSSolver

# Simple registry to allow selecting a subset of solvers from config
SOLVER_REGISTRY = {
    "random": RandomSolver,
    "heuristic": HeuristicSolver,
    "entropy": EntropySolver,
    "mcts": MCTSSolver,
}

def load_profile_yaml(profile: str) -> dict:
    repo = Path(__file__).resolve().parents[1]
    yml = repo / "config" / f"{profile}.yaml"
    if not yml.exists():
        raise FileNotFoundError(f"Config profile not found: {yml}")
    with yml.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def run_profile(profile: str = "fast_dev"):
    # 1) load YAML and set CONFIG
    cfg = load_profile_yaml(profile)
    CONFIG = set_config(cfg)

    # 2) instantiate solvers (names may be overridden via YAML)
    solver_names = cfg.get("solvers")
    if solver_names:
        try:
            solvers = [SOLVER_REGISTRY[name.lower()]() for name in solver_names]
        except KeyError as e:
            raise ValueError(f"Unknown solver name: {e.args[0]}")
    else:
        solvers = [cls() for cls in SOLVER_REGISTRY.values()]

    # 3) run benchmark
    rows, metrics = run_benchmark(solvers, mode=CONFIG["mode"])
    summarize_with_cis(rows)

    # 4) snapshot config for reproducibility
    paths = config_paths()
    snap_path = paths["summary"] / f"config_snapshot_{profile}.json"
    snap = dict(CONFIG)
    snap["solvers"] = [s.__class__.__name__ for s in solvers]
    snap_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"Done. CSVs in {paths['summary']}, plots in {paths['plots']}.")

