from pathlib import Path
import yaml

from .eval import run_benchmark, summarize_with_cis
from .config import set_config
from .solvers import (
    RandomSolver,
    HeuristicSolver,
    EntropySolver,
    MCTSSolver,
    PositionalHeuristicSolver,
)


SOLVER_REGISTRY = {
    "random": RandomSolver,
    "heuristic": HeuristicSolver,
    "positional": PositionalHeuristicSolver,
    "entropy": EntropySolver,
    "mcts": MCTSSolver,
}


def build_solvers_from_config(cfg: dict):
    names = cfg.get("solvers") or list(SOLVER_REGISTRY.keys())
    solvers = []
    for name in names:
        cls = SOLVER_REGISTRY.get(name.lower())
        if cls is None:
            raise ValueError(f"Unknown solver name: {name}")
        solvers.append(cls())
    return solvers


def run_profile(profile_name: str, solvers: list[str] | None = None):
    cfg_path = Path("config") / f"{profile_name}.yaml"
    with open(cfg_path) as f:
        CONFIG = yaml.safe_load(f) or {}
    if solvers is not None:
        CONFIG["solvers"] = solvers

    # propagate config for solvers that consult global settings
    set_config(CONFIG)

    mode = CONFIG.get("mode", profile_name)
    results_dir = Path(CONFIG.get("results_dir", "results/summary"))
    results_dir.mkdir(parents=True, exist_ok=True)

    log_turns = bool(CONFIG.get("log_turns", False))
    write_ci = bool(CONFIG.get("write_ci", profile_name != "smoke"))
    make_plots = bool(CONFIG.get("make_plots", False))

    n_targets = CONFIG.get("n_targets")
    repeats = int(CONFIG.get("repeats", 1))

    solver_objs = build_solvers_from_config(CONFIG)

    rows, meta = run_benchmark(
        solver_objs,
        mode=mode,
        results_dir=results_dir,
        log_turns=log_turns,
        n_targets=n_targets,
        repeats=repeats,
    )

    games_csv = Path(meta["games_csv"])
    metrics_csv = Path(meta["metrics_csv"])

    if write_ci:
        summarize_with_cis(str(games_csv))

    if make_plots:
        from .figures import build_all

        build_all(mode=mode, results_dir=str(results_dir))

    return meta


__all__ = ["run_profile", "SOLVER_REGISTRY", "build_solvers_from_config"]

