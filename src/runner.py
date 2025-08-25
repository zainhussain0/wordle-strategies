# src/runner.py
import json, os, random, math
from pathlib import Path

# --- config loading (supports either config/<profile>.yaml OR config/presets.yaml) ---
def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _load_yaml(path: Path):
    import yaml  # ensure PyYAML is in requirements.txt
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_profile_config(profile: str) -> dict:
    repo = Path(__file__).resolve().parents[1]
    cfg_dir = repo / "config"

    # 1) config/<profile>.yaml
    profile_yaml = cfg_dir / f"{profile}.yaml"
    if profile_yaml.exists():
        base = {
            "mode": "fast",
            "hard_mode": False,
            "allow_probe_outside_candidates": True,
            "answers_path": str(repo / "data" / "wordle-answers-alphabetical.txt"),
            "guesses_path": str(repo / "data" / "wordle-allowed-guesses.txt"),
            "output_dir": str(repo / "results"),
            "solvers": [{"name": "random"}, {"name": "heuristic"}, {"name": "entropy"}, {"name": "mcts"}],
            "seed": 1337,
            "repeats": 1,
            "max_targets": 200
        }
        return _deep_merge(base, _load_yaml(profile_yaml))

    # 2) config/presets.yaml with a section
    presets_yaml = cfg_dir / "presets.yaml"
    if presets_yaml.exists():
        all_presets = _load_yaml(presets_yaml)
        if profile not in all_presets:
            raise KeyError(f"Profile '{profile}' not found in {presets_yaml}. Available: {list(all_presets.keys())}")
        # merge over sensible defaults
        base = {
            "mode": "fast",
            "hard_mode": False,
            "allow_probes_outside_candidates": True,
            "answers_path": str(repo / "data" / "wordle-answers-alphabetical.txt"),
            "guesses_path": str(repo / "data" / "wordle-allowed-guesses.txt"),
            "output_dir": str(repo / "results"),
            "solvers": [{"name": "random"}, {"name": "heuristic"}, {"name": "entropy"}, {"name": "mcts"}],
            "seed": 1337,
        }
        return _deep_merge(base, all_presets.get(profile) or {})

    raise FileNotFoundError(
        "No config found. Create either config/<profile>.yaml or config/presets.yaml "
        "(with a section named after your profile)."
    )

# --- word lists into wordle_core (works whether wordle_core exposes a helper or not) ---
def _ensure_word_lists(cfg: dict):
    from . import wordle_core as core
    answers = Path(cfg.get("answers_path"))
    guesses = Path(cfg.get("guesses_path"))
    if not answers.exists() or not guesses.exists():
        raise FileNotFoundError(f"Word lists not found at:\n  {answers}\n  {guesses}")
    # If the module has a loader, use it; otherwise set globals
    if hasattr(core, "load_words_from_file"):
        # Many notebooks define this helper — reuse it if present
        target_words = core.load_words_from_file(str(answers))
        valid_words  = core.load_words_from_file(str(guesses))
        core.target_words = sorted(set(target_words))
        core.valid_words  = sorted(set(valid_words))
        core.all_valid_words = sorted(set(core.target_words) | set(core.valid_words))
    else:
        # Minimal local loader
        def _load(p: Path):
            return sorted({w.strip().lower() for w in p.read_text(encoding="utf-8").splitlines()
                           if len(w.strip()) == 5 and w.strip().isalpha()})
        core.target_words = _load(answers)
        core.valid_words  = _load(guesses)
        core.all_valid_words = sorted(set(core.target_words) | set(core.valid_words))

# --- solver factory from names in config ---
def _make_solvers(solver_specs):
    # Lazy import to avoid heavy imports when just loading config
    from .solvers import RandomSolver, HeuristicSolver, EntropySolver, MCTSSolver
    name2cls = {
        "random": RandomSolver,
        "heuristic": HeuristicSolver,
        "entropy": EntropySolver,
        "mcts": MCTSSolver,
    }
    solvers = []
    for spec in solver_specs or []:
        nm = (spec.get("name") or "").lower()
        if nm not in name2cls:
            raise ValueError(f"Unknown solver name in config: {nm}")
        solvers.append(name2cls[nm]())
    if not solvers:
        solvers = [RandomSolver(), HeuristicSolver(), EntropySolver(), MCTSSolver()]
    return solvers

def _set_seeds(seed: int | None):
    try:
        import numpy as np
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    except Exception:
        if seed is not None:
            random.seed(seed)

# --- public entrypoint used by src/cli.py ---
def run_profile(profile: str = "fast_dev"):
    cfg = load_profile_config(profile)
    _set_seeds(cfg.get("seed"))

    # ensure output dirs exist
    outdir = Path(cfg.get("output_dir", "results"))
    (outdir / "summary").mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    # load word lists into wordle_core module
    _ensure_word_lists(cfg)

    # make solvers
    solvers = _make_solvers(cfg.get("solvers"))

    # run benchmark
    from .eval import run_benchmark, summarize_with_cis
    rows, metrics = run_benchmark(solvers, mode=cfg.get("mode", "fast"))

    # write a frozen config snapshot for reproducibility
    snap = dict(cfg)
    snap["solvers"] = [s.__class__.__name__ for s in solvers]
    (outdir / "summary" / f"config_snapshot_{profile}.json").write_text(
        json.dumps(snap, indent=2), encoding="utf-8"
    )

    # optional: write CI summary with CIs
    summarize_with_cis(rows)

    print(f"Done. See '{outdir}/summary' for CSVs and '{outdir}/plots' for figures.")
