import os, json, yaml
from copy import deepcopy

DEFAULTS = {
  "mode": "fast",
  "fast_n_targets": 200, "fast_repeats": 20, "full_repeats": 1,
  "hard_mode": False, "allow_probes": True,
  "mcts": {"rollouts_per_move": 100, "ucb_c": 1.4},
  "analysis": {"log_turns": True, "topk": 10, "log_dir": "results/summary"},
}

def _deep_merge(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return b
    out = dict(a)
    for k,v in b.items():
        out[k] = _deep_merge(a.get(k), v) if isinstance(v, dict) else v
    return out

def load_config(profile="fast_dev", presets_path="config/presets.yaml"):
    with open(presets_path, "r") as f:
        presets = yaml.safe_load(f) or {}
    preset = presets.get(profile, {})
    cfg = _deep_merge(DEFAULTS, preset)
    os.makedirs("results/summary", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    print("Loaded profile:", profile)
    print(json.dumps(cfg, indent=2))
    return cfg
