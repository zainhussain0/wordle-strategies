# wordle-strategies

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### GUI
Run a simple desktop interface for interactive solving:

```bash
python -m src.gui
```

### Profiles
- **Smoke**: quick check across all solvers on a small random target subset. Heavy solvers use a
  reduced search (fewer MCTS rollouts and capped entropy candidates) to finish in seconds.
  `python -m src.cli run --profile smoke`

- **Full**: exhaustive 2309-word sweep using all solvers with repeated runs for accuracy
  `python -m src.cli run --profile full`

After each profile completes, the CLI reports the total runtime.

### Outputs
`results/summary/`
- `games_<mode>.csv` – per-game rows
- `metrics_<mode>.csv` – per-solver metrics (2 d.p.)
- `metrics_with_cis_<mode>.csv` – per-solver metrics with bootstrap 95% CIs (2 d.p.)

`results/plots/`
- `solver_bars_<mode>.png` (uses CIs if available)

### Statistical tests
After running a benchmark you can compare solvers using paired tests:

```bash
python -m src.stats results/summary/games_full.csv --metric guesses --test wilcoxon
```
This writes a table of pairwise test statistics and p-values comparing solver performance.

