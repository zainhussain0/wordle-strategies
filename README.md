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
- **Smoke** (seconds): sanity-check the pipeline  
  `python -m src.cli run --profile smoke`

- **Fast** (dev benchmark; with 95% CIs + plots):  
  `python -m src.cli run --profile fast`

- **Full** (full 2309-word sweep; with 95% CIs + plots):  
  `python -m src.cli run --profile full`

### Outputs
`results/summary/`
- `games_<mode>.csv` – per-game rows
- `metrics_<mode>.csv` – per-solver metrics (2 d.p.)
- `metrics_with_cis_<mode>.csv` – per-solver metrics with bootstrap 95% CIs (2 d.p.)

`results/plots/`
- `solver_bars_<mode>.png` (uses CIs if available)

