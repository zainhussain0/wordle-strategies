# wordle-strategies

Tools for benchmarking different approaches to solving the game **Wordle**. The
project ships with several ready-made profiles that evaluate solvers ranging
from simple random guessing to more sophisticated strategies.

## Prerequisites

- [Python](https://www.python.org/) 3.11 or later
- [Git](https://git-scm.com/) to clone the repository

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd wordle-strategies
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the benchmarks

The project provides multiple profiles that control how many games are played
and which outputs are produced.

- **Smoke** – quick check that everything works (≈10 games)
  ```bash
  python -m src.cli run --profile smoke
  ```

- **Fast** – development benchmark with confidence intervals and plots
  ```bash
  python -m src.cli run --profile fast
  ```

- **Full** – exhaustive benchmark over the entire Wordle list
  ```bash
  python -m src.cli run --profile full
  ```

To build plots from existing results, run:
```bash
python -m src.cli figures --profile <profile>
```

## Outputs

Results are written to `results/summary/`:

- `games_<mode>.csv` – per-game rows
- `metrics_<mode>.csv` – per-solver metrics (2 d.p.)
- `metrics_with_cis_<mode>.csv` – metrics with bootstrap 95% CIs (2 d.p., when enabled)

Generated plots are saved to `results/plots/` as `solver_bars_<mode>.png`.

