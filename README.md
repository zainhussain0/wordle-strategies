# Wordle Strategies

This repository contains the full codebase and reproducibility materials for the MSc Data Science Extended Research Project:
Solving Wordle: A Study of Human-Inspired and Algorithmic Strategies
University of Manchester, 2025

---

## Project Overview
This project benchmarks multiple solver strategies for the game *Wordle* under a unified simulation framework.  
Implemented solvers include:
- **Random baseline** (control)
- **Entropy maximisation** (information-theoretic)
- **Heuristic frequency solver** (psycholinguistic priors)
- **Monte Carlo Tree Search (MCTS)** (novel for Wordle)
- **Reinforcement Learning (exploratory)**

Results are reported in the accompanying report, with key metrics:
- Win rate (% of words solved within 6 guesses)
- Average number of guesses
- Runtime performance
- Best first-guess recommendations



## Repository Structure
```
wordle-strategies/
├── src/
│ ├── cli.py # Command line interface for running experiments
│ ├── solvers/ # Solver implementations
│ │ ├── random_solver.py
│ │ ├── entropy_solver.py
│ │ ├── heuristic_solver.py
│ │ ├── mcts_solver.py
│ │ └── rl_solver.py
│ ├── game/ # Wordle engine and feedback encoding
│ └── utils/ # Helper functions (logging, plotting, etc.)
├── data/
│ ├── word_lists/
│ │ ├── target_words.txt # Official solution words
│ │ └── valid_words.txt # Valid guesses
├── configs/
│ ├── config_fast.yaml # Fast evaluation (200 sampled targets)
│ └── config_full.yaml # Full evaluation (all 2,309 targets)
├── results/
│ ├── logs/ # Raw CSV outputs
│ ├── figures/ # Plots for report
│ └── tables/ # Aggregated performance tables
├── requirements.txt
└── README.md
```
## Installation

```bash
# Create environment (Python 3.10+ recommended)
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
```
## Running Experiments
The solvers and each evaluation profile can be ran through the terminal, with the code to do so shown below. For users with less technical experience, a GUI can be used instead to run the solvers:

### Using the GUI
To initialise the GUI from the repo root (after you’ve created and activated the venv and installed requirements):
```
python -m src.gui
```
#### What you will see:
A Tk window titled “Wordle Benchmark”  

A checklist of available solvers

A profile dropdown (smoke or full - fast profile was removed to minimise computational expense of GUI)  

A text area where run progress and summary are printed  

Click “Run” to execute the selected solvers under the chosen profile  

The results and plots will appear in their respective folders after runs
### Using the terminal

#### Smoke Test (To assess full functionality of code, fastest runtime)
```
python -m src.cli run --profile smoke
```
#### Fast Evaluation (≈1 hour runtime)
```
python -m src.cli run --profile fast
```
#### Full evaluation (≈12+ hours runtime)
```
python -m src.cli run --profile full
```



## Expected Outputs
CSV logs in ```results/logs/```

Summary tables in ```results/tables/```

Figures in ```results/figures/```

Example output (Fast evaluation, Entropy vs Heuristic):
| Solver    | Win Rate | Avg. Guesses | Runtime (s) |
| --------- | -------- | ------------ | ----------- |
| Random    | 72%      | 4.81         | 120         |
| Entropy   | 98%      | 3.42         | 1,150       |
| Heuristic | 93%      | 3.87         | 410         |


## Citation

If reusing this framework, please cite:  

Hussain, Z. (2025). Solving Wordle: A Study of Human-Inspired and Algorithmic Strategies. MSc Data Science Extended Research Project, The University of Manchester.

## License
This repository is provided for academic reproducibility. No commercial use without explicit permission.



