import argparse, os
from .config import load_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run","figures"])
    ap.add_argument("--profile", default=os.environ.get("WORDLE_PROFILE","fast_dev"))
    args = ap.parse_args()

    CONFIG = load_config(args.profile)

    if args.cmd == "run":
        # import here to avoid heavy imports on 'figures'
        from .eval import run_benchmark, summarize_with_cis
        from .solvers import RandomSolver, HeuristicSolver, EntropySolver, MCTSSolver
        solvers = [RandomSolver(), HeuristicSolver(), EntropySolver(), MCTSSolver()]
        rows, metrics = run_benchmark(solvers, mode=CONFIG["mode"])
        summarize_with_cis(rows)
    elif args.cmd == "figures":
        from .analysis import make_all_figures
        make_all_figures()

if __name__ == "__main__":
    main()
