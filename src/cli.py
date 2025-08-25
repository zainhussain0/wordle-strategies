# src/cli.py
import argparse, os
from .runner import run_profile

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--profile", default=os.environ.get("WORDLE_PROFILE", "fast_dev"))

    sub.add_parser("figures")

    args = ap.parse_args()
    if args.cmd == "run":
        run_profile(args.profile)
    else:
        # figures command
        from .analysis import make_all_figures
        make_all_figures()

if __name__ == "__main__":
    main()
