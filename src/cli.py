# src/cli.py
import argparse
from .runner import run_profile
from .figures import build_all_figures

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--profile", required=True)

    f = sub.add_parser("figures")

    args = p.parse_args()
    if args.cmd == "run":
        run_profile(args.profile)
    elif args.cmd == "figures":
        build_all_figures()

if __name__ == "__main__":
    main()
