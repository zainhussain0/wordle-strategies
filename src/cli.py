import argparse

from .runner import run_profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="run", choices=["run", "figures"])
    p.add_argument("--profile", default="smoke")
    args = p.parse_args()

    if args.cmd == "run":
        run_profile(args.profile)
    else:
        from .figures import build_all

        build_all(mode=args.profile, results_dir="results/summary")


if __name__ == "__main__":
    main()

