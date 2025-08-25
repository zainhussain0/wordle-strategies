#Plots/analysis
import os, json, collections
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import get_config
from .eval import GameResult  # if want to type-annotate
from .wordle_core import target_words, partitions_for_guess, entropy_of_guess
CONFIG = get_config()
SUMMARY_DIR = os.path.join("results","summary")
PLOTS_DIR   = os.path.join("results","plots")


df = pd.DataFrame(rows)

# Distribution of guesses
plt.hist(df["num_guesses"], bins=[1,2,3,4,5,6,7], rwidth=0.8)
plt.title(f"Guess distribution - {s.name}")
plt.xlabel("Guesses used")
plt.ylabel("Frequency")
plt.savefig(PLOTS_DIR + f"/{s.name}_guess_distribution.png")

# Cumulative success curve
cum = df["num_guesses"].value_counts().sort_index().cumsum() / len(df)
cum.plot(drawstyle="steps-post")
plt.title(f"Cumulative success curve - {s.name}")
plt.xlabel("Guess number")
plt.ylabel("Cumulative proportion solved")
plt.savefig(PLOTS_DIR + f"/{s.name}_cumulative_success.png")

def plot_guess_distribution(rows: List[GameResult], title="Distribution of guesses"):
    data = [(r.solver, r.guesses) for r in rows if r.success]
    if not data:
        print("No successful games to plot.")
        return
    df = pd.DataFrame(data, columns=["solver","guesses"])
    # bar of proportions per solver
    solvers = sorted(df["solver"].unique())
    maxg = 6
    fig, ax = plt.subplots(figsize=(7,4))
    for i, s in enumerate(solvers):
        sub = df[df.solver==s]["guesses"].value_counts().sort_index()
        xs = list(range(2, maxg+1))
        ys = [sub.get(k,0)/len(df[df.solver==s]) for k in xs]
        ax.plot(xs, np.cumsum(ys), marker="o", label=s)  # cumulative success curve
    ax.set_xlabel("Guess number")
    ax.set_ylabel("Cumulative proportion solved")
    ax.set_title("Cumulative proportion solved by guess")
    ax.legend()
    path = os.path.join(PLOTS_DIR, "cumulative_success.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    print("Saved:", path)

def plot_avg_guesses_bars(metrics):
    df = pd.DataFrame(metrics)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(df["solver"], df["avg_guesses_success"])
    ax.set_ylabel("Average guesses (successes)")
    ax.set_title("Avg guesses by solver")
    path = os.path.join(PLOTS_DIR, "solver_avg_guesses.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    print("Saved:", path)

def global_first_move_entropy(valid_pool=None, candidates_pool=None, out_path=None, topn=50):
    valid_pool = valid_pool or all_valid_words
    candidates_pool = candidates_pool or target_words
    scored = []
    for g in valid_pool:
        H = entropy_of_guess(g, candidates_pool)
        scored.append((g, H))
    scored.sort(key=lambda x: x[1], reverse=True)
    path = out_path or os.path.join(SUMMARY_DIR, "global_first_move_entropy.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank","word","entropy_bits"])
        for i,(g,H) in enumerate(scored, start=1):
            w.writerow([i, g, H])
    print("Wrote:", path)
    return scored[:topn]

top50 = global_first_move_entropy()

def hardest_words(rows):
    by_solver = collections.defaultdict(list)
    for r in rows:
        by_solver[r.solver].append(r)

    out = {}
    for s, lst in by_solver.items():
        fail_ct = collections.Counter([r.target for r in lst if not r.success]).most_common(20)
        # words that needed the most guesses (among successes)
        succ = [r for r in lst if r.success]
        by_word = collections.defaultdict(list)
        for r in succ:
            by_word[r.target].append(r.guesses)
        worst_solved = sorted(((w, statistics.mean(gs)) for w,gs in by_word.items()),
                              key=lambda x: x[1], reverse=True)[:20]
        out[s] = {"top_failures": fail_ct, "worst_solved": worst_solved}
    return out

hw = hardest_words(rows)
hw  # inspect or write to CSVs

turnlog = pd.read_csv(TURNLOG_CSV)

# median/IQR of |C| before guess by solver/turn
stats = (turnlog.groupby(["solver","turn"])["candidates_before"]
         .agg(median="median",
              p25=lambda s: np.percentile(s,25),
              p75=lambda s: np.percentile(s,75))
         .reset_index())

plt.figure(figsize=(7,4))
for s in sorted(stats["solver"].unique()):
    sub = stats[stats["solver"]==s]
    plt.plot(sub["turn"], sub["median"], marker="o", label=s)
plt.yscale("log")
plt.xlabel("Turn"); plt.ylabel("|Candidates| before guess (median)")
plt.title("Candidate set shrinkage over turns")
plt.legend()
path = os.path.join(PLOTS_DIR, "candidate_shrinkage.png")
plt.tight_layout(); plt.savefig(path, dpi=200); print("Saved:", path)

def run_with_forced_opener(solver: Solver, opener: str, targets: list):
    rows = []
    for t in targets:
        candidates = [w for w in target_words]
        history, seq = [], []
        solver.reset()

        # forced turn 1
        g = opener
        seq.append(g)
        patt = cached_pattern(g, t)
        history.append((g, patt))
        if g == t:
            rows.append(GameResult(solver=solver.name+f"+forced:{opener}", target=t, success=True, guesses=1, sequence=seq))
            continue
        candidates = [w for w in candidates if consistent(w, g, patt)]

        # continue policy
        for turn in range(2, 7):
            cand_before = len(candidates)
            g = solver.guess(candidates, all_valid_words if CONFIG["allow_probes"] else candidates, history, CONFIG["hard_mode"])
            seq.append(g)
            patt = cached_pattern(g, t)
            history.append((g, patt))
            if g == t:
                rows.append(GameResult(solver=solver.name+f"+forced:{opener}", target=t, success=True, guesses=turn, sequence=seq))
                break
            candidates = [w for w in candidates if consistent(w, g, patt)]
        else:
            rows.append(GameResult(solver=solver.name+f"+forced:{opener}", target=t, success=False, guesses=6, sequence=seq))
    return rows

# Example (dev subset to keep it quick)
subset = random.sample(target_words, k=200)
rows_forced = []
for opener in ["crane","slate","soare"]:
    for s in [HeuristicSolver(), EntropySolver(), MCTSSolver()]:
        rows_forced += run_with_forced_opener(s, opener, subset)
# Aggregate rows_forced as usual for a small comparison table

def bootstrap_ci(samples, fn, B=2000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(samples); stats = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        stats.append(fn(samples[idx]))
    stats.sort()
    lo = stats[int((alpha/2)*B)]
    hi = stats[int((1-alpha/2)*B)]
    return (lo, hi)

def summarize_with_cis(rows):
    by_solver = collections.defaultdict(list)
    for r in rows: by_solver[r.solver].append(r)
    out = []
    for s, lst in by_solver.items():
        succ = np.array([int(r.success) for r in lst])
        wr = succ.mean()*100
        wr_ci = [x*100 for x in bootstrap_ci(succ, np.mean)]
        gsucc = np.array([r.guesses for r in lst if r.success])
        if len(gsucc):
            avg = gsucc.mean()
            avg_ci = bootstrap_ci(gsucc, np.mean)
        else:
            avg, avg_ci = float("nan"), (float("nan"), float("nan"))
        out.append({
            "solver": s,
            "win_rate": wr, "win_rate_lo": wr_ci[0], "win_rate_hi": wr_ci[1],
            "avg_guesses": avg, "avg_lo": avg_ci[0], "avg_hi": avg_ci[1]
        })
    df = pd.DataFrame(out)
    path = os.path.join(SUMMARY_DIR, f"metrics_with_cis_{CONFIG['mode']}.csv")
    df.to_csv(path, index=False); print("Wrote:", path)
    return df

cis_df = summarize_with_cis(rows)
cis_df

# %%
def save_entropy_partitions_demo(guess="stare", fname="entropy_partitions.png"):
    # choose a mid-sized candidate set (simulate early/mid game)
    # here we just use the full target list for illustration
    cands = target_words
    buckets = partitions_for_guess(guess, cands)
    labels = ["".join(map(str,b)) for b in buckets.keys()]
    sizes = list(buckets.values())
    # sort by size
    order = np.argsort(sizes)[::-1]
    sizes = [sizes[i] for i in order]
    labels = [labels[i] for i in order]
    plt.figure(figsize=(8,3))
    plt.bar(range(len(sizes)), sizes)
    plt.xlabel("Feedback pattern bucket")
    plt.ylabel("Bucket size")
    plt.title(f"Partitions for guess '{guess}' (H={entropy_of_guess(guess, cands):.2f} bits)")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(path, dpi=200)
    print("Saved:", path)

save_entropy_partitions_demo("stare", "entropy_partitions.png")

def make_all_figures():
    # Expect summary CSVs to exist from a previous run
    # Read what you need (e.g., metrics CSV) and call your plotting functions
    # Example: plot_guess_distribution(rows) if you load rows, or create a summary plot from metrics CSVs.
    print("Implement me: read CSVs from results/summary and write figures to results/plots.")
