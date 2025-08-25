# %% [markdown]
# ## Load Word Lists
import collections
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
# %%
def load_words_from_file(path: str):
    with open(path, "r") as f:
        words = [w.strip().lower() for w in f if len(w.strip()) == 5 and w.strip().isalpha()]
    return sorted(set(words))

REPO = Path(__file__).resolve().parents[1]
ANSWERS_FILE = REPO / "data" / "wordle-answers-alphabetical.txt"
GUESSES_FILE = REPO / "data" / "wordle-allowed-guesses.txt"

target_words = load_words_from_file(ANSWERS_FILE)   # ~2309 words
valid_words  = load_words_from_file(GUESSES_FILE)   # ~10k words

# Combine for all legal guesses
all_valid_words = sorted(set(target_words) | set(valid_words))

print(f"Loaded {len(target_words)} target words, {len(valid_words)} valid guesses, {len(all_valid_words)} unique legal words.")
print("Examples (targets):", target_words[:10])
print("Examples (valids):", valid_words[:10])

# Feedback and Filtering(repeat safe)
# %%
# Tile codes: 2=green, 1=yellow, 0=grey
Pattern = Tuple[int, int, int, int, int]

def feedback_pattern(guess: str, secret: str) -> Pattern:
    """Compute Wordle feedback with correct handling of repeated letters."""
    g = list(guess)
    s = list(secret)
    res = [0]*5

    # Greens
    for i in range(5):
        if g[i] == s[i]:
            res[i] = 2
            s[i] = None  # consumed
            g[i] = None

    # Yellows
    for i in range(5):
        if g[i] is not None:
            try:
                j = s.index(g[i])
                res[i] = 1
                s[j] = None  # consume that occurrence
            except ValueError:
                pass
    return tuple(res)

def consistent(word: str, guess: str, patt: Pattern) -> bool:
    return feedback_pattern(guess, word) == patt

# Cache to speed entropy/MCTS
PATTERN_CACHE: Dict[Tuple[str,str], Pattern] = {}
def cached_pattern(g: str, c: str) -> Pattern:
    key = (g, c)
    if key not in PATTERN_CACHE:
        PATTERN_CACHE[key] = feedback_pattern(g, c)
    return PATTERN_CACHE[key]

#entropy helpers

def partitions_for_guess(guess: str, candidates: List[str]) -> Dict[Pattern, int]:
    """Count outcome pattern frequencies for a guess over candidates."""
    buckets = collections.Counter(cached_pattern(guess, c) for c in candidates)
    return buckets

def entropy_of_guess(guess: str, candidates: List[str]) -> float:
    N = len(candidates)
    if N <= 1:
        return 0.0
    buckets = partitions_for_guess(guess, candidates)
    H = 0.0
    for n in buckets.values():
        p = n / N
        H -= p * math.log(p, 2)
    return H


#MCTS helpers
def next_candidates(cands: List[str], guess: str, patt: Pattern) -> List[str]:
    return [w for w in cands if consistent(w, guess, patt)]

def rollout_policy(cands: List[str]) -> str:
    """Simple policy used during MCTS rollouts."""
    return random.choice(cands) if cands else ""

def simulate_to_terminal(cands: List[str], max_guesses_left: int = 6) -> Tuple[bool, int]:
    """Return (success, guesses_used) from this state using a fast policy."""
    guesses_used = 0
    remaining = list(cands)
    while guesses_used < max_guesses_left and remaining:
        guess = rollout_policy(remaining)
        guesses_used += 1
        # sample a plausible secret uniformly (simulation)
        secret = random.choice(remaining)
        patt = cached_pattern(guess, secret)
        remaining = next_candidates(remaining, guess, patt)
        if guess == secret:
            return True, guesses_used
    return False, guesses_used

# importable across the repo
target_words  # list[str]
valid_words   # list[str]
all_valid_words  # list[str]


