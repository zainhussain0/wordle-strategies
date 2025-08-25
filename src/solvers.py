import math, random, collections
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from .wordle_core import (
    Pattern, feedback_pattern, cached_pattern, next_candidates,
    entropy_of_guess, partitions_for_guess, simulate_to_terminal,
    target_words, valid_words, all_valid_words
)
from .config import get_config



#Base Solver class
# %%
class Solver:
    name = "Base"

    def reset(self):
        """Reset any internal state before a new game."""
        pass

    def guess(self, candidates: List[str], valid: List[str], history: List[Tuple[str, Pattern]], hard_mode: bool) -> str:
        """Return a legal guess."""
        raise NotImplementedError

#Solver with diagnostics
class WithDiagnostics:
    def reset_diag(self):
        self._diag = {
            "score_name": None,       # e.g., "entropy_bits", "heuristic_score", "ucb_Q", "none"
            "score_value": None,
            "is_probe": None,         # True if guess not in candidates
            "topk": [],               # list of (word, score) for top-K
            "extras": {},             # free-form: buckets, ucb_parts, visits, etc.
        }

    def diag(self):
        # return a copy to avoid mutation surprises
        return dict(self._diag)

# Update the base Solver to inherit this (or mix into each concrete solver)
class Solver(WithDiagnostics):
    name = "Base"
    def reset(self):
        self.reset_diag()
    def guess(self, candidates, valid, history, hard_mode):
        raise NotImplementedError


#Random Solver
class RandomSolver(Solver):
    name = "Random"
    def guess(self, candidates, valid, history, hard_mode):
        self.reset_diag()
        pool = candidates  # keep admissible
        g = random.choice(pool) if pool else random.choice(valid)
        self._diag.update({
            "score_name": "none",
            "score_value": None,
            "is_probe": (g not in candidates),
            "topk": [],
            "extras": {}
        })
        return g


#Heuristic Solver
# %%
# Letter frequencies from target_words (unigram); positional priors optional
from collections import Counter

letter_counts = Counter("".join(target_words))
total_letters = sum(letter_counts.values())
letter_freq = {ch: letter_counts[ch]/total_letters for ch in 'abcdefghijklmnopqrstuvwxyz'}

def heuristic_score(word: str, use_positional: bool=False) -> float:
    score = sum(letter_freq.get(ch, 0) for ch in set(word))  # distinct letters to encourage coverage
    # vowel coverage bonus
    vowels = set(word) & set("aeiou")
    score += 0.05*len(vowels)
    # penalty for repeats
    if len(set(word)) < 5:
        score -= 0.05
    return score

class HeuristicSolver(Solver):
    name = "Heuristic"
    def guess(self, candidates, valid, history, hard_mode):
        self.reset_diag()
        cfg = get_config()
        pool = candidates
        # rank pool by heuristic_score
        scored = [(w, heuristic_score(w)) for w in pool]
        scored.sort(key=lambda x: x[1], reverse=True)
        g, sc = scored[0]
        self._diag.update({
            "score_name": "heuristic_score",
            "score_value": float(sc),
            "is_probe": (g not in candidates),
            "topk": scored[:cfg["analysis"]["topk"]],
            "extras": {}
        })
        return g

#Entropy Solver
class EntropySolver(Solver):
    name = "Entropy"
    def guess(self, candidates, valid, history, hard_mode):
        self.reset_diag()
        cfg = get_config()
        pool = candidates if (hard_mode or not cfg["allow_probes"]) else all_valid_words

        # optional prune by heuristic for speed
        K_prune = 2000 if len(pool) > 3000 else len(pool)
        if K_prune < len(pool):
            pool = sorted(pool, key=heuristic_score, reverse=True)[:K_prune]

        scored = []
        for g in pool:
            H = entropy_of_guess(g, candidates)
            scored.append((g, H))
        scored.sort(key=lambda x: x[1], reverse=True)
        best_g, best_H = scored[0]

        # buckets for chosen
        buckets = partitions_for_guess(best_g, candidates)
        bucket_sizes = {str(p): n for p, n in buckets.items()}

        self._diag.update({
            "score_name": "entropy_bits",
            "score_value": float(best_H),
            "is_probe": (best_g not in candidates),
            "topk": [(w, float(h)) for w, h in scored[:cfg["analysis"]["topk"]]],
            "extras": {"bucket_sizes": bucket_sizes, "pool_size": len(pool)}
        })
        return best_g


#MCTS Solver
@dataclass
class Node:
    candidates: Tuple[str, ...]       # immutable for hashing
    guess: Optional[str] = None       # guess that led here (None at root)
    pattern: Optional[Pattern] = None # feedback observed at edge into this node
    parent: Optional["Node"] = None
    children: Dict[Tuple[str, Pattern], "Node"] = field(default_factory=dict)
    N: int = 0
    W: float = 0.0                    # cumulative reward (higher better)

    def ucb(self, c=1.4) -> float:
        if self.N == 0:
            return float("inf")
        Q = self.W / self.N
        return Q + c * math.sqrt(math.log(self.parent.N + 1) / self.N)

class MCTSSolver(Solver):
    name = "MCTS"
    def guess(self, candidates, valid, history, hard_mode):
        self.reset_diag()
        cfg = get_config()
        root = Node(tuple(sorted(candidates)))
        c_ucb = cfg["mcts"]["ucb_c"]
        R = cfg["mcts"]["rollouts_per_move"]
        pool = candidates if (hard_mode or not cfg["allow_probes"]) else all_valid_words

        for _ in range(R):
            node = root
            path = [node]
            # selection
            while node.children:
                node = max(node.children.values(), key=lambda n: n.ucb(c_ucb))
                path.append(node)
            # expansion
            if len(node.candidates) > 1:
                K = 50 if len(pool) > 500 else min(50, len(pool))
                guess_candidates = sorted(pool, key=heuristic_score, reverse=True)[:K]
                g = random.choice(guess_candidates)
                secret = random.choice(node.candidates)
                patt = cached_pattern(g, secret)
                new_cands = tuple(sorted(next_candidates(list(node.candidates), g, patt)))
                child = node.children.get((g, patt))
                if child is None:
                    child = Node(candidates=new_cands, guess=g, pattern=patt, parent=node)
                    node.children[(g, patt)] = child
                node = child
                path.append(node)
                # simulation
                success, _ = simulate_to_terminal(list(node.candidates), max_guesses_left=6)
                reward = 1.0 if success else 0.0
            else:
                reward = 1.0  # single candidate considered solved in rollout

            # backprop
            for n in path:
                n.N += 1
                n.W += reward

        if not root.children:
            g = max(candidates or pool, key=heuristic_score)
            self._diag.update({
                "score_name": "ucb_Q",
                "score_value": None,
                "is_probe": (g not in candidates),
                "topk": [],
                "extras": {"rollouts": R, "note": "no_children_fallback"}
            })
            return g

        # pick best child by visits
        # collect per-(guess,pattern) stats for topK
        child_stats = []
        for (g,p), n in root.children.items():
            Q = (n.W/n.N) if n.N>0 else 0.0
            U = (c_ucb * math.sqrt(math.log(root.N+1)/n.N)) if n.N>0 else float('inf')
            child_stats.append((g, n.N, Q, Q+U))
        child_stats.sort(key=lambda x: x[1], reverse=True)  # by visits
        best_g = child_stats[0][0]

        self._diag.update({
            "score_name": "ucb_Q",
            "score_value": float(child_stats[0][2]),  # Q of chosen
            "is_probe": (best_g not in candidates),
            "topk": [(g, float(Q)) for (g, visits, Q, u) in child_stats[:cfg["analysis"]["topk"]]],
            "extras": {
                "rollouts": R,
                "topk_visits": [(g, int(v)) for (g, v, Q, u) in child_stats[:cfg["analysis"]["topk"]]],
                "topk_ucb": [(g, float(u)) for (g, v, Q, u) in child_stats[:cfg["analysis"]["topk"]]]
            }
        })
        return best_g


#RL Solver Class
# %%
class RLSolver(Solver):
    name = "RL_exp"
    def __init__(self):
        self.Q = {}  # very coarse state -> action value
        self.alpha = 0.10
        self.gamma = 0.95
        self.eps   = 0.10
    def state_key(self, candidates, history):
        # coarse encoding: (len(candidates), seen_green_positions_mask, seen_yellow_letters_set)
        greens = [i for (g,p) in history for i,v in enumerate(p) if v==2]
        ys = sorted({g[i] for (g,p) in history for i,v in enumerate(p) if v==1})
        return (len(candidates), tuple(sorted(greens)), tuple(ys))
    def guess(self, candidates, valid, history, hard_mode):
        state = self.state_key(candidates, history)
        acts = candidates  # keep admissible; huge action space, this is exploratory only
        if not acts:
            return random.choice(valid)
        if random.random() < self.eps:
            return random.choice(acts)
        # greedy
        qvals = self.Q.get(state, {})
        return max(acts, key=lambda a: qvals.get(a, 0.0))
