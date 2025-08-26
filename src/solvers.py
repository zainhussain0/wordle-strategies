import math, random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from .wordle_core import (
    Pattern,
    cached_pattern,
    entropy_of_guess,
    next_candidates,
    partitions_for_guess,
    simulate_to_terminal,
    target_words,
    all_valid_words,
)
from .config import get_config


random.seed(0)


class Solver:
    """Minimal Solver interface."""

    name = "Base"

    def reset(self) -> None:
        """Reset any internal state before a new game."""
        pass

    def guess(
        self,
        candidates: List[str],
        valid: List[str],
        history: List[Tuple[str, Pattern]],
        hard_mode: bool,
    ) -> str:
        """Return a legal guess."""
        raise NotImplementedError


class RandomSolver(Solver):
    name = "Random"

    def guess(self, candidates, valid, history, hard_mode):
        pool = candidates
        return random.choice(pool) if pool else random.choice(valid)


from collections import Counter

letter_counts = Counter("".join(target_words))
total_letters = sum(letter_counts.values())
letter_freq = {ch: letter_counts[ch] / total_letters for ch in "abcdefghijklmnopqrstuvwxyz"}

pos_counts = [Counter(word[i] for word in target_words) for i in range(5)]
pos_freq = [
    {ch: pos_counts[i][ch] / len(target_words) for ch in "abcdefghijklmnopqrstuvwxyz"}
    for i in range(5)
]


def heuristic_score(word: str, use_positional: bool = False) -> float:
    score = sum(letter_freq.get(ch, 0) for ch in set(word))
    if use_positional:
        score += sum(pos_freq[i].get(ch, 0) for i, ch in enumerate(word))
    vowels = set(word) & set("aeiou")
    score += 0.05 * len(vowels)
    if len(set(word)) < 5:
        score -= 0.05
    return score


class HeuristicSolver(Solver):
    name = "Heuristic"

    def guess(self, candidates, valid, history, hard_mode):
        scored = [(w, heuristic_score(w)) for w in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        g, _ = scored[0]
        return g


class PositionalHeuristicSolver(Solver):
    name = "Positional"

    def guess(self, candidates, valid, history, hard_mode):
        scored = [(w, heuristic_score(w, use_positional=True)) for w in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        g, _ = scored[0]
        return g


class EntropySolver(Solver):
    name = "Entropy"

    def guess(self, candidates, valid, history, hard_mode):
        cfg = get_config()
        pool = candidates if (hard_mode or not cfg["allow_probes"]) else all_valid_words
        max_cand = int(cfg.get("entropy_max_candidates", 2000))
        if len(pool) > max_cand:
            pool = sorted(pool, key=heuristic_score, reverse=True)[:max_cand]
        scored = [(g, entropy_of_guess(g, candidates)) for g in pool]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_g, _ = scored[0]
        return best_g


@dataclass
class Node:
    candidates: Tuple[str, ...]
    guess: Optional[str] = None
    pattern: Optional[Pattern] = None
    parent: Optional["Node"] = None
    children: Dict[Tuple[str, Pattern], "Node"] = field(default_factory=dict)
    N: int = 0
    W: float = 0.0

    def ucb(self, c: float = 1.4) -> float:
        if self.N == 0:
            return float("inf")
        Q = self.W / self.N
        return Q + c * math.sqrt(math.log(self.parent.N + 1) / self.N)


class MCTSSolver(Solver):
    name = "MCTS"

    def guess(self, candidates, valid, history, hard_mode):
        cfg = get_config()
        root = Node(tuple(sorted(candidates)))
        c_ucb = cfg["mcts"]["ucb_c"]
        R = cfg["mcts"]["rollouts_per_move"]
        pool = candidates if (hard_mode or not cfg["allow_probes"]) else all_valid_words

        for _ in range(R):
            node = root
            path = [node]
            while node.children:
                node = max(node.children.values(), key=lambda n: n.ucb(c_ucb))
                path.append(node)
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
                success, _ = simulate_to_terminal(list(node.candidates), max_guesses_left=6)
                reward = 1.0 if success else 0.0
            else:
                reward = 1.0
            for n in path:
                n.N += 1
                n.W += reward

        if not root.children:
            return max(candidates or pool, key=heuristic_score)

        child_stats = []
        for (g, p), n in root.children.items():
            Q = (n.W / n.N) if n.N > 0 else 0.0
            U = (c_ucb * math.sqrt(math.log(root.N + 1) / n.N)) if n.N > 0 else float("inf")
            child_stats.append((g, n.N, Q, Q + U))
        child_stats.sort(key=lambda x: x[1], reverse=True)
        best_g = child_stats[0][0]
        return best_g


__all__ = [
    "Solver",
    "RandomSolver",
    "HeuristicSolver",
    "PositionalHeuristicSolver",
    "EntropySolver",
    "MCTSSolver",
]

