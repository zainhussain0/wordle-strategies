import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .base import Solver, register_solver
from ..config import get_config
from ..wordle_core import (
    Pattern,
    cached_pattern,
    next_candidates,
    simulate_to_terminal,
    all_valid_words,
)
from .heuristic_solver import heuristic_score


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


@register_solver
class MCTSSolver(Solver):
    name = "MCTS"

    def guess(
        self,
        candidates: List[str],
        valid: List[str],
        history: List[Tuple[str, Pattern]],
        hard_mode: bool,
    ) -> str:
        self.reset_diag()
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
            g = max(candidates or pool, key=heuristic_score)
            self._diag.update(
                {
                    "score_name": "ucb_Q",
                    "score_value": None,
                    "is_probe": g not in candidates,
                    "topk": [],
                    "extras": {"rollouts": R, "note": "no_children_fallback"},
                }
            )
            return g

        child_stats = []
        for (g, p), n in root.children.items():
            Q = (n.W / n.N) if n.N > 0 else 0.0
            U = (c_ucb * math.sqrt(math.log(root.N + 1) / n.N)) if n.N > 0 else float("inf")
            child_stats.append((g, n.N, Q, Q + U))
        child_stats.sort(key=lambda x: x[1], reverse=True)
        best_g = child_stats[0][0]

        self._diag.update(
            {
                "score_name": "ucb_Q",
                "score_value": float(child_stats[0][2]),
                "is_probe": best_g not in candidates,
                "topk": [(g, float(Q)) for (g, visits, Q, u) in child_stats[: cfg["analysis"]["topk"]]],
                "extras": {
                    "rollouts": R,
                    "topk_visits": [(g, int(v)) for (g, v, Q, u) in child_stats[: cfg["analysis"]["topk"]]],
                    "topk_ucb": [(g, float(u)) for (g, v, Q, u) in child_stats[: cfg["analysis"]["topk"]]],
                },
            }
        )
        return best_g

