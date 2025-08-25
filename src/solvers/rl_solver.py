import random
from typing import List, Tuple

from .base import Solver, register_solver
from ..wordle_core import Pattern


@register_solver
class RLSolver(Solver):
    name = "RL_exp"

    def __init__(self):
        self.Q = {}
        self.alpha = 0.10
        self.gamma = 0.95
        self.eps = 0.10

    def state_key(self, candidates, history):
        greens = [i for (g, p) in history for i, v in enumerate(p) if v == 2]
        ys = sorted({g[i] for (g, p) in history for i, v in enumerate(p) if v == 1})
        return (len(candidates), tuple(sorted(greens)), tuple(ys))

    def guess(
        self,
        candidates: List[str],
        valid: List[str],
        history: List[Tuple[str, Pattern]],
        hard_mode: bool,
    ) -> str:
        state = self.state_key(candidates, history)
        acts = candidates
        if not acts:
            return random.choice(valid)
        if random.random() < self.eps:
            return random.choice(acts)
        qvals = self.Q.get(state, {})
        return max(acts, key=lambda a: qvals.get(a, 0.0))

