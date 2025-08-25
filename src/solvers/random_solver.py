import random
from typing import List, Tuple

from .base import Solver, register_solver
from ..wordle_core import Pattern


@register_solver
class RandomSolver(Solver):
    name = "Random"

    def guess(
        self,
        candidates: List[str],
        valid: List[str],
        history: List[Tuple[str, Pattern]],
        hard_mode: bool,
    ) -> str:
        self.reset_diag()
        pool = candidates
        g = random.choice(pool) if pool else random.choice(valid)
        self._diag.update(
            {
                "score_name": "none",
                "score_value": None,
                "is_probe": g not in candidates,
                "topk": [],
                "extras": {},
            }
        )
        return g

