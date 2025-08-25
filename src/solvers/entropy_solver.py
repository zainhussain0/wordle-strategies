from typing import List, Tuple

from .base import Solver, register_solver
from ..config import get_config
from ..wordle_core import Pattern, entropy_of_guess, partitions_for_guess, all_valid_words
from .heuristic_solver import heuristic_score


@register_solver
class EntropySolver(Solver):
    name = "Entropy"

    def guess(
        self,
        candidates: List[str],
        valid: List[str],
        history: List[Tuple[str, Pattern]],
        hard_mode: bool,
    ) -> str:
        self.reset_diag()
        cfg = get_config()
        pool = candidates if (hard_mode or not cfg["allow_probes"]) else all_valid_words

        K_prune = 2000 if len(pool) > 3000 else len(pool)
        if K_prune < len(pool):
            pool = sorted(pool, key=heuristic_score, reverse=True)[:K_prune]

        scored = [(g, entropy_of_guess(g, candidates)) for g in pool]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_g, best_H = scored[0]

        buckets = partitions_for_guess(best_g, candidates)
        bucket_sizes = {str(p): n for p, n in buckets.items()}

        self._diag.update(
            {
                "score_name": "entropy_bits",
                "score_value": float(best_H),
                "is_probe": best_g not in candidates,
                "topk": [(w, float(h)) for w, h in scored[: cfg["analysis"]["topk"]]],
                "extras": {"bucket_sizes": bucket_sizes, "pool_size": len(pool)},
            }
        )
        return best_g

