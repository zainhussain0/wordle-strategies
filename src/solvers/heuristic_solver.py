from collections import Counter
from typing import List, Tuple

from .base import Solver, register_solver
from ..config import get_config
from ..wordle_core import Pattern, target_words


letter_counts = Counter("".join(target_words))
total_letters = sum(letter_counts.values())
letter_freq = {ch: letter_counts[ch] / total_letters for ch in "abcdefghijklmnopqrstuvwxyz"}


def heuristic_score(word: str, use_positional: bool = False) -> float:
    score = sum(letter_freq.get(ch, 0) for ch in set(word))
    vowels = set(word) & set("aeiou")
    score += 0.05 * len(vowels)
    if len(set(word)) < 5:
        score -= 0.05
    return score


@register_solver
class HeuristicSolver(Solver):
    name = "Heuristic"

    def guess(
        self,
        candidates: List[str],
        valid: List[str],
        history: List[Tuple[str, Pattern]],
        hard_mode: bool,
    ) -> str:
        self.reset_diag()
        cfg = get_config()
        pool = candidates
        scored = [(w, heuristic_score(w)) for w in pool]
        scored.sort(key=lambda x: x[1], reverse=True)
        g, sc = scored[0]
        self._diag.update(
            {
                "score_name": "heuristic_score",
                "score_value": float(sc),
                "is_probe": g not in candidates,
                "topk": scored[: cfg["analysis"]["topk"]],
                "extras": {},
            }
        )
        return g

