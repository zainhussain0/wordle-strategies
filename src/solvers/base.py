from typing import List, Tuple, Dict, Type

from ..wordle_core import Pattern

# Registry for solver classes
SOLVER_REGISTRY: Dict[str, Type["Solver"]] = {}


def register_solver(cls: Type["Solver"]) -> Type["Solver"]:
    """Class decorator to register Solver subclasses by name."""
    SOLVER_REGISTRY[cls.name.lower()] = cls
    return cls


class WithDiagnostics:
    def reset_diag(self):
        self._diag = {
            "score_name": None,
            "score_value": None,
            "is_probe": None,
            "topk": [],
            "extras": {},
        }

    def diag(self):
        return dict(self._diag)


class Solver(WithDiagnostics):
    name = "Base"

    def reset(self):
        self.reset_diag()

    def guess(
        self,
        candidates: List[str],
        valid: List[str],
        history: List[Tuple[str, Pattern]],
        hard_mode: bool,
    ) -> str:
        raise NotImplementedError

