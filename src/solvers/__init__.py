from .base import Solver, WithDiagnostics, SOLVER_REGISTRY, register_solver
from .random_solver import RandomSolver
from .heuristic_solver import HeuristicSolver, heuristic_score
from .entropy_solver import EntropySolver
from .mcts_solver import MCTSSolver
from .rl_solver import RLSolver

DEFAULT_SOLVERS = ["random", "heuristic", "entropy", "mcts"]

__all__ = [
    "Solver",
    "WithDiagnostics",
    "SOLVER_REGISTRY",
    "register_solver",
    "RandomSolver",
    "HeuristicSolver",
    "EntropySolver",
    "MCTSSolver",
    "RLSolver",
    "heuristic_score",
    "DEFAULT_SOLVERS",
]

