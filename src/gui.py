import tkinter as tk
from tkinter import ttk, messagebox

from .solvers import (
    RandomSolver,
    HeuristicSolver,
    PositionalHeuristicSolver,
    EntropySolver,
    MCTSSolver,
)
from .wordle_core import feedback_pattern, next_candidates, target_words, valid_words

SOLVER_OPTIONS = {
    "Random": RandomSolver,
    "Heuristic": HeuristicSolver,
    "Positional": PositionalHeuristicSolver,
    "Entropy": EntropySolver,
    "MCTS": MCTSSolver,
}


def solve_word(solver_cls, target: str):
    candidates = target_words.copy()
    valid = list(valid_words)
    history = []
    solver = solver_cls()
    solver.reset()
    for _ in range(6):
        guess = solver.guess(candidates, valid, history, hard_mode=False)
        patt = feedback_pattern(guess, target)
        history.append((guess, ''.join(str(x) for x in patt)))
        if all(v == 2 for v in patt):
            break
        candidates = next_candidates(candidates, guess, patt)
    return history


def run_gui():
    root = tk.Tk()
    root.title("Wordle Solver")

    solver_var = tk.StringVar(value="Heuristic")
    tk.Label(root, text="Solver:").grid(row=0, column=0, sticky="w")
    solver_menu = ttk.Combobox(root, textvariable=solver_var, values=list(SOLVER_OPTIONS.keys()), state="readonly")
    solver_menu.grid(row=0, column=1, sticky="we")

    tk.Label(root, text="Target word:").grid(row=1, column=0, sticky="w")
    target_entry = tk.Entry(root)
    target_entry.grid(row=1, column=1, sticky="we")

    output = tk.Text(root, height=10, width=40)
    output.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    def on_solve():
        target = target_entry.get().strip().lower()
        if len(target) != 5 or target not in target_words:
            messagebox.showerror("Error", "Please enter a valid 5-letter word.")
            return
        solver_cls = SOLVER_OPTIONS[solver_var.get()]
        hist = solve_word(solver_cls, target)
        output.delete("1.0", tk.END)
        for turn, (guess, patt) in enumerate(hist, 1):
            output.insert(tk.END, f"{turn}: {guess} {patt}\n")

    tk.Button(root, text="Solve", command=on_solve).grid(row=2, column=0, columnspan=2, pady=5)

    root.columnconfigure(1, weight=1)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
