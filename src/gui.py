import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

from .runner import SOLVER_REGISTRY, run_profile


def run_gui():
    root = tk.Tk()
    root.title("Wordle Benchmark")

    tk.Label(root, text="Solvers:").grid(row=0, column=0, sticky="nw")
    solver_vars: dict[str, tk.BooleanVar] = {}
    for i, name in enumerate(SOLVER_REGISTRY.keys(), start=1):
        var = tk.BooleanVar(value=False)
        solver_vars[name] = var
        tk.Checkbutton(root, text=name, variable=var).grid(row=i, column=0, sticky="w")

    tk.Label(root, text="Profile:").grid(row=0, column=1, sticky="w")
    profile_var = tk.StringVar(value="smoke")
    profile_menu = ttk.Combobox(
        root, textvariable=profile_var, values=["smoke", "full"], state="readonly"
    )
    profile_menu.grid(row=0, column=2, sticky="we")

    output = tk.Text(root, height=15, width=60)
    output.grid(row=len(SOLVER_REGISTRY) + 1, column=0, columnspan=3, padx=5, pady=5)

    def on_run():
        selected = [name for name, var in solver_vars.items() if var.get()]
        if not selected:
            messagebox.showerror("Error", "Select at least one solver")
            return
        output.delete("1.0", tk.END)
        output.insert(
            tk.END,
            f"Running profile '{profile_var.get()}' for: {', '.join(selected)}\n",
        )
        try:
            meta = run_profile(profile_var.get(), solvers=selected)
            metrics_csv = Path(meta["metrics_csv"])
            output.insert(tk.END, f"Results written to {metrics_csv}\n\n")
            try:
                output.insert(tk.END, metrics_csv.read_text())
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Error", str(e))

    tk.Button(root, text="Run", command=on_run).grid(
        row=len(SOLVER_REGISTRY) + 2, column=0, columnspan=3, pady=5
    )

    root.columnconfigure(2, weight=1)
    root.mainloop()


if __name__ == "__main__":
    run_gui()

