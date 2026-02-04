"""Tkinter dialog for entering task annotation before recording."""

import tkinter as tk
from typing import Optional, Tuple


# Module-level state to remember previous values across calls.
_prev_task_name = ""
_prev_instruction = ""


def ask_annotation() -> Optional[Tuple[str, str]]:
    """Show a dialog asking for task_name and instruction.

    Pre-fills with values from the previous call for convenience.

    Returns:
        (task_name, instruction) on OK, or None if cancelled.
    """
    global _prev_task_name, _prev_instruction

    result: Optional[Tuple[str, str]] = None

    root = tk.Tk()
    root.title("Recording Annotation")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    frame = tk.Frame(root, padx=16, pady=12)
    frame.pack()

    tk.Label(frame, text="Task Name:").grid(row=0, column=0, sticky="e", pady=4)
    task_entry = tk.Entry(frame, width=40)
    task_entry.grid(row=0, column=1, pady=4, padx=(8, 0))
    task_entry.insert(0, _prev_task_name)

    tk.Label(frame, text="Instruction:").grid(row=1, column=0, sticky="e", pady=4)
    instr_entry = tk.Entry(frame, width=40)
    instr_entry.grid(row=1, column=1, pady=4, padx=(8, 0))
    instr_entry.insert(0, _prev_instruction)

    def on_ok(_event=None):
        nonlocal result
        global _prev_task_name, _prev_instruction
        _prev_task_name = task_entry.get().strip()
        _prev_instruction = instr_entry.get().strip()
        result = (_prev_task_name, _prev_instruction)
        root.destroy()

    def on_cancel(_event=None):
        root.destroy()

    btn_frame = tk.Frame(frame)
    btn_frame.grid(row=2, column=0, columnspan=2, pady=(12, 0))
    tk.Button(btn_frame, text="OK", width=10, command=on_ok).pack(side="left", padx=4)
    tk.Button(btn_frame, text="Cancel", width=10, command=on_cancel).pack(side="left", padx=4)

    root.bind("<Return>", on_ok)
    root.bind("<Escape>", on_cancel)
    root.protocol("WM_DELETE_WINDOW", on_cancel)

    task_entry.focus_set()
    task_entry.select_range(0, tk.END)

    root.mainloop()
    return result
