import sys
import os
import json
import io
import contextlib
import trace
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nbformat

notebook_path = PROJECT_ROOT / "notebooks" / "resume_job_matching.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

exec_globals = {"__name__": "__main__"}

exec_count = 1

for cell_idx, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        code = cell.source
        print(f"Executing cell {cell_idx + 1}...")
        
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        cell.outputs = []
        
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code, exec_globals)
            
            output_text = stdout_buf.getvalue()
            if output_text:
                cell.outputs.append(nbformat.v4.new_output(
                    output_type="stream",
                    name="stdout",
                    text=output_text
                ))
            cell.execution_count = exec_count
            exec_count += 1
            print(f"Cell {cell_idx + 1} executed successfully.")
        except Exception as e:
            print(f"Error in cell {cell_idx + 1}: {e}")
            err_text = stdout_buf.getvalue() + "\n" + str(e)
            cell.outputs.append(nbformat.v4.new_output(
                output_type="stream",
                name="stderr",
                text=err_text
            ))
            cell.execution_count = exec_count
            exec_count += 1
            break

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"\nNotebook execution completed and saved to {notebook_path}")
