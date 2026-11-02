"""Execute the two data notebooks in order in an in-process IPython session.

Useful for a local audit without a separate Jupyter kernel server.
"""
import os
from pathlib import Path

import nbformat
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    config = Config()
    config.HistoryManager.enabled = False
    config.HistoryManager.hist_file = ":memory:"
    config.TerminalInteractiveShell.simple_prompt = True
    for filename in ("00_dataset_validation.ipynb", "01_eda_rdd2022.ipynb"):
        path = ROOT / "notebooks" / filename
        notebook = nbformat.read(path, as_version=4)
        shell = TerminalInteractiveShell.instance(config=config)
        shell.reset(new_session=True)
        shell.run_line_magic("matplotlib", "inline")
        count = 0
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            count += 1
            print(f"{filename}: cell {index}", flush=True)
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
            if not result.success:
                print(captured.stdout)
                print(captured.stderr)
                raise RuntimeError(f"{filename}: cell {index} failed") from (
                    result.error_before_exec or result.error_in_exec)
            cell.execution_count = count
            cell.outputs = []
            if captured.stdout:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
            if captured.stderr:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
            for output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data,
                                                          metadata=output.metadata))
        nbformat.write(notebook, path)
        print(f"{filename}: {count} code cells OK", flush=True)
    print("Both data notebooks executed successfully.", flush=True)


if __name__ == "__main__":
    main()
