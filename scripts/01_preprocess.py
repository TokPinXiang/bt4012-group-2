"""Execute the complete, self-contained data notebook and its validation.

Inputs: data/raw/HI-Small_Trans.csv and HI-Small_Patterns.txt.
Outputs: data/processed/{train,validation,test,pattern_labels}.csv.
Owner: Kang Min Gu. The notebook is the single source of preprocessing logic.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true",
                        help="Check source-file placement without rebuilding data.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    required = [root / "data" / "raw" / name for name in
                ("HI-Small_Trans.csv", "HI-Small_Patterns.txt")]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing inputs: " + ", ".join(missing) +
                     ". Download the two HI-Small files listed in README.md.")
    if args.check_inputs:
        for path in required:
            print(f"Found {path.relative_to(root)} ({path.stat().st_size:,} bytes)")
        return 0

    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    notebook_path = root / "notebooks" / "01_prepare_final_data.ipynb"
    result_path = root / "outputs" / "data-preparation" / "01_prepare_final_data.executed.ipynb"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    class ProgressClient(NotebookClient):
        def process_message(self, message, cell, cell_index):
            if message.get("msg_type") == "stream":
                print(message["content"]["text"], end="", flush=True)
            return super().process_message(message, cell, cell_index)

    # Use this Python environment, without modifying the installed kernel registry.
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher",
                                "-f", "{connection_file}"]
    environment = dict(os.environ)
    runtime = root / "outputs" / ".runtime"
    for variable, directory in [("IPYTHONDIR", "ipython"),
                                ("JUPYTER_RUNTIME_DIR", "jupyter"),
                                ("MPLCONFIGDIR", "matplotlib")]:
        location = runtime / directory
        location.mkdir(parents=True, exist_ok=True)
        environment[variable] = str(location)
    client = ProgressClient(notebook, km=manager, timeout=None,
                            resources={"metadata": {"path": str(root)}})
    client.on_cell_executed = lambda **kwargs: nbformat.write(notebook, result_path)
    print("Running the data notebook with", sys.executable, flush=True)
    try:
        client.execute(env=environment)
    finally:
        nbformat.write(notebook, result_path)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    print("Validated data saved to data/processed/.")
    print("Executed notebook:", result_path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
