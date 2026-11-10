"""Execute the solution notebooks headless, to check they still run.

    QSAR_DATA=./data python run_solutions.py [notebook ...]

Executed copies are written to ./executed/. Any cell that raises aborts the
run and prints the traceback, so this doubles as a regression test after
editing the material.
"""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QSAR_DATA", str(HERE / "data"))

targets = [Path(a) for a in sys.argv[1:]] or sorted((HERE / "notebooks").glob("*_solutions.ipynb"))
out_dir = HERE / "executed"
out_dir.mkdir(exist_ok=True)

for path in targets:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    print(f"running {path.name} ...", flush=True)
    client.execute()
    nbformat.write(nb, out_dir / path.name)
    print(f"  ok -> executed/{path.name}", flush=True)
