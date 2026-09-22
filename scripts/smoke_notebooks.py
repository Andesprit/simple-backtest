"""Execute every tutorial offline using the installed checkout and real kernels."""

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run notebook cells, skipping tagged installation cells and saving diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "notebooks")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = args.notebooks or sorted((ROOT / "notebooks").glob("*.ipynb"))
    environment = {**os.environ, "SIMPLE_BACKTEST_OFFLINE": "1", "PLOTLY_RENDERER": "json"}
    for path in paths:
        print(f"Executing {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=300,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        try:
            client.execute(env=environment)
        finally:
            nbformat.write(notebook, args.output_dir / path.name)
        print(f"Passed {path.name}", flush=True)


if __name__ == "__main__":
    main()
