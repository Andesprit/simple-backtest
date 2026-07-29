"""Offline release-readiness checks for versions and published examples."""

import ast
import importlib
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import simple_backtest

ROOT = Path(__file__).parents[1]


def test_release_version_is_consistent():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lockfile = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_project = next(
        package for package in lockfile["package"] if package["name"] == "simple-backtest"
    )

    assert metadata["project"]["version"] == "0.4.0"
    assert locked_project["version"] == metadata["project"]["version"]
    assert simple_backtest.__version__ == "0.4.0"


def test_notebooks_are_clean_and_import_public_api_names():
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))

    assert len(notebooks) == 6
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []

            source = "".join(cell["source"])
            source = "\n".join(
                "" if line.lstrip().startswith(("!", "%")) else line for line in source.splitlines()
            )
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.module is None or not node.module.startswith("simple_backtest"):
                    continue
                module = importlib.import_module(node.module)
                for imported_name in node.names:
                    if imported_name.name == "*":
                        continue
                    assert hasattr(module, imported_name.name), (
                        f"{path.name} imports missing {node.module}.{imported_name.name}"
                    )
