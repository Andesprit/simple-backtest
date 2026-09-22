"""Offline release-readiness checks for versions and published examples."""

import ast
import importlib
import json
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import simple_backtest

ROOT = Path(__file__).parents[1]


def test_readme_quick_start_executes_offline(capsys):
    """Execute the actual published snippet and check its documented output."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quick_start = readme.split("## 🚀 Quick Start", 1)[1].split("## 📚 Documentation", 1)[0]
    code = re.search(r"```python\n(.*?)```", quick_start, re.DOTALL).group(1)
    output = re.search(r"```text\n(.*?)```", quick_start, re.DOTALL).group(1)
    exec(compile(code, "README.md quick start", "exec"), {})
    assert capsys.readouterr().out == output


def test_release_version_is_consistent():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lockfile = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_project = next(
        package for package in lockfile["package"] if package["name"] == "simple-backtest"
    )

    assert metadata["project"]["version"] == "0.5.0"
    assert locked_project["version"] == metadata["project"]["version"]
    assert simple_backtest.__version__ == "0.5.0"


def test_research_guide_executes_and_exports_portable_reports(tmp_path, monkeypatch):
    guide = (ROOT / "docs" / "REPRODUCIBLE_RESEARCH.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", guide, re.DOTALL).group(1)
    monkeypatch.chdir(tmp_path)
    exec(compile(code, "docs/REPRODUCIBLE_RESEARCH.md", "exec"), {})
    experiment = json.loads((tmp_path / "experiment.json").read_text())
    search = json.loads((tmp_path / "search.json").read_text())
    walk_forward = json.loads((tmp_path / "walk-forward.json").read_text())
    assert set(experiment["results"]) == {"RandomRebalance", "benchmark"}
    assert search["metadata"]["search"]["attempted_evaluations"] == 9
    assert search["metadata"]["search"]["strategy_class"].endswith("RandomRebalance")
    assert walk_forward["aggregate_metrics"]["test_bars"] == 60
    assert (tmp_path / "strategy.json").exists()


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
