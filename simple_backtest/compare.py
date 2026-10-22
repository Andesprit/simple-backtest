"""Compare two saved single-strategy runs: recorded inputs and settings first, then metrics.

Usage: python -m simple_backtest.compare BEFORE.json AFTER.json

Reads two ``StrategyResult.export_json(path)`` files (schema_version 1). It uses only
the standard library, never re-runs a strategy and never ranks the runs.
"""

import json
import sys

_MISSING = object()


def _load(path: str) -> dict:
    """Return one export as a dict, or raise ValueError with a one-line reason."""
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except OSError as error:
        raise ValueError(f"cannot read it ({error.strerror})") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"not valid JSON (line {error.lineno}, column {error.colno})") from None
    if not isinstance(data, dict):
        raise ValueError("not a StrategyResult export (expected a JSON object)")
    if data.get("schema_version") != 1:
        version = data.get("schema_version", "missing")
        raise ValueError(f"schema_version {version} is not supported; this command reads 1")
    if "folds" in data:
        raise ValueError(
            "a walk-forward export; save one fold with report.folds[0].result.export_json(path)"
        )
    results = data.get("results")
    if isinstance(results, dict) and {"columns", "index", "values"} <= set(results):
        raise ValueError(
            "a search export; save the selected run with optimizer.best_result.export_json(path)"
        )
    if isinstance(results, dict):
        names = ", ".join(list(results)[:5])
        raise ValueError(
            f"an export of several runs ({names}); save one with "
            "results.get_strategy(name).export_json(path)"
        )
    if not isinstance(data.get("metrics"), dict):
        raise ValueError("no metrics; expected the output of StrategyResult.export_json(path)")
    return data


def _fields(metadata) -> dict:
    """Recorded provenance as flat names, one level into groups such as config and data."""
    fields = {}
    for key, value in (metadata if isinstance(metadata, dict) else {}).items():
        if isinstance(value, dict):
            fields.update({f"{key}.{inner}": item for inner, item in value.items()})
        else:
            fields[key] = value
    return fields


def _show(value) -> str:
    return "not recorded" if value is _MISSING else json.dumps(value, sort_keys=True)


def _union(first: dict, second: dict) -> list:
    return list(first) + [key for key in second if key not in first]


def main(argv=None) -> int:
    paths = sys.argv[1:] if argv is None else argv
    if len(paths) != 2:
        print("usage: python -m simple_backtest.compare BEFORE.json AFTER.json", file=sys.stderr)
        return 2
    runs = []
    for path in paths:
        try:
            runs.append(_load(path))
        except ValueError as error:
            print(f"{path}: {error}", file=sys.stderr)
            return 2
    before, after = runs
    lines = [f"Before: {paths[0]} ({before.get('name', 'unnamed')})"]
    lines.append(f"After:  {paths[1]} ({after.get('name', 'unnamed')})")

    old, new = _fields(before.get("metadata")), _fields(after.get("metadata"))
    changed, one_sided, equal = [], [], 0
    for key in _union(old, new):
        pair = (old.get(key, _MISSING), new.get(key, _MISSING))
        line = f"  {key}: {_show(pair[0])} -> {_show(pair[1])}"
        if _MISSING in pair:
            one_sided.append(line)
        elif pair[0] != pair[1]:
            changed.append(line)
        else:
            equal += 1
    lines.append("")
    lines.append("Recorded differences in inputs and settings:" + ("" if changed else " none"))
    lines += changed
    if one_sided:
        lines.append("Recorded in only one file (unknown, not equal):")
        lines += one_sided
    lines.append(f"{equal} other recorded fields are equal.")

    lines.append("")
    lines.append("Metrics, before -> after (null means undefined; no ranking):")
    for key in _union(before["metrics"], after["metrics"]):
        lines.append(
            f"  {key}: {_show(before['metrics'].get(key, _MISSING))} -> "
            f"{_show(after['metrics'].get(key, _MISSING))}"
        )
    lines.append("")
    lines.append(
        "These are recorded facts only: when inputs or settings differ, this comparison "
        "does not show why a metric changed."
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
