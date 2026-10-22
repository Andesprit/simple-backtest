"""Read-only comparison of two saved single-strategy JSON exports."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from simple_backtest import Backtest, BacktestConfig, BuyAndHoldStrategy, WalkForwardOptimizer
from simple_backtest.compare import main


@pytest.fixture
def exports(tmp_path):
    close = 100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, 60)))
    data = pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close},
        index=pd.date_range("2024-01-01", periods=60),
    )
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    config = BacktestConfig.zero_commission(lookback_period=5, parallel_execution=False)
    results = Backtest(data, config).run([BuyAndHoldStrategy(shares=1)])
    results.get_strategy("BuyAndHold").export_json(before)
    config = BacktestConfig.zero_commission(
        lookback_period=5, parallel_execution=False, final_liquidation=True
    )
    Backtest(data.iloc[10:], config).run([BuyAndHoldStrategy(shares=2)]).get_strategy(
        "BuyAndHold"
    ).export_json(after)
    return before, after, results, data


def test_recorded_differences_come_before_both_metric_values(exports, capsys):
    before, after, _, _ = exports

    assert main([str(before), str(after)]) == 0

    out = capsys.readouterr().out
    settings, metrics = out.split("Metrics, before -> after")
    assert "data.rows: 60 -> 50" in settings
    assert "config.final_liquidation: false -> true" in settings
    assert 'strategy.parameters: {"name": "BuyAndHold", "shares": 1} -> {"name"' in settings
    assert "other recorded fields are equal" in settings
    saved = [json.loads(path.read_text())["metrics"]["total_return"] for path in (before, after)]
    assert f"total_return: {json.dumps(saved[0])} -> {json.dumps(saved[1])}" in metrics
    assert "does not show why" in metrics
    assert "better" not in out.lower() and "winner" not in out.lower()


def test_missing_null_and_one_sided_fields_are_never_called_equal(exports, capsys):
    before, after, _, _ = exports
    edited = json.loads(after.read_text())
    edited["metrics"]["new_metric"] = 1.5  # Only in the second file.
    edited["metrics"]["sharpe_ratio"] = None  # Undefined numbers are saved as null.
    del edited["metadata"]["data"]["fingerprint"]
    edited["metadata"]["config"]["brand_new_setting"] = 3
    after.write_text(json.dumps(edited))

    assert main([str(before), str(after)]) == 0

    out = capsys.readouterr().out
    unknown = out.split("Recorded in only one file")[1].split("Metrics")[0]
    assert "data.fingerprint: " in unknown and "-> not recorded" in unknown
    assert "config.brand_new_setting: not recorded -> 3" in unknown
    assert "new_metric: not recorded -> 1.5" in out
    assert "-> null" in out.split("sharpe_ratio:")[1].splitlines()[0]


def test_same_file_has_no_differences(exports, capsys):
    before, _, _, _ = exports

    assert main([str(before), str(before)]) == 0

    out = capsys.readouterr().out
    assert "Recorded differences in inputs and settings: none" in out
    assert "Recorded in only one file" not in out


def test_unsupported_inputs_get_one_line_reasons(exports, tmp_path, capsys):
    before, _, results, data = exports
    collection = tmp_path / "collection.json"
    results.export_json(collection)
    folds = tmp_path / "folds.json"
    WalkForwardOptimizer(train_size=0.5, n_splits=2, verbose=False).optimize_report(
        data,
        BacktestConfig.zero_commission(lookback_period=5, parallel_execution=False),
        BuyAndHoldStrategy,
        {"shares": [1]},
        metric="total_return",
    ).export_json(folds)
    newer = tmp_path / "newer.json"
    newer.write_text(json.dumps({"schema_version": 2, "metrics": {}}))
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    no_metrics = tmp_path / "no_metrics.json"
    no_metrics.write_text(json.dumps({"schema_version": 1, "name": "x"}))

    cases = {
        collection: "results.get_strategy(name).export_json(path)",
        folds: "report.folds[0].result.export_json(path)",
        newer: "schema_version 2 is not supported",
        broken: "not valid JSON (line 1, column 2)",
        no_metrics: "StrategyResult.export_json(path)",
        tmp_path / "absent.json": "cannot read it",
    }
    for path, reason in cases.items():
        assert main([str(before), str(path)]) == 2
        err = capsys.readouterr().err
        assert err.startswith(f"{path}: ") and reason in err and "Traceback" not in err
    assert main([str(before)]) == 2
    assert "usage: python -m simple_backtest.compare" in capsys.readouterr().err


def test_runs_as_a_module(exports):
    before, after, _, _ = exports

    shown = subprocess.run(
        [sys.executable, "-m", "simple_backtest.compare", str(before), str(after)],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "data.rows: 60 -> 50" in shown.stdout
