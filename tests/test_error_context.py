"""Post-run explanation of continued strategy errors (StrategyResult.explain_error)."""

import json
import pickle
from copy import deepcopy
from unittest import mock

import pandas as pd
import pytest

from simple_backtest import Backtest, BacktestConfig, Strategy
from simple_backtest.core import error_context
from simple_backtest.core.error_context import source_location
from simple_backtest.core.results import StrategyResult
from simple_backtest.utils.validation import StrategyExecutionError


class Broken(Strategy):
    """Buys on call 1, wrong-case column on call 4, sells half on the last bar; every callback fails."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def predict(self, data, trade_history):
        self.calls += 1
        if self.calls == 4:
            return self._indicator(data)
        if self.calls == 1:
            return self.buy(4)
        if self._portfolio_state["is_last_day"]:
            return self.sell(2)
        return self.hold()

    def _indicator(self, data):
        return data["close"]

    def on_trade_executed(self, trade_info):
        raise RuntimeError("ledger </td><script>x</script> é")


class FailsOnLastBar(Strategy):
    def predict(self, data, trade_history):
        if self._portfolio_state["is_last_day"]:
            raise ValueError("last bar indicator undefined")
        return self.buy(1) if not self.has_position() else self.hold()


def prices(rows=20, tz="UTC"):
    close = [100 + float(i % 7) for i in range(rows)]
    return pd.DataFrame(
        {
            "Open": close,
            "High": [c + 2 for c in close],
            "Low": [c - 2 for c in close],
            "Close": [c + 1 for c in close],
        },
        index=pd.date_range("2025-04-01", periods=rows, tz=tz),
    )


def run(strategy, data=None, **overrides):
    overrides.setdefault("error_policy", "continue")
    config = BacktestConfig.zero_commission(
        initial_capital=1000,
        lookback_period=5,
        parallel_execution=False,
        show_progress=False,
        risk_free_rate=0,
        final_liquidation=True,
        **overrides,
    )
    data = prices() if data is None else data
    return Backtest(data, config).run([strategy]).get_strategy(strategy.get_name())


@pytest.fixture(scope="module")
def broken():
    return run(Broken())


def stage_index(result, stage, last=False):
    indices = [i for i, e in enumerate(result.errors) if e["stage"] == stage]
    return indices[-1] if last else indices[0]


def test_prediction_error_points_to_user_code_and_shows_only_prior_bars(broken):
    data = prices()
    i = stage_index(broken, "prediction")
    error = broken.errors[i]
    assert error["source"]["function"] == "_indicator"
    assert error["source"]["filename"] == __file__
    assert error["source"]["raised_in"].startswith("base.py:")  # inside pandas

    text = broken.explain_error(i, price_data=data)
    assert f"error {i + 1} of {len(broken.errors)}" in text
    assert "Stage:   prediction" in text and "KeyError" in text
    assert "_indicator()" in text and "raised inside base.py" in text
    assert "submitted no order" in text and "no order was attempted" not in text
    assert "identical to the data this run used" in text
    assert "last 5 of the 5 bars of recorded lookback; predict() saw these rows" in text
    t = pd.Timestamp(error["timestamp"])
    position = data.index.get_loc(t)
    prior, execution = text.split("Bar at the error time")
    for bar in data.index[position - 5 : position]:
        assert str(bar) in prior
    assert str(data.index[position - 6]) not in text
    assert str(t) in execution and str(data.index[position + 1]) not in text
    assert "Not recorded: strategy attributes" in text
    assert f"result.errors[{i}]" in text


def test_callback_error_names_its_committed_fill_on_a_shared_bar(broken):
    last = stage_index(broken, "trade callback", last=True)
    previous = broken.errors[last - 1]
    assert previous["stage"] == "trade callback"
    assert previous["timestamp"] == broken.errors[last]["timestamp"]
    fills = [f for f in broken.trade_history if f["timestamp"] == previous["timestamp"]]
    assert len(fills) == 2 and fills[1].get("forced_liquidation")
    assert [previous["order_id"], broken.errors[last]["order_id"]] == [
        fills[0]["order_id"],
        fills[1]["order_id"],
    ]

    text = broken.explain_error(last)
    handled, others = text.split("Other fills on the same bar:")
    assert "matched by recorded order id" in handled and "[forced liquidation]" in handled
    assert fills[0]["order_id"] in others
    assert "a callback failure does not undo a fill" in text
    assert "unknown" not in text


def test_legacy_records_without_source_or_order_id_stay_qualified(broken):
    legacy = deepcopy(broken._to_dict())
    for error in legacy["errors"]:
        error.pop("source")
        error.pop("order_id", None)
    legacy.pop("metadata")
    result = StrategyResult("Broken", legacy)
    last = len(result.errors) - 1
    text = result.explain_error(last, price_data=prices())
    assert "Where:   not recorded (this result predates source recording)" in text
    assert "which one the callback was handling is unknown" in text
    assert "not verified: this result has no recorded data fingerprint" in text
    assert "not proven to be what the strategy saw" in text


def test_last_bar_prediction_error_still_shows_final_liquidation():
    result = run(FailsOnLastBar())
    (error,) = result.errors
    assert error["stage"] == "prediction"
    text = result.explain_error()
    assert "final liquidation can still fill" in text
    assert "[forced liquidation]" in text


def test_supplied_prices_that_differ_or_do_not_fit_are_labelled_or_refused(broken):
    i = stage_index(broken, "prediction")
    t = pd.Timestamp(broken.errors[i]["timestamp"])
    data = prices()
    partial = broken.explain_error(i, price_data=data.iloc[:-3])
    assert "NOT the data this run used" in partial
    assert "not proven to be what the strategy saw" in partial
    assert "predict() saw these rows" not in partial
    changed = data.copy()
    changed[["High", "Close"]] += 0.5
    assert "fingerprint differs" in broken.explain_error(i, price_data=changed)

    naive = broken.explain_error(i, price_data=data.tz_localize(None))
    assert "timezone (None) does not match" in naive
    assert "Bar at the error time" not in naive and f"result.errors[{i}]" in naive
    missing = broken.explain_error(i, price_data=data.drop(t))
    assert f"{t} is not in the supplied data" in missing
    bad = broken.explain_error(i, price_data=data.assign(High=data["Low"] - 5))
    assert "not valid price data" in bad and len(bad) < 3000

    short = broken.explain_error(i, price_data=data, context_bars=2)
    assert "last 2 of the 5 bars of recorded lookback" in short


def test_arguments_are_validated(broken):
    for index in (True, 1.0, "0"):
        with pytest.raises(TypeError, match="error_index must be an integer"):
            broken.explain_error(index)
    with pytest.raises(IndexError, match=f"from 0 to {len(broken.errors) - 1}"):
        broken.explain_error(len(broken.errors))
    assert broken.explain_error(-1).startswith(f"Broken: error {len(broken.errors)} of")
    for bars, kind in ((True, TypeError), (0, ValueError), (201, ValueError), (2.5, TypeError)):
        with pytest.raises(kind, match="context_bars"):
            broken.explain_error(0, context_bars=bars)
    with pytest.raises(TypeError, match="price_data must be a pandas DataFrame"):
        broken.explain_error(0, price_data=[1, 2])


def test_no_errors_explains_how_errors_get_recorded():
    class Hold(Strategy):
        def predict(self, data, trade_history):
            return self.hold()

    text = run(Hold()).explain_error()
    assert "no strategy errors were recorded" in text and "error_policy='continue'" in text


def test_source_capture_is_primitive_portable_and_bounded(broken, tmp_path):
    for error in broken.errors:
        source = error["source"]
        assert {type(v) for v in source.values()} <= {str, int}
        assert len(json.dumps(source)) < 700
    pickle.dumps(broken.errors)
    path = tmp_path / "result.json"
    broken.export_json(path)
    loaded = json.loads(path.read_text())
    assert loaded["errors"][0]["source"] == broken.errors[0]["source"]
    restored = StrategyResult("Broken", {**loaded, "portfolio_values": None, "returns": None})
    assert restored.explain_error(0) == broken.explain_error(0)

    assert source_location(ValueError("never raised")) is None
    record = {"timestamp": "2025-01-01", "stage": "prediction", "source": None}
    assert "no traceback was available" in error_context.error_where(record)


def test_frames_only_in_library_paths_fall_back_to_the_innermost_frame():
    code = compile(
        "def helper():\n    raise ValueError('x')\n", "/fake/site-packages/strat.py", "exec"
    )
    namespace = {}
    exec(code, namespace)
    try:
        namespace["helper"]()
    except ValueError as error:
        caught = error
    with mock.patch.object(error_context, "_LIBRARY_PATHS", ("/fake/", __file__)):
        location = source_location(caught)
    assert location["frame"] == "innermost" and location["function"] == "helper"
    assert location["filename"].endswith("strat.py")
    assert "innermost frame" in error_context.error_where({"source": location})


def test_long_names_are_bounded():
    long_name = "/" + "d" * 500 + "/s.py"
    namespace = {}
    exec(compile("def f():\n    raise ValueError('x')\n", long_name, "exec"), namespace)
    try:
        namespace["f"]()
    except ValueError as error:
        location = source_location(error)
    assert len(location["filename"]) == 200 and location["filename"].endswith("...")


def test_inspection_never_reruns_or_changes_the_result(broken):
    before = deepcopy(broken._to_dict())
    data = prices()
    data_before = data.copy()
    with (
        mock.patch.object(Backtest, "run", side_effect=AssertionError("replayed")),
        mock.patch.object(Broken, "predict", side_effect=AssertionError("replayed")),
    ):
        for i in range(len(broken.errors)):
            broken.explain_error(i, price_data=data)
    pd.testing.assert_frame_equal(data, data_before)
    assert json.dumps(broken._to_dict(), default=str) == json.dumps(before, default=str)


def test_raise_policy_and_successful_runs_are_unchanged():
    with pytest.raises(StrategyExecutionError) as caught:
        run(Broken(), error_policy="raise")
    assert isinstance(caught.value.__cause__, RuntimeError)

    class Clean(Strategy):
        def predict(self, data, trade_history):
            return self.buy(1) if not self.has_position() else self.hold()

    result = run(Clean())
    assert result.errors == [] and all("order_id" in f for f in result.trade_history)


def test_logged_warning_count_is_one_per_error():
    with mock.patch("simple_backtest.core.backtest.logger.warning") as warning:
        result = run(Broken())
    assert warning.call_count == len(result.errors)


def test_report_previews_source_and_stage_escaped(broken, tmp_path):
    path = broken.export_html(tmp_path / "r.html")
    html = path.read_text()
    section = html[html.index("Strategy errors") :]
    assert "<th>where</th>" in section and "<th>meaning</th>" in section
    assert "_indicator()" in section and "it was not undone" in section
    assert "<script>x</script>" not in html
    assert "&lt;/td&gt;&lt;script&gt;x&lt;/script&gt;" in section
    assert "result.explain_error(i, price_data=data)" in section


def test_print_points_to_explain_error(broken):
    assert (
        "Call result.explain_error() to see where and when it failed; raw records are in result.errors."
        in " ".join(str(broken).split())
    )


def test_readme_error_inspection_example_runs():
    import re
    from pathlib import Path

    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("### Inspecting a continued strategy error") :]
    code = re.search(r"```python\n(.*?)```", section, re.DOTALL).group(1)
    namespace = {}
    with mock.patch("builtins.print"):
        exec(compile(code, "README.md error inspection", "exec"), namespace)
    assert namespace["raw"]["source"]["function"] == "predict"
