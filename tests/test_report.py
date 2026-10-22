"""Tests for the self-contained offline HTML report."""

import json
import re
from copy import deepcopy
from html.parser import HTMLParser
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from plotly.offline import get_plotlyjs

from simple_backtest import (
    Backtest,
    BacktestConfig,
    BuyAndHoldStrategy,
    GridSearchOptimizer,
    MovingAverageStrategy,
    Strategy,
)
from simple_backtest.core.results import StrategyResult

EVIL = 'Evil</script><img src=x onerror=alert(1)> "q" & ünï'


class Hold(Strategy):
    def predict(self, data, trade_history):
        return self.hold()


class BrokenOnce(Strategy):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.calls = 0

    def predict(self, data, trade_history):
        self.calls += 1
        if self.calls == 3:
            raise RuntimeError("boom </script><b>bad</b> & co " + "x" * 300)
        return self.hold()


class Page(HTMLParser):
    """Collect what a reader sees with scripts ignored, plus scripts and references."""

    def __init__(self, html):
        super().__init__()
        self.text, self.scripts, self.refs, self.tags, self._script = [], [], [], [], False
        self.feed(html)
        self.visible = " ".join(" ".join(self.text).split())

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self._script = tag == "script"
        self.refs += [v for k, v in attrs if k in ("src", "href") and not v.startswith("#")]

    def handle_endtag(self, tag):
        self._script = False

    def handle_data(self, data):
        (self.scripts if self._script else self.text).append(data)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(54)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 60)))
    return pd.DataFrame(
        {"Open": close, "High": close * 1.02, "Low": close * 0.98, "Close": close},
        index=pd.date_range("2024-01-01", periods=60),
    )


@pytest.fixture(scope="module")
def results(data):
    config = BacktestConfig.zero_commission(
        initial_capital=1000,
        lookback_period=4,
        parallel_execution=False,
        random_seed=4,
        error_policy="continue",
    )
    return Backtest(data, config).run(
        [
            BuyAndHoldStrategy(shares=1),
            Hold(),
            BuyAndHoldStrategy(shares=10000, name="TooBig"),
            BrokenOnce(name=EVIL),
        ]
    )


def export(result, tmp_path, **kwargs):
    with mock.patch.object(Backtest, "run", side_effect=AssertionError("export re-ran")):
        path = result.export_html(tmp_path / "report.html", **kwargs)
    return path.read_text(encoding="utf-8")


def test_comparison_report_is_one_offline_file_with_static_answers(results, data, tmp_path):
    before = data.copy()
    snapshot = {name: deepcopy(results[name]) for name in results}
    html = export(results, tmp_path, price_data=data)
    page = Page(html)

    assert list(tmp_path.iterdir()) == [tmp_path / "report.html"]
    assert html.count(get_plotlyjs()) == 1
    assert page.refs == [] and "img" not in page.tags and "link" not in page.tags
    assert not re.findall(r"https?://", html.replace(get_plotlyjs(), ""))
    chart_ids = re.findall(r'<div class="chart" id="([^"]+)"', html)
    assert chart_ids == [f"chart-{n}" for n in range(1, 6)]  # equity + 4 trade charts

    for answer in (
        "Best total return",
        "benchmark",
        "No orders were placed",
        "insufficient_cash",
        "strategy error",
        "n/a (no closed trades)",
        "Input data span",
        "Trading span",
        "in-sample",
    ):
        assert answer in page.visible
    assert "Trade price charts are not included" not in page.visible
    data_scripts = [s for s in page.scripts if s.lstrip().startswith("{")]
    assert len(data_scripts) == 5 and all(json.loads(s)["data"] for s in data_scripts)

    pd.testing.assert_frame_equal(data, before)
    for name in results:
        assert results[name]["trade_history"] == snapshot[name]["trade_history"]
        pd.testing.assert_series_equal(
            results[name]["portfolio_values"], snapshot[name]["portfolio_values"]
        )


def test_user_text_stays_inert_and_previews_are_bounded(results, tmp_path):
    html = export(results, tmp_path)
    page = Page(html)
    assert EVIL in page.visible
    assert "boom </script><b>bad</b> & co" in page.visible
    assert html.count("</script>") == len(page.scripts)  # only our own closing tags
    assert "<img" not in html.replace(get_plotlyjs(), "")
    assert "x" * 300 not in page.visible
    assert "shortened; export_json keeps the full text" in page.visible
    for script in page.scripts:
        if script.lstrip().startswith("{"):
            assert EVIL in [trace.get("name") for trace in json.loads(script)["data"]]


def test_without_price_data_report_explains_optional_trade_context(results, tmp_path):
    page = Page(export(results, tmp_path))
    assert "Trade price charts are not included" in page.visible
    assert "Trade price chart not included" in page.visible
    assert len(re.findall(r'class="chart"', export(results, tmp_path))) == 1


def test_partial_prices_keep_incomplete_data_guidance(results, data, tmp_path):
    page = Page(export(results, tmp_path, price_data=data.drop(data.index[10:13])))
    assert "Incomplete price data: 3 of" in page.visible


def test_single_result_has_assumptions_and_no_benchmark(results, data, tmp_path):
    result = results.get_strategy("BuyAndHold")
    page = Page(export(result, tmp_path, price_data=data))
    assert "No benchmark available for comparison" in page.visible
    for label in ("Strategy class", "Execution price", "Spread (bps)", "Random seed", "4"):
        assert label in page.visible
    assert "Hold" not in page.visible.replace("BuyAndHold", "")


def test_assumptions_list_values_that_differ_per_run(results, tmp_path):
    visible = Page(export(results, tmp_path)).visible
    assumptions = visible[visible.index("Assumptions As recorded") :]
    assert 'TooBig: {"shares": 10000' in assumptions
    assert 'BuyAndHold: {"shares": 1,' in assumptions
    assert "benchmark: buy_and_hold" in assumptions


def test_legacy_result_marks_missing_records_without_inventing_reasons(results, data, tmp_path):
    hold = results.get_strategy("Hold")
    legacy = StrategyResult(
        "Legacy",
        {k: getattr(hold, k) for k in ("metrics", "portfolio_values", "trade_history", "returns")},
    )
    page = Page(export(legacy, tmp_path))
    assert "not recorded" in page.visible
    assert "No orders were placed" not in page.visible
    assert "No fills: no order was executed in this run." in page.visible
    notes = " ".join(a.text for a in legacy.plot_trades(data).layout.annotations)
    assert "No orders were placed" not in notes and "No fills" in notes


def test_errors_without_order_records_do_not_claim_the_strategy_held(results, data, tmp_path):
    hold = results.get_strategy("Hold")
    fields = ("metrics", "portfolio_values", "trade_history", "returns")
    legacy = StrategyResult("Legacy", {**{k: getattr(hold, k) for k in fields}, "errors": []})

    page = Page(export(legacy, tmp_path))
    notes = " ".join(a.text for a in legacy.plot_trades(data).layout.annotations)

    for text in (page.visible, notes):
        assert "No orders were placed" not in text and "No fills" in text
    assert "No orders were placed" in " ".join(
        a.text for a in hold.plot_trades(data).layout.annotations
    )


def test_selected_search_trial_exports_without_another_simulation(data, tmp_path):
    optimizer = GridSearchOptimizer(verbose=False)
    optimizer.optimize(
        data=data,
        config=BacktestConfig.default(lookback_period=20, parallel_execution=False),
        strategy_class=MovingAverageStrategy,
        param_space={"short_window": [3, 5], "long_window": [10], "shares": [2]},
        metric="sharpe_ratio",
    )
    best = optimizer.best_result
    page = Page(export(best, tmp_path, price_data=data))
    assert best.name in page.visible and "in-sample" in page.visible
    assert '"short_window": ' in page.visible
