"""Tests for the one-file offline walk-forward fold review."""

import base64
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
    GridSearchOptimizer,
    MovingAverageStrategy,
    Strategy,
    WalkForwardOptimizer,
)
from simple_backtest.optimization.results import WalkForwardReport

EVIL = 'Evil</script><img src=x onerror=alert(1)> "q" & ünï'


class Page(HTMLParser):
    """Collect IDs, outside references, static text and chart payloads without scripts."""

    def __init__(self):
        super().__init__()
        self.ids, self.refs, self.text, self.payloads = [], [], [], {}
        self._payload = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        self.refs += [
            value
            for key, value in attrs.items()
            if key in ("src", "href") and value and not value.startswith("#")
        ]
        if tag == "script":
            self._payload = attrs.get("id") if attrs.get("type") == "application/json" else ""

    def handle_endtag(self, tag):
        if tag == "script":
            self._payload = None

    def handle_data(self, data):
        if self._payload:
            self.payloads[self._payload] = json.loads(data)
        elif self._payload is None:
            self.text.append(data)


def read(path):
    html = path.read_text(encoding="utf-8")
    page = Page()
    page.feed(html)
    return html, page, " ".join(page.text)


def values(array):
    """Plotly stores numeric arrays as base64 'bdata'; decode them like Plotly.js does."""
    if isinstance(array, dict):
        return np.frombuffer(base64.b64decode(array["bdata"]), dtype=array["dtype"])
    return array


def prices(rows=160):
    rng = np.random.default_rng(115)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.008, rows)))
    return pd.DataFrame(
        {"Open": close, "High": close * 1.02, "Low": close * 0.98, "Close": close},
        index=pd.date_range("2024-01-01", periods=rows, tz="UTC"),
    )


def config(**overrides):
    return BacktestConfig.zero_commission(
        initial_capital=10000,
        lookback_period=20,
        parallel_execution=False,
        risk_free_rate=0,
        random_seed=11,
        periods_per_year=365,
        **overrides,
    )


class Greedy(Strategy):
    """Unaffordable orders every bar and a strategy error every seventh bar."""

    def __init__(self, tag="x", shares=1e9, name=None):
        super().__init__(name=name)
        self.tag, self.shares, self.bars = tag, shares, 0

    def predict(self, data, trade_history):
        self.bars += 1
        if self.bars % 7 == 0:
            raise ValueError(f"boom {self.tag} " + "y" * 300)
        return self.buy(self.shares)


@pytest.fixture(scope="module")
def data():
    return prices()


@pytest.fixture(scope="module")
def report(data):
    return WalkForwardOptimizer(train_size=0.5, n_splits=3, verbose=False).optimize_report(
        data,
        config(),
        MovingAverageStrategy,
        {"short_window": [3, 7], "long_window": [12, 20], "shares": [5]},
        metric="total_return",
    )


def test_one_offline_page_reviews_every_fold_without_rerunning(report, data, tmp_path):
    before_prices = data.copy(deep=True)
    before_summary = report.summary.copy(deep=True)
    before_folds = [
        json.dumps([f.result._to_dict(), f.benchmark._to_dict()], default=str) for f in report.folds
    ]
    with (
        mock.patch.object(Backtest, "run", side_effect=AssertionError("re-simulated")),
        mock.patch.object(GridSearchOptimizer, "optimize", side_effect=AssertionError("searched")),
    ):
        path = report.export_html(tmp_path / "wf.html", price_data=data)

    assert path == tmp_path / "wf.html"
    html, page, text = read(path)
    assert html.count(get_plotlyjs()) == 1
    assert len(page.ids) == len(set(page.ids))
    assert page.refs == []
    assert "Cash, positions, strategy state and random numbers reset" in text
    assert "no fold is picked as best" in text
    assert "Search objective and base optimizer: not recorded" in text
    for fold in report.folds:
        row = report.summary[report.summary["fold"] == fold.number].iloc[0]
        train = f"{row['train_start'].date()} to {row['train_end'].date()}"
        test = f"{row['test_start'].date()} to {row['test_end'].date()}"
        assert f"chosen on training rows {train} and tested on rows {test}" in text
        assert f"{test} ({len(fold.result.returns)} bars)" in text
        assert json.dumps(fold.parameters) in text
        metrics = dict(fold.result._metric_rows())
        benchmark = dict(fold.benchmark._metric_rows())
        assert f"Total Return {metrics['Total Return']} {benchmark['Total Return']}" in " ".join(
            text.split()
        )
        assert f"Fills ({len(fold.result.trade_history)})" in text
    statistics = report.aggregate_metrics
    assert f"Mean fold return (unweighted) {statistics['mean_fold_return']:.2f}%" in " ".join(
        text.split()
    )
    assert "winner" not in text.lower()

    assert data.equals(before_prices)
    assert report.summary.equals(before_summary)
    assert before_folds == [
        json.dumps([f.result._to_dict(), f.benchmark._to_dict()], default=str) for f in report.folds
    ]


def test_each_fold_keeps_its_own_curves_from_its_initial_cash(report, data, tmp_path):
    _, page, _ = read(report.export_html(tmp_path / "wf.html", price_data=data))
    equity = [
        figure
        for figure in page.payloads.values()
        if any(trace.get("name") == "benchmark" for trace in figure["data"])
    ]
    assert len(equity) == len(report.folds)
    for figure, fold in zip(equity, report.folds):
        strategy = next(t for t in figure["data"] if t.get("name") == fold.result.name)
        start = fold.result.portfolio_values.index[0]
        assert pd.Timestamp(strategy["x"][0]) == start  # pre-test cash point is kept
        assert values(strategy["y"])[0] == pytest.approx(fold.result.portfolio_values.iloc[0])
        assert pd.Timestamp(strategy["x"][-1]) == fold.result.portfolio_values.index[-1]
    # Trade charts are one per fold too, so every fold has two charts.
    assert len(page.payloads) == 2 * len(report.folds)


def test_without_prices_or_with_partial_prices_the_page_explains_what_is_missing(
    report, data, tmp_path
):
    html, page, text = read(report.export_html(tmp_path / "none.html"))
    assert len(page.payloads) == len(report.folds)  # equity only
    assert "Trade price charts are not included because no price data was given" in text
    assert html.count(get_plotlyjs()) == 1

    second = report.folds[1].result.returns.index
    partial = data.drop(second[3:6])
    partial = partial[partial.index < report.folds[2].result.returns.index[0]]
    _, page, text = read(report.export_html(tmp_path / "partial.html", price_data=partial))
    assert "Incomplete price data: 3 of" in text
    assert "price_data has no prices between" in text  # fold 3 is outside the given prices
    assert len(page.payloads) == len(report.folds) + 2


def test_hostile_text_rejections_and_errors_are_escaped_and_bounded(tmp_path):
    data = prices()
    report = WalkForwardOptimizer(train_size=0.5, n_splits=2, verbose=False).optimize_report(
        data,
        config(error_policy="continue"),
        Greedy,
        {"tag": [EVIL], "name": [EVIL]},
        metric="total_return",
    )
    html, page, text = read(report.export_html(tmp_path / "evil.html", price_data=data))
    assert "<img" not in re.sub(r"<script.*?</script>", "", html, flags=re.S)
    assert "</script><img" not in html
    assert EVIL in text
    assert len(page.ids) == len(set(page.ids))
    assert "No fills" in text
    assert "orders rejected; main reason: insufficient_cash" in text
    assert "Strategy errors (" in text
    assert "shortened; export_json keeps the full text" in text


def test_missing_or_unusual_fold_records_are_shown_as_not_recorded(report, tmp_path):
    folds = deepcopy(report.folds)
    folds[0].number = EVIL
    folds[1].number = 1  # also numbered 1 elsewhere in the summary: no row is guessed
    summary = report.summary.copy()
    summary.loc[summary["fold"] == 2, "fold"] = 1
    broken = WalkForwardReport(summary, folds)
    html, page, text = read(broken.export_html(tmp_path / "odd.html"))
    assert [i for i in page.ids if re.fullmatch(r"fold-\d+", i)] == ["fold-1", "fold-2", "fold-3"]
    assert f"Fold {EVIL}" in text
    assert "chosen on training rows not recorded and tested on rows not recorded" in text
    third = report.summary[report.summary["fold"] == 3].iloc[0]
    assert f"{third['train_start'].date()} to {third['train_end'].date()}" in text

    no_summary = WalkForwardReport(pd.DataFrame(), deepcopy(report.folds))
    _, _, text = read(no_summary.export_html(tmp_path / "bare.html"))
    assert text.count("chosen on training rows not recorded") == len(report.folds)


def test_empty_report_has_an_honest_page_without_runtime(tmp_path):
    html, page, text = read(WalkForwardReport(pd.DataFrame(), []).export_html(tmp_path / "e.html"))
    assert "No folds available" in text
    assert get_plotlyjs()[:200] not in html
    assert page.payloads == {}
