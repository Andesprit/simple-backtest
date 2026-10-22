"""Self-contained offline HTML report composed from existing results and charts."""

import json
from collections import Counter
from html import escape
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs

from simple_backtest.core.error_context import error_where, stage_meaning
from simple_backtest.core.results import _format_value
from simple_backtest.utils.validation import _normalize_ohlcv_columns
from simple_backtest.visualization.plotter import (
    _run_notes,
    _trade_window,
    plot_equity_curve,
    plot_strategy_trades,
)

PREVIEW_ROWS = 10
MESSAGE_CHARS = 160
_MISSING = object()

_STYLE = """
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0 auto;max-width:1100px;
padding:16px;color:#1d2330;background:#fff}
h1{margin:0 0 4px}h2{margin-top:32px;border-bottom:1px solid #d8dde6;padding-bottom:4px}
nav{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #d8dde6;
overflow-x:auto;white-space:nowrap}
nav a{margin-right:14px}table{border-collapse:collapse;margin:8px 0;display:block;overflow-x:auto}
th,td{border:1px solid #d8dde6;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#f3f5f8}.muted{color:#5b6475}.note{background:#fff7e0;padding:8px 12px}
.chart{min-height:120px}td ul{margin:0;padding-left:18px}
"""

_PLOT = """
if (window.Plotly) {
  document.querySelectorAll('.chart').forEach(function (div) {
    var figure = JSON.parse(document.getElementById(div.id + '-data').textContent);
    div.textContent = '';
    Plotly.newPlot(div, figure.data, figure.layout, {responsive: true});
  });
}
"""


def write_report(path, results, price_data=None) -> Path:
    """Write one HTML file that needs no network, server or Python to read.

    :param path: Output file path
    :param results: BacktestResults to report; it is read, never changed or re-run
    :param price_data: Optional price DataFrame given to Backtest, adding trade charts
    :return: The written path
    """
    path = Path(path)
    path.write_text(build_report(results, price_data), encoding="utf-8")
    return path


def build_report(results, price_data=None) -> str:
    """Compose the static answers first, then the interactive charts, into one page."""
    from simple_backtest import __version__

    strategies = {name: results.get_strategy(name) for name in results.list_strategies()}
    runs = dict(strategies)
    if results.benchmark is not None:
        runs["benchmark"] = results.benchmark

    trade_figures, coverage = {}, {}
    if price_data is not None:
        prices = _normalize_ohlcv_columns(price_data)
        for name, result in strategies.items():
            coverage[name] = _trade_window(prices, result._plot_dict(), name)[1]
        trade_figures = plot_strategy_trades(
            price_data, {name: result._plot_dict() for name, result in strategies.items()}
        )

    chart = _Charts()

    sections = [_summary(results, strategies, price_data is not None)]
    sections.append(
        '<section id="equity"><h2>Equity and drawdown</h2>'
        + chart(
            plot_equity_curve(
                {name: result._plot_dict() for name, result in runs.items()}, show_drawdown=True
            )
        )
        + "</section>"
    )
    for index, (name, result) in enumerate(strategies.items(), start=1):
        trade_chart = (
            chart(trade_figures[name])
            if name in trade_figures
            else '<p class="muted">Trade price chart not included: pass price_data, the '
            "DataFrame you gave to Backtest, to export_html.</p>"
        )
        sections.append(
            _strategy_section(
                f"strategy-{index}", name, result, coverage.get(name, []), trade_chart
            )
        )
    sections.append(_assumptions_section(runs))

    nav = ['<a href="#summary">Summary</a>', '<a href="#equity">Equity and drawdown</a>']
    nav += [
        f'<a href="#strategy-{index}">{escape(name)}</a>'
        for index, name in enumerate(strategies, start=1)
    ]
    nav.append('<a href="#assumptions">Assumptions</a>')
    count = len(strategies)
    return _document(
        "Backtest report",
        f"{count} strateg{'y' if count == 1 else 'ies'}; "
        f"made with simple-backtest {escape(__version__)}. Simulated on the supplied data with "
        "the assumptions listed below: not a forecast or a recommendation. A result picked by a "
        "parameter search is in-sample, so its numbers do not show robustness.",
        nav,
        sections,
        "Every record (fills, rejected orders, errors, equity) is in results.export_json(path).",
        chart,
    )


class _Charts:
    """Give each figure a page-unique ID and an inert JSON payload for the one Plotly runtime."""

    def __init__(self):
        self.ids = []

    def __call__(self, figure) -> str:
        chart_id = f"chart-{len(self.ids) + 1}"
        self.ids.append(chart_id)
        # Plotly JSON already escapes "<"; the replace keeps any "</script>" inert regardless.
        payload = figure.to_json().replace("</", "<\\/")
        return (
            f'<div class="chart" id="{chart_id}"><p class="muted">Interactive chart: open this '
            "file in a browser with JavaScript enabled. The tables hold the same answers.</p></div>"
            f'<script type="application/json" id="{chart_id}-data">{payload}</script>'
        )


def _document(title, intro, nav, sections, footer, charts) -> str:
    """Wrap escaped sections in one page; embed Plotly once, only if a chart was added."""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title><style>{_STYLE}</style></head><body>"
        f'<header><h1>{title}</h1><p class="muted">{intro}</p></header>'
        f"<nav>{''.join(nav)}</nav><main>{''.join(sections)}</main>"
        f'<footer><p class="muted">{footer}</p></footer>'
        + (f"<script>{get_plotlyjs()}</script><script>{_PLOT}</script>" if charts.ids else "")
        + "</body></html>"
    )


def _summary(results, strategies, has_prices: bool) -> str:
    table = results._comparison_table()
    html = '<section id="summary"><h2>Summary</h2>'
    if table.empty:
        return html + "<p>No strategies available.</p></section>"
    html += table.to_html(escape=True, border=0, max_rows=None, max_cols=None)
    html += f"<p><strong>{escape(results._return_verdict())}</strong></p>"
    notes = [
        f"{name}: {note}"
        for name, result in strategies.items()
        for note in _run_notes(name, result._plot_dict())
    ]
    if results.benchmark is not None:
        notes += [f"benchmark: {note}" for note in results.benchmark._diagnostic_notes(120)]
    if notes:
        html += (
            "<h3>What to check</h3><ul>"
            + "".join(f"<li>{escape(note)}</li>" for note in notes)
            + "</ul>"
        )
    if not has_prices:
        html += (
            '<p class="note">Trade price charts are not included because no price data was '
            "given. Call export_html(path, price_data=data) with the DataFrame you gave to "
            "Backtest to add them.</p>"
        )
    return html + "</section>"


def _strategy_section(section_id, name, result, coverage, trade_chart) -> str:
    html = f'<section id="{section_id}"><h2>{escape(name)}</h2><table>'
    html += "".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in result._metric_rows()
    )
    html += "</table>"
    return html + _run_details(name, result, coverage, trade_chart) + "</section>"


def _run_details(name, result, coverage, trade_chart) -> str:
    """Notes, trade chart, fills, rejections and errors of one run, bounded for display."""
    html = ""
    notes = coverage + _run_notes(name, result._plot_dict())
    if notes:
        html += "<ul>" + "".join(f"<li>{escape(note)}</li>" for note in notes) + "</ul>"
    html += trade_chart

    trades = result.trade_history
    if trades:
        columns = ["timestamp", "signal", "shares", "price", "commission", "pnl"]
        html += _preview("Fills", trades, [c for c in columns if any(c in t for t in trades)])
    rejected = Counter(
        outcome.get("reason") or "unknown"
        for outcome in result.order_outcomes
        if outcome.get("status") == "rejected"
    )
    if rejected:
        html += "<details><summary>Rejected orders by reason</summary><table>"
        html += "<tr><th>Reason</th><th>Orders</th></tr>" + "".join(
            f"<tr><td>{escape(str(reason))}</td><td>{count}</td></tr>"
            for reason, count in rejected.most_common()
        )
        html += "</table></details>"
    if result.errors:
        rows = [
            {**error, "where": error_where(error), "meaning": stage_meaning(error, short=True)}
            for error in result.errors[:PREVIEW_ROWS]
        ]
        html += _preview(
            "Strategy errors",
            rows + result.errors[PREVIEW_ROWS:],
            ["timestamp", "stage", "error_type", "message", "where", "meaning"],
        )
        html += (
            '<p class="muted">In Python, result.explain_error(i, price_data=data) shows the '
            "fills and prior bars around error i without re-running the strategy.</p>"
        )
    return html


def _preview(title, records, columns) -> str:
    shown = records[:PREVIEW_ROWS]
    label = f"{title} ({len(records)})"
    if len(records) > len(shown):
        label += f", showing the first {len(shown)}; export_json keeps all"
    html = f"<details><summary>{escape(label)}</summary><table><tr>"
    html += "".join(f"<th>{escape(column)}</th>" for column in columns) + "</tr>"
    for record in shown:
        html += (
            "<tr>"
            + "".join(f"<td>{escape(_cell(record.get(column, '')))}</td>" for column in columns)
            + "</tr>"
        )
    return html + "</table></details>"


def _cell(value) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    text = str(value)
    if len(text) > MESSAGE_CHARS:
        return text[: MESSAGE_CHARS - 3] + "... (shortened; export_json keeps the full text)"
    return text


def _assumptions_section(runs, section_id="assumptions", title="Assumptions") -> str:
    rows = {name: _assumptions(result) for name, result in runs.items()}
    html = (
        f'<section id="{section_id}"><h2>{escape(title)}</h2><p class="muted">As recorded when each '
        "run was simulated. A value that differs between runs is listed per run. "
        "“not recorded” means this result carries no such record.</p><table>"
    )
    for label in next(iter(rows.values()), {}):
        values = {name: row[label] for name, row in rows.items()}
        if len(set(values.values())) == 1:
            cell = escape(next(iter(values.values())))
        else:
            cell = (
                "<ul>"
                + "".join(
                    f"<li>{escape(name)}: {escape(value)}</li>" for name, value in values.items()
                )
                + "</ul>"
            )
        html += f"<tr><th>{escape(label)}</th><td>{cell}</td></tr>"
    return html + "</table></section>"


def _assumptions(result) -> dict:
    meta = result.metadata
    config = meta.get("config", {})
    data = meta.get("data", {})
    strategy = meta.get("strategy", {})
    returns = result.returns
    trading = (
        f"{_date(returns.index[0])} to {_date(returns.index[-1])} ({len(returns)} bars)"
        if returns is not None and len(returns)
        else "unavailable"
    )
    input_span = (
        f"{data['start']} to {data['end']} ({data['rows']} rows, including warm-up)"
        if {"start", "end", "rows"} <= set(data)
        else "not recorded"
    )
    commission = _show(config.get("commission_type", _MISSING))
    if "commission_value" in config:
        commission += f" {_show(config['commission_value'])}"
    return {
        "Strategy class": _show(strategy.get("class", _MISSING)),
        "Strategy parameters": _show(strategy.get("parameters", _MISSING)),
        "Initial capital": _show(config.get("initial_capital", _MISSING)),
        "Input data span": input_span,
        "Trading span": trading,
        "Warm-up bars (lookback_period)": _show(config.get("lookback_period", _MISSING)),
        "Execution price": _show(config.get("execution_price", _MISSING)),
        "Custom execution price": _show(meta.get("custom_execution_price", _MISSING)),
        "Commission": commission,
        "Custom commission": _show(meta.get("custom_commission", _MISSING)),
        "Spread (bps)": _show(config.get("spread_bps", _MISSING)),
        "Slippage (bps)": _show(config.get("slippage_bps", _MISSING)),
        "Max volume participation": _show(config.get("max_volume_participation", _MISSING)),
        "Final liquidation": _show(config.get("final_liquidation", _MISSING)),
        "Periods per year (annualization)": _show(meta.get("periods_per_year", _MISSING)),
        "Risk-free rate": _show(config.get("risk_free_rate", _MISSING)),
        "Random seed": _show(meta.get("random_seed", _MISSING)),
        "Error policy": _show(config.get("error_policy", _MISSING)),
        "Data fingerprint": _show(data.get("fingerprint", _MISSING)),
        "Data timezone": _show(data.get("timezone", _MISSING)),
        "Package version": _show(meta.get("package_version", _MISSING)),
    }


def _show(value) -> str:
    if value is _MISSING:
        return "not recorded"
    if value is None:
        return "none"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _date(timestamp) -> str:
    timestamp = pd.Timestamp(timestamp)
    return str(timestamp.date()) if timestamp == timestamp.normalize() else str(timestamp)


_FOLD_STATISTICS = (
    ("fold_count", "Folds", ".0f", ""),
    ("test_bars", "Test bars, all folds", ".0f", ""),
    ("total_trades", "Trades, all folds", ".0f", ""),
    ("rejected_orders", "Rejected orders, all folds", ".0f", ""),
    ("mean_fold_return", "Mean fold return (unweighted)", ".2f", "%"),
    ("median_fold_return", "Median fold return", ".2f", "%"),
    ("worst_fold_return", "Worst fold return", ".2f", "%"),
    ("maximum_fold_drawdown", "Largest single-fold max drawdown", ".2f", "%"),
)


def write_walk_forward_report(path, report, price_data=None) -> Path:
    """Write one offline HTML file reviewing every independent walk-forward test fold.

    :param path: Output file path
    :param report: WalkForwardReport to review; it is read, never changed or re-run
    :param price_data: Optional price DataFrame given to the optimizer, adding trade charts
    :return: The written path
    """
    path = Path(path)
    path.write_text(build_walk_forward_report(report, price_data), encoding="utf-8")
    return path


def build_walk_forward_report(report, price_data=None) -> str:
    """Compose fold boundaries, selections and fold statistics, then one section per fold."""
    from simple_backtest import __version__

    chart = _Charts()
    prices = _normalize_ohlcv_columns(price_data) if price_data is not None else None
    records = [_fold_record(report.summary, fold.number) for fold in report.folds]
    sections = [_walk_forward_summary(report, records, prices is not None)]
    nav = ['<a href="#summary">Summary</a>']
    for position, (fold, record) in enumerate(zip(report.folds, records), start=1):
        # IDs come from the fold's position, never from user-controlled names or numbers.
        section_id = f"fold-{position}"
        label = f"Fold {fold.number}"
        nav.append(f'<a href="#{section_id}">{escape(label)}</a>')
        sections.append(_fold_section(section_id, label, fold, record, prices, chart))
    count = len(report.folds)
    return _document(
        "Walk-forward report",
        f"{count} independent test fold{'' if count == 1 else 's'}; made with simple-backtest "
        f"{escape(__version__)}. Each fold's parameters were chosen on earlier training rows "
        "only, then traded once on its test rows with fresh cash and no positions. Simulated "
        "on the supplied data: not a forecast or a recommendation, and out-of-sample results "
        "on one history do not prove robustness.",
        nav,
        sections,
        "Every fold record (summary, parameters, equity, fills, rejected orders, errors) is in "
        "report.export_json(path).",
        chart,
    )


def _fold_record(summary, number) -> dict | None:
    """The single summary row for this fold number, or None rather than a guessed row."""
    if not isinstance(summary, pd.DataFrame) or "fold" not in summary:
        return None
    rows = [row for row in summary.to_dict("records") if row["fold"] == number]
    return rows[0] if len(rows) == 1 else None


def _span(record, start, end) -> str:
    if record is None or start not in record or end not in record:
        return "not recorded"
    return f"{_date(record[start])} to {_date(record[end])}"


def _metric(result, label) -> str:
    return dict(result._metric_rows())[label]


def _walk_forward_summary(report, records, has_prices: bool) -> str:
    html = (
        '<section id="summary"><h2>Summary</h2><p class="note"><strong>Each fold is a separate '
        "test.</strong> Cash, positions, strategy state and random numbers reset at the start of "
        "every fold. Parameters were chosen using only the training rows before each test. Test "
        "results are not added, chained or compounded into one portfolio, and no fold is picked "
        "as best: the fold statistics below summarize independent folds.</p>"
    )
    if not report.folds:
        return html + "<p>No folds available: this report holds no test results.</p></section>"
    html += (
        "<table><tr><th>Fold</th><th>Training rows (chose the parameters)</th>"
        "<th>Test rows (traded)</th><th>Selected parameters</th><th>Test return</th>"
        "<th>Benchmark return</th><th>Max drawdown</th><th>Trades</th></tr>"
    )
    for fold, record in zip(report.folds, records):
        cells = (
            str(fold.number),
            _span(record, "train_start", "train_end"),
            _span(record, "test_start", "test_end") + _bars(fold.result),
            _show(fold.parameters),
            _metric(fold.result, "Total Return"),
            _metric(fold.benchmark, "Total Return"),
            _metric(fold.result, "Max Drawdown"),
            _metric(fold.result, "Trades"),
        )
        html += "<tr>" + "".join(f"<td>{escape(_cell(cell))}</td>" for cell in cells) + "</tr>"
    html += "</table>"
    try:
        statistics = report.aggregate_metrics
    except (KeyError, TypeError, ValueError) as error:
        html += f"<p>Fold statistics unavailable: {escape(_cell(repr(error)))}.</p>"
    else:
        html += "<h3>Statistics over independent folds</h3><table>" + "".join(
            f"<tr><th>{label}</th>"
            f"<td>{escape(_format_value(statistics.get(key), precision, suffix))}</td></tr>"
            for key, label, precision, suffix in _FOLD_STATISTICS
        )
        html += "</table>"
    html += (
        "<p>Search objective and base optimizer: not recorded in this report, so this page "
        "does not say which metric chose the parameters.</p>"
    )
    notes = [
        f"Fold {fold.number} {run}: {note}"
        for fold in report.folds
        for run, notes in (
            (fold.result.name, _run_notes(fold.result.name, fold.result._plot_dict())),
            ("benchmark", fold.benchmark._diagnostic_notes(120)),
        )
        for note in notes
    ]
    if notes:
        html += (
            "<h3>What to check</h3><ul>"
            + "".join(f"<li>{escape(note)}</li>" for note in notes)
            + "</ul>"
        )
    if not has_prices:
        html += (
            '<p class="note">Trade price charts are not included because no price data was '
            "given. Call export_html(path, price_data=data) with the DataFrame you gave to the "
            "optimizer to add them.</p>"
        )
    return html + "</section>"


def _bars(result) -> str:
    returns = result.returns
    return f" ({len(returns)} bars)" if returns is not None else ""


def _fold_section(section_id, label, fold, record, prices, chart) -> str:
    name, result, benchmark = fold.result.name, fold.result, fold.benchmark
    html = (
        f'<section id="{section_id}"><h2>{escape(label)}: {escape(name)}</h2>'
        f"<p>Parameters {escape(_show(fold.parameters))} were chosen on training rows "
        f"{escape(_span(record, 'train_start', 'train_end'))} and tested on rows "
        f"{escape(_span(record, 'test_start', 'test_end'))}. This fold started with fresh cash "
        "and no positions; its curves are not joined to other folds.</p>"
        f"<table><tr><th>Metric</th><th>{escape(name)}</th><th>benchmark</th></tr>"
    )
    html += "".join(
        f"<tr><th>{escape(metric)}</th><td>{escape(value)}</td>"
        f"<td>{escape(_metric(benchmark, metric))}</td></tr>"
        for metric, value in result._metric_rows()
    )
    html += "</table>" + chart(
        plot_equity_curve(
            {name: result._plot_dict(), "benchmark": benchmark._plot_dict()}, show_drawdown=True
        )
    )
    coverage = []
    if prices is None:
        trade_chart = (
            '<p class="muted">Trade price chart not included: pass price_data, the DataFrame you '
            "gave to the optimizer, to export_html.</p>"
        )
    else:
        try:
            coverage = _trade_window(prices, result._plot_dict(), name)[1]
            trade_chart = chart(plot_strategy_trades(prices, {name: result._plot_dict()})[name])
        except ValueError as error:
            trade_chart = (
                f'<p class="note">Trade price chart not included: {escape(str(error))}</p>'
            )
    html += _run_details(name, result, coverage, trade_chart)
    html += (
        '<p class="muted">Input data span below is this test\'s evaluation slice, including its '
        "warm-up rows; the training rows are listed above.</p>"
    )
    runs = {name: result, "benchmark": benchmark}
    assumptions = _assumptions_section(runs, f"{section_id}-assumptions", f"{label} assumptions")
    return html + assumptions + "</section>"
