"""Self-contained offline HTML report composed from existing results and charts."""

import json
from collections import Counter
from html import escape
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs

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

    charts = []

    def chart(figure) -> str:
        chart_id = f"chart-{len(charts) + 1}"
        charts.append(chart_id)
        # Plotly JSON already escapes "<"; the replace keeps any "</script>" inert regardless.
        payload = figure.to_json().replace("</", "<\\/")
        return (
            f'<div class="chart" id="{chart_id}"><p class="muted">Interactive chart: open this '
            "file in a browser with JavaScript enabled. The tables hold the same answers.</p></div>"
            f'<script type="application/json" id="{chart_id}-data">{payload}</script>'
        )

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
        sections.append(_strategy_section(index, name, result, coverage.get(name, []), trade_chart))
    sections.append(_assumptions_section(runs))

    nav = ['<a href="#summary">Summary</a>', '<a href="#equity">Equity and drawdown</a>']
    nav += [
        f'<a href="#strategy-{index}">{escape(name)}</a>'
        for index, name in enumerate(strategies, start=1)
    ]
    nav.append('<a href="#assumptions">Assumptions</a>')
    count = len(strategies)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Backtest report</title><style>{_STYLE}</style></head><body>"
        "<header><h1>Backtest report</h1>"
        f'<p class="muted">{count} strateg{"y" if count == 1 else "ies"}; '
        f"made with simple-backtest {escape(__version__)}. Simulated on the supplied data with "
        "the assumptions listed below: not a forecast or a recommendation. A result picked by a "
        "parameter search is in-sample, so its numbers do not show robustness.</p></header>"
        f"<nav>{''.join(nav)}</nav><main>{''.join(sections)}</main>"
        '<footer><p class="muted">Every record (fills, rejected orders, errors, equity) is in '
        "results.export_json(path).</p></footer>"
        + (f"<script>{get_plotlyjs()}</script><script>{_PLOT}</script>" if charts else "")
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


def _strategy_section(index, name, result, coverage, trade_chart) -> str:
    html = f'<section id="strategy-{index}"><h2>{escape(name)}</h2><table>'
    html += "".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in result._metric_rows()
    )
    html += "</table>"
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
        html += _preview(
            "Strategy errors", result.errors, ["timestamp", "stage", "error_type", "message"]
        )
    return html + "</section>"


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


def _assumptions_section(runs) -> str:
    rows = {name: _assumptions(result) for name, result in runs.items()}
    html = (
        '<section id="assumptions"><h2>Assumptions</h2><p class="muted">As recorded when each '
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
