"""Comprehensive metrics calculator for backtest results."""

from typing import Any, Dict, List

import pandas as pd

from simple_backtest.metrics import definitions as defs


def calculate_metrics(
    trade_history: List[Dict[str, Any]],
    portfolio_values: pd.Series,
    benchmark_values: pd.Series,
    initial_capital: float,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    exposure: pd.Series | None = None,
) -> Dict[str, float]:
    """Calculate comprehensive performance metrics.

    Args:
        trade_history: List of trade dictionaries from Portfolio
        portfolio_values: Series of portfolio values over time (indexed by date)
        benchmark_values: Series of benchmark portfolio values over time
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)

    Returns:
        Dictionary of metrics with the following keys:
        - Returns metrics: total_return, cagr
        - Risk metrics: volatility, sharpe_ratio, sortino_ratio, calmar_ratio,
                       max_drawdown, max_drawdown_duration
        - Trade statistics: total_trades, win_rate, avg_win, avg_loss,
                          profit_factor, expectancy
        - Portfolio metrics: final_value, peak_value, exposure_time
        - Benchmark comparison: alpha, beta, information_ratio

    Example:
        >>> metrics = calculate_metrics(
        ...     trade_history=[...],
        ...     portfolio_values=pd.Series([1000, 1050, 1100]),
        ...     benchmark_values=pd.Series([1000, 1020, 1040]),
        ...     initial_capital=1000.0,
        ...     risk_free_rate=0.02
        ... )
    """
    if len(portfolio_values) == 0:
        return _empty_metrics()

    # Calculate returns
    strategy_returns = portfolio_values.pct_change().dropna()
    benchmark_returns = benchmark_values.pct_change().dropna()

    # Time period calculations
    final_value = portfolio_values.iloc[-1]
    peak_value = portfolio_values.max()

    # Calculate years for CAGR
    if isinstance(portfolio_values.index, pd.DatetimeIndex):
        days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        years = days / 365.25
    else:
        # Assume daily data if no datetime index
        years = len(portfolio_values) / periods_per_year

    # Returns metrics
    total_return = defs.calculate_total_return(initial_capital, final_value)
    cagr = defs.calculate_cagr(initial_capital, final_value, years)

    # Risk metrics
    volatility = defs.calculate_volatility(strategy_returns, periods_per_year)
    sharpe_ratio = defs.calculate_sharpe_ratio(strategy_returns, risk_free_rate, periods_per_year)
    sortino_ratio = defs.calculate_sortino_ratio(strategy_returns, risk_free_rate, periods_per_year)

    # Drawdown metrics
    drawdown_metrics = defs.calculate_max_drawdown(portfolio_values)
    max_drawdown = drawdown_metrics["max_drawdown"]
    max_drawdown_duration = drawdown_metrics["max_drawdown_duration"]

    calmar_ratio = defs.calculate_calmar_ratio(cagr, max_drawdown)

    # Trade statistics
    total_trades = len([t for t in trade_history if t.get("signal") in ["buy", "sell"]])
    win_rate = defs.calculate_win_rate(trade_history)

    # Calculate average win/loss
    sell_trades = [t for t in trade_history if t.get("signal") == "sell"]
    wins = [t["pnl"] for t in sell_trades if t.get("pnl", 0) > 0]
    losses = [t["pnl"] for t in sell_trades if t.get("pnl", 0) < 0]

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    profit_factor = defs.calculate_profit_factor(trade_history)
    expectancy = defs.calculate_expectancy(trade_history)

    # Portfolio metrics
    exposure_time = (
        float(exposure.astype(bool).mean() * 100)
        if exposure is not None and not exposure.empty
        else 0.0
    )

    # Benchmark comparison
    alpha_beta = defs.calculate_alpha_beta(strategy_returns, benchmark_returns, periods_per_year)
    information_ratio = defs.calculate_information_ratio(
        strategy_returns, benchmark_returns, periods_per_year
    )

    return {
        # Returns
        "total_return": total_return,
        "cagr": cagr,
        # Risk
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": max_drawdown_duration,
        # Trades
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        # Portfolio
        "final_value": final_value,
        "peak_value": peak_value,
        "exposure_time": exposure_time,
        # Benchmark
        "alpha": alpha_beta["alpha"],
        "beta": alpha_beta["beta"],
        "information_ratio": information_ratio,
    }


def _empty_metrics() -> Dict[str, float]:
    """Return empty metrics dictionary with all keys set to 0.

    Returns:
        Dictionary with all metric keys set to 0.0
    """
    return {
        "total_return": 0.0,
        "cagr": 0.0,
        "volatility": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_duration": 0,
        "total_trades": 0,
        "win_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "final_value": 0.0,
        "peak_value": 0.0,
        "exposure_time": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "information_ratio": 0.0,
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary as readable string.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Formatted string representation

    Example:
        >>> print(format_metrics(metrics))
        Performance Metrics
        ===================
        Returns:
          Total Return: 15.50%
          CAGR: 12.30%
        ...
    """
    output = ["Performance Metrics", "=" * 50, ""]

    # Returns
    output.append("Returns:")
    output.append(f"  Total Return: {metrics['total_return']:.2f}%")
    output.append(f"  CAGR: {metrics['cagr']:.2f}%")
    output.append("")

    # Risk
    output.append("Risk Metrics:")
    output.append(f"  Volatility: {metrics['volatility']:.2f}%")
    output.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    output.append(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    output.append(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    output.append(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    output.append(f"  Max Drawdown Duration: {metrics['max_drawdown_duration']} periods")
    output.append("")

    # Trades
    output.append("Trade Statistics:")
    output.append(f"  Total Trades: {metrics['total_trades']}")
    output.append(f"  Win Rate: {metrics['win_rate']:.2f}%")
    output.append(f"  Average Win: ${metrics['avg_win']:.2f}")
    output.append(f"  Average Loss: ${metrics['avg_loss']:.2f}")
    output.append(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    output.append(f"  Expectancy: ${metrics['expectancy']:.2f}")
    output.append("")

    # Portfolio
    output.append("Portfolio Metrics:")
    output.append(f"  Final Value: ${metrics['final_value']:.2f}")
    output.append(f"  Peak Value: ${metrics['peak_value']:.2f}")
    output.append(f"  Exposure Time: {metrics['exposure_time']:.2f}%")
    output.append("")

    # Benchmark
    output.append("Benchmark Comparison:")
    output.append(f"  Alpha: {metrics['alpha']:.2f}%")
    output.append(f"  Beta: {metrics['beta']:.2f}")
    output.append(f"  Information Ratio: {metrics['information_ratio']:.2f}")

    return "\n".join(output)
