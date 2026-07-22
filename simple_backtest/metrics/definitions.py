"""Metric definitions and calculation formulas."""

from typing import Any, Dict

import numpy as np
import pandas as pd


def calculate_total_return(initial_value: float, final_value: float) -> float:
    """Calculate total return percentage.

    Args:
        initial_value: Starting portfolio value
        final_value: Ending portfolio value

    Returns:
        Total return as percentage
    """
    return ((final_value - initial_value) / initial_value) * 100


def calculate_cagr(initial_value: float, final_value: float, years: float) -> float:
    """Calculate Compound Annual Growth Rate.

    Args:
        initial_value: Starting portfolio value
        final_value: Ending portfolio value
        years: Number of years in period

    Returns:
        CAGR as percentage
    """
    if years == 0:
        return 0.0
    return (((final_value / initial_value) ** (1 / years)) - 1) * 100


def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility.

    Args:
        returns: Series of period returns (not percentages)
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly)

    Returns:
        Annualized volatility as percentage
    """
    std = returns.std()
    return 0.0 if pd.isna(std) else std * np.sqrt(periods_per_year) * 100


def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Calculate annualized Sharpe Ratio.

    Args:
        returns: Series of period returns (not percentages)
        risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
        periods_per_year: Number of periods in a year

    Returns:
        Sharpe Ratio
    """
    # Convert annual risk-free rate to period rate
    period_rf_rate = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    # Calculate excess returns
    excess_returns = returns - period_rf_rate

    if pd.isna(excess_returns.std()) or excess_returns.std() == 0:
        return 0.0

    # Annualize
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Calculate annualized Sortino Ratio (downside deviation).

    Args:
        returns: Series of period returns (not percentages)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year

    Returns:
        Sortino Ratio
    """
    # Convert annual risk-free rate to period rate
    period_rf_rate = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    # Calculate excess returns
    excess_returns = returns - period_rf_rate

    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or pd.isna(downside_returns.std()) or downside_returns.std() == 0:
        return 0.0

    downside_std = downside_returns.std()

    # Annualize
    return (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)


def calculate_max_drawdown(portfolio_values: pd.Series) -> Dict[str, Any]:
    """Calculate maximum drawdown and its duration.

    Args:
        portfolio_values: Series of portfolio values over time

    Returns:
        Dictionary with max_drawdown (percentage) and max_drawdown_duration (days)
    """
    # Calculate running maximum
    running_max = portfolio_values.cummax()

    # Calculate drawdown at each point
    drawdown = (portfolio_values - running_max) / running_max * 100

    # Find maximum drawdown
    max_drawdown = drawdown.min()

    # Calculate drawdown duration
    # Find all drawdown periods
    in_drawdown = drawdown < 0
    drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()

    max_duration = 0
    if in_drawdown.any():
        # Get duration of each drawdown period
        durations = drawdown_periods[in_drawdown].value_counts()
        max_duration = int(durations.max()) if len(durations) > 0 else 0

    return {"max_drawdown": abs(max_drawdown), "max_drawdown_duration": max_duration}


def calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """Calculate Calmar Ratio (CAGR / Max Drawdown).

    Args:
        cagr: Compound Annual Growth Rate (percentage)
        max_drawdown: Maximum drawdown (percentage)

    Returns:
        Calmar Ratio
    """
    if max_drawdown == 0:
        return 0.0
    return cagr / max_drawdown


def calculate_win_rate(trade_history: list) -> float:
    """Calculate win rate from trade history.

    Args:
        trade_history: List of trade dictionaries with 'pnl' field

    Returns:
        Win rate as percentage
    """
    sell_trades = [t for t in trade_history if t.get("signal") == "sell"]

    if not sell_trades:
        return 0.0

    winning_trades = sum(1 for t in sell_trades if t.get("pnl", 0) > 0)
    return (winning_trades / len(sell_trades)) * 100


def calculate_profit_factor(trade_history: list) -> float:
    """Calculate profit factor (total wins / total losses).

    Args:
        trade_history: List of trade dictionaries with 'pnl' field

    Returns:
        Profit factor
    """
    sell_trades = [t for t in trade_history if t.get("signal") == "sell"]

    if not sell_trades:
        return 0.0

    total_wins = sum(t.get("pnl", 0) for t in sell_trades if t.get("pnl", 0) > 0)
    total_losses = abs(sum(t.get("pnl", 0) for t in sell_trades if t.get("pnl", 0) < 0))

    if total_losses == 0:
        return float("inf") if total_wins > 0 else 0.0

    return total_wins / total_losses


def calculate_expectancy(trade_history: list) -> float:
    """Calculate expectancy (average profit per trade).

    Args:
        trade_history: List of trade dictionaries with 'pnl' field

    Returns:
        Expectancy (average PnL per trade)
    """
    sell_trades = [t for t in trade_history if t.get("signal") == "sell"]

    if not sell_trades:
        return 0.0

    total_pnl = sum(t.get("pnl", 0) for t in sell_trades)
    return total_pnl / len(sell_trades)


def calculate_alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Calculate alpha and beta relative to benchmark.

    Args:
        strategy_returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns

    Returns:
        Dictionary with 'alpha' and 'beta'
    """
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) < 2 or aligned["benchmark"].var() == 0:
        return {"alpha": 0.0, "beta": 0.0}

    covariance = aligned["strategy"].cov(aligned["benchmark"])
    beta = covariance / aligned["benchmark"].var()
    intercept = aligned["strategy"].mean() - beta * aligned["benchmark"].mean()
    alpha = intercept * periods_per_year * 100

    return {"alpha": alpha, "beta": beta}


def calculate_information_ratio(
    strategy_returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Calculate Information Ratio.

    Args:
        strategy_returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns
        periods_per_year: Number of periods in a year

    Returns:
        Information Ratio
    """
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return 0.0

    # Calculate tracking error (excess returns)
    tracking_error = aligned["strategy"] - aligned["benchmark"]

    if pd.isna(tracking_error.std()) or tracking_error.std() == 0:
        return 0.0

    # Annualize
    return (tracking_error.mean() / tracking_error.std()) * np.sqrt(periods_per_year)
