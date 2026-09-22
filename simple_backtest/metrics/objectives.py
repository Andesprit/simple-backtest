"""Optimization direction for built-in performance metrics."""

import pandas as pd

MINIMIZE_METRICS = frozenset(
    {
        "max_drawdown",
        "max_drawdown_duration",
        "volatility",
    }
)


def metric_is_maximized(metric: str) -> bool:
    """Return whether a metric is better when its value is larger."""
    return metric not in MINIMIZE_METRICS


def rank_results(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rank defined objectives, retaining legitimate unbounded values like profit factor."""
    if results.empty:
        return results
    if metric not in results:
        raise ValueError(f"Metric '{metric}' not found. Available metrics: {list(results.columns)}")
    return (
        results.dropna(subset=[metric])
        .sort_values(metric, ascending=not metric_is_maximized(metric), kind="stable")
        .reset_index(drop=True)
    )
