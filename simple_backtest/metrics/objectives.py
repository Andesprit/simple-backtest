"""Optimization direction for built-in performance metrics."""

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
