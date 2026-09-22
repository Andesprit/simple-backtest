"""Detailed out-of-sample reports with explicit independent-fold accounting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from simple_backtest.core.results import StrategyResult
from simple_backtest.core.serialization import write_json


@dataclass
class WalkForwardFold:
    """Selected parameters and complete strategy/benchmark results for one test fold."""

    number: int
    parameters: dict[str, Any]
    result: StrategyResult
    benchmark: StrategyResult


@dataclass
class WalkForwardReport:
    """Independent test folds, each with fresh cash, positions, strategy state, and RNG.

    No continuous tradable equity curve is implied. Open lots are marked at each
    fold's final close unless the configured liquidation policy closes them.
    """

    summary: pd.DataFrame
    folds: list[WalkForwardFold]

    @property
    def aggregate_metrics(self) -> dict[str, float | int]:
        """Summarize independent folds without adding their cash balances or P&L."""
        returns = pd.Series([fold.result.metrics["total_return"] for fold in self.folds])
        return {
            "fold_count": len(self.folds),
            "test_bars": sum(len(fold.result.returns) for fold in self.folds),
            "total_trades": sum(int(fold.result.metrics["total_trades"]) for fold in self.folds),
            "rejected_orders": sum(
                order["status"] == "rejected"
                for fold in self.folds
                for order in fold.result.order_outcomes
            ),
            "mean_fold_return": float(returns.mean()),
            "median_fold_return": float(returns.median()),
            "worst_fold_return": float(returns.min()),
            "maximum_fold_drawdown": max(
                fold.result.metrics["max_drawdown"] for fold in self.folds
            ),
        }

    def export_json(self, path: str | Path) -> None:
        """Save fold summaries, parameters, equity, fills, diagnostics, and provenance."""
        write_json(
            path,
            {
                "schema_version": 1,
                "capital_policy": "reset",
                "position_policy": "reset",
                "summary": self.summary,
                "aggregate_metrics": self.aggregate_metrics,
                "folds": [
                    {
                        "number": fold.number,
                        "parameters": fold.parameters,
                        "result": fold.result._to_dict(),
                        "benchmark": fold.benchmark._to_dict(),
                    }
                    for fold in self.folds
                ],
            },
        )
