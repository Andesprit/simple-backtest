"""Base optimizer class for strategy parameter optimization."""

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Type

import pandas as pd

from simple_backtest.config.settings import BacktestConfig
from simple_backtest.core.backtest import Backtest
from simple_backtest.core.results import StrategyResult, _metric_hint
from simple_backtest.core.serialization import json_value, write_json
from simple_backtest.metrics.objectives import metric_is_maximized, rank_results
from simple_backtest.strategy.base import Strategy

if TYPE_CHECKING:
    import plotly.graph_objects as go

_PERCENT_METRICS = frozenset(
    {"total_return", "cagr", "volatility", "max_drawdown", "win_rate", "exposure_time", "alpha"}
)


class Optimizer(ABC):
    """Abstract base class for parameter optimization.

    Users can create custom optimizers by inheriting from this class
    and implementing the optimize() method.

    Example:
        class MyOptimizer(Optimizer):
            def optimize(self, data, config, strategy_class, param_space, metric):
                # Your optimization logic here
                return results_df
    """

    def __init__(self, name: str | None = None):
        """Initialize optimizer.

        :param name: Optimizer name (auto-generated if None)
        """
        self._name = name or self.__class__.__name__
        self.failures: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}
        # Selection state of the last search: no search, "running" (stopped early if
        # still set), "failed" or "done", plus only the best evaluation so far.
        self._status: str | None = None
        self._objective: str | None = None
        self._param_names: list[str] = []
        self._best: Dict[str, Any] | None = None
        self._last_result: StrategyResult | None = None
        # Compact evidence of every evaluation: parameter positions, status, objective,
        # trial and seed. No results or histories are kept.
        self._space: Dict[str, List[Any]] = {}
        self._space_snapshot: Dict[str, List[Any]] = {}
        self._evaluations: list[tuple[tuple[int | None, ...], str, Any, Any, Any]] = []
        self._trial_context: Dict[str, Any] = {}
        self._snapshot_error: str | None = None

    def _prepare_search(
        self, param_space: Dict[str, List[Any]], metric: str | None = None
    ) -> Dict[str, List[Any]]:
        """Validate finite parameter lists and remove duplicate values in insertion order."""
        self.failures = []
        self.summary = {"attempted_evaluations": 0, "unique_candidates": 0}
        self._status, self._objective = "running", metric
        self._param_names = list(param_space)
        self._best = self._last_result = None
        self._evaluations, self._trial_context = [], {}
        self._snapshot_error: str | None = None
        cleaned = {}
        for name, values in param_space.items():
            if not isinstance(values, (list, tuple, range)) or not values:
                raise ValueError(f"Parameter '{name}' must have a non-empty sequence of candidates")
            unique: list[Any] = []
            for value in values:
                if value not in unique:
                    unique.append(value)
            cleaned[name] = unique
        # An empty mapping deliberately evaluates the strategy's defaults once.
        self._space = cleaned
        # Inspection needs a detached copy; a noncopyable value only disables inspection.
        try:
            self._space_snapshot = deepcopy(cleaned)
        except Exception as error:
            self._space_snapshot = {}
            self._snapshot_error = f"{type(error).__name__}: {error}"
        return cleaned

    def _finish_search(
        self,
        rows: list[dict[str, Any]],
        metric: str,
        metadata: dict[str, Any],
        strategy_class: Type[Strategy],
    ) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        ranked = rank_results(frame, metric)
        self.summary.update(
            successful_evaluations=len(frame),
            failed_evaluations=len(self.failures),
            undefined_objectives=len(frame) - len(ranked),
        )
        if not rows and self.failures:
            valid_parameters = [
                name
                for name, parameter in signature(strategy_class.__init__).parameters.items()
                if name != "self"
                and parameter.kind
                in (
                    parameter.POSITIONAL_OR_KEYWORD,
                    parameter.KEYWORD_ONLY,
                )
            ]
            first = self.failures[0]
            self._status = "failed"
            raise ValueError(
                f"All {len(self.failures)} candidate evaluations failed for "
                f"{strategy_class.__name__}. First failure for {first['parameters']}: "
                f"{first['error_type']}: {first['message']}. "
                f"Valid parameters: {valid_parameters}. "
                "Correct param_space names and values (or lookback_period for history failures) "
                "and retry. Individual failures are available in optimizer.failures."
            )
        self._status = "done"
        if rows and ranked.empty:
            # One bounded completion notice; an empty frame alone looks like success.
            warnings.warn(f"{self.get_name()}: {self._no_winner_reason()}", stacklevel=3)
        ranked.attrs["search"] = json_value(
            {
                **metadata,
                **self.summary,
                "objective": metric,
                "failures": self.failures.copy(),
            }
        )
        return ranked

    def _consider(
        self, parameters: Dict[str, Any], metrics: Dict[str, Any], metric: str, **trial: Any
    ) -> None:
        """Keep only the best evaluated result so far, matching rank_results' stable order."""
        result, self._last_result = self._last_result, None
        if metric not in metrics:
            raise ValueError(f"Metric '{metric}' not found. {_metric_hint(metric, list(metrics))}")
        score = metrics[metric]
        self._record_evaluation(
            parameters, "undefined" if pd.isna(score) else "defined", score, trial
        )
        if pd.isna(score):
            return
        best = self._best
        if best is None or (
            score > best["score"] if metric_is_maximized(metric) else score < best["score"]
        ):
            self._best = {
                "score": score,
                "parameters": {name: parameters[name] for name in self._param_names},
                "result": result,
                "evidence_key": self._evaluations[-1][0],
                **trial,
            }

    def _no_winner_reason(self) -> str:
        """Explain why the last search has no selected candidate and how to retry."""
        if self._status is None:
            return "No search has run yet; call optimizer.optimize(...) first."
        if self._status == "running":
            return "The last search stopped with an error before finishing; fix it and rerun."
        if self._status == "failed":
            return (
                f"All {len(self.failures)} candidate evaluations failed; "
                "see optimizer.failures for each error."
            )
        retry = (
            " Retry with metric='total_return', or check"
            if self._objective != "total_return"
            else " Check"
        )
        return (
            f"No candidate had a defined {self._objective} "
            f"({self.summary['successful_evaluations']} evaluated, "
            f"{self.summary['undefined_objectives']} undefined, "
            f"{self.summary['failed_evaluations']} failed), so the ranked results are empty."
            f"{retry} that prices vary and risk_free_rate suits the objective."
        )

    def _selected(self) -> Dict[str, Any]:
        # A tentative best from an unfinished search is never a selection.
        if self._status != "done" or self._best is None:
            raise ValueError(self._no_winner_reason())
        return self._best

    def _record_evaluation(
        self, parameters: Dict[str, Any], status: str, score: Any, trial: Dict[str, Any]
    ) -> None:
        def position(name: str) -> int | None:
            values = self._space.get(name, [])
            value = parameters.get(name)
            for index, candidate in enumerate(values):
                if candidate is value:
                    return index
            return next((i for i, candidate in enumerate(values) if candidate == value), None)

        key = tuple(position(name) for name in self._param_names)
        score = None if status != "defined" else score
        self._evaluations.append(
            (key, status, score, trial.get("trial"), trial.get("simulation_seed"))
        )

    def _record_failure(self, parameters: Dict[str, Any], error: Exception) -> None:
        """Record an expected invalid parameter combination."""
        self._record_evaluation(parameters, "failed", None, self._trial_context)
        self.failures.append(
            {
                "parameters": parameters.copy(),
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )

    @staticmethod
    def export_json(results: pd.DataFrame, path: str | Path) -> None:
        """Save ranked candidates and the provenance attached to their DataFrame."""
        write_json(path, {"schema_version": 1, "results": results, "metadata": results.attrs})

    @abstractmethod
    def optimize(
        self,
        data: pd.DataFrame,
        config: BacktestConfig,
        strategy_class: Type[Strategy],
        param_space: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """Run optimization to find best parameters.

        :param data: OHLCV DataFrame with DatetimeIndex
        :param config: Backtest configuration
        :param strategy_class: Strategy class to optimize
        :param param_space: Dict of param_name: [values] to search
        :param metric: Metric to optimize (e.g., 'sharpe_ratio', 'total_return')
        :return: DataFrame of results sorted by metric (best first)
        """
        pass

    def get_name(self) -> str:
        """Get optimizer name."""
        return self._name

    def _run_backtest(
        self,
        data: pd.DataFrame,
        config: BacktestConfig,
        strategy: Strategy,
        backtest: Backtest | None = None,
    ) -> Dict[str, Any]:
        """Helper to run a single backtest and return metrics.

        :param data: OHLCV DataFrame
        :param config: Backtest configuration
        :param strategy: Strategy instance to test
        :return: Dict with parameters and metrics
        """
        bt = backtest or Backtest(data, config)
        # Avoid one rejection warning per candidate; regular runs still report rejections.
        bt._warn_rejected_orders = False
        results = bt.run([strategy])
        strategy_result = results.get_strategy(strategy.get_name())
        self._last_result = strategy_result
        return strategy_result.metrics

    @property
    def best_parameters(self) -> Dict[str, Any]:
        """Original parameter objects of the last search's top-ranked candidate."""
        return self._selected()["parameters"].copy()

    @property
    def best_result(self) -> StrategyResult:
        """The evaluated StrategyResult of the last search's top-ranked candidate.

        With repeats, this is the single in-sample trial that ranked first, not an average.
        """
        selected = self._selected()
        if selected["result"] is None:
            raise ValueError(
                f"{self.get_name()}._run_backtest returned metrics without keeping the full "
                "result. Run Backtest(data, config).run([strategy_class("
                "**optimizer.best_parameters)]) to inspect it."
            )
        return selected["result"]

    def _checked_evaluations(self) -> list[tuple[tuple[int | None, ...], str, Any, Any, Any]]:
        """Compact evidence of the last completed search, refused when incomplete."""
        if self._status not in ("done", "failed"):
            raise ValueError(self._no_winner_reason())
        if self._snapshot_error:
            raise ValueError(
                f"{self.get_name()} cannot show a per-combination summary because a "
                f"param_space value could not be copied ({self._snapshot_error}). The search "
                "itself is complete: use the ranked DataFrame, optimizer.failures and "
                "optimizer.best_result."
            )
        records = self._evaluations
        counts = {
            status: sum(record[1] == status for record in records)
            for status in ("defined", "undefined", "failed")
        }
        summary = self.summary
        if (
            len(records) != summary.get("attempted_evaluations")
            or counts["defined"] + counts["undefined"] != summary.get("successful_evaluations")
            or counts["undefined"] != summary.get("undefined_objectives")
            or counts["failed"] != len(self.failures)
            or any(None in record[0] for record in records)
        ):
            raise ValueError(
                f"{self.get_name()} recorded per-evaluation evidence for {len(records)} of "
                f"{summary.get('attempted_evaluations')} attempted evaluations, so no grouped "
                "summary is shown. Optimizers must call _consider for each successful evaluation "
                "and _record_failure for each failed one; inspect the ranked DataFrame and "
                "optimizer.failures instead."
            )
        return records

    def trial_summary(self) -> pd.DataFrame:
        """Observed objective per tested parameter combination of the last search.

        One row per combination, in the order first tested, with attempted, defined,
        undefined and failed evaluation counts and the observed min/median/max objective
        of its defined evaluations. All values are in-sample observations: they are not a
        robustness estimate and never change the selected candidate. ``contains_selected``
        marks the combination holding the top-ranked trial and ``selected_trial`` describes
        that single trial. Performs no new evaluations.
        """
        records = self._checked_evaluations()
        objective = self._objective or ""
        names = self._param_names
        groups: Dict[tuple, list] = {}
        for record in records:
            groups.setdefault(record[0], []).append(record)
        best = self._best if self._status == "done" else None
        best_key = None if best is None else best["evidence_key"]
        statistics = [f"{objective}_{stat}" for stat in ("min", "median", "max")]
        reserved = {"attempted", "defined", "undefined", "failed", *statistics}
        reserved |= {"contains_selected", "selected_trial", "note"}
        # A parameter named like a summary field keeps its value under "param:<name>".
        columns = {name: f"param:{name}" if name in reserved else name for name in names}
        rows = []
        for key, items in groups.items():
            scores = pd.Series([score for _, status, score, *_ in items if status == "defined"])
            scores = scores.astype(float)
            counts = {
                status: sum(item[1] == status for item in items)
                for status in ("defined", "undefined", "failed")
            }
            row: Dict[str, Any] = {
                columns[name]: deepcopy(self._space_snapshot[name][index])
                for name, index in zip(names, key)
            }
            row.update(
                attempted=len(items),
                **counts,
                **dict(zip(statistics, (scores.min(), scores.median(), scores.max()))),
                contains_selected=key == best_key,
                selected_trial="",
                note=self._trial_note(counts, len(items), objective),
            )
            if key == best_key and best is not None:
                trial = (
                    f"trial {best['trial']}, simulation_seed={best['simulation_seed']}, "
                    if "trial" in best
                    else ""
                )
                row["selected_trial"] = f"{trial}{objective}={best['score']:.6g}"
            rows.append(row)
        frame = pd.DataFrame(rows)
        for column in columns.values():
            # Keep the original parameter objects (no int-to-float upcasting).
            frame[column] = pd.Series([row[column] for row in rows], dtype=object)
        frame.attrs["trial_summary"] = {
            "objective": objective,
            "direction": "higher is better"
            if metric_is_maximized(objective)
            else "lower is better",
            "unit": "%" if objective in _PERCENT_METRICS else "",
            "scope": "in-sample observations; not a robustness estimate or a selection rule",
            "parameter_columns": columns,
        }
        return frame

    @staticmethod
    def _trial_note(counts: Dict[str, int], attempted: int, objective: str) -> str:
        parts = []
        if counts["failed"] == attempted:
            parts.append(f"all {attempted} evaluations failed; see optimizer.failures")
        elif counts["defined"] == 0:
            parts.append(
                f"no defined {objective}; check that prices vary and risk_free_rate "
                "suits the objective, or retry with metric='total_return'"
            )
        elif counts["defined"] == 1:
            parts.append("single observation")
        excluded = [
            f"{counts[status]} {status}"
            for status in ("undefined", "failed")
            if counts[status] and counts["defined"]
        ]
        if excluded:
            parts.append(" and ".join(excluded) + " excluded from the statistics")
        return "; ".join(parts)

    def plot_trials(self) -> "go.Figure":
        """Plotly Figure of every observed objective value per tested combination.

        Uses the same evidence as trial_summary(); performs no new evaluations.
        """
        from simple_backtest.visualization.plotter import plot_search_trials

        summary = self.trial_summary()
        # Same first-tested order as the summary rows.
        positions = {key: i for i, key in enumerate(dict.fromkeys(r[0] for r in self._evaluations))}
        points = [
            {"combination": positions[key], "objective": score, "trial": trial, "seed": seed}
            for key, status, score, trial, seed in self._evaluations
            if status == "defined"
        ]
        best = self._best if self._status == "done" else None
        return plot_search_trials(
            summary,
            self._param_names,
            points,
            None if best is None else best["score"],
            title=self.get_name(),
        )

    def __str__(self) -> str:
        """Concise outcome of the last search."""
        if self._status != "done" or self._best is None:
            return f"{self.get_name()}: {self._no_winner_reason()}"
        best, summary = self._best, self.summary
        direction = "higher" if metric_is_maximized(self._objective or "") else "lower"
        parameters = ", ".join(f"{k}={v!r}" for k, v in best["parameters"].items())
        lines = [
            f"{self.get_name()}: best {self._objective} = {best['score']:.4g} "
            f"({direction} is better)",
            f"  parameters: {parameters or 'strategy defaults'}",
            f"  {summary['successful_evaluations']} evaluated, "
            f"{summary['failed_evaluations']} failed, "
            f"{summary['undefined_objectives']} undefined {self._objective}",
        ]
        if "trial" in best:
            lines.append(
                f"  selected trial {best['trial']} (simulation_seed={best['simulation_seed']}): "
                "one in-sample trial, not an average across repeats"
            )
            lines.append("  spread across repeats per combination: optimizer.trial_summary()")
        details = "optimizer.best_result" if best["result"] is not None else "best_parameters"
        failures = ", optimizer.failures" if self.failures else ""
        lines.append(f"  inspect: {details}{failures}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self._name}')"
