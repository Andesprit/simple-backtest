"""Post-run explanation of recorded strategy errors; nothing here re-runs a strategy."""

import sysconfig
import traceback
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import simple_backtest

_TEXT = 200
_MESSAGE = 300
MAX_CONTEXT_BARS = 200
# Frames from this package, the standard library and installed packages are rarely
# where a strategy bug is written; the first frame outside them is the likely callsite.
_LIBRARY_PATHS = tuple(
    {
        str(Path(simple_backtest.__file__).parent),
        *(sysconfig.get_paths()[key] for key in ("stdlib", "purelib", "platlib")),
    }
)

_STAGE_MEANING = {
    "prediction": (
        "predict() or the check of its returned order raised, so this prediction submitted "
        "no order. predict() saw only bars before {t}. On the last bar, final liquidation "
        "can still fill; see the fills below."
    ),
    "trade callback": (
        "on_trade_executed() raised after the fill at {t} was committed. The fill, cash and "
        "position stay in trade_history; a callback failure does not undo a fill."
    ),
    "execution": (
        "Executing the strategy's order at {t} raised (this includes custom commission or "
        "execution-price functions). Check the fills below to see whether one was recorded."
    ),
    "final liquidation": (
        "Closing the open position on the last bar ({t}) raised. Check the fills below to see "
        "whether one was recorded."
    ),
}


def _short(text: Any, limit: int = _TEXT) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def source_location(error: BaseException) -> Dict[str, Any] | None:
    """Return a small, primitive record of where ``error`` was raised, or None.

    Only names and line numbers are kept: no exception, frame or local variable.
    """
    frames = [
        (frame.f_code.co_filename, frame.f_code.co_name, line)
        for frame, line in traceback.walk_tb(error.__traceback__)
    ]
    if not frames:
        return None
    outside = [f for f in frames if not f[0].startswith(_LIBRARY_PATHS)]
    filename, function, line = (outside or frames)[-1]
    location = {
        "filename": _short(filename),
        "function": _short(function, 100),
        "line": line,
        "frame": "first outside libraries" if outside else "innermost",
    }
    if outside and frames[-1] != outside[-1]:
        name, function, line = frames[-1]
        location["raised_in"] = _short(f"{Path(name).name}:{line} in {function}")
    return location


def error_where(error: Dict[str, Any]) -> str:
    """One line naming the recorded source of a strategy error."""
    if "source" not in error:
        return "not recorded (this result predates source recording)"
    source = error["source"]
    if not isinstance(source, dict):
        return "not recorded (no traceback was available)"
    where = f"{source.get('filename')}:{source.get('line')} in {source.get('function')}()"
    if source.get("frame") == "innermost":
        where += (
            " (innermost frame; no frame outside simple_backtest, Python or installed packages)"
        )
    if source.get("raised_in"):
        where += f", raised inside {source['raised_in']}"
    return _short(where, 2 * _TEXT)


_SHORT_MEANING = {
    "prediction": "This prediction submitted no order; predict() saw only earlier bars.",
    "trade callback": "The fill was committed before the callback failed; it was not undone.",
    "execution": "Executing the order raised; see Fills for whether one was recorded.",
    "final liquidation": "Closing the position raised; see Fills for whether one was recorded.",
}


def stage_meaning(error: Dict[str, Any], short: bool = False) -> str:
    """What a stage implies about timing and fills, for one recorded error."""
    stage = error.get("stage")
    meanings = _SHORT_MEANING if short else _STAGE_MEANING
    template = meanings.get(stage, "Unknown stage {stage!r}; read the raw record.")
    return template.format(t=error.get("timestamp"), stage=stage)


def _fill_line(fill: Dict[str, Any]) -> str:
    text = (
        f"{fill.get('signal')} {fill.get('shares')} @ {fill.get('price')} "
        f"(order {fill.get('order_id')}; after it: cash {fill.get('cash')}, "
        f"shares {fill.get('total_shares')})"
    )
    return text + (" [forced liquidation]" if fill.get("forced_liquidation") else "")


def _fill_lines(error: Dict[str, Any], fills: List[Dict[str, Any]], t) -> List[str]:
    same_bar = [fill for fill in fills if pd.Timestamp(fill["timestamp"]) == t]
    callback = error.get("stage") == "trade callback"
    order_id = error.get("order_id")
    handled = [fill for fill in same_bar if callback and fill.get("order_id") == order_id]
    if handled and order_id is not None:
        lines = ["Fill the failing callback was handling (matched by recorded order id):"]
        lines.append("  " + _fill_line(handled[0]))
        others = [fill for fill in same_bar if fill is not handled[0]]
        if others:
            lines.append("Other fills on the same bar:")
            lines += ["  " + _fill_line(fill) for fill in others]
        return lines
    lines = []
    if callback and order_id is not None:
        lines.append(f"Recorded order id {_short(order_id, 60)} was not found in trade_history.")
    if not same_bar:
        return lines + [f"No fills recorded at {t}."]
    lines.append(f"Fills recorded at {t}:")
    lines += ["  " + _fill_line(fill) for fill in same_bar]
    if callback and len(same_bar) > 1:
        lines.append(
            "Several fills share this bar and none is linked to this record by order id, "
            "so which one the callback was handling is unknown."
        )
    return lines


def _price_lines(result, stage, t, price_data, context_bars: int) -> List[str]:
    from simple_backtest.core.serialization import data_fingerprint
    from simple_backtest.utils.validation import validate_dataframe

    refused = "Price context not shown: "
    try:
        data = validate_dataframe(price_data)
    except Exception as error:
        return [refused + "the supplied data is not valid price data: " + _short(error, _MESSAGE)]
    if (data.index.tz is None) != (t.tz is None):
        return [
            refused + f"the supplied index timezone ({data.index.tz}) does not match the error "
            f"time {t}. Pass prices with the same timezone you gave to Backtest."
        ]
    if t not in data.index:
        return [refused + f"{t} is not in the supplied data."]

    metadata = result._metadata
    lookback = (metadata.get("config") or {}).get("lookback_period")
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        lookback = None  # without it the rows predict() saw are unknown
    recorded = (metadata.get("data") or {}).get("fingerprint")
    if recorded is None:
        exact, check = False, "not verified: this result has no recorded data fingerprint"
    elif data_fingerprint(data) == recorded:
        exact, check = True, "identical to the data this run used (fingerprint checked)"
    else:
        exact, check = (
            False,
            "NOT the data this run used (fingerprint differs: changed, partial or extra rows)",
        )

    position = data.index.get_loc(t)
    count = min(context_bars, lookback) if isinstance(lookback, int) else context_bars
    prior = data.iloc[max(0, position - count) : position]
    columns = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in data.columns]
    exact = exact and lookback is not None
    if exact:
        shown = f"last {len(prior)} of the {lookback} bars of recorded lookback"
    else:
        shown = f"{len(prior)} bars before the error" + (
            "; lookback not recorded" if lookback is None else ""
        )
    lines = [f"Supplied prices: {check}."]
    if exact and stage == "prediction":
        lines.append(f"Bars before {t} ({shown}; predict() saw these rows):")
    else:
        lines.append(
            f"Supplied context before {t} ({shown}; not proven to be what the strategy saw):"
        )
    lines += ["  " + row for row in prior[columns].to_string().splitlines()]
    lines.append("Bar at the error time (orders execute here; predict() never sees it):")
    lines += ["  " + row for row in data.loc[[t], columns].to_string().splitlines()]
    return lines


def explain_error(result, error_index=0, price_data=None, context_bars=5) -> str:
    """Build the text returned by StrategyResult.explain_error."""
    errors = result.errors
    if isinstance(context_bars, bool) or not isinstance(context_bars, int):
        raise TypeError(f"context_bars must be an integer; got {context_bars!r}.")
    if not 1 <= context_bars <= MAX_CONTEXT_BARS:
        raise ValueError(f"context_bars must be from 1 to {MAX_CONTEXT_BARS}; got {context_bars}.")
    if price_data is not None and not isinstance(price_data, pd.DataFrame):
        raise TypeError(
            f"price_data must be a pandas DataFrame or None; got {type(price_data).__name__}."
        )
    if not errors:
        return (
            f"{result.name}: no strategy errors were recorded. Errors are recorded only with "
            "error_policy='continue'; the default 'raise' stops the run at the first one."
        )
    if isinstance(error_index, bool) or not isinstance(error_index, int):
        raise TypeError(f"error_index must be an integer; got {error_index!r}.")
    if not -len(errors) <= error_index < len(errors):
        raise IndexError(
            f"error_index must be from 0 to {len(errors) - 1} ({result.name} recorded "
            f"{len(errors)} error{'s' if len(errors) != 1 else ''}); got {error_index}."
        )
    index = error_index % len(errors)
    error = errors[index]
    t = pd.Timestamp(error.get("timestamp"))
    lines = [
        f"{result.name}: error {index + 1} of {len(errors)}",
        f"When:    {t}",
        f"Stage:   {error.get('stage')}",
        f"Error:   {error.get('error_type')}: {_short(error.get('message', ''), _MESSAGE)}",
        f"Where:   {error_where(error)}",
        f"Meaning: {stage_meaning({**error, 'timestamp': t})}",
    ]
    lines += _fill_lines(error, result.trade_history, t)
    if price_data is not None:
        lines += _price_lines(result, error.get("stage"), t, price_data, context_bars)
    lines.append(
        "Not recorded: strategy attributes, indicator values, local variables and callback "
        "arguments at the time of the error."
    )
    lines.append(f"Full record: result.errors[{index}]; every record: result.export_json(path).")
    return "\n".join(lines)
