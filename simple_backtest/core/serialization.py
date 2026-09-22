"""Portable experiment snapshots without pickling executable Python objects."""

import hashlib
import json
from datetime import date, datetime
from math import isfinite, isnan
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_value(value: Any) -> Any:
    """Encode portable values; undefined numbers become null, infinities explicit strings."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if isfinite(number) else (None if isnan(number) else str(number))
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return {"index": json_value(value.index.tolist()), "values": json_value(value.tolist())}
    if isinstance(value, pd.DataFrame):
        return {
            "columns": json_value(value.columns.tolist()),
            "index": json_value(value.index.tolist()),
            "values": json_value(value.values.tolist()),
        }
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in value]
    descriptor = {"type": f"{type(value).__module__}.{type(value).__qualname__}"}
    if hasattr(value, "get_params"):
        descriptor["parameters"] = json_value(value.get_params(deep=False))
    else:
        descriptor["capture"] = "opaque; supply a portable value through get_parameters()"
    return descriptor


def write_json(path: str | Path, payload: Any) -> None:
    """Write schema-versioned JSON suitable for comparing saved experiments."""
    Path(path).write_text(
        json.dumps(json_value(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def data_fingerprint(data: pd.DataFrame) -> str:
    """Hash values, timestamps, column order, and dtypes of normalized input data."""
    schema = json.dumps(
        {
            "columns": json_value(data.columns.tolist()),
            "dtypes": [str(dtype) for dtype in data.dtypes],
            "index_dtype": str(data.index.dtype),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(schema.encode())
    digest.update(pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes())
    return digest.hexdigest()
