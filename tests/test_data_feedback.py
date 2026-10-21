"""User-facing data errors and the one-step fixes they suggest."""

from pathlib import Path

import pandas as pd
import pytest

from simple_backtest import Backtest, BacktestConfig, BuyAndHoldStrategy
from simple_backtest.utils.validation import (
    DataValidationError,
    DateRangeError,
    validate_dataframe,
    validate_date_range,
)


@pytest.fixture
def prices():
    dates = pd.date_range("2023-01-01", periods=50, name="Date")
    return pd.DataFrame(
        {"Open": 100.0, "High": 110.0, "Low": 90.0, "Close": 105.0, "Volume": 1000}, index=dates
    )


def apply_fix(error, df=None):
    line = str(error).split("→ Fix:\n  ", 1)[1].splitlines()[0]
    scope = {"df": df, "pd": pd}
    exec(line, scope)
    return scope["df"]


@pytest.mark.parametrize("value", ["prices.csv", Path("prices.csv"), [], 123, None])
def test_backtest_non_dataframe_explains_csv_load(value):
    with pytest.raises(DataValidationError) as caught:
        Backtest(value, BacktestConfig(parallel_execution=False))
    message = str(caught.value)
    assert "DataFrame" in message and "read_csv" in message
    assert "index_col='Date'" in message and "parse_dates=True" in message


@pytest.mark.parametrize("date_name", ["Date", "date", "Datetime", "timestamp"])
@pytest.mark.parametrize("adj_name", [None, "Adj Close", "adj close", "ADJ CLOSE"])
def test_combined_structure_gap_and_sort_advice_works_in_one_fix(prices, date_name, adj_name):
    source = prices.copy()
    source.iloc[15, :4] = float("nan")
    if adj_name:
        source = source.rename(columns={"Close": adj_name})
    source = source.iloc[::-1].rename_axis(date_name).reset_index()
    original = source.copy(deep=True)
    with pytest.raises(DataValidationError) as caught:
        validate_dataframe(source)
    message = str(caught.value)
    assert date_name in message and "DatetimeIndex" in message
    assert "2023-01-16" in message and "NaN" in message
    assert "sorted" in message
    assert "fillna(0)" not in message and "fillna(method=" not in message
    if adj_name:
        assert repr(adj_name) + ": 'Close'" in message
        assert "unadjusted" in message and "Open/High/Low" in message
    fixed = apply_fix(caught.value, source)
    if adj_name:
        # This fixture already has consistent OHLC; apply the explicitly conditional rename.
        assert adj_name in fixed.columns and "Close" not in fixed.columns
        assert "only if Open/High/Low are already adjusted" in message
        fixed = fixed.rename(columns={adj_name: "Close"})
    result = Backtest(fixed, BacktestConfig(lookback_period=10, parallel_execution=False)).run(
        [BuyAndHoldStrategy(shares=1)]
    )
    assert result.get_strategy("BuyAndHold").trade_history
    pd.testing.assert_frame_equal(source, original)


def test_nan_dates_are_reported_per_column_and_leading_gap_fix_works(prices):
    prices.iloc[0, :4] = float("nan")
    prices.iloc[20, 3] = float("nan")
    with pytest.raises(DataValidationError) as caught:
        validate_dataframe(prices)
    message = str(caught.value)
    assert "2023-01-01" in message and "2023-01-21" in message
    assert "Close" in message and "Open" in message
    assert ".ffill()" not in message and ".dropna(subset=" in message
    fixed = apply_fix(caught.value, prices)
    validate_dataframe(fixed)


@pytest.mark.parametrize("multiindex", [False, True])
def test_public_validation_returns_normalized_copy_without_mutating_caller(prices, multiindex):
    source = prices.rename(columns=str.lower)
    if multiindex:
        source.columns = pd.MultiIndex.from_product([source.columns, ["ABC"]])
    original = source.copy(deep=True)
    normalized = validate_dataframe(source)
    pd.testing.assert_frame_equal(normalized, prices)
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize("rows", [1, 2, 25])
def test_too_few_rows_preserves_exception_and_gives_viable_fix(prices, rows):
    with pytest.raises(ValueError) as caught:
        Backtest(prices.head(rows), BacktestConfig(lookback_period=30))
    message = str(caught.value)
    assert str(rows) in message and "32" in message
    assert "supply more data" in message.lower()
    if rows >= 3:
        assert "lookback_period=23" in message
        Backtest(prices.head(rows), BacktestConfig(lookback_period=23))
    else:
        assert "lookback_period=0" not in message


def test_direct_date_range_validation_keeps_exception_type_and_fix(prices):
    with pytest.raises(DateRangeError, match="lookback_period=23"):
        validate_date_range(prices.head(25), None, None, 30)


def test_lowercase_original_frame_can_plot_both_result_entrypoints(prices):
    source = prices.rename(columns=str.lower)
    original = source.copy(deep=True)
    results = Backtest(source, BacktestConfig(lookback_period=10, parallel_execution=False)).run(
        [BuyAndHoldStrategy(shares=1)]
    )
    single = results.get_strategy("BuyAndHold").plot_trades(source)
    all_figures = results.plot_trades(source)
    for figure in (single, all_figures["BuyAndHold"]):
        assert list(figure.data[0].y) == list(prices.loc[list(figure.data[0].x), "Close"])
    pd.testing.assert_frame_equal(source, original)


def test_one_trading_bar_boundary_preserves_date_range_error_and_gives_working_fix(prices):
    with pytest.raises(DateRangeError, match="lookback_period=29") as caught:
        Backtest(prices.head(31), BacktestConfig(lookback_period=30))
    assert "32 rows" in str(caught.value)
    Backtest(prices.head(31), BacktestConfig(lookback_period=29))


def test_unusual_index_still_gets_structural_validation_error(prices):
    prices.index = prices.index.to_period("D")
    with pytest.raises(DataValidationError, match="DatetimeIndex"):
        validate_dataframe(prices)


@pytest.mark.parametrize("adj_close", [False, True])
def test_combined_fix_uses_original_lowercase_frame(prices, adj_close):
    source = prices.rename(columns=str.lower)
    if adj_close:
        source = source.rename(columns={"close": "adj close"})
    source.iloc[20, :4] = float("nan")
    source = source.iloc[::-1].rename_axis("date").reset_index()
    with pytest.raises(DataValidationError) as caught:
        validate_dataframe(source)
    fixed = apply_fix(caught.value, source)
    if adj_close:
        assert "adj close" in fixed.columns and "Close" not in fixed.columns
        fixed = fixed.rename(columns={"adj close": "Close"})
    Backtest(fixed, BacktestConfig(lookback_period=10, parallel_execution=False))


def test_multiindex_nan_fix_works_on_original_frame(prices):
    prices.iloc[20, :4] = float("nan")
    prices.columns = pd.MultiIndex.from_product([prices.columns.str.lower(), ["ABC"]])
    with pytest.raises(DataValidationError) as caught:
        validate_dataframe(prices)
    fixed = apply_fix(caught.value, prices)
    validate_dataframe(fixed)


def test_missing_timestamp_remains_a_validation_error(prices):
    dates = list(prices.index)
    dates[20] = pd.NaT
    prices.index = pd.DatetimeIndex(dates)
    with pytest.raises(DataValidationError):
        validate_dataframe(prices)


@pytest.mark.parametrize(
    "column,value,rule",
    [
        ("High", 80.0, "High below Low"),
        ("Open", 80.0, "Open below Low"),
        ("Open", 120.0, "Open above High"),
        ("Close", 80.0, "Close below Low"),
        ("Close", 120.0, "Close above High"),
    ],
)
def test_ohlc_error_names_rules_counts_and_first_dates(prices, column, value, rule):
    prices.loc[prices.index[10:14], column] = value
    original = prices.copy(deep=True)
    with pytest.raises(DataValidationError) as caught:
        validate_dataframe(prices)
    message = str(caught.value)
    assert "4 rows with invalid OHLC relationships" in message
    assert f"{rule}: 4 rows" in message
    assert all(f"2023-01-{day}" in message for day in (11, 12, 13))
    assert "2023-01-14" not in message
    assert "source" in message and "column-wise" in message
    pd.testing.assert_frame_equal(prices, original)


def test_adjusted_close_error_explains_missing_raw_prices(prices):
    # This represents a user's earlier rename; provenance is no longer available.
    prices["Close"] = prices["Close"] * 0.8
    with pytest.raises(DataValidationError) as caught:
        validate_dataframe(prices)
    message = str(caught.value)
    assert "Close below Low: 50 rows" in message
    assert "If" in message and "unadjusted Open/High/Low" in message
    assert "supply unadjusted Close" in message
    assert "supply consistently adjusted OHLC" in message
    assert "cannot recover" in message and "adjustment factor" in message
    assert "column-wise" not in message


def test_ohlc_warning_has_same_diagnostics_and_does_not_change_prices(prices):
    prices.loc[prices.index[20], "Open"] = 80.0
    with pytest.raises(DataValidationError) as caught:
        validate_dataframe(prices)
    with pytest.warns(UserWarning) as warned:
        result = validate_dataframe(prices, strict=False)
    assert len(warned) == 1
    assert str(warned[0].message) == str(caught.value)
    pd.testing.assert_frame_equal(result, prices)


def test_both_close_series_use_raw_close_without_adjustment(prices):
    prices["Adj Close"] = prices["Close"] * 0.8
    normalized = validate_dataframe(prices)
    pd.testing.assert_frame_equal(normalized, prices)


@pytest.fixture
def trending_prices():
    close = pd.Series([100.0 + 2 * i for i in range(120)],
                      index=pd.bdate_range("2023-01-02", periods=120, name="Date"))
    return pd.DataFrame({"Open": close, "High": close * 1.001, "Low": close * 0.999,
                         "Close": close, "Volume": 1e6})


@pytest.mark.parametrize("lowercase", [False, True])
def test_partial_gap_suggested_fix_reaches_result_in_one_step(trending_prices, lowercase):
    source = trending_prices.copy()
    source.loc[source.index[30], "Close"] = float("nan")
    source.loc[source.index[31], "Open"] = float("nan")
    source["Adj Close"] = source["Close"] * 0.98
    if lowercase:
        source = source.rename(columns=str.lower)
    source = source.iloc[::-1].reset_index()
    original = source.copy(deep=True)
    with pytest.raises(DataValidationError) as caught:
        validate_dataframe(source)
    assert ".ffill()" not in str(caught.value)
    assert "partly missing" in str(caught.value)
    fixed = apply_fix(caught.value, source)
    result = Backtest(fixed, BacktestConfig(parallel_execution=False)).run(
        [BuyAndHoldStrategy(shares=1)]
    )
    assert len(fixed) == 118 and result.get_strategy("BuyAndHold").trade_history
    pd.testing.assert_frame_equal(source, original)


def test_whole_price_gap_fix_preserves_consistent_bars(trending_prices):
    source = trending_prices.copy()
    source.iloc[[0, 30], :4] = float("nan")
    with pytest.raises(DataValidationError) as caught:
        validate_dataframe(source)
    message = str(caught.value)
    assert ".ffill()" in message and "all OHLC prices" in message
    fixed = apply_fix(caught.value, source)
    validate_dataframe(fixed)
    assert len(fixed) == 119
    pd.testing.assert_series_equal(fixed.iloc[29, :4], source.iloc[29, :4], check_names=False)


def test_adj_close_only_error_does_not_offer_unconditional_rename(trending_prices):
    source = trending_prices.drop(columns="Close").assign(
        **{"Adj Close": trending_prices["Close"] * 0.98}
    ).reset_index()
    with pytest.raises(DataValidationError) as caught:
        validate_dataframe(source)
    message = str(caught.value)
    assert "probably unadjusted" in message
    assert "supply unadjusted Close" in message and "supply consistently adjusted OHLC" in message
    assert "Adj Close below Low: 120 rows" in message and "2023-01-02" in message
    assert "cannot recover" in message
    assert "only if Open/High/Low are already adjusted" in message
    preprocessed = apply_fix(caught.value, source)
    assert isinstance(preprocessed.index, pd.DatetimeIndex)
    assert "Adj Close" in preprocessed and "Close" not in preprocessed
    # The provider supplies raw Close; the already-suggested date fix needs no retry.
    fixed = preprocessed.assign(Close=trending_prices["Close"])
    Backtest(fixed, BacktestConfig(parallel_execution=False)).run([BuyAndHoldStrategy(shares=1)])


def test_both_close_csv_fix_retains_raw_prices(trending_prices):
    source = trending_prices.assign(**{"Adj Close": trending_prices["Close"] * 0.98}).reset_index()
    with pytest.raises(DataValidationError) as caught:
        validate_dataframe(source)
    message = str(caught.value)
    assert "rename" not in message and "adjust" not in message
    fixed = apply_fix(caught.value, source)
    pd.testing.assert_series_equal(fixed["Close"], trending_prices["Close"], check_freq=False)
    Backtest(fixed, BacktestConfig(parallel_execution=False)).run([BuyAndHoldStrategy(shares=1)])


@pytest.mark.parametrize("with_adj_close", [False, True])
def test_isolated_ohlc_violations_do_not_suggest_mixed_prices(trending_prices, with_adj_close):
    source = trending_prices.copy()
    if with_adj_close:
        source["Adj Close"] = source["Close"] * 0.98
    source.loc[source.index[30], "Close"] = source.iloc[29]["Close"]
    source.loc[source.index[31], "Open"] = source.iloc[30]["Open"]
    with pytest.raises(DataValidationError) as caught:
        validate_dataframe(source)
    message = str(caught.value)
    assert "2 rows with invalid OHLC" in message and "column-wise" in message
    assert "unadjusted" not in message


def test_copied_adjusted_close_supports_hint_even_for_isolated_rows(trending_prices):
    source = trending_prices.assign(**{"Adj Close": trending_prices["Close"] * 0.98})
    source.loc[source.index[30], "Close"] = source.iloc[30]["Adj Close"]
    with pytest.raises(DataValidationError) as caught:
        validate_dataframe(source)
    assert "unadjusted Open/High/Low" in str(caught.value)


def test_partial_gap_with_adj_close_only_preprocesses_existing_columns(trending_prices):
    source = trending_prices.rename(columns={"Close": "adj close"}).rename(columns=str.lower)
    source.loc[source.index[30], "open"] = float("nan")
    source = source.reset_index()
    with pytest.raises(DataValidationError) as caught:
        validate_dataframe(source)
    fixed = apply_fix(caught.value, source)
    assert len(fixed) == 119 and "adj close" in fixed and "Close" not in fixed
    # This fixture's OHLC prices already share one basis, so its conditional rename is valid.
    validate_dataframe(fixed.rename(columns={"adj close": "Close"}))


def test_invalid_adj_close_strings_still_get_missing_close_error(prices):
    source = prices.drop(columns="Close").assign(**{"Adj Close": "bad"})
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataframe(source)
