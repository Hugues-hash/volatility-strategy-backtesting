"""
Turning prices into the inputs the models need.

Three things get built here:

- daily log returns,
- a daily volatility estimate from the price range (Garman and Klass, 1980), which
  is steadier than just squaring the daily return because it also uses the high and
  the low,
- the HAR features: yesterday's volatility averaged over one day, one week and one
  month, which is the classic way to summarise how volatile things have been.

The Garman-Klass series is also the yardstick we later score the forecasts against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def garman_klass_volatility(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Daily volatility from the day's open, high, low and close (Garman and Klass, 1980).

    Using the full range makes this about five times more accurate than squaring the
    close to close return, so it is a cleaner target to forecast. The output is a
    volatility (a standard deviation), not a variance.
    """
    log_hl = np.log(high / low)      # how wide the day's range was
    log_co = np.log(close / open_)   # how far it travelled open to close
    var = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2

    # Rounding can push the estimate a hair below zero, so clip before the square root.
    var = var.clip(lower=0.0)
    return np.sqrt(var).rename("rv")


def har_lagged_features(rv: pd.Series) -> pd.DataFrame:
    """
    The three HAR features, each using only data up to and including today.

        rv_d = today's volatility
        rv_w = the average over the last 5 days   (a week)
        rv_m = the average over the last 22 days   (a month)

    Short, medium and long memory in one small set of inputs.
    """
    daily = rv
    weekly = rv.rolling(window=5, min_periods=5).mean()
    monthly = rv.rolling(window=22, min_periods=22).mean()
    return pd.DataFrame({"rv_d": daily, "rv_w": weekly, "rv_m": monthly})


def one_day_ahead_target(rv: pd.Series) -> pd.Series:
    """What the Random Forest tries to predict: tomorrow's volatility."""
    return rv.shift(-1).rename("rv_next")


def build_feature_table(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Run the whole feature pipeline and return one tidy table.

    Given clean OHLCV prices, it produces the log return, the Garman-Klass volatility,
    the three HAR lags and the next day target. Rows left empty by the rolling and
    shift steps are dropped, so the result is ready to model.
    """
    log_ret = np.log(prices["Close"] / prices["Close"].shift(1)).rename("log_return")
    rv = garman_klass_volatility(prices["Open"], prices["High"], prices["Low"], prices["Close"])
    har = har_lagged_features(rv)
    target = one_day_ahead_target(rv)

    table = pd.concat([log_ret, rv, har, target], axis=1)
    return table.dropna()
