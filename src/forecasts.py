"""
The six volatility forecasts.

Every model here answers the same question: how volatile will tomorrow's return be?
And they all follow the same rule so the rest of the code can treat them alike:

    vol_hat[day D] is the forecast for day D, made using only data up to day D-1.

The six models, from simplest to most involved:

    rolling  a plain 21 day standard deviation of returns
    ewma     the RiskMetrics exponentially weighted volatility
    garch    GARCH(1,1), the workhorse volatility model
    gjr      GJR-GARCH, which lets a fall raise volatility more than a rise
    egarch   EGARCH, another way of modelling that same asymmetry
    rf       a Random Forest on the HAR features

The GARCH models are refit every 21 days on a moving window. Between refits we keep
the fitted parameters and just roll the recursion forward with the real returns,
which is a normal one step ahead forecast and saves refitting every single day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# Simple models
def rolling_vol_forecast(returns: pd.Series, window: int = 21) -> pd.Series:
    """A rolling standard deviation of returns, shifted one day so it never sees today."""
    rstd = returns.rolling(window=window, min_periods=window).std()
    return rstd.shift(1).rename("rolling")


def ewma_vol_forecast(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """
    RiskMetrics EWMA volatility.

    Tomorrow's variance is a blend of today's variance and today's squared return:
    var = lam * var_yesterday + (1 - lam) * return_yesterday^2. A higher lam means
    the estimate reacts more slowly. 0.94 is the standard daily setting.
    """
    r = returns.dropna()
    var = np.empty(len(r))
    seed = float(np.var(r.iloc[: min(len(r), 21)], ddof=1))  # start from the first month's variance
    prev = seed
    for i in range(len(r)):
        # The forecast for day i uses the squared return of day i-1, so no look ahead.
        var[i] = seed if i == 0 else lam * prev + (1.0 - lam) * float(r.iloc[i - 1]) ** 2
        prev = var[i]
    return pd.Series(np.sqrt(var), index=r.index, name="ewma")


# GARCH family
def _garch_spec(kind: str) -> dict:
    """The arch package settings for each GARCH variant. The 'o' term adds asymmetry."""
    kind = kind.lower()
    if kind == "garch":
        return {"vol": "GARCH", "p": 1, "q": 1}
    if kind == "gjr":
        return {"vol": "GARCH", "p": 1, "o": 1, "q": 1}
    if kind == "egarch":
        return {"vol": "EGARCH", "p": 1, "o": 1, "q": 1}
    raise ValueError(f"unknown garch kind: {kind}")


def garch_vol_forecast(returns: pd.Series, kind: str = "garch", fit_window: int = 1500, refit_every: int = 21,
                       min_train: int = 750, unstable_daily: float = 0.25, cap_daily: float = 0.15) -> pd.Series:
    """
    Rolling one step ahead GARCH forecast.

    We refit every refit_every days on the last fit_window returns, then hold those
    parameters fixed and filter the volatility forward over the next block of days
    using the real returns. Returns are scaled to percent while fitting because the
    optimiser behaves better on bigger numbers.

    Two guards keep the odd bad fit from spoiling the series. A short early sample can
    occasionally give EGARCH parameters that blow up when rolled forward. If a block's
    forecast tops unstable_daily (about 400 percent a year) we treat the fit as broken
    and carry the last good value forward. A final cap at cap_daily (about 240 percent
    a year, well above any real crisis) trims milder spikes. Both bind on only a
    handful of early days.
    """
    from arch import arch_model

    spec = _garch_spec(kind)
    r = returns.dropna() * 100.0
    n = len(r)
    vol_hat = pd.Series(index=r.index, dtype=float, name=kind if kind != "garch" else "garch")

    s = min_train
    while s < n:
        block_end = min(s + refit_every, n)   # this block runs from day s up to block_end
        block_len = block_end - s
        train_start = max(0, s - fit_window)

        try:
            # Fit on the window ending at s, i.e. only on the past.
            res = arch_model(r.iloc[train_start:s], mean="Constant", dist="normal", **spec).fit(disp="off")
            # Refilter over the training window plus this block. The last block_len
            # values are the one step ahead forecasts for the block's days.
            filt = r.iloc[train_start:block_end]
            cvol = arch_model(filt, mean="Constant", dist="normal", **spec).fix(res.params).conditional_volatility
            block_vals = np.asarray(cvol)[-block_len:] / 100.0
        except Exception:
            block_vals = np.full(block_len, np.nan)

        # Guard 1: an unstable fit that explodes. Carry the last good value forward.
        if (not np.all(np.isfinite(block_vals))) or (np.nanmax(block_vals) > unstable_daily):
            last = vol_hat.dropna()
            block_vals = np.full(block_len, last.iloc[-1] if last.size else np.nan)

        # Guard 2: a final safety cap above any real crisis level.
        vol_hat.iloc[s:block_end] = np.clip(block_vals, 0.0, cap_daily)
        s = block_end

    return vol_hat


# Random Forest on the HAR features
@dataclass
class WalkForwardSplit:
    """One train and test slice from the walk forward loop."""
    train_idx: pd.Index
    test_idx: pd.Index
    step: int


def walk_forward_splits(index: pd.Index, train_size: int = 504, test_size: int = 21, step_size: int = 21) -> Iterator[WalkForwardSplit]:
    """Walk a train then test window forward through time, yielding each slice."""
    n = len(index)
    start = 0
    step = 0
    while True:
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > n:   # stop once the test block runs off the end
            break
        yield WalkForwardSplit(index[start:train_end], index[train_end:test_end], step)
        start += step_size
        step += 1


def rf_vol_forecast(features: pd.DataFrame, scale_k: float = 1.0, train_size: int = 504, test_size: int = 21,
                    step_size: int = 21, random_state: int = 1) -> pd.Series:
    """
    Walk forward Random Forest forecast of tomorrow's volatility.

    The inputs are the three HAR lags and the target is next day Garman-Klass
    volatility. We retrain on a rolling 504 day window and predict the next 21 days,
    then step forward. scale_k lifts the intraday Garman-Klass scale onto the traded
    close to close scale (see the pipeline for where scale_k comes from).
    """
    from sklearn.ensemble import RandomForestRegressor

    X = features[["rv_d", "rv_w", "rv_m"]]
    y = features["rv_next"]

    preds = {}
    for sp in walk_forward_splits(features.index, train_size, test_size, step_size):
        model = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, n_jobs=-1, random_state=random_state)
        model.fit(X.loc[sp.train_idx], y.loc[sp.train_idx])
        for date, val in zip(sp.test_idx, model.predict(X.loc[sp.test_idx])):
            preds[date] = float(val)

    # The prediction on row t is for the volatility of day t+1, so shift it onto t+1.
    pred_next = pd.Series(preds).sort_index()
    return (pred_next.shift(1) * scale_k).rename("rf")


# Put every forecast on one aligned frame
def build_forecasts(features: pd.DataFrame, scale_k: float, garch_kinds: tuple[str, ...] = ("garch", "gjr", "egarch")) -> pd.DataFrame:
    """
    Run all six models and return one frame: a column per model, plus the realised
    volatility and the return, over the dates where every forecast exists.
    """
    returns = features["log_return"]
    realized = (features["rv"] * scale_k).rename("realized")  # the yardstick, on the traded scale

    cols = {"rolling": rolling_vol_forecast(returns, window=21), "ewma": ewma_vol_forecast(returns, lam=0.94)}
    for kind in garch_kinds:
        cols[kind] = garch_vol_forecast(returns, kind=kind)
    cols["rf"] = rf_vol_forecast(features, scale_k=scale_k)

    out = pd.DataFrame(cols)
    out["realized"] = realized
    out["log_return"] = returns
    return out.dropna()  # trim the warm up period where some models have no value yet
