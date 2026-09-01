"""
Scoring the forecasts, and testing whether the differences are real.

Three losses measure how close each forecast is to realised volatility. QLIKE is the
one we lead with: it copes well with a noisy volatility proxy and it punishes under
prediction harder than over prediction, which is what you want for risk. The
Diebold-Mariano test then asks whether a gap in QLIKE between two forecasts is big
enough to be real, or just noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mse(realized: pd.Series, forecast: pd.Series) -> float:
    """Mean squared error, the plain average of the squared misses."""
    return float(np.mean((realized - forecast) ** 2))


def mae(realized: pd.Series, forecast: pd.Series) -> float:
    """Mean absolute error, the average size of the misses."""
    return float(np.mean(np.abs(realized - forecast)))


def qlike(realized: pd.Series, forecast: pd.Series, eps: float = 1e-12) -> float:
    """
    QLIKE loss, worked out on variances (so we square the vols first). Lower is
    better. It leans on the forecast harder when it under predicts a big move, which
    is the mistake that hurts most in risk work.
    """
    s2 = np.maximum(np.asarray(realized) ** 2, eps)   # realised variance, floored away from zero
    h = np.maximum(np.asarray(forecast) ** 2, eps)    # forecast variance
    return float(np.mean(s2 / h - np.log(s2 / h) - 1.0))


def qlike_loss_series(realized: pd.Series, forecast: pd.Series, eps: float = 1e-12) -> pd.Series:
    """The per day QLIKE loss, which the Diebold-Mariano test needs day by day."""
    s2 = np.maximum(realized.values ** 2, eps)
    h = np.maximum(forecast.values ** 2, eps)
    return pd.Series(s2 / h - np.log(s2 / h) - 1.0, index=realized.index)


def accuracy_table(realized: pd.Series, forecasts: pd.DataFrame) -> pd.DataFrame:
    """One row per forecast with MSE, MAE and QLIKE, sorted best QLIKE first."""
    rows = {name: {"MSE": mse(realized, forecasts[name]), "MAE": mae(realized, forecasts[name]),
                   "QLIKE": qlike(realized, forecasts[name])} for name in forecasts.columns}
    return pd.DataFrame(rows).T.sort_values("QLIKE")


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, harvey_correction: bool = True) -> tuple[float, float]:
    """
    Diebold-Mariano test on two per day loss series (one step ahead).

    It looks at the daily difference d = loss_a - loss_b. A positive statistic means
    model A loses more, so model B is the better forecast. Returns the statistic and
    a two sided p value; a small p value means the two really do differ.
    """
    d = (loss_a - loss_b).dropna()
    n = len(d)
    if d.var(ddof=1) == 0 or n < 3:   # identical or too little data to tell
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(d.var(ddof=1) / n)
    if harvey_correction:
        stat *= np.sqrt((n - 1) / n)   # small sample tweak, barely matters when n is large
    p = 2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


def dm_matrix(realized: pd.Series, forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    A table of Diebold-Mariano p values for every pair of forecasts. Entry (row,
    column) is the p value for whether that pair differs in accuracy.
    """
    losses = {name: qlike_loss_series(realized, forecasts[name]) for name in forecasts.columns}
    names = list(forecasts.columns)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            out.loc[a, b] = np.nan if a == b else round(diebold_mariano(losses[a], losses[b])[1], 4)
    return out
