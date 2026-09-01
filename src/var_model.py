"""
Value at Risk, and the tests a risk team uses to check it.

Value at Risk (VaR) is a loss line. A 99 percent one day VaR is the loss you should
only breach on about 1 day in 100. We build it from each volatility forecast and
then check it the way a market risk validation team would: count the days the loss
was worse than the line, and test whether there are too many of them and whether
they clump together.

    Kupiec               are there roughly the right number of breaches
    Christoffersen ind.  are the breaches spread out, or do they cluster in crises
    Christoffersen cc    the two above combined into one test

A good 99 percent VaR breaks on about 1 percent of days, with the breaches scattered
rather than bunched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def estimate_t_dof(std_resid: pd.Series, lo: float = 3.0, hi: float = 100.0) -> float:
    """
    Guess the Student-t degrees of freedom from how fat the tails are.

    Fewer degrees of freedom means fatter tails. We back it out from the excess
    kurtosis of the standardised residuals (return / forecast vol) and keep it in a
    sensible range. Near normal residuals return the upper bound.
    """
    z = std_resid.dropna()
    ex_kurt = float(stats.kurtosis(z, fisher=True, bias=False))
    if ex_kurt <= 0:   # tails no fatter than normal
        return hi
    nu = 4.0 + 6.0 / ex_kurt
    return float(min(max(nu, lo), hi))


def var_quantile(alpha: float, dist: str = "normal", nu: float | None = None) -> float:
    """
    The multiplier z in VaR = z * forecast vol.

    alpha is the confidence, e.g. 0.99. For the Student-t we scale the quantile so
    the distribution still has a variance of one, which keeps it comparable to the
    normal.
    """
    p = 1.0 - alpha
    if dist == "normal":
        return float(-stats.norm.ppf(p))            # e.g. 2.33 at 99 percent
    if dist == "t":
        if nu is None or nu <= 2:
            raise ValueError("Student-t VaR needs nu > 2")
        scale = np.sqrt((nu - 2.0) / nu)            # rescale to unit variance
        return float(-stats.t.ppf(p, nu) * scale)
    raise ValueError(f"unknown dist: {dist}")


def var_series(vol_hat: pd.Series, alpha: float, dist: str = "normal", nu: float | None = None) -> pd.Series:
    """The daily VaR as a positive loss number: VaR = z * forecast vol."""
    return (var_quantile(alpha, dist=dist, nu=nu) * vol_hat).rename("var")


def kupiec_pof(n: int, x: int, alpha: float) -> tuple[float, float]:
    """
    Kupiec test: is the breach count about right?

    n days, x breaches, expected breach rate p = 1 - alpha. Returns the test
    statistic and its p value. A small p value means the count is off.
    """
    p = 1.0 - alpha
    if x == 0:
        lr = -2.0 * (n * np.log(1.0 - p))
        return float(lr), float(1.0 - stats.chi2.cdf(lr, 1))
    pi = x / n   # the breach rate we actually saw
    ll_null = (n - x) * np.log(1.0 - p) + x * np.log(p)     # likelihood if the rate is p
    ll_alt = (n - x) * np.log(1.0 - pi) + x * np.log(pi)    # likelihood at the observed rate
    lr = -2.0 * (ll_null - ll_alt)
    return float(lr), float(1.0 - stats.chi2.cdf(lr, 1))


def christoffersen_independence(exceptions: pd.Series) -> tuple[float, float]:
    """
    Christoffersen test: do breaches cluster?

    It counts how often a breach follows a breach versus follows a calm day. If a
    breach today makes a breach tomorrow more likely, the breaches are clustering and
    the p value is small.
    """
    e = exceptions.dropna().astype(int).values
    n00 = n01 = n10 = n11 = 0   # counts of every yesterday->today pair of (calm=0, breach=1)
    for prev, cur in zip(e[:-1], e[1:]):
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0   # breach chance after a calm day
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0   # breach chance after a breach
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)             # breach chance overall
    if pi in (0.0, 1.0) or (n10 + n11) == 0:
        return 0.0, 1.0

    def ll(p, a, b):
        # Log likelihood of a breaches and b calm days at probability p, guarding log(0).
        term = 0.0
        if p > 0 and a > 0:
            term += a * np.log(p)
        if p < 1 and b > 0:
            term += b * np.log(1.0 - p)
        return term

    ll_null = ll(pi, n01 + n11, n00 + n10)                 # one rate for everything
    ll_alt = ll(pi01, n01, n00) + ll(pi11, n11, n10)       # different rates after calm vs breach
    lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    return float(lr), float(1.0 - stats.chi2.cdf(lr, 1))


def var_backtest(returns: pd.Series, vol_hat: pd.Series, alpha: float = 0.99, dist: str = "normal", nu: float | None = None) -> dict:
    """
    Backtest one VaR model at one confidence level.

    A breach is a day whose return is worse than minus the VaR. Returns the breach
    count and rate plus the Kupiec, independence and combined (conditional coverage)
    statistics with their p values.
    """
    idx = returns.index.intersection(vol_hat.index)
    r = returns.loc[idx]
    var = var_series(vol_hat.loc[idx], alpha, dist=dist, nu=nu)
    exceptions = (r < -var).astype(int)   # 1 on a breach day, 0 otherwise

    n, x = int(len(r)), int(exceptions.sum())
    lr_uc, p_uc = kupiec_pof(n, x, alpha)
    lr_ind, p_ind = christoffersen_independence(exceptions)
    lr_cc = lr_uc + lr_ind                                  # conditional coverage adds the two up
    p_cc = float(1.0 - stats.chi2.cdf(lr_cc, 2))

    return {"alpha": alpha, "dist": dist, "n": n, "exceptions": x,
            "exception_rate": x / n if n else np.nan, "expected_rate": 1.0 - alpha,
            "kupiec_lr": lr_uc, "kupiec_p": p_uc, "independence_lr": lr_ind, "independence_p": p_ind,
            "cc_lr": lr_cc, "cc_p": p_cc}
