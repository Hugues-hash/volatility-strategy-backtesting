"""
Volatility targeting.

The idea is simple. Pick a volatility you are comfortable running, say 10 percent a
year. Each day, look at the forecast for tomorrow. If the forecast is calm, hold
more of the asset. If it is stormy, hold less. Size the position so its forecast
volatility equals the target, and park whatever is left in cash.

    weight = target daily vol / forecast vol

The weight for a given day uses only the forecast made the day before, so nothing
here peeks at the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def volatility_target_weights(vol_hat: pd.Series, target_vol: float = 0.10, weight_cap: float = 1.5,
                              weight_floor: float = 0.0) -> pd.Series:
    """
    Turn a daily volatility forecast into a daily position weight.

    target_vol is an annual number like 0.10. weight_cap limits leverage, so 1.0
    means never borrow and 1.5 lets the position scale up a bit in calm markets.
    """
    # The target is annual, the forecast is daily, so put the target on a daily footing.
    target_daily = target_vol / np.sqrt(TRADING_DAYS)

    # Smaller forecast means a bigger position, and vice versa. Then keep the weight
    # inside sensible bounds so it never goes negative or takes on wild leverage.
    w = target_daily / vol_hat
    return w.clip(lower=weight_floor, upper=weight_cap).rename("weight")
