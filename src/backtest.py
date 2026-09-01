"""
Backtest and performance metrics for the volatility targeting strategy.

We work in simple returns here, not log returns, so the equity curve compounds the
way real money does. Each day we hold `weight` in the asset and the rest in cash,
and we pay a small cost whenever the weight changes. Cash earns nothing, which is a
touch conservative and keeps the comparison with buy and hold honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # trading days in a year, used for annualising


def run_backtest(weights: pd.Series, log_returns: pd.Series, cost_per_turn: float = 0.0005) -> pd.DataFrame:
    """Run one strategy and return its full daily record."""
    # Keep only the dates both series share, then line them up.
    idx = weights.index.intersection(log_returns.index)
    w = weights.loc[idx].astype(float)

    # Log returns compound by adding, simple returns by multiplying. We want the
    # second one for a money-like equity curve, so convert with expm1.
    asset = np.expm1(log_returns.loc[idx].astype(float))

    # Turnover is how far the weight moved since yesterday. Day one counts the full
    # weight, because we open the position from cash.
    turnover = w.diff().abs().fillna(w.abs())

    gross = w * asset                       # return before trading costs
    net = gross - cost_per_turn * turnover  # return after paying to trade

    out = pd.DataFrame({"asset_ret": asset, "weight": w, "turnover": turnover, "gross_ret": gross, "net_ret": net})
    out["equity"] = (1.0 + out["net_ret"]).cumprod()  # growth of one pound
    return out


def performance_metrics(returns: pd.Series, name: str = "strategy") -> pd.Series:
    """The usual scorecard: annual return, annual vol, Sharpe, Sortino, drawdown, Calmar."""
    r = returns.dropna()
    equity = (1.0 + r).cumprod()

    # Annual return is the compound growth rate, not the plain average of daily returns.
    years = len(r) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0

    ann_vol = r.std() * np.sqrt(TRADING_DAYS)            # daily vol scaled up to a year
    sharpe = r.mean() / r.std() * np.sqrt(TRADING_DAYS)  # return per unit of risk, risk-free rate set to 0

    # Sortino is Sharpe but it only treats down days as risk.
    downside = r[r < 0].std()
    sortino = r.mean() / downside * np.sqrt(TRADING_DAYS) if downside > 0 else np.nan

    # Drawdown is how far below its own peak the equity has fallen. We want the worst one.
    drawdown = equity / equity.cummax() - 1.0
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan  # return earned per unit of worst-case pain

    return pd.Series({"ann_return": cagr, "ann_vol": ann_vol, "sharpe": sharpe, "sortino": sortino,
                      "max_drawdown": max_dd, "calmar": calmar, "pct_positive_days": float((r > 0).mean())}, name=name)


def buy_and_hold(log_returns: pd.Series) -> pd.DataFrame:
    """The benchmark: hold the asset fully, weight fixed at 1, and never trade."""
    w = pd.Series(1.0, index=log_returns.index, name="weight")
    return run_backtest(w, log_returns, cost_per_turn=0.0)


def compare_strategies(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack the scorecards for several backtests into one table, one row each."""
    rows = {}
    for name, bt in results.items():
        m = performance_metrics(bt["net_ret"], name=name)
        m["avg_ann_turnover"] = float(bt["turnover"].mean() * TRADING_DAYS)  # how much it trades in a year
        m["avg_weight"] = float(bt["weight"].mean())                          # average exposure to the asset
        rows[name] = m
    return pd.DataFrame(rows).T
