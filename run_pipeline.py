"""
End to end pipeline.

Runs the whole project in order and saves every table and figure used in the
notebooks and the README:

    1. load and clean the price data, build features
    2. estimate the Garman-Klass to close scale factor on the burn in sample
    3. build the six volatility forecasts (cached to the processed folder)
    4. score the forecasts, run the Diebold-Mariano test
    5. run the volatility targeting strategy and the backtest
    6. build and backtest the VaR models

Run on the default SPY data with:

    python run_pipeline.py

Run on a second index by pointing it at another OHLC CSV. Outputs then go to their
own subfolders so the SPY results are not overwritten:

    python run_pipeline.py --asset ftse --data data/raw/ftse.csv
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.data_loader import load_ohlc, validate_prices
from src.features import build_feature_table
from src.forecasts import build_forecasts
from src.strategy import volatility_target_weights
from src.backtest import run_backtest, buy_and_hold, compare_strategies, performance_metrics
from src.var_model import var_backtest, estimate_t_dof
from src.evaluation import accuracy_table, dm_matrix

MODELS = ["rolling", "ewma", "garch", "gjr", "egarch", "rf"]
ANN = np.sqrt(252)


def output_dirs(asset: str) -> tuple[Path, Path]:
    """SPY keeps the default folders. Any other asset gets its own subfolder."""
    if asset == "spy":
        proc, fig = Path("data/processed"), Path("reports/figures")
    else:
        proc, fig = Path(f"data/processed/{asset}"), Path(f"reports/figures/{asset}")
    proc.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    return proc, fig


def get_forecasts(feat: pd.DataFrame, scale_k: float, proc: Path, rebuild: bool = False) -> pd.DataFrame:
    path = proc / "forecasts.csv"
    if path.exists() and not rebuild:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    fc = build_forecasts(feat, scale_k=scale_k)
    fc.to_csv(path)
    return fc


def main(asset: str = "spy", data_path: str = "data/raw/spy.csv", rebuild: bool = False):
    proc, fig = output_dirs(asset)
    label = asset.upper()

    # 1. data and features
    prices = load_ohlc(data_path)
    print(f"[{label}] data:", validate_prices(prices))
    feat = build_feature_table(prices)

    # 2. scale factor on the burn in sample only (first 504 days)
    burn = feat.iloc[:504]
    scale_k = float(burn["log_return"].std() / burn["rv"].mean())

    # 3. forecasts
    fc = get_forecasts(feat, scale_k, proc, rebuild=rebuild)
    r = fc["log_return"]
    realized = fc["realized"]
    meta = {
        "asset": asset,
        "scale_k": scale_k,
        "eval_start": str(fc.index.min().date()),
        "eval_end": str(fc.index.max().date()),
        "n_days": int(len(fc)),
    }

    # 4. forecast accuracy and Diebold-Mariano
    acc = accuracy_table(realized, fc[MODELS])
    acc.to_csv(proc / "accuracy.csv")
    dm_matrix(realized, fc[MODELS]).to_csv(proc / "dm_pvalues.csv")
    print(f"\n[{label}] forecast accuracy (sorted by QLIKE):\n", acc.round(5).to_string())

    # 5. strategy and backtest
    results = {"buy_and_hold": buy_and_hold(r)}
    for m in MODELS:
        w = volatility_target_weights(fc[m], target_vol=0.10, weight_cap=1.5)
        results[f"vt_{m}"] = run_backtest(w, r, cost_per_turn=0.0005)
    summary = compare_strategies(results)
    summary.to_csv(proc / "backtest_summary.csv")
    equity = pd.DataFrame({name: bt["equity"] for name, bt in results.items()})
    equity.to_csv(proc / "equity_curves.csv")
    print(f"\n[{label}] backtest summary:\n", summary.round(3).to_string())

    # target volatility sensitivity on the gjr strategy
    sens_rows = {}
    for tv in [0.08, 0.10, 0.12, 0.15]:
        w = volatility_target_weights(fc["gjr"], target_vol=tv, weight_cap=1.5)
        bt = run_backtest(w, r, cost_per_turn=0.0005)
        m = performance_metrics(bt["net_ret"], name=f"target_{int(tv*100)}")
        m["realized_vol"] = bt["net_ret"].std() * ANN
        m["avg_weight"] = bt["weight"].mean()
        sens_rows[f"target_{int(tv*100)}pct"] = m
    pd.DataFrame(sens_rows).T.to_csv(proc / "target_vol_sensitivity.csv")

    # crisis sub periods
    crises = {
        "GFC_2008": ("2008-09-01", "2009-06-30"),
        "COVID_2020": ("2020-02-15", "2020-06-30"),
        "Bear_2022": ("2022-01-01", "2022-12-31"),
    }
    crisis_rows = {}
    for cname, (a, b) in crises.items():
        for sname in ["buy_and_hold", "vt_ewma", "vt_gjr"]:
            bt = results[sname].loc[a:b]
            if len(bt) == 0:
                continue
            e = (1 + bt["net_ret"]).cumprod()
            crisis_rows[f"{cname}::{sname}"] = {
                "total_return": (1 + bt["net_ret"]).prod() - 1,
                "ann_vol": bt["net_ret"].std() * ANN,
                "max_drawdown": (e / e.cummax() - 1).min(),
            }
    pd.DataFrame(crisis_rows).T.to_csv(proc / "crisis_performance.csv")

    # 6. VaR validation
    var_rows = []
    for m in MODELS:
        nu = estimate_t_dof(r / fc[m])
        for alpha in [0.99, 0.95]:
            for dist in ["normal", "t"]:
                res = var_backtest(r, fc[m], alpha=alpha, dist=dist, nu=nu if dist == "t" else None)
                res["model"] = m
                res["nu"] = nu if dist == "t" else np.nan
                var_rows.append(res)
    var_tbl = pd.DataFrame(var_rows)
    var_tbl.to_csv(proc / "var_summary.csv", index=False)
    v99 = var_tbl[var_tbl["alpha"] == 0.99]
    print(f"\n[{label}] VaR 99% (exception rate and conditional coverage p):")
    print(v99[["model", "dist", "exception_rate", "kupiec_p", "independence_p", "cc_p"]].round(3).to_string(index=False))

    json.dump(meta, open(proc / "meta.json", "w"), indent=2)

    _make_figures(prices, feat, fc, results, equity, var_tbl, scale_k, fig, label)
    print(f"\n[{label}] saved tables to {proc} and figures to {fig}")
    return meta


def _make_figures(prices, feat, fc, results, equity, var_tbl, scale_k, fig, label):
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    fig1, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(prices.index, prices["Close"], color="#1f4e79", lw=0.8)
    ax[0].set_title(f"{label} close")
    ax[1].plot(feat.index, feat["rv"] * scale_k * ANN * 100, color="#c00000", lw=0.6)
    ax[1].set_title("Annualised realised volatility (Garman-Klass, close scale)")
    ax[1].set_ylabel("%")
    fig1.tight_layout(); fig1.savefig(fig / "01_price_and_vol.png"); plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(10, 4.5))
    win = fc.loc["2020-01-01":"2020-07-31"]
    if len(win):
        ax.plot(win.index, win["realized"] * ANN * 100, color="black", lw=1.3, label="realised")
        for m, c in [("ewma", "#2e75b6"), ("gjr", "#c55a11")]:
            ax.plot(win.index, win[m] * ANN * 100, lw=1.1, label=m, color=c)
        ax.set_title(f"{label}: forecast vs realised, COVID 2020"); ax.set_ylabel("annualised %"); ax.legend()
    fig2.tight_layout(); fig2.savefig(fig / "02_forecast_vs_realised_covid.png"); plt.close(fig2)

    fig3, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity["buy_and_hold"], color="black", lw=1.4, label="buy and hold")
    for m, c in [("vt_ewma", "#2e75b6"), ("vt_gjr", "#c55a11"), ("vt_rf", "#548235")]:
        ax.plot(equity.index, equity[m], lw=1.1, label=m, color=c)
    ax.set_yscale("log"); ax.set_title(f"{label}: growth of 1 dollar, log scale (net of costs)"); ax.legend()
    fig3.tight_layout(); fig3.savefig(fig / "03_equity_curves.png"); plt.close(fig3)

    fig4, ax = plt.subplots(figsize=(10, 4.5))
    for name, c in [("buy_and_hold", "black"), ("vt_ewma", "#2e75b6"), ("vt_gjr", "#c55a11")]:
        e = results[name]["equity"]
        ax.plot(e.index, (e / e.cummax() - 1) * 100, lw=1.0, label=name, color=c)
    ax.set_title(f"{label}: drawdown"); ax.set_ylabel("%"); ax.legend()
    fig4.tight_layout(); fig4.savefig(fig / "04_drawdowns.png"); plt.close(fig4)

    summ = compare_strategies(results)
    fig5, ax = plt.subplots(figsize=(9, 4.5))
    order = summ["sharpe"].sort_values()
    colors = ["#808080" if i == "buy_and_hold" else "#1f4e79" for i in order.index]
    ax.barh(order.index, order.values, color=colors)
    ax.set_title(f"{label}: Sharpe ratio, net of costs")
    ax.axvline(summ.loc["buy_and_hold", "sharpe"], color="#c00000", ls="--", lw=1)
    fig5.tight_layout(); fig5.savefig(fig / "05_sharpe_bar.png"); plt.close(fig5)

    v = var_tbl[var_tbl["alpha"] == 0.99]
    piv = v.pivot(index="model", columns="dist", values="exception_rate")[["normal", "t"]] * 100
    fig6, ax = plt.subplots(figsize=(9, 4.5))
    piv.plot(kind="bar", ax=ax, color={"normal": "#c55a11", "t": "#2e75b6"})
    ax.axhline(1.0, color="black", ls="--", lw=1, label="expected 1%")
    ax.set_title(f"{label}: 99% VaR exception rate, normal vs Student-t"); ax.set_ylabel("%"); ax.legend()
    fig6.tight_layout(); fig6.savefig(fig / "06_var_exception_rates.png"); plt.close(fig6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="spy", help="asset label, controls the output subfolder")
    ap.add_argument("--data", default="data/raw/spy.csv", help="path to the OHLC CSV")
    ap.add_argument("--rebuild", action="store_true", help="refit the forecasts from scratch")
    args = ap.parse_args()
    main(asset=args.asset, data_path=args.data, rebuild=args.rebuild)
