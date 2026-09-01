"""
Side by side comparison of the SPY and FTSE 100 results.

Reads the processed tables that run_pipeline.py saves for each asset and prints a
compact comparison, then saves a comparison table and a figure. Run after the
pipeline has been run for both assets:

    python run_pipeline.py
    python run_pipeline.py --asset ftse --data data/raw/ftse.csv
    python compare_assets.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ASSETS = {"SPY": Path("data/processed"), "FTSE": Path("data/processed/ftse")}


def load(asset_dir: Path):
    acc = pd.read_csv(asset_dir / "accuracy.csv", index_col=0)
    bt = pd.read_csv(asset_dir / "backtest_summary.csv", index_col=0)
    var = pd.read_csv(asset_dir / "var_summary.csv")
    return acc, bt, var


def main():
    rows = []
    bts = {}
    for name, d in ASSETS.items():
        acc, bt, var = load(d)
        bts[name] = bt
        best_fc = acc["QLIKE"].idxmin()
        v99t = var[(var.alpha == 0.99) & (var.dist == "t")].set_index("model")
        gjr_ok = "yes" if v99t.loc["gjr", "cc_p"] >= 0.05 else "no"
        rows.append({
            "asset": name,
            "best_forecast": best_fc,
            "bh_sharpe": round(bt.loc["buy_and_hold", "sharpe"], 3),
            "bh_max_dd": round(bt.loc["buy_and_hold", "max_drawdown"], 3),
            "best_vt_sharpe": round(bt.loc[[i for i in bt.index if i.startswith("vt_")], "sharpe"].max(), 3),
            "vt_helps_sharpe": "yes" if bt.loc[[i for i in bt.index if i.startswith("vt_")], "sharpe"].max() > bt.loc["buy_and_hold", "sharpe"] else "no",
            "best_vt_max_dd": round(bt.loc[[i for i in bt.index if i.startswith("vt_")], "max_drawdown"].max(), 3),
            "gjr_t_var99_valid": gjr_ok,
        })
    comp = pd.DataFrame(rows).set_index("asset")
    comp.to_csv("data/processed/asset_comparison.csv")
    print(comp.to_string())

    # figure: Sharpe and max drawdown, buy and hold vs two vol targeting variants
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    variants = ["buy_and_hold", "vt_ewma", "vt_gjr"]
    labels = ["buy & hold", "vt ewma", "vt gjr"]
    x = np.arange(len(ASSETS))
    w = 0.25
    colors = ["#000000", "#2e75b6", "#c55a11"]

    for k, (v, c) in enumerate(zip(variants, colors)):
        sharpe = [bts[a].loc[v, "sharpe"] for a in ASSETS]
        ax[0].bar(x + (k - 1) * w, sharpe, w, label=labels[k], color=c)
        dd = [bts[a].loc[v, "max_drawdown"] * 100 for a in ASSETS]
        ax[1].bar(x + (k - 1) * w, dd, w, label=labels[k], color=c)

    ax[0].set_xticks(x); ax[0].set_xticklabels(list(ASSETS)); ax[0].set_title("Sharpe ratio")
    ax[0].legend()
    ax[1].set_xticks(x); ax[1].set_xticklabels(list(ASSETS)); ax[1].set_title("Max drawdown (%)")
    fig.suptitle("SPY vs FTSE 100: buy and hold against volatility targeting")
    fig.tight_layout()
    fig.savefig("reports/figures/asset_comparison.png")
    plt.close(fig)
    print("\nsaved data/processed/asset_comparison.csv and reports/figures/asset_comparison.png")


if __name__ == "__main__":
    main()
