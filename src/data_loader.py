"""
Loading price data.

The project runs on daily OHLC prices (open, high, low, close). The main entry
point is load_ohlc, which reads a CSV from any of the usual sources and hands back a
clean, sorted frame. There is also a small yfinance helper to refresh the cache, but
you never need it to run the project because the data ships with the repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_spy(start: str = "2005-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Pull daily SPY prices from Yahoo Finance. Only used to refresh the cached CSV."""
    import yfinance as yf

    raw = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False)
    # yfinance sometimes returns a two level column index. Flatten it if so.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[PRICE_COLUMNS].copy()
    df.index.name = "Date"
    return clean_prices(df)


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by date, drop duplicate and empty rows, and force a proper date index."""
    out = df.copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]        # one row per day
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out.index = pd.to_datetime(out.index)
    return out[PRICE_COLUMNS].astype(float)


def load_prices(path: str | Path) -> pd.DataFrame:
    """Read a already clean price CSV that has a Date column."""
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df.sort_index()


def load_ohlc(path: str | Path) -> pd.DataFrame:
    """
    Read a daily OHLC CSV from whatever source and return clean OHLCV columns.

    This copes with the three files you actually run into: the tidy
    Date,Open,High,Low,Close,Volume from Stooq, the Yahoo version that adds an Adj
    Close column, and the messy yfinance dump where the header row says Price and the
    next two rows are the ticker and a blank date. Column names are matched without
    caring about case, and Volume is optional.
    """
    raw = pd.read_csv(path)

    # Spot the messy yfinance dump (first header cell is "Price") and reread it,
    # skipping the two junk rows underneath the header.
    first = str(raw.columns[0]).strip().lower()
    if first in ("price", "unnamed: 0", "") and any(str(c).strip().lower() == "close" for c in raw.columns):
        raw = pd.read_csv(path, skiprows=[1, 2])
        raw = raw.rename(columns={raw.columns[0]: "Date"})

    # Find the date column, parse it, and make it the index.
    raw.columns = [str(c).strip() for c in raw.columns]
    date_col = next((c for c in raw.columns if c.lower() in ("date", "datetime", "time")), raw.columns[0])
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).set_index(date_col).sort_index()
    raw.index.name = "Date"

    # Match the OHLC columns by name, ignoring case. Prefer the plain Close over Adj
    # Close so the four prices stay consistent for the Garman-Klass estimator.
    lower = {c.lower(): c for c in raw.columns}
    pick = lambda *names: next((lower[n] for n in names if n in lower), None)
    o, h, l, c = pick("open"), pick("high"), pick("low"), pick("close", "adj close", "adjclose")
    v = pick("volume", "vol")
    if None in (o, h, l, c):
        raise ValueError(f"could not find OHLC columns in {list(raw.columns)}")

    out = pd.DataFrame({"Open": raw[o], "High": raw[h], "Low": raw[l], "Close": raw[c]})
    out["Volume"] = raw[v] if v is not None else np.nan
    out = out.apply(pd.to_numeric, errors="coerce")

    # Drop duplicate days, rows missing a price, and any non positive price (bad ticks).
    out = out[~out.index.duplicated(keep="first")].dropna(subset=["Open", "High", "Low", "Close"])
    out = out[(out[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    return out.astype(float)


def validate_prices(df: pd.DataFrame) -> dict:
    """A few quick sanity checks, returned as a dict so a notebook can print them."""
    report = {
        "n_rows": len(df),
        "first_date": str(df.index.min().date()) if len(df) else None,
        "last_date": str(df.index.max().date()) if len(df) else None,
        "missing_values": int(df.isna().sum().sum()),
        "non_positive_prices": int((df[["Open", "High", "Low", "Close"]] <= 0).sum().sum()),
        "high_lt_low_count": int((df["High"] < df["Low"]).sum()),   # high should never be below low
        "duplicated_dates": int(df.index.duplicated().sum()),
    }
    if len(df) > 1:
        gaps = df.index.to_series().diff().dt.days
        report["max_gap_days"] = int(gaps.max())  # biggest calendar gap, weekends push this to 3 or so
    return report


def compute_log_returns(close: pd.Series) -> pd.Series:
    """Daily log returns of the close. Log returns add up over time, which is handy."""
    return np.log(close / close.shift(1)).rename("log_return")
