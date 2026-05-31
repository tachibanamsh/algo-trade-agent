"""
Data loading utilities.

Timeline (why the date shift matters):
  JP market closes 15:30 JST = 06:30 ET  (same calendar day)
  US market closes 16:00 ET  = next day 06:00 JST

So US CC return on calendar day t is the *signal* that predicts
JP OC return on calendar day t+1 (next Japan trading day).

Alignment strategy used here:
  1. Download US and JP prices on their own calendars.
  2. Find the intersection of US and JP trading days (common_dates).
  3. US CC[t] on common_dates → signal → JP OC[t+1] on common_dates.
  This matches the academic convention used in the paper.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from .config import JP_TICKERS, US_TICKERS


def _download(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Return {'open': df, 'close': df} with tickers as columns."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance returns (Price, Ticker) multi-index for multiple tickers
        opens  = raw["Open"][tickers]
        closes = raw["Close"][tickers]
    else:
        # single ticker — normalise to same shape
        opens  = raw[["Open"]].rename(columns={"Open":  tickers[0]})
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]})

    return {"open": opens, "close": closes}


def load_us(start: str, end: str) -> pd.DataFrame:
    """US sector ETF close-to-close returns, indexed by US trading dates."""
    prices = _download(US_TICKERS, start=start, end=end)
    return prices["close"].pct_change()


def load_jp(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Japan sector ETF returns, indexed by JP trading dates.
    Returns (cc, oc): close-to-close, open-to-close.
    """
    prices = _download(JP_TICKERS, start=start, end=end)
    cc = prices["close"].pct_change()
    oc = prices["close"] / prices["open"] - 1
    return cc, oc


def align(us_cc: pd.DataFrame, jp_cc: pd.DataFrame, jp_oc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Restrict all three DataFrames to the intersection of US and JP trading days,
    dropping rows where any ticker has NaN.
    """
    common = us_cc.index.intersection(jp_cc.index).intersection(jp_oc.index)
    us  = us_cc.loc[common].dropna()
    jcc = jp_cc.loc[common].dropna()
    joc = jp_oc.loc[common].dropna()

    # Re-align to the intersection after dropping NaN rows
    shared = us.index.intersection(jcc.index).intersection(joc.index)
    return us.loc[shared], jcc.loc[shared], joc.loc[shared]
