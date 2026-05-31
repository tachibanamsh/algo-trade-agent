"""
Compute today's trading signal from live market data.

Called once per day after US market close (JST morning, before JP open).

Flow
────
  1. Download last (L + buffer) days of US and JP CC returns via yfinance.
  2. Re-use the same subspace-regularised PCA algorithm as the backtest.
  3. Return {jp_yf_ticker: weight} dict ready to pass to MoomooExecutor.

The prior correlation matrix C0 is recomputed from the fixed in-sample
period each call — it is deterministic and fast (~seconds).
"""

from __future__ import annotations

import logging
import warnings

import pandas as pd

from strategy.config import (
    CFULL_END,
    CFULL_START,
    JP_TICKERS,
    K,
    LAMBDA,
    Q,
    US_TICKERS,
    WINDOW_L,
)
from strategy.data import align, load_jp, load_us
from strategy.portfolio import compute_weights
from strategy.signal import build_prior_correlation, build_prior_subspace, compute_signals

logger = logging.getLogger(__name__)


def compute_today_weights(
    L: int = WINDOW_L,
    lambda_: float = LAMBDA,
    k: int = K,
    q: float = Q,
    cfull_start: str = CFULL_START,
    cfull_end: str = CFULL_END,
) -> dict[str, float]:
    """
    Compute portfolio weights for today's Japan trading session.

    Returns
    ───────
    dict mapping yfinance ticker (e.g. '1617.T') to weight float.
    Positive = long, negative = short.
    Empty dict if signal cannot be computed (e.g. insufficient history).
    """
    # Need L window days + today's US return.  Add 40 calendar-day buffer for
    # holidays, weekends, and missing data.
    fetch_days = int(L * 1.6) + 40

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    fetch_start = (pd.Timestamp.today() - pd.Timedelta(days=fetch_days)).strftime("%Y-%m-%d")

    logger.info("Downloading recent price data (%s → %s) …", fetch_start, today)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        us_cc_recent          = load_us(fetch_start, today)
        jp_cc_recent, _       = load_jp(fetch_start, today)  # OC not needed here

    # Also load C0 estimation data
    logger.info("Downloading C0 estimation data (%s → %s) …", cfull_start, cfull_end)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        us_cc_full       = load_us(cfull_start, cfull_end)
        jp_cc_full, _    = load_jp(cfull_start, cfull_end)

    # Align recent data on common trading days
    dummy_oc = jp_cc_recent.copy()   # placeholder; align() needs jp_oc arg
    us_cc_recent, jp_cc_recent, _ = align(us_cc_recent, jp_cc_recent, dummy_oc)

    if len(us_cc_recent) < L + 1:
        logger.error("Not enough data (%d rows, need %d+1)", len(us_cc_recent), L)
        return {}

    # Build prior
    V0 = build_prior_subspace()
    cc_full = pd.concat([us_cc_full[US_TICKERS], jp_cc_full[JP_TICKERS]], axis=1).dropna()
    C0 = build_prior_correlation(V0, cc_full)

    # Compute signal for all recent dates; we only care about the last one
    signals_df = compute_signals(us_cc_recent, jp_cc_recent, C0, L=L, lambda_=lambda_, k=k)

    last_signal = signals_df.dropna(how="any").tail(1)
    if last_signal.empty:
        logger.error("Signal is NaN for the most recent date — cannot trade today")
        return {}

    signal_date = last_signal.index[0]
    logger.info("Signal computed for %s", signal_date.date())

    weights_df = compute_weights(last_signal, q=q)
    weights_row = weights_df.loc[signal_date]

    return weights_row[weights_row != 0].to_dict()
