"""
Portfolio construction (Section 2.2).

At each date t the signal ŝ_J,t+1 is used to build an equal-weight
long/short portfolio:
  Long  (weight +1/|L_{t+1}|) : top-q fraction of signal
  Short (weight -1/|S_{t+1}|) : bottom-q fraction of signal

The weights sum to zero and |Σ|w_j|| = 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import JP_TICKERS, Q


def compute_weights(signals: pd.DataFrame, q: float = Q) -> pd.DataFrame:
    """
    Convert a signal DataFrame into equal-weight long/short portfolio weights.

    Parameters
    ----------
    signals : DataFrame (dates × JP_TICKERS), NaN rows are skipped.
    q       : quantile cutoff (default 0.3 → top/bottom 30 %).

    Returns
    -------
    weights : same shape as signals, values in {+1/n_q, 0, -1/n_q}.
    """
    weights = pd.DataFrame(0.0, index=signals.index, columns=JP_TICKERS)

    for t, row in signals.iterrows():
        valid = row.dropna()
        if valid.empty:
            continue

        n   = len(valid)
        n_q = max(1, int(np.floor(q * n)))  # e.g. floor(0.3 × 17) = 5

        sorted_idx = valid.sort_values().index
        short_set  = sorted_idx[:n_q]   # bottom q  → short
        long_set   = sorted_idx[-n_q:]  # top q     → long

        weights.loc[t, long_set]  =  1.0 / n_q
        weights.loc[t, short_set] = -1.0 / n_q

    return weights
