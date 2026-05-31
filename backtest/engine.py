"""
Backtesting engine.

Execution logic
───────────────
  weights[t]  ──signal computed at US close on day t──▶  JP OC[t+1]

In DataFrame terms: shift weights forward by 1 row so that weights[t]
is paired with jp_oc[t+1].

Metrics (Section 4.2)
─────────────────────
  AR   = annualised return  (×252)
  RISK = annualised volatility
  R/R  = AR / RISK
  MDD  = maximum drawdown
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.config import JP_TICKERS


def compute_strategy_returns(
    weights: pd.DataFrame,
    jp_oc: pd.DataFrame,
) -> pd.Series:
    """
    Compute daily strategy returns.

    weights[t] (signal date) → applied to jp_oc[t+1] (execution date).

    Parameters
    ----------
    weights : (dates × JP_TICKERS), output of portfolio.compute_weights()
    jp_oc   : (dates × JP_TICKERS), JP open-to-close returns, same index

    Returns
    -------
    pd.Series of daily returns indexed by execution date (t+1).
    """
    # shift weights forward: weights_shifted.loc[t+1] = weights.loc[t]
    weights_shifted = weights.reindex(jp_oc.index).shift(1)

    daily = (weights_shifted * jp_oc[JP_TICKERS]).sum(axis=1)
    return daily.loc[daily.index[1]:]   # drop first date (no signal yet)


def compute_metrics(returns: pd.Series) -> dict[str, float]:
    """
    Compute AR, RISK, R/R, MDD as defined in Section 4.2.
    Returns values as plain floats (percentages where noted).
    """
    r = returns.dropna()

    ann_ret = r.mean() * 252
    ann_vol = r.std()  * np.sqrt(252)
    rr      = ann_ret / ann_vol if ann_vol > 1e-12 else float("nan")

    cum    = (1.0 + r).cumprod()
    peak   = cum.cummax()
    mdd    = ((cum - peak) / peak).min()

    return {
        "AR (%)":   ann_ret * 100,
        "RISK (%)": ann_vol * 100,
        "R/R":      rr,
        "MDD (%)":  mdd    * 100,
    }


def run_backtest(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    jp_oc: pd.DataFrame,
    cfull_us_cc: pd.DataFrame,
    cfull_jp_cc: pd.DataFrame,
    L: int,
    lambda_: float,
    k: int,
    q: float,
) -> tuple[pd.Series, dict[str, float]]:
    """
    End-to-end backtest: signal → weights → returns → metrics.

    Parameters
    ----------
    us_cc / jp_cc / jp_oc : aligned return DataFrames (same index)
    cfull_us_cc / cfull_jp_cc : in-sample data for estimating C0
    L, lambda_, k, q : strategy hyperparameters

    Returns
    -------
    (daily_returns, metrics_dict)
    """
    from strategy.signal import build_prior_subspace, build_prior_correlation, compute_signals
    from strategy.portfolio import compute_weights

    # Build fixed prior
    V0 = build_prior_subspace()
    cc_full = pd.concat([cfull_us_cc, cfull_jp_cc], axis=1)
    C0 = build_prior_correlation(V0, cc_full)

    print(f"C0 built. Shape: {C0.shape}")

    # Rolling signals
    signals = compute_signals(us_cc, jp_cc, C0, L=L, lambda_=lambda_, k=k)
    n_signal = signals.notna().all(axis=1).sum()
    print(f"Signals computed: {n_signal} non-NaN dates out of {len(signals)}")

    # Portfolio weights
    weights = compute_weights(signals, q=q)

    # Strategy returns
    daily = compute_strategy_returns(weights, jp_oc)

    metrics = compute_metrics(daily)
    return daily, metrics
