"""
Entry point: run the PCA-SUB lead-lag backtest.

Usage
─────
  uv run python main.py                      # default: 2015-01-01 → today
  uv run python main.py --start 2018-01-01 --end 2024-12-31
  uv run python main.py --plot               # show cumulative-return chart
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from backtest.engine import run_backtest
from strategy.config import (
    CFULL_END,
    CFULL_START,
    K,
    LAMBDA,
    Q,
    WINDOW_L,
)
from strategy.data import align, load_jp, load_us


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lead-lag PCA-SUB strategy backtest")
    p.add_argument("--start",   default="2015-01-01", help="Backtest start date (YYYY-MM-DD)")
    p.add_argument("--end",     default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="Backtest end date")
    p.add_argument("--window",  type=int,   default=WINDOW_L, help=f"Estimation window L (default {WINDOW_L})")
    p.add_argument("--lambda_", type=float, default=LAMBDA,   help=f"Regularisation λ (default {LAMBDA})")
    p.add_argument("--k",       type=int,   default=K,        help=f"Top eigenvectors K (default {K})")
    p.add_argument("--q",       type=float, default=Q,        help=f"Long/short quantile q (default {Q})")
    p.add_argument("--plot",    action="store_true",           help="Show cumulative return plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load & align data ───────────────────────────────────────────────────
    print(f"Downloading data ({CFULL_START} → {args.end}) …")
    data_start = CFULL_START  # must include C0 estimation period

    us_cc_all          = load_us(data_start, args.end)
    jp_cc_all, jp_oc_all = load_jp(data_start, args.end)

    us_cc_all, jp_cc_all, jp_oc_all = align(us_cc_all, jp_cc_all, jp_oc_all)

    # In-sample window for C0
    cfull_us = us_cc_all.loc[CFULL_START:CFULL_END]
    cfull_jp = jp_cc_all.loc[CFULL_START:CFULL_END]

    # Backtest window
    bt_us  = us_cc_all.loc[args.start:args.end]
    bt_jcc = jp_cc_all.loc[args.start:args.end]
    bt_joc = jp_oc_all.loc[args.start:args.end]

    print(f"Backtest period : {bt_us.index[0].date()} → {bt_us.index[-1].date()}  ({len(bt_us)} days)")
    print(f"C0 estimation   : {cfull_us.index[0].date()} → {cfull_us.index[-1].date()}  ({len(cfull_us)} days)")
    print()

    # ── Run backtest ────────────────────────────────────────────────────────
    daily, metrics = run_backtest(
        us_cc=bt_us,
        jp_cc=bt_jcc,
        jp_oc=bt_joc,
        cfull_us_cc=cfull_us,
        cfull_jp_cc=cfull_jp,
        L=args.window,
        lambda_=args.lambda_,
        k=args.k,
        q=args.q,
    )

    # ── Print results ───────────────────────────────────────────────────────
    print("\n── PCA-SUB Lead-Lag Strategy Performance ──")
    print(f"  AR      : {metrics['AR (%)']:+.2f} %")
    print(f"  RISK    : {metrics['RISK (%)']:.2f} %")
    print(f"  R/R     : {metrics['R/R']:.3f}")
    print(f"  MDD     : {metrics['MDD (%)']:.2f} %")
    print()

    # ── Optional plot ───────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping plot (pip install matplotlib)")
            sys.exit(0)

        cum = (1 + daily).cumprod()
        fig, ax = plt.subplots(figsize=(12, 5))
        cum.plot(ax=ax, label="PCA-SUB", color="steelblue")
        ax.set_title("Lead-Lag PCA-SUB: cumulative return")
        ax.set_ylabel("Cumulative return (gross)")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
