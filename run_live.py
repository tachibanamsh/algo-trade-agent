"""
Daily live trading entry point.

Run this script twice per trading day:

  ① Before Japan market open (08:55 JST):
       uv run python run_live.py open

  ② After Japan market close (15:35 JST):
       uv run python run_live.py close

Environment variables
─────────────────────
  MOOMOO_OPEND_HOST      OpenD host        (default: 127.0.0.1)
  MOOMOO_OPEND_PORT      OpenD port        (default: 11111)
  MOOMOO_SECURITY_FIRM   e.g. FUTUSG / FUTUSECURITIES
  MOOMOO_TRADE_PASSWORD  plain-text password (required for REAL)
  MOOMOO_TRD_ENV         SIMULATE (default) or REAL
  MOOMOO_ALLOW_SHORT     1 to enable short selling, 0 = long-only (default)
  MOOMOO_NOTIONAL_JPY    JPY amount to deploy per side  (default: 80% of cash)

Dry-run mode
────────────
  Pass --dry-run to print orders without sending them to OpenD.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv()  # .env を自動ロード（なければ何もしない）

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_open(dry_run: bool) -> None:
    """Compute today's signal and place opening orders."""
    from execution.live_signal import compute_today_weights
    from execution.moomoo_executor import MoomooExecutor

    logger.info("=== OPEN: computing signal ===")
    weights = compute_today_weights()

    if not weights:
        logger.error("No signal — aborting")
        sys.exit(1)

    long_legs  = {t: w for t, w in weights.items() if w > 0}
    short_legs = {t: w for t, w in weights.items() if w < 0}
    logger.info("Long  (%d): %s", len(long_legs),  list(long_legs.keys()))
    logger.info("Short (%d): %s", len(short_legs), list(short_legs.keys()))

    if dry_run:
        logger.info("[DRY RUN] Would place %d orders — skipping OpenD", len(weights))
        return

    with MoomooExecutor() as ex:
        # Determine notional: env override or 80 % of available cash
        env_notional = os.environ.get("MOOMOO_NOTIONAL_JPY")
        if env_notional:
            notional = float(env_notional)
        else:
            cash     = ex.get_cash_jpy()
            notional = cash * 0.8
            logger.info("Cash: ¥{:,.0f}  →  Notional: ¥{:,.0f}".format(cash, notional))

        if notional <= 0:
            logger.error("Notional is ¥0 — check account cash")
            sys.exit(1)

        results = ex.execute_weights(weights, notional)
        logger.info("=== OPEN done: %d order(s) placed ===", len(results))
        for r in results:
            logger.info("  order_id=%-20s  code=%-10s  qty=%s  side=%s",
                        r.get("order_id"), r.get("code"), r.get("qty"), r.get("trd_side"))


def cmd_close(dry_run: bool) -> None:
    """Close all open JP positions."""
    from execution.moomoo_executor import MoomooExecutor

    logger.info("=== CLOSE: flattening all JP positions ===")

    if dry_run:
        logger.info("[DRY RUN] Would close all positions — skipping OpenD")
        return

    with MoomooExecutor() as ex:
        results = ex.close_all_positions()
        logger.info("=== CLOSE done: %d position(s) closed ===", len(results))
        for r in results:
            logger.info("  order_id=%-20s  code=%-10s  qty=%s  side=%s",
                        r.get("order_id"), r.get("code"), r.get("qty"), r.get("trd_side"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Lead-lag live trader")
    parser.add_argument("command", choices=["open", "close"], help="open: place orders; close: flatten positions")
    parser.add_argument("--dry-run", action="store_true", help="Print planned orders without sending to OpenD")
    args = parser.parse_args()

    if args.command == "open":
        cmd_open(dry_run=args.dry_run)
    else:
        cmd_close(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
