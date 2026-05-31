"""
moomoo-api execution wrapper.

Ticker format conversion
────────────────────────
  yfinance : 1617.T  →  moomoo : JP.1617
  The ".T" suffix is dropped and "JP." prefix is added.

Order flow (daily cycle)
────────────────────────
  08:55 JST  run_live.py open  → execute_open()
               - compute today's signal weights
               - get prices for each JP ETF
               - place MARKET orders (long side BUY, short side SELL)

  15:35 JST  run_live.py close → execute_close()
               - query open positions
               - place opposite MARKET orders to flatten all positions

Short selling note
──────────────────
  Short positions require margin/borrowing permission on the moomoo account.
  Set ALLOW_SHORT=False in env to skip short legs (long-only mode).
"""

from __future__ import annotations

import logging
import math
import os

from moomoo import OpenQuoteContext, OpenSecTradeContext, RET_OK, SecurityFirm, SubType, TrdMarket

logger = logging.getLogger(__name__)


# ── Ticker conversion ────────────────────────────────────────────────────────

def to_moomoo_code(yf_ticker: str) -> str:
    """Convert yfinance ticker to moomoo code. '1617.T' → 'JP.1617'."""
    return "JP." + yf_ticker.replace(".T", "")


def to_yf_ticker(moomoo_code: str) -> str:
    """Convert moomoo code to yfinance ticker. 'JP.1617' → '1617.T'."""
    return moomoo_code.replace("JP.", "") + ".T"


# ── Executor ─────────────────────────────────────────────────────────────────

class MoomooExecutor:
    """
    Thin wrapper around moomoo-api for this strategy's execution needs.

    Parameters read from environment variables (override via __init__ kwargs):
      MOOMOO_OPEND_HOST        default 127.0.0.1
      MOOMOO_OPEND_PORT        default 11111
      MOOMOO_SECURITY_FIRM     e.g. FUTUSG / FUTUSECURITIES
      MOOMOO_TRADE_PASSWORD    plain-text trade password (for REAL accounts)
      MOOMOO_TRD_ENV           SIMULATE (default) or REAL
      MOOMOO_ALLOW_SHORT       1 to allow short selling, 0 for long-only (default 0)
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        security_firm: str | None = None,
        trade_password: str | None = None,
        trd_env: str | None = None,
        allow_short: bool | None = None,
    ):
        self.host          = host          or os.environ.get("MOOMOO_OPEND_HOST", "127.0.0.1")
        self.port          = port          or int(os.environ.get("MOOMOO_OPEND_PORT", "11111"))
        self.security_firm = security_firm or os.environ.get("MOOMOO_SECURITY_FIRM")
        self.trade_password = trade_password or os.environ.get("MOOMOO_TRADE_PASSWORD")
        self.trd_env       = trd_env       or os.environ.get("MOOMOO_TRD_ENV", "SIMULATE")
        allow_env = os.environ.get("MOOMOO_ALLOW_SHORT", "0")
        self.allow_short   = allow_short if allow_short is not None else (allow_env == "1")
        # Manual acc_id override (set MOOMOO_ACC_ID in .env to skip auto-detection)
        env_acc_id = os.environ.get("MOOMOO_ACC_ID", "")
        self._manual_acc_id: int | None = int(env_acc_id) if env_acc_id.isdigit() else None

        self._quote_ctx: OpenQuoteContext | None = None
        self._trade_ctx: OpenSecTradeContext | None = None
        self._acc_id: int = 0

    # ── Connection lifecycle ─────────────────────────────────────────────────

    def connect(self) -> None:
        """Open quote and trade contexts, unlock trade if password is set."""
        self._quote_ctx = OpenQuoteContext(host=self.host, port=self.port)

        kwargs: dict = {"host": self.host, "port": self.port}
        if self.security_firm:
            firm_enum = getattr(SecurityFirm, self.security_firm, None)
            if firm_enum:
                kwargs["security_firm"] = firm_enum
        self._trade_ctx = OpenSecTradeContext(**kwargs)

        if self.trade_password and self.trd_env == "REAL":
            ret, data = self._trade_ctx.unlock_trade(
                password=self.trade_password, is_unlock=True
            )
            if ret != RET_OK:
                raise RuntimeError(f"unlock_trade failed: {data}")
            logger.info("Trade unlocked for REAL account")

        self._acc_id = self._find_account_id()
        logger.info("Connected. acc_id=%s  trd_env=%s", self._acc_id, self.trd_env)

    def close(self) -> None:
        """Close both contexts."""
        if self._trade_ctx:
            self._trade_ctx.close()
            self._trade_ctx = None
        if self._quote_ctx:
            self._quote_ctx.close()
            self._quote_ctx = None

    def __enter__(self) -> "MoomooExecutor":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Account ──────────────────────────────────────────────────────────────

    def _find_account_id(self) -> int:
        """
        Return the acc_id to use, in order of priority:
          1. MOOMOO_ACC_ID env var (manual override)
          2. First account with trd_env + JP market_auth
          3. First account with trd_env (any market)
          4. acc_id=0  (moomoo auto-selects; last resort)
        """
        if self._manual_acc_id is not None:
            logger.info("Using manually specified acc_id=%d", self._manual_acc_id)
            return self._manual_acc_id

        ret, data = self._trade_ctx.get_acc_list()
        if ret != RET_OK:
            raise RuntimeError(f"get_acc_list failed: {data}")

        accounts = data.to_dict("records") if not data.empty else []

        # Debug: show all accounts
        for row in accounts:
            logger.info(
                "Account: acc_id=%s  trd_env=%s  trdmarket_auth=%s  security_firm=%s",
                row.get("acc_id"), row.get("trd_env"),
                row.get("trdmarket_auth") or row.get("market_auth"),
                row.get("security_firm"),
            )

        # Priority 1: matching env + JP market
        for row in accounts:
            if row.get("trd_env") != self.trd_env:
                continue
            market_auth = row.get("trdmarket_auth") or row.get("market_auth") or []
            if "JP" in market_auth:
                return int(row["acc_id"])

        # Priority 2: matching env only
        for row in accounts:
            if row.get("trd_env") == self.trd_env:
                acc_id = int(row["acc_id"])
                logger.warning(
                    "No JP-specific account found. Using first %s account: acc_id=%d",
                    self.trd_env, acc_id,
                )
                return acc_id

        # Priority 3: no matching account — use 0 (moomoo auto-select)
        logger.warning(
            "get_acc_list returned 0 accounts for trd_env=%s. "
            "Falling back to acc_id=0. "
            "Tip: set MOOMOO_ACC_ID in .env if you know your account ID.",
            self.trd_env,
        )
        return 0

    def get_cash_jpy(self) -> float:
        """Return available cash in JPY for the active account."""
        ret, data = self._trade_ctx.accinfo_query(
            trd_env=self.trd_env,
            acc_id=self._acc_id,
            currency="JPY",
        )
        if ret != RET_OK:
            raise RuntimeError(f"accinfo_query failed: {data}")
        records = data.to_dict("records")
        return float(records[0].get("cash", 0)) if records else 0.0

    # ── Prices ───────────────────────────────────────────────────────────────

    def get_prices(self, yf_tickers: list[str]) -> dict[str, float]:
        """
        Return {yf_ticker: last_price} for a list of JP ETFs.
        Uses the previous close if real-time price is unavailable.
        """
        codes = [to_moomoo_code(t) for t in yf_tickers]

        ret, _ = self._quote_ctx.subscribe(codes, [SubType.QUOTE], subscribe_push=False)
        if ret != RET_OK:
            logger.warning("subscribe failed, continuing without real-time prices")

        ret, data = self._quote_ctx.get_stock_quote(codes)
        if ret != RET_OK:
            raise RuntimeError(f"get_stock_quote failed: {data}")

        prices: dict[str, float] = {}
        for row in data.to_dict("records"):
            yf_t = to_yf_ticker(row["code"])
            # prefer last_price; fall back to prev_close_price
            price = row.get("last_price") or row.get("prev_close_price") or 0.0
            prices[yf_t] = float(price)
        return prices

    # ── Orders ───────────────────────────────────────────────────────────────

    def _place_market(self, yf_ticker: str, qty: int, side: str) -> dict:
        """
        Place a MARKET order.
        side : 'BUY' or 'SELL'
        qty  : must be > 0
        """
        code = to_moomoo_code(yf_ticker)
        ret, data = self._trade_ctx.place_order(
            price=0.0,          # ignored for MARKET
            qty=qty,
            code=code,
            trd_side=side,
            order_type="MARKET",
            time_in_force="DAY",
            trd_env=self.trd_env,
            acc_id=self._acc_id,
            remark="algo-trade-agent",
        )
        if ret != RET_OK:
            raise RuntimeError(f"place_order {side} {code} ×{qty} failed: {data}")

        result = data.to_dict("records")[0] if not data.empty else {}
        logger.info("Placed %s %s ×%d  order_id=%s", side, code, qty, result.get("order_id"))
        return result

    def execute_weights(
        self,
        weights: dict[str, float],
        total_notional_jpy: float,
    ) -> list[dict]:
        """
        Translate portfolio weights into market orders.

        Parameters
        ──────────
        weights : {yf_ticker: weight}
                  Positive = long, negative = short.
                  Typical sum of positives ≈ 1, sum of negatives ≈ −1 (eq-weight L/S).
        total_notional_jpy : JPY amount to deploy on each side (long side + short side).
                             e.g. pass available cash; the actual spend is half long, half short.

        Returns list of order result dicts.
        """
        long_tickers  = {t: w for t, w in weights.items() if w > 0}
        short_tickers = {t: w for t, w in weights.items() if w < 0}

        if not long_tickers:
            logger.warning("No long positions in weights — nothing to trade")
            return []

        prices = self.get_prices(list(weights.keys()))

        # Each side gets half the notional; weight is equal so divide evenly
        long_notional_each  = total_notional_jpy / 2 / max(len(long_tickers), 1)
        short_notional_each = total_notional_jpy / 2 / max(len(short_tickers), 1)

        results: list[dict] = []

        for ticker, w in long_tickers.items():
            price = prices.get(ticker, 0.0)
            if price <= 0:
                logger.warning("No price for %s — skipping long", ticker)
                continue
            qty = math.floor(long_notional_each / price)
            if qty <= 0:
                logger.warning("Computed qty=0 for %s (price=%.2f, notional=%.0f)", ticker, price, long_notional_each)
                continue
            results.append(self._place_market(ticker, qty, "BUY"))

        if self.allow_short:
            for ticker, w in short_tickers.items():
                price = prices.get(ticker, 0.0)
                if price <= 0:
                    logger.warning("No price for %s — skipping short", ticker)
                    continue
                qty = math.floor(short_notional_each / price)
                if qty <= 0:
                    continue
                results.append(self._place_market(ticker, qty, "SELL"))
        else:
            logger.info("allow_short=False → skipping %d short positions", len(short_tickers))

        return results

    # ── Close all positions ──────────────────────────────────────────────────

    def close_all_positions(self) -> list[dict]:
        """
        Flatten all open JP positions by placing opposite MARKET orders.
        Call this at end-of-day (after JP market close).
        """
        ret, data = self._trade_ctx.position_list_query(
            position_market=TrdMarket.JP,
            trd_env=self.trd_env,
            acc_id=self._acc_id,
        )
        if ret != RET_OK:
            raise RuntimeError(f"position_list_query failed: {data}")

        if data is None or data.empty:
            logger.info("No open positions to close")
            return []

        results: list[dict] = []
        for row in data.to_dict("records"):
            code  = row["code"]               # e.g. JP.1617
            qty   = int(row.get("qty", 0))
            side  = row.get("position_side", "LONG")

            if qty <= 0:
                continue

            yf_t       = to_yf_ticker(code)
            close_side = "SELL" if side == "LONG" else "BUY"
            logger.info("Closing %s %s ×%d", close_side, code, qty)
            results.append(self._place_market(yf_t, qty, close_side))

        return results
