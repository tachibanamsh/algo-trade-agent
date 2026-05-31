# --- Universe (Section 4.1) ---

US_TICKERS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
JP_TICKERS = [f"{i}.T" for i in range(1617, 1634)]  # 1617.T ... 1633.T (17 sectors)

# Cyclical / Defensive labels (Section 4.1)
US_CYCLICAL  = {"XLB", "XLE", "XLF", "XLRE"}
US_DEFENSIVE = {"XLK", "XLP", "XLU", "XLV"}
JP_CYCLICAL  = {"1618.T", "1625.T", "1629.T", "1631.T"}
JP_DEFENSIVE = {"1617.T", "1621.T", "1627.T", "1630.T"}

# --- Model parameters (Section 4.3) ---

WINDOW_L = 60    # rolling estimation window (business days)
LAMBDA   = 0.9   # regularization weight  (eq. 13)
K        = 3     # number of top eigenvectors kept
Q        = 0.3   # long / short quantile (top/bottom 30 %)

# Prior correlation C0 is estimated once on this in-sample period.
# Must start after XLC (2018-06) and XLRE (2015-10) were listed.
CFULL_START = "2019-01-01"
CFULL_END   = "2023-01-01"
