"""
Lead-lag signal via Subspace-Regularized PCA (Section 3 of the paper).

Algorithm per date t
────────────────────
1. Estimation window W_t = {t-L, …, t-1}  (L days of US+JP CC returns)
2. Standardise: z_{i,τ} = (r_{i,τ} - μ_{i,t}) / σ_{i,t}
3. Rolling correlation: C_t = Z_t^T Z_t / L
4. Regularise:  C_reg = (1-λ)·C_t + λ·C0          (eq. 13, λ=0.9)
5. Eigen-decompose C_reg → top-K eigenvectors V_K ∈ R^{N×K}
6. Split: V_U (US block, N_U×K), V_J (JP block, N_J×K)
7. Standardise US return at t: z_U,t
8. Factor score: f_t = V_U^T · z_U,t              (eq. 18)
9. Signal:       ŝ_J,t+1 = V_J · f_t              (eq. 19)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from .config import (
    JP_CYCLICAL,
    JP_DEFENSIVE,
    JP_TICKERS,
    K,
    LAMBDA,
    US_CYCLICAL,
    US_DEFENSIVE,
    US_TICKERS,
    WINDOW_L,
)

N_U = len(US_TICKERS)
N_J = len(JP_TICKERS)
N   = N_U + N_J
ALL_TICKERS = US_TICKERS + JP_TICKERS


# ── Prior subspace ──────────────────────────────────────────────────────────

def build_prior_subspace() -> np.ndarray:
    """
    Build V0 ∈ R^{N×3} with three mutually orthogonal prior directions
    (Section 3.1):
      v1 – global (equal weight)
      v2 – country spread (US+, JP-)
      v3 – cyclical vs defensive
    """
    # v1: global
    v1 = np.ones(N) / np.sqrt(N)

    # v2: country spread, orthogonalised against v1
    raw2 = np.concatenate([np.ones(N_U), -np.ones(N_J)])
    raw2 -= raw2 @ v1 * v1
    v2 = raw2 / np.linalg.norm(raw2)

    # v3: cyclical (+1) / defensive (−1), orthogonalised against v1, v2
    raw3 = np.zeros(N)
    for i, t in enumerate(US_TICKERS):
        if t in US_CYCLICAL:
            raw3[i] = 1.0
        elif t in US_DEFENSIVE:
            raw3[i] = -1.0
    for j, t in enumerate(JP_TICKERS):
        if t in JP_CYCLICAL:
            raw3[N_U + j] = 1.0
        elif t in JP_DEFENSIVE:
            raw3[N_U + j] = -1.0
    raw3 -= raw3 @ v1 * v1
    raw3 -= raw3 @ v2 * v2
    norm3 = np.linalg.norm(raw3)
    v3 = raw3 / norm3 if norm3 > 1e-12 else raw3

    return np.column_stack([v1, v2, v3])  # (N, 3)


# ── Prior correlation C0 ────────────────────────────────────────────────────

def build_prior_correlation(V0: np.ndarray, cc_full: pd.DataFrame) -> np.ndarray:
    """
    Build the fixed prior correlation matrix C0 (eqs. 8–12) from a long
    in-sample period.

    Steps:
      Standardise cc_full → Z_full
      C_full  = Z_full^T Z_full / T_full
      D0      = diag(V0^T C_full V0)
      C0_raw  = V0 · D0 · V0^T
      C0      = Δ^{-1/2} C0_raw Δ^{-1/2}  (normalise to correlation)
    """
    data = cc_full[ALL_TICKERS].dropna()
    if data.empty:
        raise ValueError(
            "No rows remain after dropna() in the C0 estimation period. "
            "Likely some tickers (XLRE listed 2015-10, XLC listed 2018-06) "
            "did not exist yet. Set CFULL_START to 2019-01-01 or later."
        )
    mu    = data.mean()
    sigma = data.std(ddof=0).replace(0.0, np.nan)
    Z     = ((data - mu) / sigma).dropna()

    C_full = (Z.values.T @ Z.values) / len(Z)

    D0_diag = np.diag(V0.T @ C_full @ V0)
    C0_raw  = V0 @ np.diag(D0_diag) @ V0.T

    diag_vals  = np.maximum(np.diag(C0_raw), 1e-12)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(diag_vals))
    C0 = D_inv_sqrt @ C0_raw @ D_inv_sqrt
    np.fill_diagonal(C0, 1.0)

    return C0


# ── Rolling signal computation ───────────────────────────────────────────────

def compute_signals(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    C0: np.ndarray,
    L: int = WINDOW_L,
    lambda_: float = LAMBDA,
    k: int = K,
) -> pd.DataFrame:
    """
    Compute the lead-lag signal for every date in the shared index of
    us_cc and jp_cc.

    Parameters
    ----------
    us_cc : US close-to-close returns, columns = US_TICKERS
    jp_cc : JP close-to-close returns, columns = JP_TICKERS
    C0    : prior correlation matrix (N × N), from build_prior_correlation()
    L     : estimation window length
    lambda_ : regularisation weight
    k     : number of top eigenvectors

    Returns
    -------
    DataFrame indexed by the same dates, columns = JP_TICKERS.
    signals.loc[t] is the signal used to trade JP OC on day t+1.
    NaN rows mean insufficient data.
    """
    # Combine into joint N-column return matrix on common dates
    combined = pd.concat(
        [us_cc[US_TICKERS], jp_cc[JP_TICKERS]], axis=1
    ).dropna(how="any")

    dates   = combined.index
    n_dates = len(dates)
    signals = np.full((n_dates, N_J), np.nan)

    for i in range(L, n_dates):
        # --- Window [t-L, t-1] ---
        window = combined.iloc[i - L : i].values  # (L, N)

        mu    = window.mean(axis=0)
        sigma = window.std(axis=0, ddof=0)
        sigma[sigma < 1e-12] = np.nan

        Z = (window - mu) / sigma  # (L, N)
        if np.any(np.isnan(Z)):
            continue

        # Sample correlation (diagonal = 1 by construction)
        Ct = (Z.T @ Z) / L  # (N, N)

        # Regularised correlation (eq. 13)
        C_reg = (1.0 - lambda_) * Ct + lambda_ * C0

        # Eigen-decompose; eigh returns eigenvalues ascending → take last k cols
        _, eigvecs = eigh(C_reg)
        V_K = eigvecs[:, -k:]   # (N, k)

        V_U_k = V_K[:N_U, :]   # (N_U, k)
        V_J_k = V_K[N_U:, :]   # (N_J, k)

        # Standardise today's US return using the window stats (eq. 17)
        us_today = combined.iloc[i, :N_U].values
        us_mu    = mu[:N_U]
        us_sigma = sigma[:N_U]
        if np.any(np.isnan(us_sigma)):
            continue
        z_U = (us_today - us_mu) / us_sigma  # (N_U,)

        # Factor score (eq. 18) and signal (eq. 19)
        ft        = V_U_k.T @ z_U     # (k,)
        signals[i] = V_J_k @ ft       # (N_J,)

    return pd.DataFrame(signals, index=dates, columns=JP_TICKERS)
