"""
signal.py — the heart of the PCA-residual stat-arb strategy.

Turns a window of daily returns into s-scores, in three stages (read GUIDELINES.md §1.1):

    Stage 1  PCA      : returns -> common factors (the market/sector forces)
    Stage 2  RESIDUAL : each stock's return minus its common part (regress on factors)
    Stage 3  S-SCORE  : standardize the residual via an OU mean-reversion fit

DESIGN PRINCIPLES baked in here
------------------------------
- PCA DECOMPOSES, it does not predict. The reversion bet is the strategy's
  hypothesis, applied later — not produced here.
- The residual is modeled as Ornstein-Uhlenbeck (mean-reverting). The s-score is
  the residual expressed in its OWN standard deviations.
- Look-ahead safety is the CALLER's responsibility: pass in returns only up to and
  including bar T's close; act on bar T+1's open. This module computes on whatever
  window it's given and never peeks forward.

This file is pure, testable logic. No I/O, no framework, no network. The harness
calls these functions each bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class SignalResult:
    """Everything the strategy needs for one bar, for one set of stocks."""
    s_score: pd.Series              # index=symbol -> s-score (the trading signal)
    ou_speed: pd.Series             # index=symbol -> mean-reversion speed (kappa)
    residual_last: pd.Series        # index=symbol -> latest residual level
    factor_var_explained: np.ndarray  # variance explained by each kept component
    n_factors: int
    universe: list = field(default_factory=list)  # symbols PCA was estimated on


# --------------------------------------------------------------------------- #
# Stage 1 — PCA: find the common factors
# --------------------------------------------------------------------------- #
def compute_pca_factors(returns: pd.DataFrame, n_factors: int = 5):
    """
    Run PCA on a (days x stocks) returns matrix.

    Returns
    -------
    factors : (days x n_factors) DataFrame — the common factor TIME SERIES
              (how each common force moved each day).
    loadings: (stocks x n_factors) DataFrame — each stock's exposure to each
              factor (NOT used directly downstream; we re-regress for residuals,
              which is equivalent because components are orthogonal, but explicit).
    var_explained : fraction of total variance each component explains.

    Standardization: we z-score each stock's return series (mean 0, std 1) before
    PCA, so a single high-variance stock doesn't dominate the factors. This is the
    correlation-matrix form of PCA (standard for this application).
    """
    R = returns.dropna(axis=1, how="any")          # need complete columns for PCA
    if R.shape[1] < n_factors + 1:
        raise ValueError(
            f"PCA needs more stocks than factors: have {R.shape[1]} usable stocks, "
            f"asked for {n_factors} factors. (Demo data with few stocks is degenerate "
            f"— see GUIDELINES.md §7b.)"
        )

    # z-score each stock (column) over the window
    mu = R.mean(axis=0)
    sigma = R.std(axis=0, ddof=1).replace(0, np.nan)
    Z = (R - mu) / sigma
    Z = Z.dropna(axis=1, how="any")                # drop any zero-variance stock

    # PCA via SVD on the standardized matrix
    # Z = U S Vt ; principal directions are rows of Vt (eigenvectors of corr matrix)
    U, S, Vt = np.linalg.svd(Z.values, full_matrices=False)
    k = min(n_factors, Vt.shape[0])

    # factor time series = projection of Z onto the top-k directions
    # (U[:, :k] * S[:k]) gives the scores; equivalently Z @ Vt[:k].T
    factor_scores = Z.values @ Vt[:k].T            # (days x k)
    loadings = Vt[:k].T                            # (stocks x k)

    var_explained = (S[:k] ** 2) / (S ** 2).sum()

    factors_df = pd.DataFrame(
        factor_scores, index=Z.index,
        columns=[f"F{i+1}" for i in range(k)],
    )
    loadings_df = pd.DataFrame(
        loadings, index=Z.columns,
        columns=[f"F{i+1}" for i in range(k)],
    )
    return factors_df, loadings_df, var_explained


# --------------------------------------------------------------------------- #
# Stage 2 — RESIDUAL: strip the common part from each stock
# --------------------------------------------------------------------------- #
def compute_residuals(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """
    For each stock, regress its return series on the common factors and keep the
    residual (the part the factors can't explain).

        stock_return_t = b0 + b1*F1_t + ... + bk*Fk_t + residual_t

    The betas ARE the exposures; the residual IS the idiosyncratic move we trade.

    Works for ANY stock passed in — including names NOT in the PCA estimation set
    (e.g. an orphaned holding). Such a stock is simply PROJECTED onto the existing
    factors. This is the mechanism that lets us still score dropped-out holdings
    (GUIDELINES.md §5).

    Returns (days x stocks) DataFrame of residuals.
    """
    F = factors.copy()
    F["_const"] = 1.0
    Fv = F.values                                   # (days x (k+1))

    # least-squares betas for all stocks at once: (Fv^+ ) @ returns
    # Use lstsq per alignment with the factor index.
    resid = {}
    # restrict returns to the factor window (same dates)
    R = returns.loc[F.index]
    FtF_inv_Ft = np.linalg.pinv(Fv)                 # ((k+1) x days)
    for col in R.columns:
        y = R[col].values
        if np.isnan(y).any():
            # align on non-nan rows for this stock
            mask = ~np.isnan(y)
            if mask.sum() < Fv.shape[1] + 1:
                resid[col] = pd.Series(np.nan, index=R.index)
                continue
            beta = np.linalg.lstsq(Fv[mask], y[mask], rcond=None)[0]
            fitted = Fv @ beta
            resid[col] = pd.Series(y - fitted, index=R.index)
        else:
            beta = FtF_inv_Ft @ y
            fitted = Fv @ beta
            resid[col] = pd.Series(y - fitted, index=R.index)

    return pd.DataFrame(resid, index=R.index)


# --------------------------------------------------------------------------- #
# Stage 3 — S-SCORE: standardize residual via an OU fit
# --------------------------------------------------------------------------- #
def _ou_fit(resid_series: np.ndarray):
    """
    Fit an Ornstein-Uhlenbeck process to a residual series by modeling its
    CUMULATIVE sum as mean-reverting (Avellaneda-Lee construction).

    We form X_t = cumulative sum of residuals, then fit the AR(1):
        X_t = a + b * X_{t-1} + noise
    Mean-reversion speed kappa = -ln(b) (only meaningful for 0<b<1).
    Equilibrium mean = a / (1 - b).
    Equilibrium std (sigma_eq) from the residual variance of the AR(1) fit.

    Returns (s_score_of_last_point, kappa, equilibrium_mean, last_X).
    Returns NaNs if the fit is degenerate or non-mean-reverting.
    """
    x = np.asarray(resid_series, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 30:                                 # need enough points for a stable fit
        return np.nan, np.nan, np.nan, np.nan

    X = np.cumsum(x)                                # the OU level
    X_lag = X[:-1]
    X_now = X[1:]

    # AR(1) regression X_now = a + b X_lag
    A = np.vstack([np.ones_like(X_lag), X_lag]).T
    coef, *_ = np.linalg.lstsq(A, X_now, rcond=None)
    a, b = coef[0], coef[1]

    # mean-reversion requires 0 < b < 1
    if not (0.0 < b < 1.0):
        return np.nan, np.nan, np.nan, X[-1]

    kappa = -np.log(b)
    eq_mean = a / (1.0 - b)

    # residual variance of the AR(1) fit -> equilibrium variance of X
    fitted = A @ coef
    eps = X_now - fitted
    var_eps = np.var(eps, ddof=2) if len(eps) > 2 else np.var(eps)
    # equilibrium variance of an AR(1): var_eps / (1 - b^2)
    sigma_eq = np.sqrt(var_eps / (1.0 - b**2)) if (1.0 - b**2) > 0 else np.nan
    if not np.isfinite(sigma_eq) or sigma_eq == 0:
        return np.nan, kappa, eq_mean, X[-1]

    s_score = (X[-1] - eq_mean) / sigma_eq
    return s_score, kappa, eq_mean, X[-1]


def compute_s_scores(residuals: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    For each stock's residual series, fit OU and return:
        s_score : Series(symbol -> s-score of the latest point)
        ou_speed: Series(symbol -> kappa)
    """
    s_scores, speeds = {}, {}
    for col in residuals.columns:
        s, kappa, _eq, _last = _ou_fit(residuals[col].values)
        s_scores[col] = s
        speeds[col] = kappa
    return pd.Series(s_scores), pd.Series(speeds)


# --------------------------------------------------------------------------- #
# Orchestrator — one call, full pipeline for one bar
# --------------------------------------------------------------------------- #
def compute_signal(
    returns_window: pd.DataFrame,
    n_factors: int = 5,
    score_symbols: list | None = None,
) -> SignalResult:
    """
    Full signal pipeline for ONE bar.

    Parameters
    ----------
    returns_window : (days x stocks) daily returns, ending at bar T (no look-ahead).
                     Columns = the liquid universe used to ESTIMATE the factors.
    n_factors      : number of principal components to keep as common factors.
    score_symbols  : symbols to actually compute s-scores for. Defaults to the
                     universe columns. Pass a SUPERSET (universe ∪ holdings) to
                     also score orphaned holdings (GUIDELINES.md §5) — they get
                     projected onto the universe's factors.

    Returns SignalResult.
    """
    # Stage 1: factors from the (liquid) universe
    factors, _loadings, var_explained = compute_pca_factors(returns_window, n_factors)

    # which stocks to score
    if score_symbols is None:
        score_cols = list(returns_window.columns)
    else:
        score_cols = list(score_symbols)
    # returns for the stocks we want to score (may include names beyond the universe)
    score_returns = returns_window.reindex(columns=score_cols)

    # Stage 2: residuals for all scored stocks (projection onto factors)
    residuals = compute_residuals(score_returns, factors)

    # Stage 3: s-scores + OU speeds
    s_scores, ou_speed = compute_s_scores(residuals)

    return SignalResult(
        s_score=s_scores,
        ou_speed=ou_speed,
        residual_last=residuals.iloc[-1],
        factor_var_explained=var_explained,
        n_factors=factors.shape[1],
        universe=list(returns_window.columns),
    )
