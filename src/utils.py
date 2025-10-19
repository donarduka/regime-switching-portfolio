from __future__ import annotations
import numpy as np
import pandas as pd

_TRADING_DAYS = 252.0
_SQRT_TRADING_DAYS = np.sqrt(_TRADING_DAYS)

def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index()
    return np.log(prices).diff().dropna(how="all")

def sharpe_ratio(daily_returns: pd.Series | np.ndarray, rf_daily: float = 0.0) -> float:
    x = np.asarray(daily_returns, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    excess = x - rf_daily
    mu = np.nanmean(excess)
    sigma = np.nanstd(excess, ddof=1)
    if sigma == 0 or not np.isfinite(sigma):
        return np.nan
    return _SQRT_TRADING_DAYS * mu / sigma

def max_drawdown(equity_curve: pd.Series | np.ndarray) -> float:
    x = np.asarray(equity_curve, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
    roll_max = np.maximum.accumulate(x)
    dd = x / roll_max - 1.0
    return float(np.nanmin(dd))

def annualise_return(daily_ret_mean: float) -> float:
    # daily_ret_mean should be the mean of daily returns
    return (1.0 + daily_ret_mean) ** _TRADING_DAYS - 1.0

def annualise_vol(daily_vol: float) -> float:
    return daily_vol * _SQRT_TRADING_DAYS

def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    """EWMA covariance (RiskMetrics). returns: T x N (recent last)."""
    x = returns.to_numpy(dtype=float)
    T, N = x.shape
    if T < 2:
        return np.eye(N) * 1e-6
    S = np.cov(x[:2].T)  # seed with first two rows 
    for t in range(2, T):
        r = x[t:t+1].T
        S = lam * S + (1 - lam) * (r @ r.T)
    # ensure PSD-ish
    S = S + 1e-6 * np.eye(N)
    return S

def realised_vol_annualised(weights: np.ndarray, cov_daily: np.ndarray) -> float:
    # vol = sqrt(w' * Sigma * w) * sqrt(252)
    return float(np.sqrt(weights @ cov_daily @ weights) * np.sqrt(252.0))

def scale_to_target_vol(weights: np.ndarray, cov_daily: np.ndarray, target_vol: float) -> np.ndarray:
    cur_vol = realised_vol_annualised(weights, cov_daily)
    if cur_vol <= 1e-12:
        return weights
    scale = target_vol / cur_vol
    w = weights * scale
    # project back to simplex to ensure valid weights;
    # scale down only, so sum(w) <= 1
    return project_weights_to_simplex(w)

# project weights to simplex {w >= 0, sum w = 1}
def project_weights_to_simplex(v: np.ndarray) -> np.ndarray:
    v = v.ravel()
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = (u * (np.arange(1, u.size + 1)) > (cssv - 1)).nonzero()[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)