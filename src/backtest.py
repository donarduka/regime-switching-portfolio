from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import ewma_cov, project_weights_to_simplex

def _period_end_flags(index: pd.DatetimeIndex, freq: str = 'M') -> np.ndarray:
    # True on the last date present in each period of 'freq' (e.g., 'M', 'Q')
    p = index.to_period(freq).to_numpy()
    return np.r_[p[1:] != p[:-1], True]

# In-sample backtest (soft regime mixture)
def run_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime_probs: pd.DataFrame,
    regime_weights: dict,
    tcost: float = 0.0005,
    prob_threshold: float = 0.0,
    rebalance_freq: str = 'M'
):
    idx = returns.index
    tickers = list(returns.columns)
    T, N = returns.shape

    is_reb = _period_end_flags(idx, rebalance_freq)
    W = np.zeros((T, N), dtype=float)
    w = np.full(N, 1.0 / N)
    W[0] = w
    cost_factor = np.ones(T, dtype=float)
    P = regime_probs.values  # (T, K)

    for t in range(1, T):
        if is_reb[t]:
            pk = P[t]
            if prob_threshold > 0.0 and pk.max() < prob_threshold:
                W[t] = w
                continue
            w_mix = np.zeros(N)
            for k, wk in regime_weights.items():
                w_mix += pk[int(k)] * wk
            w_mix = project_weights_to_simplex(w_mix)
            turnover = np.abs(w_mix - w).sum()
            if turnover > 0:
                cost_factor[t] = 1.0 - tcost * turnover
            w = w_mix
        W[t] = w

    R = returns.values
    port_ret = np.einsum('tn,tn->t', R, W)
    equity = np.cumprod((1.0 + port_ret) * cost_factor, dtype=float)
    return pd.Series(equity, index=idx, name='equity'), pd.DataFrame(W, index=idx, columns=tickers)

# Rolling OOS backtest (re-fit model each period) with vol targeting
def run_backtest_rolling(
    returns: pd.DataFrame,
    fit_fn,
    weight_fn,
    window: int = 252*3,
    rebalance_freq: str = 'M',
    tcost: float = 0.0005,
    target_vol: float | None = 0.12,
    prob_threshold: float = 0.0
):
    idx = returns.index
    tickers = list(returns.columns)
    T, N = returns.shape
    is_reb = _period_end_flags(idx, rebalance_freq)
    W = np.zeros((T, N), dtype=float)
    w = np.full(N, 1.0 / N)
    W[0] = w
    cost_factor = np.ones(T, dtype=float)

    from .regime_model import posterior_probs, viterbi_path, regime_stats_by_label

    for t in range(1, T):
        if is_reb[t]:
            start = max(0, t - window)
            r_win = returns.iloc[start:t]
            if len(r_win) < max(126, N*20):
                W[t] = w
                continue

            model = fit_fn(r_win)
            probs = posterior_probs(model, r_win)
            labels = viterbi_path(model, r_win)
            stats = regime_stats_by_label(r_win, labels)
            reg_w = weight_fn(stats)

            pk = probs.iloc[-1].to_numpy()
            if prob_threshold > 0.0 and pk.max() < prob_threshold:
                W[t] = w
                continue

            w_mix = np.zeros(N)
            for k, wk in reg_w.items():
                w_mix += pk[int(k)] * wk

            # Volatility targeting
            if target_vol is not None and target_vol > 0:
                cov_fore = ewma_cov(r_win, lam=0.94)
                cur_vol = np.sqrt(w_mix @ cov_fore @ w_mix) * np.sqrt(252.0)
                if cur_vol > 1e-12:
                    scale = target_vol / cur_vol
                    w_mix *= scale

            w_new = project_weights_to_simplex(w_mix)
            turnover = np.abs(w_new - w).sum()
            if turnover > 0:
                cost_factor[t] = 1.0 - tcost * turnover
            w = w_new
        W[t] = w

    R = returns.values
    port_ret = np.einsum('tn,tn->t', R, W)
    equity = np.cumprod((1.0 + port_ret) * cost_factor, dtype=float)
    return pd.Series(equity, index=idx, name='equity'), pd.DataFrame(W, index=idx, columns=tickers)

# Static baseline (no regime switching)
def static_backtest(returns: pd.DataFrame, static_w: np.ndarray):
    R = returns.values
    port_ret = R @ static_w
    equity = np.cumprod(1.0 + port_ret)
    return pd.Series(equity, index=returns.index, name='equity')