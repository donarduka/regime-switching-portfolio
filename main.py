from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import fetch_prices
from src.utils import to_log_returns, sharpe_ratio, max_drawdown, annualise_return, annualise_vol
from src.regime_model import fit_hmm, posterior_probs, viterbi_path, regime_stats_by_label
from src.optimiser import per_regime_weights, mean_variance_long_only
from src.backtest import run_backtest, static_backtest, run_backtest_rolling
from src.plotting import plot_regimes, plot_equity_curves, plot_weights

def parse_args():
    p = argparse.ArgumentParser(description='Regime-Switching Portfolio Optimiser')
    p.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'TLT', 'GLD'])
    p.add_argument('--start', type=str, default='2012-01-01')
    p.add_argument('--end', type=str, default='2025-01-01')
    p.add_argument('--states', type=int, default=3)
    p.add_argument('--gamma', type=float, default=5.0)
    p.add_argument('--cap', type=float, default=0.6)
    p.add_argument('--tcost', type=float, default=0.0005)
    p.add_argument('--prob_threshold', type=float, default=0.0)
    p.add_argument('--rebalance', type=str, default='M', choices=['M','Q'])
    p.add_argument('--rolling', type=int, default=0, help='1=use rolling OOS backtest')
    p.add_argument('--window_days', type=int, default=252*3, help='rolling fit lookback window')
    p.add_argument('--target_vol', type=float, default=0.12, help='annual target vol; set 0 to disable')
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path('results'); plot_dir = out_dir / 'plots'
    out_dir.mkdir(exist_ok=True); plot_dir.mkdir(exist_ok=True)

    prices = fetch_prices(args.tickers, args.start, args.end)
    returns = to_log_returns(prices).dropna()

    # Always fit once for plotting regimes (not used if rolling)
    model_full = fit_hmm(returns, n_states=args.states, random_state=42)
    labels_full = viterbi_path(model_full, returns)
    plot_regimes(labels_full.index, labels_full, str(plot_dir / 'regimes.png'))

    if args.rolling:
        #rolling OOS + vol targeting
        def fit_fn(r_win):
            return fit_hmm(r_win, n_states=args.states, random_state=42)

        def weight_fn(stats):
            #reuse optimiser with cap and shrinkage
            return per_regime_weights(stats, gamma=args.gamma, cap=args.cap, shrink_alpha=0.10)

        tv = None if args.target_vol <= 0 else args.target_vol
        eq_dyn, w_hist = run_backtest_rolling(
            returns, fit_fn, weight_fn,
            window=args.window_days,
            rebalance_freq=args.rebalance,
            tcost=args.tcost,
            target_vol=tv,
            prob_threshold=args.prob_threshold
        )
    else:
        # single in-sample fit (soft mixture)
        probs = posterior_probs(model_full, returns)
        stats = regime_stats_by_label(returns, labels_full)
        reg_w = per_regime_weights(stats, gamma=args.gamma, cap=args.cap, shrink_alpha=0.10)
        eq_dyn, w_hist = run_backtest(
            prices, returns, probs, reg_w,
            tcost=args.tcost, prob_threshold=args.prob_threshold, rebalance_freq=args.rebalance
        )

    # Static comparator
    mu_all = returns.mean().values
    cov_all = returns.cov().values + 1e-6 * np.eye(returns.shape[1])
    w_static = mean_variance_long_only(mu_all, cov_all, gamma=args.gamma, cap=args.cap)
    from src.backtest import static_backtest
    eq_static = static_backtest(returns, w_static)

    daily_dyn = eq_dyn.pct_change().to_numpy()[1:]
    daily_static = eq_static.pct_change().to_numpy()[1:]
    metrics = pd.DataFrame({
        'Sharpe': [sharpe_ratio(daily_dyn), sharpe_ratio(daily_static)],
        'MaxDrawdown': [max_drawdown(eq_dyn), max_drawdown(eq_static)],
        'AnnualisedReturn': [annualise_return(np.nanmean(daily_dyn)), annualise_return(np.nanmean(daily_static))],
        'AnnualisedVol': [annualise_vol(np.nanstd(daily_dyn, ddof=1)), annualise_vol(np.nanstd(daily_static, ddof=1))],
    }, index=['Dynamic', 'Static'])

    metrics.to_csv(out_dir / 'metrics.csv', index=True)
    w_hist.to_csv(out_dir / 'weights_history.csv')

    plot_equity_curves(eq_dyn, eq_static, str(plot_dir / 'equity_curves.png'))
    plot_weights(w_hist, str(plot_dir / 'weights.png'))

    print('--- Summary ---')
    print(metrics.round(4))
    print(f'Outputs saved to: {out_dir.resolve()}')

if __name__ == '__main__':
    main()