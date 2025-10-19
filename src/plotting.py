from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
# plotting functions for main.py
# regime plot
def plot_regimes(index: pd.DatetimeIndex, labels: pd.Series, save_path: str):
    plt.figure(figsize=(10, 3))
    plt.plot(index, labels.values, lw=1)
    plt.title('Most Likely Regime (Viterbi)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
# equity curves comparison
def plot_equity_curves(eq_dyn, eq_static, save_path: str):
    plt.figure(figsize=(10, 4))
    plt.plot(eq_dyn.index, eq_dyn.values, label='Dynamic (Regime Switching)')
    plt.plot(eq_static.index, eq_static.values, label='Static')
    plt.legend()
    plt.title('Equity Curves')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
# portfolio weights over time
def plot_weights(weights_hist, save_path: str):
    plt.figure(figsize=(10, 4))
    for c in weights_hist.columns:
        plt.plot(weights_hist.index, weights_hist[c].values, label=c, alpha=0.85)
    plt.legend(ncol=3, fontsize=8)
    plt.title('Portfolio Weights Over Time')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()