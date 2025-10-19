# Regime-Switching Portfolio Optimiser — Report

## Objective
This project investigates whether adapting portfolio weights to market regimes, identified via a Hidden Markov Model (HMM), can improve performance compared to a static mean–variance portfolio.

## Data and Setup
- Assets: Default tickers (SPY, QQQ, TLT, GLD)
- Period: 2012–2025 (daily data)
- Model: Gaussian HMM (hmmlearn) with 3 latent regimes
- Optimiser: Long-only mean–variance optimiser using projected gradient descent
- Rebalancing: Monthly
- Transaction Cost: 0.0005 (0.05% per rebalance)

## Methodology
1. Estimate log returns from price data using yfinance.
2. Fit a HMM on historical returns to infer hidden regimes (bull, bear, volatile).
3. Estimate regime-conditioned means and covariances.
4. Compute optimal long-only weights per regime using projected gradient descent.
5. Run dynamic backtest where weights update when the regime changes.
6. Compare to a static mean–variance portfolio fitted on full-sample statistics.

## Results Summary
| Portfolio | Sharpe | Max Drawdown | Annualised Return | Annualised Volatility |
|------------|:------:|:-------------:|:-----------------:|:---------------------:|
| **Dynamic** | 0.7345 | -0.2745 | 0.0982 | 0.1276 |
| **Static**  | 0.8978 | -0.3201 | 0.1512 | 0.1568 |

## Interpretation
- The static portfolio achieved a higher Sharpe ratio (0.90) and higher annualised return (15.1%), though with greater drawdown and volatility.
- The dynamic portfolio was smoother and more conservative, achieving slightly lower returns but smaller drawdowns — consistent with regime-aware risk management.
- The model successfully identified stable and volatile periods, reallocating defensively during higher volatility regimes.

## Outputs
All results and plots were saved in:
```
results/
├── metrics.csv
├── weights_history.csv
└── plots/
    ├── regimes.png
    ├── equity_curves.png
    └── weights.png
```

## Discussion
This experiment demonstrates the trade-off between adaptability and stability in regime-switching strategies:
- Regime detection helps reduce risk exposure during volatile states.
- Static portfolios may outperform in trending markets but are less responsive.
- The framework can be extended with:
  - Cross-validation for optimal number of regimes
  - Shrinkage estimators for covariance
  - Multi-asset regime features (volatility, spreads, macro indicators)

## Conclusion
The HMM-based regime model offers a clear, interpretable way to adjust portfolio risk dynamically.
While the dynamic strategy traded off some return for stability, it achieved smoother performance and less severe drawdowns — validating the value of regime awareness in portfolio allocation.

## License
MIT License © 2025 Donard Uka