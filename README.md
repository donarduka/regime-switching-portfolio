# Regime-Switching Portfolio Optimiser

A research-style project that learns hidden market regimes (e.g., bull, bear, volatile) from returns using a Hidden Markov Model (HMM) and adapts portfolio weights per regime using a simple mean-variance optimiser with non-negative weights and a budget constraint. Includes a clean backtest with transaction costs and rebalancing.

## Features
- Hidden regime detection with hmmlearn
- Regime-conditioned mean/variance estimation
- Simple, dependency-light, long-only optimiser (no shorting)
- Monthly rebalancing with transaction costs
- Clear plots and metrics for comparison vs. a static portfolio

## Quickstart
```bash
pip install -r requirements.txt

# Run end-to-end (default tickers + dates)
python main.py

# Customise
python main.py --tickers AAPL MSFT SPY TLT --start 2015-01-01 --end 2025-01-01 --states 3 --tcost 0.0005
```
Outputs will be saved in `results/` and `results/plots/`.

## Results (sample run)
| Portfolio | Sharpe | Max Drawdown | Ann. Return | Ann. Vol |
|-----------|:------:|:------------:|:-----------:|:--------:|
| Dynamic   | 0.7345 |    -0.2745   |    0.0982   |  0.1276  |
| Static    | 0.8978 |    -0.3201   |    0.1512   |  0.1568  |

![Equity curves](results/plots/equity_curves.png)


## Repository Layout
```
regime-switching-portfolio/
├── src/
│   ├── data_loader.py
│   ├── regime_model.py
│   ├── optimiser.py
│   ├── backtest.py
│   ├── plotting.py
│   └── utils.py
├── results/
│   └── plots/
├── main.py
├── requirements.txt
├── REPORT.md
└── README.md
```

## Notes
- The optimiser uses projected gradient descent onto the simplex for stability and zero-dependency usage. If scipy or cvxpy is installed, you can extend easily.
- This is a research demo. In production, you would add cross-validation for the number of regimes, stronger stationarity checks, and more robust estimation/shrinkage of covariance matrices.

## License
MIT

