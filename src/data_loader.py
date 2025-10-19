from __future__ import annotations
import pandas as pd
import yfinance as yf

def fetch_prices(tickers, start: str, end: str) -> pd.DataFrame:
    if isinstance(tickers, str):
        tickers = [tickers]
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # yfinance can return a multiindex but only need 'Close' prices
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"].copy()
    else:
        prices = df.rename(columns={"Close": tickers[0]})
    prices = prices.sort_index().dropna(how="all")
    return prices.astype("float64")
