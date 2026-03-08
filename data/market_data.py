"""
Market Data Loader — Downloads and caches NIFTY 50 stock data via yfinance.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NIFTY50_TICKERS, STOCK_SECTOR_MAP, SECTORS,
    START_DATE, END_DATE, BENCHMARK_TICKER,
)


class MarketDataLoader:
    def __init__(self):
        self._prices: dict[str, pd.DataFrame] = {}
        self._benchmark: pd.Series | None = None
        self._loaded = False

    def fetch_all(self) -> None:
        if self._loaded:
            return
        print("  Downloading NIFTY 50 stocks...")
        for ticker in NIFTY50_TICKERS:
            try:
                df = self._fetch_single(ticker)
                if len(df) > 100:
                    self._prices[ticker] = df
            except Exception as e:
                print(f"    Warning: {ticker} failed — {e}")

        print(f"  Downloaded {len(self._prices)} stocks successfully.")

        # Benchmark
        print("  Downloading NIFTY 50 index...")
        try:
            bench_df = self._fetch_single(BENCHMARK_TICKER)
            self._benchmark = bench_df["close"]
        except Exception as e:
            print(f"    Warning: Benchmark failed — {e}")
            # Fallback: equal-weight of all stocks
            all_closes = pd.DataFrame({
                t: self._prices[t]["close"] for t in self._prices
            })
            self._benchmark = all_closes.mean(axis=1)

        self._loaded = True
        print("  Data loading complete.")

    def _fetch_single(self, ticker: str) -> pd.DataFrame:
        raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.dropna(subset=["close"], inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    def get_returns(self, ticker: str) -> pd.Series:
        return np.log(self._prices[ticker]["close"] /
                      self._prices[ticker]["close"].shift(1)).dropna()

    def get_close(self, ticker: str) -> pd.Series:
        return self._prices[ticker]["close"]

    def get_all_returns(self) -> pd.DataFrame:
        returns = {}
        for ticker in self._prices:
            returns[ticker] = self.get_returns(ticker)
        return pd.DataFrame(returns).dropna(how="all")

    def get_sector_returns(self) -> pd.DataFrame:
        """Equal-weight average of constituent stock returns per sector."""
        all_ret = self.get_all_returns()
        sector_ret = {}
        for sector in SECTORS:
            tickers = [t for t, s in STOCK_SECTOR_MAP.items()
                       if s == sector and t in all_ret.columns]
            if tickers:
                sector_ret[sector] = all_ret[tickers].mean(axis=1)
        return pd.DataFrame(sector_ret).dropna()

    def get_benchmark(self) -> pd.Series:
        return self._benchmark

    def get_all_closes(self) -> pd.DataFrame:
        return pd.DataFrame({
            t: self._prices[t]["close"] for t in self._prices
        }).dropna(how="all")

    @property
    def tickers(self) -> list[str]:
        return list(self._prices.keys())
