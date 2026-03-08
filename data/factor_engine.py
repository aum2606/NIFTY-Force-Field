"""
Factor Engine — Computes Momentum, Value, and Volatility factor scores.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MOMENTUM_PERIOD, ROLLING_WINDOW, LOOKBACK_WINDOW, VOL_WINDOW,
    FACTORS, STOCK_SECTOR_MAP, SECTORS,
)


class FactorEngine:
    def __init__(self):
        self._factor_scores: pd.DataFrame | None = None
        self._sector_factor_scores: dict[str, pd.DataFrame] | None = None

    def compute(self, closes: pd.DataFrame, returns: pd.DataFrame) -> None:
        """Compute all factor scores for each stock.

        Args:
            closes: DataFrame with columns=tickers, index=dates
            returns: DataFrame with columns=tickers, index=dates (log returns)
        """
        momentum = self._compute_momentum(closes)
        value = self._compute_value(closes)
        volatility = self._compute_volatility(returns)

        # Stack into a single DataFrame with MultiIndex columns (ticker, factor)
        frames = {}
        for ticker in closes.columns:
            if ticker in momentum.columns:
                frames[(ticker, "Momentum")] = momentum[ticker]
            if ticker in value.columns:
                frames[(ticker, "Value")] = value[ticker]
            if ticker in volatility.columns:
                frames[(ticker, "Volatility")] = volatility[ticker]

        self._factor_scores = pd.DataFrame(frames).dropna(how="all")

        # Compute sector-level average factor scores
        self._sector_factor_scores = {}
        for date in self._factor_scores.index:
            row = {}
            for sector in SECTORS:
                tickers = [t for t, s in STOCK_SECTOR_MAP.items()
                           if s == sector and (t, "Momentum") in self._factor_scores.columns]
                if not tickers:
                    row[sector] = {f: 0.0 for f in FACTORS}
                    continue
                sector_vals = {}
                for factor in FACTORS:
                    vals = [self._factor_scores.loc[date, (t, factor)]
                            for t in tickers
                            if (t, factor) in self._factor_scores.columns
                            and not np.isnan(self._factor_scores.loc[date, (t, factor)])]
                    sector_vals[factor] = np.mean(vals) if vals else 0.0
                row[sector] = sector_vals
            self._sector_factor_scores[date] = row

    def _compute_momentum(self, closes: pd.DataFrame) -> pd.DataFrame:
        """20-day return, Z-scored over 60-day rolling window."""
        mom = closes.pct_change(MOMENTUM_PERIOD)
        mom_z = (mom - mom.rolling(ROLLING_WINDOW).mean()) / mom.rolling(ROLLING_WINDOW).std()
        return mom_z

    def _compute_value(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Negative deviation from 252-day rolling mean (mean-reversion proxy)."""
        deviation = (closes - closes.rolling(LOOKBACK_WINDOW).mean()) / closes.rolling(LOOKBACK_WINDOW).std()
        return -deviation

    def _compute_volatility(self, returns: pd.DataFrame) -> pd.DataFrame:
        """20-day realized vol, Z-scored over 60-day window."""
        realized_vol = returns.rolling(VOL_WINDOW).std() * np.sqrt(252)
        vol_z = (realized_vol - realized_vol.rolling(ROLLING_WINDOW).mean()) / realized_vol.rolling(ROLLING_WINDOW).std()
        return vol_z

    def get_sector_factor_matrix(self, date) -> np.ndarray:
        """Returns (n_sectors x n_factors) matrix for a given date."""
        if self._sector_factor_scores is None:
            return np.zeros((len(SECTORS), len(FACTORS)))
        if date not in self._sector_factor_scores:
            # Find nearest date
            dates = sorted(self._sector_factor_scores.keys())
            nearest = min(dates, key=lambda d: abs((d - date).total_seconds()))
            date = nearest
        row = self._sector_factor_scores[date]
        matrix = np.zeros((len(SECTORS), len(FACTORS)))
        for i, sector in enumerate(SECTORS):
            for j, factor in enumerate(FACTORS):
                matrix[i, j] = row.get(sector, {}).get(factor, 0.0)
        return matrix

    def get_factor_series(self, sector: str, factor: str) -> pd.Series:
        """Get time series of a specific sector-factor combination."""
        if self._sector_factor_scores is None:
            return pd.Series(dtype=float)
        dates = sorted(self._sector_factor_scores.keys())
        values = [self._sector_factor_scores[d].get(sector, {}).get(factor, 0.0)
                  for d in dates]
        return pd.Series(values, index=dates)

    @property
    def dates(self) -> list:
        if self._sector_factor_scores is None:
            return []
        return sorted(self._sector_factor_scores.keys())
