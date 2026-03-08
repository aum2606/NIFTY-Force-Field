"""
Trend Intensity Decay — K(rho; Phi, rho)
==========================================
Ornstein-Uhlenbeck mean-reversion applied to trend strength.
K(rho) = Phi * exp(-theta * rho)
"""

import numpy as np
import pandas as pd
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SECTORS, FACTORS, TREND_DECAY_PHI


def _ou_halflife(s: pd.Series) -> float:
    """Estimate OU half-life via regression: dX = beta * X_lag + eps."""
    s = s.dropna()
    if len(s) < 6:
        return np.nan
    lag = s.shift(1).dropna()
    diff = s.diff().dropna()
    lag, diff = lag.align(diff, join="inner")
    if len(lag) < 5:
        return np.nan
    if np.std(lag.values) < 1e-10:
        return np.nan
    slope, *_ = stats.linregress(lag.values, diff.values)
    if slope < 0:
        hl = float(-np.log(2) / slope)
        return np.clip(hl, 5, 252)
    return np.nan


class TrendDecayModel:
    def __init__(self):
        self._theta: dict[tuple[str, str], float] = {}

    def fit(self, factor_engine, sector_returns: pd.DataFrame) -> None:
        """Fit OU process to each sector-factor momentum series."""
        for sector in SECTORS:
            for factor in FACTORS:
                series = factor_engine.get_factor_series(sector, factor)
                if len(series) < 30:
                    self._theta[(sector, factor)] = 0.03  # default
                    continue
                hl = _ou_halflife(series)
                if np.isnan(hl) or hl <= 0:
                    self._theta[(sector, factor)] = 0.03
                else:
                    self._theta[(sector, factor)] = np.log(2) / hl

    def compute_decay_surface(self, factor_engine, date) -> np.ndarray:
        """Compute decay kernel at a given date.

        Returns (n_sectors x n_factors) matrix.
        The decay value represents current trend intensity after decay.
        """
        matrix = factor_engine.get_sector_factor_matrix(date)
        result = np.zeros_like(matrix)

        for i, sector in enumerate(SECTORS):
            for j, factor in enumerate(FACTORS):
                phi = matrix[i, j]  # current trend strength
                theta = self._theta.get((sector, factor), 0.03)
                # Decay applied: stronger theta = faster decay = lower retained intensity
                # We use the absolute phi as initial strength, apply decay
                result[i, j] = phi * TREND_DECAY_PHI * np.exp(-theta)

        return result
