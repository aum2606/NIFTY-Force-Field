"""
Factor-Sector Coupling — X(s, f; rho)
========================================
Rolling cross-correlation between factor returns and sector returns.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SECTORS, FACTORS, COUPLING_RHO_WINDOW


class FactorSectorCoupling:
    def __init__(self):
        self._coupling: dict = {}  # date -> (n_sectors x n_factors) matrix

    def compute(self, sector_returns: pd.DataFrame, factor_engine) -> None:
        """Compute rolling correlation between each sector's returns and
        the cross-sectional average factor score for that factor."""
        dates = factor_engine.dates
        if len(dates) < COUPLING_RHO_WINDOW + 1:
            return

        # Build sector factor score time series
        sector_factor_ts = {}
        for sector in SECTORS:
            for factor in FACTORS:
                series = factor_engine.get_factor_series(sector, factor)
                sector_factor_ts[(sector, factor)] = series

        # Compute rolling correlations
        for idx in range(COUPLING_RHO_WINDOW, len(dates)):
            date = dates[idx]
            window_dates = dates[idx - COUPLING_RHO_WINDOW:idx]
            matrix = np.zeros((len(SECTORS), len(FACTORS)))

            for i, sector in enumerate(SECTORS):
                if sector not in sector_returns.columns:
                    continue
                # Get sector returns for window
                sec_ret = sector_returns[sector].reindex(window_dates).dropna()
                if len(sec_ret) < 20:
                    continue

                for j, factor in enumerate(FACTORS):
                    fac_series = sector_factor_ts.get((sector, factor))
                    if fac_series is None:
                        continue
                    fac_window = fac_series.reindex(sec_ret.index).dropna()
                    common = sec_ret.index.intersection(fac_window.index)
                    if len(common) < 20:
                        continue
                    corr = np.corrcoef(sec_ret.loc[common].values,
                                       fac_window.loc[common].values)[0, 1]
                    matrix[i, j] = corr if not np.isnan(corr) else 0.0

            self._coupling[date] = matrix

    def get_coupling_at(self, date) -> np.ndarray:
        if date in self._coupling:
            return self._coupling[date]
        if not self._coupling:
            return np.zeros((len(SECTORS), len(FACTORS)))
        dates = sorted(self._coupling.keys())
        nearest = min(dates, key=lambda d: abs((d - date).total_seconds()))
        return self._coupling[nearest]
