"""
Return Accumulation Nodes — Sigma_c C_c(s, f, tau; Omega)
============================================================
Identifies regions in sector-factor space where returns consistently
cluster above a threshold — the "hot spots" / attractors.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SECTORS, FACTORS, STOCK_SECTOR_MAP,
    RETURN_NODE_THRESHOLD, ROLLING_WINDOW,
)


class ReturnAccumulationNodes:
    def __init__(self):
        self._nodes: dict = {}  # date -> (n_sectors x n_factors) matrix

    def compute(self, all_returns: pd.DataFrame, factor_engine) -> None:
        """Detect return accumulation nodes.

        For each (sector, factor) cell, compute the rolling average return
        of stocks ranked by that factor. A cell is a "node" if the accumulated
        return exceeds RETURN_NODE_THRESHOLD standard deviations.
        """
        dates = factor_engine.dates
        if len(dates) < ROLLING_WINDOW + 1:
            return

        # Precompute sector stock lists
        sector_stocks = {}
        for sector in SECTORS:
            tickers = [t for t, s in STOCK_SECTOR_MAP.items()
                       if s == sector and t in all_returns.columns]
            sector_stocks[sector] = tickers

        # Build rolling accumulated returns per sector
        for idx in range(ROLLING_WINDOW, len(dates)):
            date = dates[idx]
            window_dates = dates[idx - ROLLING_WINDOW:idx]
            matrix = np.zeros((len(SECTORS), len(FACTORS)))

            for i, sector in enumerate(SECTORS):
                tickers = sector_stocks[sector]
                if not tickers:
                    continue

                # Get sector returns in window
                sec_returns = all_returns[tickers].reindex(window_dates).dropna(how="all")
                if len(sec_returns) < 10:
                    continue

                accumulated = sec_returns.sum().mean()  # mean accumulated return
                accumulated_std = sec_returns.sum().std()

                for j, factor in enumerate(FACTORS):
                    # Get factor scores to weight the returns
                    factor_matrix = factor_engine.get_sector_factor_matrix(date)
                    factor_strength = abs(factor_matrix[i, j])

                    # Node intensity: Z-score of accumulated return * factor alignment
                    if accumulated_std > 0:
                        z_score = accumulated / accumulated_std
                        node_intensity = max(0, abs(z_score) * factor_strength - RETURN_NODE_THRESHOLD)
                        # Preserve sign
                        matrix[i, j] = node_intensity * np.sign(z_score)
                    else:
                        matrix[i, j] = 0.0

            self._nodes[date] = matrix

    def get_nodes_at(self, date) -> np.ndarray:
        if date in self._nodes:
            return self._nodes[date]
        if not self._nodes:
            return np.zeros((len(SECTORS), len(FACTORS)))
        dates = sorted(self._nodes.keys())
        nearest = min(dates, key=lambda d: abs((d - date).total_seconds()))
        return self._nodes[nearest]
