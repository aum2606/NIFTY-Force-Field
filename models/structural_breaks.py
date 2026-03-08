"""
Structural Break Zones — Sigma_j T_j(s, f, tau; Psi, pi_r)
=============================================================
CUSUM-based regime change detection with exponential fade.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SECTORS, FACTORS, CUSUM_THRESHOLD, CUSUM_DRIFT,
    BREAK_FADE_HALFLIFE,
)


class StructuralBreakDetector:
    def __init__(self):
        self._breaks: dict = {}  # date -> (n_sectors x n_factors) matrix

    def compute(self, factor_engine) -> None:
        """Detect structural breaks using CUSUM on each sector-factor series."""
        dates = factor_engine.dates
        if len(dates) < 30:
            return

        for i, sector in enumerate(SECTORS):
            for j, factor in enumerate(FACTORS):
                series = factor_engine.get_factor_series(sector, factor)
                if len(series) < 30:
                    continue

                # Run CUSUM
                values = series.values
                mu = np.nanmean(values[:min(60, len(values))])
                sigma = np.nanstd(values[:min(60, len(values))])
                if sigma < 1e-8:
                    sigma = 1.0

                s_pos = 0.0
                s_neg = 0.0
                k = CUSUM_DRIFT * sigma

                break_intensities = np.zeros(len(dates))
                active_breaks = []  # list of (break_date_idx, intensity)

                for t in range(len(dates)):
                    if t < len(values):
                        x = values[t] if not np.isnan(values[t]) else mu
                    else:
                        x = mu

                    s_pos = max(0, s_pos + (x - mu) - k)
                    s_neg = max(0, s_neg - (x - mu) - k)

                    h = CUSUM_THRESHOLD * sigma

                    # Check for new break
                    if s_pos > h or s_neg > h:
                        intensity = max(s_pos, s_neg) / h
                        active_breaks.append((t, intensity))
                        s_pos = 0.0
                        s_neg = 0.0

                    # Compute total break intensity with exponential fade
                    total = 0.0
                    fade_rate = np.log(2) / BREAK_FADE_HALFLIFE
                    surviving = []
                    for (bt, bi) in active_breaks:
                        elapsed = t - bt
                        faded = bi * np.exp(-fade_rate * elapsed)
                        if faded > 0.01:
                            total += faded
                            surviving.append((bt, bi))
                    active_breaks = surviving
                    break_intensities[t] = total

                # Store into per-date matrices
                for t, date in enumerate(dates):
                    if date not in self._breaks:
                        self._breaks[date] = np.zeros((len(SECTORS), len(FACTORS)))
                    self._breaks[date][i, j] = break_intensities[t]

    def get_breaks_at(self, date) -> np.ndarray:
        if date in self._breaks:
            return self._breaks[date]
        if not self._breaks:
            return np.zeros((len(SECTORS), len(FACTORS)))
        dates = sorted(self._breaks.keys())
        nearest = min(dates, key=lambda d: abs((d - date).total_seconds()))
        return self._breaks[nearest]

    def has_active_break(self, date, sector_idx: int) -> bool:
        breaks = self.get_breaks_at(date)
        return np.any(breaks[sector_idx, :] > 0.5)
