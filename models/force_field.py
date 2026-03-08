"""
Force Field Assembler — Gamma(s, f, tau)
==========================================
Composite force field combining all four components:
  Gamma = w_K * K + w_X * X + w_C * sum(C) + w_T * sum(T)
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SECTORS, FACTORS, ANIMATION_STEP_DAYS,
    W_TREND_DECAY, W_FACTOR_SECTOR, W_RETURN_NODES, W_STRUCT_BREAKS,
)
from data.market_data import MarketDataLoader
from data.factor_engine import FactorEngine
from models.trend_decay import TrendDecayModel
from models.factor_sector import FactorSectorCoupling
from models.return_nodes import ReturnAccumulationNodes
from models.structural_breaks import StructuralBreakDetector


class ForceFieldResult:
    def __init__(self, dates, surfaces, signals, break_detector):
        self.dates = dates  # list of dates (animation frames)
        self.surfaces = surfaces  # dict: date -> (n_sectors x n_factors) ndarray
        self.signals = signals  # pd.DataFrame with columns: date, sector, signal, weight
        self.break_detector = break_detector

    def get_surface_at(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self.dates):
            return np.zeros((len(SECTORS), len(FACTORS)))
        return self.surfaces[self.dates[idx]]

    def n_frames(self) -> int:
        return len(self.dates)


class ForceField:
    def __init__(self):
        self.data_loader = MarketDataLoader()
        self.factor_engine = FactorEngine()
        self.trend_decay = TrendDecayModel()
        self.factor_sector = FactorSectorCoupling()
        self.return_nodes = ReturnAccumulationNodes()
        self.structural_breaks = StructuralBreakDetector()

    def compute(self) -> ForceFieldResult:
        """Run the full pipeline and return the force field result."""
        # 1. Load data
        print("\n[1/6] Loading market data...")
        self.data_loader.fetch_all()

        closes = self.data_loader.get_all_closes()
        all_returns = self.data_loader.get_all_returns()
        sector_returns = self.data_loader.get_sector_returns()

        # 2. Compute factors
        print("[2/6] Computing factor scores...")
        self.factor_engine.compute(closes, all_returns)

        # 3. Fit trend decay
        print("[3/6] Fitting trend decay model...")
        self.trend_decay.fit(self.factor_engine, sector_returns)

        # 4. Compute factor-sector coupling
        print("[4/6] Computing factor-sector coupling...")
        self.factor_sector.compute(sector_returns, self.factor_engine)

        # 5. Compute return accumulation nodes
        print("[5/6] Detecting return accumulation nodes...")
        self.return_nodes.compute(all_returns, self.factor_engine)

        # 6. Detect structural breaks
        print("[6/6] Detecting structural breaks...")
        self.structural_breaks.compute(self.factor_engine)

        # Assemble composite force field
        print("Assembling force field...")
        all_dates = self.factor_engine.dates
        # Sample every ANIMATION_STEP_DAYS for animation frames
        frame_dates = all_dates[::ANIMATION_STEP_DAYS]
        if not frame_dates:
            frame_dates = all_dates

        surfaces = {}
        for date in frame_dates:
            K = self.trend_decay.compute_decay_surface(self.factor_engine, date)
            X = self.factor_sector.get_coupling_at(date)
            C = self.return_nodes.get_nodes_at(date)
            T = self.structural_breaks.get_breaks_at(date)

            gamma = (W_TREND_DECAY * K +
                     W_FACTOR_SECTOR * X +
                     W_RETURN_NODES * C +
                     W_STRUCT_BREAKS * T)

            # Smooth the surface for better 3D visualization
            gamma = self._smooth_surface(gamma)

            # Scale up for visual drama and signal strength
            gamma = gamma * 3.0
            surfaces[date] = gamma

        # Generate trading signals
        print("Generating trading signals...")
        signals = self._extract_signals(frame_dates, surfaces)

        print(f"Force field computed: {len(frame_dates)} frames")
        return ForceFieldResult(frame_dates, surfaces, signals, self.structural_breaks)

    def _smooth_surface(self, matrix: np.ndarray) -> np.ndarray:
        """Apply light Gaussian-like smoothing for visual appeal."""
        n_s, n_f = matrix.shape
        # Pad and smooth along sector dimension
        smoothed = matrix.copy()
        for j in range(n_f):
            for i in range(n_s):
                neighbors = []
                for di in [-1, 0, 1]:
                    ni = i + di
                    if 0 <= ni < n_s:
                        weight = 1.0 if di == 0 else 0.3
                        neighbors.append(matrix[ni, j] * weight)
                smoothed[i, j] = sum(neighbors) / (1.0 + 0.3 * (len(neighbors) - 1))
        return smoothed

    def _extract_signals(self, dates, surfaces) -> pd.DataFrame:
        """Extract trading signals from force field topology."""
        records = []
        prev_forces = None

        for date in dates:
            gamma = surfaces[date]
            # Sector force = mean across factors
            sector_forces = gamma.mean(axis=1)  # (n_sectors,)

            if prev_forces is not None:
                gradient = sector_forces - prev_forces
            else:
                gradient = np.zeros_like(sector_forces)

            for i, sector in enumerate(SECTORS):
                force = sector_forces[i]
                grad = gradient[i]
                has_break = self.structural_breaks.has_active_break(date, i)

                # Signal logic — use adaptive threshold based on force distribution
                if has_break:
                    signal = 0  # flatten during breaks
                elif force > 0.05 and grad > -0.1:
                    signal = 1  # long
                elif force < -0.05 and grad < 0.1:
                    signal = -1  # short
                else:
                    signal = 0

                # Weight proportional to force strength (scaled up for leverage)
                weight = min(abs(force) * 3.0, 1.0) if signal != 0 else 0.0

                records.append({
                    "date": date,
                    "sector": sector,
                    "signal": signal,
                    "weight": weight,
                    "force": force,
                    "gradient": grad,
                    "break_active": has_break,
                })

            prev_forces = sector_forces.copy()

        return pd.DataFrame(records)
