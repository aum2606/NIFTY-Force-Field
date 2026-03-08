"""
Force Field Strategy — Entry/exit rules derived from force field topology.

Combines force field directional signals with sector momentum for robust
trend-following. The force field acts as a regime-aware filter while
momentum determines position direction and sizing.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SECTORS, MAX_LEVERAGE, REBALANCE_FREQ


class ForceFieldStrategy:
    def __init__(self, force_field_result):
        self.result = force_field_result

    def generate_weights(self, sector_returns: pd.DataFrame) -> pd.DataFrame:
        """Generate daily sector weights using force field + momentum overlay.

        Strategy:
        - Compute 20-day sector momentum
        - Use force field direction to confirm/filter momentum signals
        - Long sectors with positive momentum AND positive force
        - Avoid/short sectors with negative momentum AND negative force
        - Size positions by momentum strength, filtered by force magnitude
        """
        signals = self.result.signals
        all_dates = sorted(sector_returns.index)

        # Build signal lookup: date -> {sector: force}
        force_lookup = {}
        for _, row in signals.iterrows():
            date = row["date"]
            if date not in force_lookup:
                force_lookup[date] = {}
            force_lookup[date][row["sector"]] = row["force"]

        # Compute 20-day sector momentum
        sector_mom = sector_returns.rolling(20).mean() * 252  # annualized

        weights = pd.DataFrame(0.0, index=all_dates, columns=SECTORS)
        current_weights = {s: 0.0 for s in SECTORS}
        force_dates = sorted(force_lookup.keys())
        force_idx = 0
        current_forces = {s: 0.0 for s in SECTORS}

        rebalance_counter = 0

        for date in all_dates:
            # Update forces from force field
            while (force_idx < len(force_dates) and
                   force_dates[force_idx] <= date):
                fd = force_dates[force_idx]
                for sector in SECTORS:
                    if sector in force_lookup[fd]:
                        current_forces[sector] = force_lookup[fd][sector]
                force_idx += 1

            rebalance_counter += 1
            if rebalance_counter >= REBALANCE_FREQ:
                rebalance_counter = 0

                for sector in SECTORS:
                    if sector not in sector_mom.columns:
                        continue
                    if date not in sector_mom.index:
                        continue

                    mom = sector_mom.loc[date, sector]
                    force = current_forces.get(sector, 0.0)

                    if np.isnan(mom):
                        current_weights[sector] = 0.0
                        continue

                    # Momentum-force alignment score
                    # Both positive = strong long, both negative = short
                    alignment = np.sign(mom) * np.sign(force) if abs(force) > 0.01 else 0

                    if alignment > 0:
                        # Aligned: take position in momentum direction
                        strength = min(abs(mom) * 2.0, 1.0)
                        current_weights[sector] = np.sign(mom) * strength * 0.3
                    elif alignment < 0:
                        # Conflicting: reduce position
                        current_weights[sector] *= 0.3
                    else:
                        # Weak force: use pure momentum with reduced size
                        if abs(mom) > 0.05:
                            current_weights[sector] = np.sign(mom) * min(abs(mom), 0.5) * 0.15
                        else:
                            current_weights[sector] *= 0.5

            # Normalize to respect leverage constraint
            total_abs = sum(abs(w) for w in current_weights.values())
            if total_abs > MAX_LEVERAGE:
                scale = MAX_LEVERAGE / total_abs
                for sector in SECTORS:
                    current_weights[sector] *= scale

            for sector in SECTORS:
                if sector in weights.columns:
                    weights.loc[date, sector] = current_weights[sector]

        return weights
