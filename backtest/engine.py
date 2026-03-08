"""
Backtest Engine — Vectorized PnL computation.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INITIAL_CAPITAL, TRANSACTION_COST_BPS, SECTORS


class BacktestResult:
    def __init__(self, equity: pd.Series, benchmark: pd.Series,
                 positions: pd.DataFrame, leverage: pd.Series, stats: dict):
        self.equity = equity
        self.benchmark = benchmark
        self.positions = positions
        self.leverage = leverage
        self.stats = stats


class BacktestEngine:
    def run(self, weights: pd.DataFrame, sector_returns: pd.DataFrame,
            benchmark_prices: pd.Series) -> BacktestResult:
        """Run vectorized backtest.

        Args:
            weights: DataFrame (dates x sectors) with position weights
            sector_returns: DataFrame (dates x sectors) with daily returns
            benchmark_prices: Series of benchmark close prices
        """
        # Align dates
        common_dates = weights.index.intersection(sector_returns.index)
        common_dates = common_dates.sort_values()
        weights = weights.loc[common_dates]
        sector_returns = sector_returns.loc[common_dates]

        # Fill missing sectors with 0
        for col in weights.columns:
            if col not in sector_returns.columns:
                sector_returns[col] = 0.0

        # Daily PnL
        prev_weights = weights.shift(1).fillna(0)
        daily_returns = (prev_weights * sector_returns[weights.columns]).sum(axis=1)

        # Transaction costs on rebalance days
        weight_changes = weights.diff().abs().sum(axis=1).fillna(0)
        txn_costs = weight_changes * TRANSACTION_COST_BPS / 10000
        daily_returns -= txn_costs

        # Equity curve
        equity = INITIAL_CAPITAL * (1 + daily_returns).cumprod()

        # Benchmark (normalized to same starting capital)
        bench = benchmark_prices.reindex(common_dates).ffill().dropna()
        if len(bench) > 0:
            benchmark_equity = INITIAL_CAPITAL * (bench / bench.iloc[0])
        else:
            benchmark_equity = pd.Series(INITIAL_CAPITAL, index=common_dates)

        # Leverage
        leverage = weights.abs().sum(axis=1)

        # Position count
        positions = (weights.abs() > 0.01).sum(axis=1)

        # Stats
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else 0
        n_years = len(common_dates) / 252
        cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        daily_ret = daily_returns.dropna()
        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                  if daily_ret.std() > 0 else 0)

        # Max drawdown
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()

        stats = {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "n_trades": int((weight_changes > 0.01).sum()),
            "avg_leverage": float(leverage.mean()),
        }

        return BacktestResult(
            equity=equity,
            benchmark=benchmark_equity,
            positions=positions.astype(int),
            leverage=leverage,
            stats=stats,
        )
