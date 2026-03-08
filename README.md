# NIFTY Force Field — Indian Market Quant Research Engine

<p align="center">
  <strong>Γ(s,f,τ) = K(ρ;Φ,ρ) + X(s,f;ρ) + ΣC<sub>c</sub>(s,f,τ;Ω) + ΣT<sub>j</sub>(s,f,τ;Ψ,π<sub>r</sub>)</strong>
</p>

A quantitative research engine that constructs a **3D force field** over Indian equity markets (NIFTY 50), mapping the interaction between **10 sectors** and **3 systematic factors** through time. The force field drives a momentum-filtered trading strategy that is backtested and visualized in a real-time animated dashboard.

---

## What It Does

1. **Downloads** 5 years of OHLCV data for 30 NIFTY 50 stocks via yfinance
2. **Computes** three factor scores per stock: Momentum, Value, Volatility
3. **Builds a composite force field** from four mathematical components:
   - **Trend Intensity Decay (K)** — Ornstein-Uhlenbeck mean-reversion model
   - **Factor-Sector Coupling (X)** — Rolling 60-day cross-correlation
   - **Return Accumulation Nodes (C)** — Z-score hot-spot detection
   - **Structural Break Zones (T)** — CUSUM regime-change detection with exponential fade
4. **Generates trading signals** from force field topology + sector momentum alignment
5. **Runs a vectorized backtest** with transaction costs and leverage constraints
6. **Renders an animated dashboard** with:
   - 3D surface plot of the evolving force field (236 frames)
   - 2D heatmap cross-section
   - Backtest equity curve: Strategy vs NIFTY 50
   - Play/Pause controls with 4 speed levels (0.5x, 1x, 2x, 4x)
   - MathJax formula display and live stats panel

---

## Performance (Backtest)

| Metric         | Value       |
|----------------|-------------|
| Total Return   | ~103%       |
| CAGR           | ~15.3%      |
| Sharpe Ratio   | ~0.85       |
| Max Drawdown   | ~-18%       |
| Avg Leverage   | ~0.9x       |
| vs NIFTY 50    | Outperforms |

> *Results based on 5-year historical backtest. Past performance is not indicative of future results.*

---

## Project Structure

```
nifty_force_field/
├── config.py                  # Central configuration (tickers, sectors, params, colors)
├── run.py                     # Entry point — runs pipeline + dashboard
├── requirements.txt           # Python dependencies
│
├── data/
│   ├── market_data.py         # yfinance data loader (30 stocks + ^NSEI index)
│   └── factor_engine.py       # Momentum / Value / Volatility factor computation
│
├── models/
│   ├── trend_decay.py         # Ornstein-Uhlenbeck trend intensity decay (K)
│   ├── factor_sector.py       # Rolling cross-correlation coupling (X)
│   ├── return_nodes.py        # Return accumulation hot-spot detection (C)
│   ├── structural_breaks.py   # CUSUM structural break detector (T)
│   └── force_field.py         # Composite Γ assembler + signal extraction
│
├── backtest/
│   ├── strategy.py            # Momentum × force-field alignment strategy
│   └── engine.py              # Vectorized PnL engine with transaction costs
│
└── dashboard/
    ├── app.py                 # Dash application with clientside JS animation
    └── components.py          # Layout helpers
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Internet connection (for yfinance data download)

### Installation

```bash
cd nifty_force_field
pip install -r requirements.txt
```

### Run

```bash
python run.py
```

The pipeline takes ~1-2 minutes on first run (downloading data + computing models). Once ready, open:

```
http://127.0.0.1:8050
```

### Controls

| Button | Action                        |
|--------|-------------------------------|
| PLAY   | Start animation               |
| PAUSE  | Pause animation               |
| 0.5x   | Slow speed (300ms per frame)  |
| 1x     | Normal speed (150ms per frame)|
| 2x     | Fast speed (75ms per frame)   |
| 4x     | Max speed (40ms per frame)    |

---

## Tech Stack

- **Data**: yfinance, pandas, numpy
- **Models**: scipy (linregress, OU fitting), CUSUM statistics
- **Visualization**: Plotly (3D Surface, Heatmap, Scatter), Dash
- **Animation**: Clientside JavaScript callbacks (zero-latency), pre-serialized frames
- **Interpolation**: scipy.ndimage.zoom for smooth 3D terrain

---

## Configuration

All parameters are tunable in `config.py`:

- **Force field weights**: `W_TREND_DECAY`, `W_FACTOR_SECTOR`, `W_RETURN_NODES`, `W_STRUCT_BREAKS`
- **Model parameters**: OU decay (`TREND_DECAY_PHI`), CUSUM threshold/drift, break fade half-life
- **Backtest settings**: leverage cap, rebalance frequency, transaction costs
- **Animation**: step size (days between frames), default interval

---

## Disclaimer

This project is for **educational and research purposes only**. It is not financial advice. The backtest results are based on historical data and do not guarantee future performance. Always consult a qualified financial advisor before making investment decisions.

---

## License

MIT
