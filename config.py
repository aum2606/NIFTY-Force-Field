"""
NIFTY Force Field — Configuration
===================================
Central config for all tickers, sector mappings, and tunable parameters.
"""

from datetime import datetime, timedelta

# ── Time Range ─────────────────────────────────────────────────
HISTORY_YEARS = 5
END_DATE = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=HISTORY_YEARS * 365 + 30)).strftime("%Y-%m-%d")

# ── NIFTY 50 Stock Universe (.NS suffix for NSE) ──────────────
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "ITC.NS",
    "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "M&M.NS", "TATASTEEL.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "BAJAJFINSV.NS", "TECHM.NS", "NESTLEIND.NS",
]

# ── Sector Definitions ─────────────────────────────────────────
SECTORS = [
    "Bank", "IT", "Pharma", "Auto", "FMCG",
    "Metal", "Energy", "Realty", "Infra", "Media",
]

STOCK_SECTOR_MAP = {
    # Bank
    "HDFCBANK.NS": "Bank", "ICICIBANK.NS": "Bank", "SBIN.NS": "Bank",
    "KOTAKBANK.NS": "Bank", "AXISBANK.NS": "Bank", "BAJFINANCE.NS": "Bank",
    "BAJAJFINSV.NS": "Bank",
    # IT
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT",
    "WIPRO.NS": "IT", "TECHM.NS": "IT",
    # Pharma
    "SUNPHARMA.NS": "Pharma",
    # Auto
    "MARUTI.NS": "Auto", "M&M.NS": "Auto",
    # FMCG
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    "ASIANPAINT.NS": "FMCG", "TITAN.NS": "FMCG",
    # Metal
    "TATASTEEL.NS": "Metal",
    # Energy
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "NTPC.NS": "Energy",
    "POWERGRID.NS": "Energy",
    # Infra
    "LT.NS": "Infra", "ULTRACEMCO.NS": "Infra", "ADANIENT.NS": "Infra",
    "ADANIPORTS.NS": "Infra",
    # Realty / Media — mapped to nearest available
    "BHARTIARTL.NS": "Media",
}

# ── Factor Names ───────────────────────────────────────────────
FACTORS = ["Momentum", "Value", "Volatility"]

# ── Factor Computation ─────────────────────────────────────────
MOMENTUM_PERIOD = 20
ROLLING_WINDOW = 60
LOOKBACK_WINDOW = 252
VOL_WINDOW = 20

# ── Force Field Weights ───────────────────────────────────────
W_TREND_DECAY = 0.30
W_FACTOR_SECTOR = 0.30
W_RETURN_NODES = 0.25
W_STRUCT_BREAKS = 0.15

# ── Model Parameters ──────────────────────────────────────────
TREND_DECAY_PHI = 0.95
COUPLING_RHO_WINDOW = 60
RETURN_NODE_THRESHOLD = 1.5
CUSUM_THRESHOLD = 2.0
CUSUM_DRIFT = 0.5
BREAK_FADE_HALFLIFE = 10

# ── Backtest ───────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000_000  # 1 crore INR
MAX_LEVERAGE = 2.0
REBALANCE_FREQ = 5  # trading days
TRANSACTION_COST_BPS = 10
ENTRY_THRESHOLD = 0.5
EXIT_THRESHOLD = 0.2
COOLDOWN_PERIOD = 10

# ── Animation ──────────────────────────────────────────────────
ANIMATION_STEP_DAYS = 5
ANIMATION_INTERVAL_MS = 150

# ── Dashboard ──────────────────────────────────────────────────
DASH_HOST = "127.0.0.1"
DASH_PORT = 8050

# ── NIFTY 50 Index Ticker ─────────────────────────────────────
BENCHMARK_TICKER = "^NSEI"

# ── Color Scheme — Vibrant Neon Quant Aesthetic ───────────────
# 3D Surface: deep space purple → electric blue → hot cyan → neon green → molten gold
COLORSCALE = [
    [0.00, "#0d0221"],   # deep space purple
    [0.15, "#1a0a4e"],   # dark indigo
    [0.30, "#3d1ca8"],   # electric violet
    [0.45, "#0077ff"],   # vivid blue
    [0.60, "#00e5ff"],   # hot cyan
    [0.75, "#00ff87"],   # neon green
    [0.90, "#ffaa00"],   # molten amber
    [1.00, "#ff3366"],   # hot pink peak
]

# Heatmap uses a diverging scale (negative=cool, positive=hot)
HEATMAP_COLORSCALE = [
    [0.00, "#0d0887"],   # deep indigo
    [0.25, "#7201a8"],   # purple
    [0.50, "#1a1a2e"],   # dark neutral (zero)
    [0.75, "#f77f00"],   # orange
    [1.00, "#fcbf49"],   # bright gold
]

BG_COLOR = "#05050f"      # near-black with blue tint
PANEL_BG = "#0c0c24"      # dark navy panel
GRID_COLOR = "#1e1e4a"    # subtle purple grid
TEXT_COLOR = "#d4d4f0"     # soft lavender white
ACCENT_CYAN = "#00e5ff"    # electric cyan (primary accent)
ACCENT_MAGENTA = "#ff3daa"  # hot magenta (secondary accent)
ACCENT_YELLOW = "#ffaa00"  # warm amber
STRATEGY_GREEN = "#00ff87"  # neon green
BENCHMARK_WHITE = "#7a8ba8" # steel blue-gray
