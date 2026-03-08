"""
NIFTY Force Field — Interactive Dash Dashboard
=================================================
Animated 3D force field surface + 2D heatmap + backtest chart.
Uses clientside callbacks for smooth, lag-free animation.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, clientside_callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import zoom
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SECTORS, FACTORS, COLORSCALE, HEATMAP_COLORSCALE,
    BG_COLOR, PANEL_BG, GRID_COLOR, TEXT_COLOR,
    ACCENT_CYAN, ACCENT_MAGENTA, ACCENT_YELLOW, STRATEGY_GREEN, BENCHMARK_WHITE,  # noqa: F401
    ANIMATION_INTERVAL_MS, INITIAL_CAPITAL,
)

# ── Precomputed data (filled by precompute()) ────────────────────
_FF_RESULT = None
_BACKTEST_RESULT = None
_PREBUILT = {}  # Holds pre-serialized frame data for clientside animation


def precompute():
    """Run full pipeline + pre-serialize all animation frames."""
    global _FF_RESULT, _BACKTEST_RESULT, _PREBUILT

    from models.force_field import ForceField
    from backtest.strategy import ForceFieldStrategy
    from backtest.engine import BacktestEngine

    ff = ForceField()
    _FF_RESULT = ff.compute()

    strategy = ForceFieldStrategy(_FF_RESULT)
    sector_returns = ff.data_loader.get_sector_returns()
    weights = strategy.generate_weights(sector_returns)

    engine = BacktestEngine()
    _BACKTEST_RESULT = engine.run(
        weights, sector_returns, ff.data_loader.get_benchmark()
    )

    # Pre-serialize ALL frames for clientside animation
    print("  Pre-rendering animation frames...")
    n = _FF_RESULT.n_frames()
    all_surfaces = []
    all_dates = []
    for i in range(n):
        surface = _FF_RESULT.get_surface_at(i)
        smooth = zoom(surface, (3, 5), order=3)
        all_surfaces.append(np.round(smooth, 4).tolist())
        d = _FF_RESULT.dates[i]
        all_dates.append(d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d))

    # Pre-build backtest data
    equity = _BACKTEST_RESULT.equity
    benchmark = _BACKTEST_RESULT.benchmark
    eq_norm = (equity / equity.iloc[0] * 100)
    bm_norm = benchmark.reindex(equity.index).ffill()
    if len(bm_norm.dropna()) > 0:
        bm_norm = bm_norm / bm_norm.dropna().iloc[0] * 100

    # Build info for each frame
    all_info = []
    for i, date in enumerate(all_dates):
        nearest = equity.index[equity.index <= _FF_RESULT.dates[i]]
        eq_val = float(equity.loc[nearest[-1]]) if len(nearest) > 0 else INITIAL_CAPITAL
        pos_nearest = _BACKTEST_RESULT.positions.index[_BACKTEST_RESULT.positions.index <= _FF_RESULT.dates[i]]
        pos_val = int(_BACKTEST_RESULT.positions.loc[pos_nearest[-1]]) if len(pos_nearest) > 0 else 0
        lev_nearest = _BACKTEST_RESULT.leverage.index[_BACKTEST_RESULT.leverage.index <= _FF_RESULT.dates[i]]
        lev_val = float(_BACKTEST_RESULT.leverage.loc[lev_nearest[-1]]) if len(lev_nearest) > 0 else 0
        all_info.append(f"{date}  |  Strategy: INR {eq_val:,.0f}  |  Positions: {pos_val}  |  Leverage: {lev_val:.2f}x  |  Frame: {i+1}/{n}")

    _PREBUILT = {
        "surfaces": all_surfaces,
        "dates": all_dates,
        "info": all_info,
        "n_frames": n,
        "sectors": SECTORS,
        "factors": FACTORS,
        # Heatmap raw (non-interpolated) surfaces
        "heatmaps": [np.round(_FF_RESULT.get_surface_at(i), 4).tolist() for i in range(n)],
    }

    print(f"  {n} frames pre-rendered. Starting dashboard...")


# ── Figure Builders ──────────────────────────────────────────────

def _build_3d_figure(surface_z):
    """Build the 3D surface figure with dramatic neon aesthetic."""
    n_rows, n_cols = len(surface_z), len(surface_z[0])
    x = np.linspace(0, len(FACTORS) - 1, n_cols).tolist()
    y = np.linspace(0, len(SECTORS) - 1, n_rows).tolist()

    fig = go.Figure(data=[go.Surface(
        z=surface_z,
        x=x,
        y=y,
        colorscale=COLORSCALE,
        lighting=dict(ambient=0.4, diffuse=0.7, specular=0.8, roughness=0.15, fresnel=0.5),
        lightposition=dict(x=-50000, y=50000, z=90000),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="#ff3daa", project_z=True, width=2),
            x=dict(show=True, usecolormap=True, highlightcolor=ACCENT_CYAN, width=1),
        ),
        showscale=True,
        opacity=0.95,
        colorbar=dict(
            title=dict(text="Γ Force", font=dict(color=ACCENT_CYAN, size=11, family="monospace")),
            tickfont=dict(color=TEXT_COLOR, size=9),
            len=0.75, thickness=18,
            bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR, borderwidth=1,
            x=1.02, outlinecolor=GRID_COLOR,
        ),
    )])

    # Watermark
    fig.add_annotation(
        text="NIFTY FORCE FIELD", xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=30, color="rgba(0,229,255,0.06)", family="monospace"),
    )

    fig.update_layout(
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        margin=dict(l=0, r=30, t=30, b=0),
        scene=dict(
            bgcolor=BG_COLOR,
            xaxis=dict(
                title=dict(text="Factors", font=dict(color=ACCENT_CYAN, size=12)),
                tickvals=list(range(len(FACTORS))), ticktext=FACTORS,
                tickfont=dict(color="#8888cc", size=9),
                gridcolor="#2a2a5e", showbackground=True, backgroundcolor="#08081a",
                zerolinecolor="#3d3d7a",
            ),
            yaxis=dict(
                title=dict(text="Sectors", font=dict(color=ACCENT_MAGENTA, size=12)),
                tickvals=list(range(len(SECTORS))), ticktext=SECTORS,
                tickfont=dict(color="#8888cc", size=8),
                gridcolor="#2a2a5e", showbackground=True, backgroundcolor="#08081a",
                zerolinecolor="#3d3d7a",
            ),
            zaxis=dict(
                title=dict(text="Γ", font=dict(color=STRATEGY_GREEN, size=14)),
                tickfont=dict(color="#8888cc", size=9),
                gridcolor="#2a2a5e", showbackground=True, backgroundcolor="#08081a",
                zerolinecolor="#3d3d7a",
            ),
            camera=dict(eye=dict(x=1.8, y=-1.5, z=1.0)),
            aspectratio=dict(x=1.5, y=2.0, z=0.8),
        ),
        font=dict(family="Courier New, monospace", color=TEXT_COLOR),
        uirevision="constant",
    )
    return fig


def _build_heatmap_figure(heatmap_z):
    """Build the 2D heatmap with diverging purple-gold scale."""
    fig = go.Figure(data=[go.Heatmap(
        z=heatmap_z, x=FACTORS, y=SECTORS,
        colorscale=HEATMAP_COLORSCALE, showscale=True,
        colorbar=dict(
            tickfont=dict(color="#8888cc", size=8), len=0.9, thickness=10,
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", x=1.02,
        ),
        hoverongaps=False,
        xgap=2, ygap=2,  # cell borders for clarity
    )])
    fig.update_layout(
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        margin=dict(l=70, r=30, t=25, b=30),
        xaxis=dict(tickfont=dict(color="#aa88dd", size=9), gridcolor=GRID_COLOR),
        yaxis=dict(tickfont=dict(color="#aa88dd", size=9), gridcolor=GRID_COLOR),
        title=dict(text="FIELD CROSS-SECTION", font=dict(size=11, color=ACCENT_MAGENTA, family="monospace"), x=0.5),
        font=dict(family="Courier New, monospace", color=TEXT_COLOR),
        uirevision="constant",
    )
    return fig


def _build_backtest_figure():
    """Build the backtest chart with gradient fills and neon styling."""
    equity = _BACKTEST_RESULT.equity
    benchmark = _BACKTEST_RESULT.benchmark
    leverage = _BACKTEST_RESULT.leverage

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    eq_norm = equity / equity.iloc[0] * 100
    bm_norm = benchmark.reindex(equity.index).ffill()
    if len(bm_norm.dropna()) > 0:
        bm_norm = bm_norm / bm_norm.dropna().iloc[0] * 100
    else:
        bm_norm = pd.Series(100, index=equity.index)

    dates_str = [d.strftime("%Y-%m-%d") for d in equity.index]
    bm_dates_str = [d.strftime("%Y-%m-%d") for d in bm_norm.index]

    # Strategy with glow fill
    fig.add_trace(go.Scatter(
        x=dates_str, y=eq_norm.values.tolist(),
        mode="lines", line=dict(color=STRATEGY_GREEN, width=2.5),
        name="Strategy", fill="tozeroy",
        fillcolor="rgba(0,255,135,0.08)",
    ), secondary_y=False)

    # Benchmark with subtle fill
    fig.add_trace(go.Scatter(
        x=bm_dates_str, y=bm_norm.values.tolist(),
        mode="lines", line=dict(color=BENCHMARK_WHITE, width=1.5, dash="dot"),
        name="NIFTY 50", fill="tozeroy",
        fillcolor="rgba(122,139,168,0.04)",
    ), secondary_y=False)

    # Leverage bars — magenta tinted
    lev = leverage.reindex(equity.index).fillna(0)
    step = max(1, len(lev) // 200)
    lev_down = lev.iloc[::step]
    lev_dates_str = [d.strftime("%Y-%m-%d") for d in lev_down.index]

    fig.add_trace(go.Bar(
        x=lev_dates_str, y=lev_down.values.tolist(),
        marker_color=ACCENT_MAGENTA, opacity=0.2, name="Leverage",
    ), secondary_y=True)

    # Endpoint labels with glow effect
    fig.add_annotation(
        x=dates_str[-1], y=float(eq_norm.iloc[-1]),
        text=f"<b>Strategy {eq_norm.iloc[-1]:.0f}</b>", showarrow=True,
        arrowhead=0, arrowcolor=STRATEGY_GREEN, arrowwidth=1,
        font=dict(color=STRATEGY_GREEN, size=11, family="monospace"),
        xanchor="left", xshift=10, ax=30, ay=-15,
        bgcolor="rgba(0,255,135,0.1)", bordercolor=STRATEGY_GREEN, borderwidth=1, borderpad=3,
    )
    fig.add_annotation(
        x=bm_dates_str[-1], y=float(bm_norm.dropna().iloc[-1]),
        text=f"NIFTY 50 {bm_norm.dropna().iloc[-1]:.0f}", showarrow=False,
        font=dict(color=BENCHMARK_WHITE, size=10, family="monospace"), xanchor="left", xshift=5,
    )

    # Vertical marker line (trace index 3)
    fig.add_trace(go.Scatter(
        x=[dates_str[0], dates_str[0]],
        y=[0, float(eq_norm.max()) * 1.1],
        mode="lines",
        line=dict(color=ACCENT_CYAN, width=2, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ), secondary_y=False)

    fig.update_layout(
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        margin=dict(l=50, r=70, t=25, b=30),
        xaxis=dict(
            tickfont=dict(color="#8888cc", size=9), gridcolor="#1a1a3a",
            showgrid=True, gridwidth=1,
        ),
        yaxis=dict(
            title=dict(text="Value (base=100)", font=dict(color=ACCENT_CYAN, size=10)),
            tickfont=dict(color="#8888cc", size=9), gridcolor="#1a1a3a", showgrid=True,
        ),
        yaxis2=dict(
            title=dict(text="Leverage", font=dict(color=ACCENT_MAGENTA, size=10)),
            tickfont=dict(color=ACCENT_MAGENTA, size=9), showgrid=False, range=[0, 5],
        ),
        legend=dict(
            font=dict(color=TEXT_COLOR, size=10), bgcolor="rgba(12,12,36,0.8)",
            bordercolor=GRID_COLOR, borderwidth=1, x=0.01, y=0.99,
        ),
        font=dict(family="Courier New, monospace"),
        uirevision="constant",
    )
    return fig


def _build_stats_html():
    """Build stats panel with color-coded metrics."""
    stats = _BACKTEST_RESULT.stats
    # Each row: (label, value, color)
    rows = [
        ("CAGR", f"{stats['cagr']:.1%}", STRATEGY_GREEN if stats['cagr'] > 0 else ACCENT_MAGENTA),
        ("Sharpe", f"{stats['sharpe']:.2f}", ACCENT_CYAN if stats['sharpe'] > 1 else ACCENT_YELLOW),
        ("Max DD", f"{stats['max_drawdown']:.1%}", ACCENT_MAGENTA),
        ("Total Return", f"{stats['total_return']:.1%}", STRATEGY_GREEN if stats['total_return'] > 0 else ACCENT_MAGENTA),
        ("Trades", f"{stats['n_trades']}", "#8888cc"),
        ("Avg Leverage", f"{stats['avg_leverage']:.2f}x", ACCENT_CYAN),
    ]
    children = [html.Div("PERFORMANCE", style={
        "fontSize": "11px", "fontWeight": "700", "color": ACCENT_MAGENTA,
        "letterSpacing": "3px", "marginBottom": "12px",
        "borderBottom": f"2px solid {ACCENT_MAGENTA}", "paddingBottom": "6px",
    })]
    for label, value, color in rows:
        children.append(html.Div(style={
            "display": "flex", "justifyContent": "space-between",
            "padding": "5px 0", "borderBottom": f"1px solid {GRID_COLOR}",
        }, children=[
            html.Span(label, style={"color": "#7a7aaa", "fontSize": "11px"}),
            html.Span(value, style={"color": color, "fontWeight": "700", "fontSize": "12px"}),
        ]))
    return children


# ── App & Layout ─────────────────────────────────────────────────

app = dash.Dash(__name__, title="NIFTY Force Field", update_title=None)


def serve_layout():
    """Deferred layout — builds initial figures from precomputed data."""
    if not _PREBUILT:
        return html.Div("Pipeline not initialized.", style={"color": "red", "padding": "40px"})

    fig_3d = _build_3d_figure(_PREBUILT["surfaces"][0])
    fig_hm = _build_heatmap_figure(_PREBUILT["heatmaps"][0])
    fig_bt = _build_backtest_figure()

    return html.Div(
        style={
            "fontFamily": "'Courier New', monospace",
            "backgroundColor": BG_COLOR,
            "minHeight": "100vh",
            "padding": "0",
            "color": TEXT_COLOR,
        },
        children=[
            # Formula header — gradient top bar
            html.Div(style={
                "background": "linear-gradient(180deg, #0d0221 0%, #05050f 100%)",
                "padding": "20px 30px 10px", "textAlign": "center",
                "borderBottom": f"1px solid #3d1ca8",
            }, children=[
                dcc.Markdown(
                    r"$$\Gamma(s,f,\tau) = \underbrace{\mathcal{K}(\rho;\Phi,\rho)}_{\text{trend intensity decay}} + \underbrace{\mathcal{X}(s,f;\rho)}_{\text{factor-sector coupling}} + \underbrace{\sum_c C_c(s,f,\tau;\Omega)}_{\text{return accumulation nodes}} + \underbrace{\sum_j \mathcal{T}_j(s,f,\tau;\Psi,\pi_r)}_{\text{structural break zones}}$$",
                    mathjax=True, style={"color": "#c4b5fd", "fontSize": "16px", "margin": "0"},
                ),
            ]),

            # Title + controls
            html.Div(style={
                "backgroundColor": BG_COLOR, "padding": "10px 30px",
                "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                "borderBottom": f"1px solid #3d1ca8",
            }, children=[
                html.Div([
                    html.Span("NIFTY FORCE FIELD", style={
                        "fontSize": "15px", "fontWeight": "700", "color": ACCENT_CYAN,
                        "letterSpacing": "4px", "fontFamily": "monospace",
                        "textShadow": "0 0 10px rgba(0,229,255,0.4)",
                    }),
                    html.Span(" | ", style={"color": "#3d1ca8", "marginLeft": "8px"}),
                    html.Span("Indian Market Quant Engine", style={
                        "fontSize": "12px", "color": ACCENT_MAGENTA, "fontFamily": "monospace",
                        "letterSpacing": "1px", "textShadow": "0 0 8px rgba(255,61,170,0.3)",
                    }),
                ]),
                html.Div(style={"display": "flex", "gap": "12px", "alignItems": "center"}, children=[
                    html.Button("PLAY", id="btn-play", n_clicks=0, style={
                        "backgroundColor": "rgba(0,229,255,0.1)", "border": f"1px solid {ACCENT_CYAN}",
                        "color": ACCENT_CYAN, "padding": "6px 22px", "borderRadius": "4px",
                        "cursor": "pointer", "fontSize": "12px", "fontWeight": "700",
                        "letterSpacing": "2px", "fontFamily": "monospace",
                        "boxShadow": "0 0 8px rgba(0,229,255,0.2)",
                    }),
                    html.Button("PAUSE", id="btn-pause", n_clicks=0, style={
                        "backgroundColor": "rgba(255,61,170,0.05)", "border": f"1px solid {ACCENT_MAGENTA}",
                        "color": ACCENT_MAGENTA, "padding": "6px 22px", "borderRadius": "4px",
                        "cursor": "pointer", "fontSize": "12px", "fontWeight": "700",
                        "letterSpacing": "2px", "fontFamily": "monospace",
                    }),
                    # Speed control
                    html.Div(style={"display": "flex", "alignItems": "center", "gap": "8px",
                                    "marginLeft": "12px",
                                    "backgroundColor": "rgba(61,28,168,0.15)",
                                    "padding": "4px 10px", "borderRadius": "4px",
                                    "border": "1px solid #2a2a5e"}, children=[
                        html.Span("SPEED", style={
                            "fontSize": "10px", "color": "#7a7aaa", "fontFamily": "monospace",
                            "letterSpacing": "2px",
                        }),
                        html.Button("0.5x", id="btn-speed-slow", n_clicks=0, style={
                            "backgroundColor": "transparent", "border": f"1px solid #2a2a5e",
                            "color": "#7a7aaa", "padding": "4px 10px", "borderRadius": "3px",
                            "cursor": "pointer", "fontSize": "10px", "fontFamily": "monospace",
                        }),
                        html.Button("1x", id="btn-speed-normal", n_clicks=0, style={
                            "backgroundColor": "rgba(0,229,255,0.1)", "border": f"1px solid {ACCENT_CYAN}",
                            "color": ACCENT_CYAN, "padding": "4px 10px", "borderRadius": "3px",
                            "cursor": "pointer", "fontSize": "10px", "fontFamily": "monospace",
                        }),
                        html.Button("2x", id="btn-speed-fast", n_clicks=0, style={
                            "backgroundColor": "transparent", "border": f"1px solid #2a2a5e",
                            "color": "#7a7aaa", "padding": "4px 10px", "borderRadius": "3px",
                            "cursor": "pointer", "fontSize": "10px", "fontFamily": "monospace",
                        }),
                        html.Button("4x", id="btn-speed-max", n_clicks=0, style={
                            "backgroundColor": "transparent", "border": f"1px solid #2a2a5e",
                            "color": "#7a7aaa", "padding": "4px 10px", "borderRadius": "3px",
                            "cursor": "pointer", "fontSize": "10px", "fontFamily": "monospace",
                        }),
                    ]),
                ]),
            ]),

            # Info bar — with gradient accent
            html.Div(id="info-bar", style={
                "background": "linear-gradient(90deg, rgba(61,28,168,0.15) 0%, rgba(0,229,255,0.05) 100%)",
                "padding": "8px 30px", "fontFamily": "monospace",
                "fontSize": "13px", "color": ACCENT_CYAN,
                "borderBottom": f"1px solid {GRID_COLOR}",
            }, children=_PREBUILT["info"][0]),

            # Main: 3D surface + heatmap
            html.Div(style={"display": "flex", "padding": "0 25px"}, children=[
                html.Div(style={"flex": "2", "padding": "5px", "minHeight": "450px"}, children=[
                    dcc.Graph(id="graph-3d", figure=fig_3d, style={"height": "450px"},
                              config={"displayModeBar": True, "displaylogo": False,
                                      "modeBarButtonsToRemove": ["toImage"]}),
                ]),
                html.Div(style={"flex": "1", "padding": "5px", "display": "flex", "flexDirection": "column", "gap": "5px"}, children=[
                    dcc.Graph(id="graph-hm", figure=fig_hm, style={"height": "220px"},
                              config={"displayModeBar": False}),
                    html.Div(style={
                        "background": "linear-gradient(180deg, #0c0c24 0%, #0d0221 100%)",
                        "border": f"1px solid #2a2a5e",
                        "borderRadius": "6px", "padding": "14px", "fontFamily": "monospace",
                        "fontSize": "11px", "color": TEXT_COLOR, "flex": "1",
                        "boxShadow": "0 2px 12px rgba(61,28,168,0.15)",
                    }, children=_build_stats_html()),
                ]),
            ]),

            # Backtest
            html.Div(style={"padding": "0 25px 10px"}, children=[
                dcc.Graph(id="graph-bt", figure=fig_bt, style={"height": "280px"},
                          config={"displayModeBar": False}),
            ]),

            # Stores & interval
            dcc.Store(id="store-frames", data=json.dumps(_PREBUILT)),
            dcc.Store(id="store-idx", data=0),
            dcc.Interval(id="anim-tick", interval=ANIMATION_INTERVAL_MS, disabled=True, n_intervals=0),
        ],
    )


app.layout = serve_layout


# ── Clientside Callbacks (run in browser JS — zero lag) ──────────

# Play/Pause/Speed control — all in one callback
clientside_callback(
    """
    function(playClicks, pauseClicks, slowClicks, normalClicks, fastClicks, maxClicks, currentDisabled, currentInterval) {
        const ctx = dash_clientside.callback_context;
        if (!ctx.triggered || ctx.triggered.length === 0) return [true, 150];
        const trigger = ctx.triggered[0].prop_id.split('.')[0];

        // Speed map: button id -> interval ms (lower = faster)
        const speeds = {
            'btn-speed-slow': 300,
            'btn-speed-normal': 150,
            'btn-speed-fast': 75,
            'btn-speed-max': 40
        };

        if (trigger === 'btn-play') {
            return [false, currentInterval || 150];
        } else if (trigger === 'btn-pause') {
            return [true, currentInterval || 150];
        } else if (speeds[trigger] !== undefined) {
            // Speed button pressed — update interval, keep play/pause state
            return [currentDisabled, speeds[trigger]];
        }
        return [currentDisabled, currentInterval || 150];
    }
    """,
    Output("anim-tick", "disabled"),
    Output("anim-tick", "interval"),
    Input("btn-play", "n_clicks"),
    Input("btn-pause", "n_clicks"),
    Input("btn-speed-slow", "n_clicks"),
    Input("btn-speed-normal", "n_clicks"),
    Input("btn-speed-fast", "n_clicks"),
    Input("btn-speed-max", "n_clicks"),
    State("anim-tick", "disabled"),
    State("anim-tick", "interval"),
    prevent_initial_call=True,
)

# Advance frame index
clientside_callback(
    """
    function(n, currentIdx, framesJson) {
        if (!framesJson) return 0;
        const data = JSON.parse(framesJson);
        return (currentIdx + 1) % data.n_frames;
    }
    """,
    Output("store-idx", "data"),
    Input("anim-tick", "n_intervals"),
    State("store-idx", "data"),
    State("store-frames", "data"),
    prevent_initial_call=True,
)

# Single unified animation callback — updates all charts + info via JS
# Uses Plotly.restyle for the 3D surface (only swaps z-data, preserves camera)
clientside_callback(
    """
    function(idx, framesJson, existing3dFig, existingHmFig, existingBtFig) {
        if (!framesJson || idx === undefined) {
            return [
                dash_clientside.no_update,
                dash_clientside.no_update,
                dash_clientside.no_update,
                dash_clientside.no_update
            ];
        }

        // Parse once, cache in window for subsequent calls
        if (!window._ffCache || window._ffCacheKey !== framesJson) {
            window._ffCache = JSON.parse(framesJson);
            window._ffCacheKey = framesJson;
        }
        const data = window._ffCache;

        // 3D Surface: shallow-clone figure, swap z-data only
        const fig3d = Object.assign({}, existing3dFig);
        fig3d.data = existing3dFig.data.map(function(trace, i) {
            if (i === 0) {
                return Object.assign({}, trace, {z: data.surfaces[idx]});
            }
            return trace;
        });

        // Heatmap: swap z-data only
        const hmFig = Object.assign({}, existingHmFig);
        hmFig.data = existingHmFig.data.map(function(trace, i) {
            if (i === 0) {
                return Object.assign({}, trace, {z: data.heatmaps[idx]});
            }
            return trace;
        });

        // Backtest: move vertical line (trace index 3)
        const btFig = Object.assign({}, existingBtFig);
        const dateStr = data.dates[idx];
        btFig.data = existingBtFig.data.map(function(trace, i) {
            if (i === 3) {
                return Object.assign({}, trace, {x: [dateStr, dateStr]});
            }
            return trace;
        });

        return [fig3d, hmFig, btFig, data.info[idx]];
    }
    """,
    Output("graph-3d", "figure"),
    Output("graph-hm", "figure"),
    Output("graph-bt", "figure"),
    Output("info-bar", "children"),
    Input("store-idx", "data"),
    State("store-frames", "data"),
    State("graph-3d", "figure"),
    State("graph-hm", "figure"),
    State("graph-bt", "figure"),
    prevent_initial_call=True,
)
