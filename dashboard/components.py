"""
Dashboard Components — Reusable layout builders for the Force Field dashboard.
"""

from dash import dcc, html
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BG_COLOR, PANEL_BG, TEXT_COLOR, ACCENT_CYAN, GRID_COLOR


def build_formula_header():
    """Math formula bar matching the video — Γ(s,f,τ) equation."""
    return html.Div(
        style={
            "backgroundColor": BG_COLOR,
            "padding": "20px 30px 10px",
            "textAlign": "center",
        },
        children=[
            dcc.Markdown(
                r"""$$\Gamma(s,f,\tau) = \underbrace{\mathcal{K}(\rho;\Phi,\rho)}_{\text{trend intensity decay}} + \underbrace{\mathcal{X}(s,f;\rho)}_{\text{factor-sector coupling}} + \underbrace{\sum_c C_c(s,f,\tau;\Omega)}_{\text{return accumulation nodes}} + \underbrace{\sum_j \mathcal{T}_j(s,f,\tau;\Psi,\pi_r)}_{\text{structural break zones}}$$""",
                mathjax=True,
                style={
                    "color": TEXT_COLOR,
                    "fontSize": "16px",
                    "margin": "0",
                },
            ),
        ],
    )


def build_title_bar():
    """Title and controls bar."""
    return html.Div(
        style={
            "backgroundColor": BG_COLOR,
            "padding": "10px 30px",
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "borderBottom": f"1px solid {GRID_COLOR}",
        },
        children=[
            html.Div([
                html.Span("NIFTY FORCE FIELD", style={
                    "fontSize": "14px",
                    "fontWeight": "700",
                    "color": ACCENT_CYAN,
                    "letterSpacing": "3px",
                    "fontFamily": "monospace",
                }),
                html.Span(" | Indian Market Quant Engine", style={
                    "fontSize": "12px",
                    "color": "#555",
                    "marginLeft": "10px",
                }),
            ]),
            html.Div(
                style={"display": "flex", "gap": "10px", "alignItems": "center"},
                children=[
                    html.Button(
                        "PLAY",
                        id="btn-play",
                        n_clicks=0,
                        style={
                            "backgroundColor": "transparent",
                            "border": f"1px solid {ACCENT_CYAN}",
                            "color": ACCENT_CYAN,
                            "padding": "6px 20px",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                            "fontSize": "12px",
                            "fontWeight": "700",
                            "letterSpacing": "2px",
                            "fontFamily": "monospace",
                        },
                    ),
                    html.Button(
                        "PAUSE",
                        id="btn-pause",
                        n_clicks=0,
                        style={
                            "backgroundColor": "transparent",
                            "border": f"1px solid #555",
                            "color": "#555",
                            "padding": "6px 20px",
                            "borderRadius": "4px",
                            "cursor": "pointer",
                            "fontSize": "12px",
                            "fontWeight": "700",
                            "letterSpacing": "2px",
                            "fontFamily": "monospace",
                        },
                    ),
                    html.Div([
                        html.Label("Speed", style={
                            "fontSize": "10px", "color": "#555",
                            "marginRight": "8px", "fontFamily": "monospace",
                        }),
                        dcc.Slider(
                            id="slider-speed",
                            min=50, max=500, step=50, value=150,
                            marks=None,
                            tooltip={"placement": "bottom"},
                        ),
                    ], style={"width": "150px"}),
                ],
            ),
        ],
    )


def build_info_bar():
    """Strategy info header that updates with animation."""
    return html.Div(
        id="info-bar",
        style={
            "backgroundColor": BG_COLOR,
            "padding": "8px 30px",
            "fontFamily": "monospace",
            "fontSize": "13px",
            "color": ACCENT_CYAN,
            "borderBottom": f"1px solid {GRID_COLOR}",
        },
        children="Loading...",
    )


def build_3d_surface_panel():
    """Main 3D force field surface plot."""
    return html.Div(
        style={
            "flex": "2",
            "padding": "5px",
            "minHeight": "450px",
        },
        children=[
            dcc.Loading(
                type="circle",
                color=ACCENT_CYAN,
                children=dcc.Graph(
                    id="graph-3d-surface",
                    style={"height": "450px"},
                    config={
                        "displayModeBar": True,
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["toImage"],
                    },
                ),
            ),
        ],
    )


def build_heatmap_panel():
    """2D force field heatmap (small panel)."""
    return html.Div(
        style={
            "flex": "1",
            "padding": "5px",
            "display": "flex",
            "flexDirection": "column",
            "gap": "5px",
        },
        children=[
            dcc.Graph(
                id="graph-heatmap",
                style={"height": "220px"},
                config={"displayModeBar": False},
            ),
            # Stats panel
            html.Div(
                id="stats-panel",
                style={
                    "backgroundColor": PANEL_BG,
                    "border": f"1px solid {GRID_COLOR}",
                    "borderRadius": "4px",
                    "padding": "12px",
                    "fontFamily": "monospace",
                    "fontSize": "11px",
                    "color": TEXT_COLOR,
                    "flex": "1",
                },
                children="Loading stats...",
            ),
        ],
    )


def build_backtest_panel():
    """Bottom backtest chart panel."""
    return html.Div(
        style={
            "padding": "5px 5px 10px",
        },
        children=[
            dcc.Graph(
                id="graph-backtest",
                style={"height": "280px"},
                config={
                    "displayModeBar": False,
                },
            ),
        ],
    )
