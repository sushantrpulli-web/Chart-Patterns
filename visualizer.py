"""
Interactive Plotly Visualizer
================================
``draw_patterns()`` produces a fully interactive candlestick chart with every
detected pattern drawn as colored overlays, target/stop lines, and rich hover
annotations.

Drawing conventions
-------------------
* **Bullish** patterns  → green shapes and labels
* **Bearish** patterns  → red shapes and labels
* **Neutral** patterns  → gold/yellow shapes
* Target price          → dashed green horizontal line
* Stop price            → dashed red horizontal line
* Harmonic legs         → zigzag line X→A→B→C→D with Fibonacci labels
* Curved patterns       → smooth spline (Cup, Rounding) via plotly ``spline``
* All other patterns    → straight trendlines / polygon outlines
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from patterns.base import PatternResult


# ---------------------------------------------------------------------------
# Colour mapping
# ---------------------------------------------------------------------------

_COLOURS: Dict[str, str] = {
    "bullish": "rgba(0,200,100,0.85)",
    "bearish": "rgba(220,50,50,0.85)",
    "neutral": "rgba(220,180,0,0.85)",
    "bullish_fill": "rgba(0,200,100,0.08)",
    "bearish_fill": "rgba(220,50,50,0.08)",
    "neutral_fill": "rgba(220,180,0,0.08)",
    "target": "rgba(0,200,100,0.70)",
    "stop": "rgba(220,50,50,0.70)",
}

_LINE_WIDTH = 1.8


def _colour(direction: str) -> str:
    return _COLOURS.get(direction, _COLOURS["neutral"])


def _fill_colour(direction: str) -> str:
    return _COLOURS.get(direction + "_fill", "rgba(200,200,200,0.05)")


# ---------------------------------------------------------------------------
# X-axis helpers (support both integer-indexed and datetime-indexed DFs)
# ---------------------------------------------------------------------------


def _x_vals(df: pd.DataFrame, indices: Sequence[int]) -> List[Any]:
    """Map integer positions to actual DataFrame index labels (dates or ints)."""
    return [df.index[i] for i in indices]


def _x_val(df: pd.DataFrame, idx: int) -> Any:
    return df.index[idx]


# ---------------------------------------------------------------------------
# Pattern-specific drawing functions
# ---------------------------------------------------------------------------


def _draw_line(
    fig: go.Figure,
    df: pd.DataFrame,
    x1: int,
    y1: float,
    x2: int,
    y2: float,
    colour: str,
    name: str,
    dash: str = "solid",
    width: float = _LINE_WIDTH,
    row: int = 1,
    col: int = 1,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=_x_vals(df, [x1, x2]),
            y=[y1, y2],
            mode="lines",
            line=dict(color=colour, width=width, dash=dash),
            name=name,
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def _draw_spline(
    fig: go.Figure,
    df: pd.DataFrame,
    indices: List[int],
    prices: List[float],
    colour: str,
    name: str,
    row: int = 1,
    col: int = 1,
) -> None:
    """Draw a smooth spline through a sequence of (index, price) points."""
    fig.add_trace(
        go.Scatter(
            x=_x_vals(df, indices),
            y=prices,
            mode="lines",
            line=dict(color=colour, width=_LINE_WIDTH, shape="spline"),
            name=name,
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def _draw_horizontal(
    fig: go.Figure,
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    price: float,
    colour: str,
    label: str,
    dash: str = "dash",
    row: int = 1,
    col: int = 1,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=_x_vals(df, [start_idx, min(end_idx + 20, len(df) - 1)]),
            y=[price, price],
            mode="lines",
            line=dict(color=colour, width=1.2, dash=dash),
            name=label,
            showlegend=False,
            hoverinfo="text",
            hovertext=f"{label}: {price:.4f}",
        ),
        row=row,
        col=col,
    )


def _draw_label(
    fig: go.Figure,
    df: pd.DataFrame,
    idx: int,
    price: float,
    text: str,
    colour: str,
    row: int = 1,
    col: int = 1,
) -> None:
    fig.add_annotation(
        x=_x_val(df, idx),
        y=price,
        text=text,
        showarrow=False,
        font=dict(size=9, color=colour),
        bgcolor="rgba(0,0,0,0.55)",
        bordercolor=colour,
        borderwidth=1,
        row=row,
        col=col,
    )


# ---------------------------------------------------------------------------
# Per-pattern drawing dispatch
# ---------------------------------------------------------------------------


def _draw_cup_and_handle(fig: go.Figure, df: pd.DataFrame, r: PatternResult, colour: str) -> None:
    kp = r.key_points
    ll = kp.get("left_lip", {})
    bot = kp.get("bottom", {})
    rl = kp.get("right_lip", {})
    hl = kp.get("handle_low", {})

    if ll and bot and rl:
        # Smooth arc through left_lip → bottom → right_lip
        indices = [ll["idx"], bot["idx"], rl["idx"]]
        prices = [ll["price"], bot["price"], rl["price"]]
        _draw_spline(fig, df, indices, prices, colour, r.pattern_name)

    if rl and hl:
        # Handle: sloped rectangle outline
        _draw_line(fig, df, rl["idx"], rl["price"], hl["idx"], hl["price"], colour, "handle")
        # Horizontal back to right-lip level
        _draw_horizontal(fig, df, hl["idx"], rl["idx"] + (rl["idx"] - hl["idx"]), rl["price"], colour, "breakout_level", dash="dot")


def _draw_head_and_shoulders(fig: go.Figure, df: pd.DataFrame, r: PatternResult, colour: str) -> None:
    kp = r.key_points
    pts = ["left_shoulder", "left_trough", "head", "right_trough", "right_shoulder"]
    seq = [(kp[p]["idx"], kp[p]["price"]) for p in pts if p in kp]
    if len(seq) >= 2:
        idxs = [s[0] for s in seq]
        prices = [s[1] for s in seq]
        fig.add_trace(
            go.Scatter(
                x=_x_vals(df, idxs),
                y=prices,
                mode="lines+markers",
                line=dict(color=colour, width=_LINE_WIDTH),
                marker=dict(size=6, color=colour),
                name=r.pattern_name,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    # Neckline
    nl = kp.get("neckline_left", {})
    nr = kp.get("neckline_right", {})
    if nl and nr:
        _draw_line(fig, df, nl["idx"], nl["price"], nr["idx"], nr["price"], colour, "neckline", dash="dot")


def _draw_triangle_or_channel(fig: go.Figure, df: pd.DataFrame, r: PatternResult, colour: str) -> None:
    kp = r.key_points
    # Upper trendline
    ul = kp.get("upper_left", {})
    ur = kp.get("upper_right", {})
    if ul and ur:
        _draw_line(fig, df, ul["idx"], ul["price"], ur["idx"], ur["price"], colour, "upper")

    # Lower trendline
    ll = kp.get("lower_left", kp.get("support_left", {}))
    lr = kp.get("lower_right", kp.get("support_right", {}))
    if ll and lr:
        _draw_line(fig, df, ll["idx"], ll["price"], lr["idx"], lr["price"], colour, "lower")

    # Fill between trendlines
    if ul and ur and ll and lr:
        upper_xs = _x_vals(df, [ul["idx"], ur["idx"]])
        lower_xs = _x_vals(df, [ll["idx"], lr["idx"]])
        fig.add_trace(
            go.Scatter(
                x=upper_xs + lower_xs[::-1],
                y=[ul["price"], ur["price"]] + [lr["price"], ll["price"]],
                fill="toself",
                fillcolor=_fill_colour(r.direction),
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name=r.pattern_name,
            )
        )


def _draw_flag_or_pennant(fig: go.Figure, df: pd.DataFrame, r: PatternResult, colour: str) -> None:
    kp = r.key_points
    ps = kp.get("pole_start", {})
    pe = kp.get("pole_end", {})
    fs = kp.get("flag_start", kp.get("pennant_start", {}))
    fe = kp.get("flag_end", kp.get("pennant_end", {}))

    # Pole
    if ps and pe:
        _draw_line(fig, df, ps["idx"], ps["price"], pe["idx"], pe["price"], colour, "pole", width=3.0)

    # Body (flag/pennant outline)
    if fs and fe:
        _draw_line(fig, df, fs["idx"], fs["price"], fe["idx"], fe["price"], colour, "body")


def _draw_double_triple(fig: go.Figure, df: pd.DataFrame, r: PatternResult, colour: str) -> None:
    kp = r.key_points
    # Connect all peaks with horizontal dotted line
    peaks = [(v["idx"], v["price"]) for k, v in kp.items() if k.startswith("peak")]
    if len(peaks) >= 2:
        mean_p = np.mean([p[1] for p in peaks])
        _draw_horizontal(fig, df, peaks[0][0], peaks[-1][0], float(mean_p), colour, "peak level", dash="dot")

    # Neckline
    nl = kp.get("neckline", {})
    valleys = [(v["idx"], v["price"]) for k, v in kp.items() if k.startswith("valley")]
    if nl:
        _draw_horizontal(fig, df, r.start_idx, r.end_idx, nl["price"], colour, "neckline", dash="dot")

    # Dot markers on peaks and valleys
    dots = [v for k, v in kp.items() if k.startswith("peak") or k.startswith("valley")]
    if dots:
        fig.add_trace(
            go.Scatter(
                x=_x_vals(df, [d["idx"] for d in dots]),
                y=[d["price"] for d in dots],
                mode="markers",
                marker=dict(size=8, color=colour, symbol="circle"),
                showlegend=False,
                hoverinfo="skip",
                name=r.pattern_name,
            )
        )


def _draw_rounding(fig: go.Figure, df: pd.DataFrame, r: PatternResult, colour: str) -> None:
    kp = r.key_points
    left = kp.get("left", {})
    bot = kp.get("bottom", {})
    right = kp.get("right", {})
    if left and bot and right:
        indices = [left["idx"], bot["idx"], right["idx"]]
        prices = [left["price"], bot["price"], right["price"]]
        _draw_spline(fig, df, indices, prices, colour, r.pattern_name)


def _draw_v_pattern(fig: go.Figure, df: pd.DataFrame, r: PatternResult, colour: str) -> None:
    kp = r.key_points
    ds = kp.get("decline_start", {})
    vt = kp.get("vertex", {})
    re = kp.get("rally_end", {})
    if ds and vt and re:
        fig.add_trace(
            go.Scatter(
                x=_x_vals(df, [ds["idx"], vt["idx"], re["idx"]]),
                y=[ds["price"], vt["price"], re["price"]],
                mode="lines+markers",
                line=dict(color=colour, width=_LINE_WIDTH + 0.5),
                marker=dict(size=7, color=colour),
                showlegend=False,
                hoverinfo="skip",
                name=r.pattern_name,
            )
        )


def _draw_harmonic(fig: go.Figure, df: pd.DataFrame, r: PatternResult, colour: str) -> None:
    kp = r.key_points
    pts_order = ["X", "A", "B", "C", "D"]
    pts = [(p, kp[p]) for p in pts_order if p in kp]

    if len(pts) >= 2:
        idxs = [pt[1]["idx"] for pt in pts]
        prices = [pt[1]["price"] for pt in pts]
        labels = [pt[0] for pt in pts]

        fig.add_trace(
            go.Scatter(
                x=_x_vals(df, idxs),
                y=prices,
                mode="lines+markers+text",
                text=labels,
                textposition="top center",
                line=dict(color=colour, width=_LINE_WIDTH),
                marker=dict(size=7, color=colour),
                showlegend=False,
                hoverinfo="skip",
                name=r.pattern_name,
            )
        )


# ---------------------------------------------------------------------------
# Generic fallback drawer
# ---------------------------------------------------------------------------


def _draw_generic(fig: go.Figure, df: pd.DataFrame, r: PatternResult, colour: str) -> None:
    kp = r.key_points
    idxs = sorted([v["idx"] for v in kp.values()])
    prices = [kp[k]["price"] for k in sorted(kp, key=lambda k: kp[k]["idx"])]
    if len(idxs) >= 2:
        fig.add_trace(
            go.Scatter(
                x=_x_vals(df, idxs),
                y=prices,
                mode="lines+markers",
                line=dict(color=colour, width=_LINE_WIDTH),
                marker=dict(size=5, color=colour),
                showlegend=False,
                hoverinfo="skip",
                name=r.pattern_name,
            )
        )


# Map pattern name fragments to drawing functions
_DRAW_DISPATCH = {
    "cup": _draw_cup_and_handle,
    "head and shoulders": _draw_head_and_shoulders,
    "inverse head and shoulders": _draw_head_and_shoulders,
    "ascending triangle": _draw_triangle_or_channel,
    "descending triangle": _draw_triangle_or_channel,
    "symmetrical triangle": _draw_triangle_or_channel,
    "rectangle": _draw_triangle_or_channel,
    "rising channel": _draw_triangle_or_channel,
    "falling channel": _draw_triangle_or_channel,
    "rising wedge": _draw_triangle_or_channel,
    "falling wedge": _draw_triangle_or_channel,
    "flag": _draw_flag_or_pennant,
    "pennant": _draw_flag_or_pennant,
    "double top": _draw_double_triple,
    "double bottom": _draw_double_triple,
    "triple top": _draw_double_triple,
    "triple bottom": _draw_double_triple,
    "rounding": _draw_rounding,
    "v-bottom": _draw_v_pattern,
    "v-top": _draw_v_pattern,
    "abcd": _draw_harmonic,
    "gartley": _draw_harmonic,
    "butterfly": _draw_harmonic,
    "bat": _draw_harmonic,
    "crab": _draw_harmonic,
}


def _get_draw_fn(pattern_name: str):
    pn = pattern_name.lower()
    for key, fn in _DRAW_DISPATCH.items():
        if key in pn:
            return fn
    return _draw_generic


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def draw_patterns(
    ohlcv_df: pd.DataFrame,
    patterns: List[PatternResult],
    title: str = "Chart Pattern Analysis",
    show_volume: bool = True,
    show_targets: bool = True,
    height: int = 800,
) -> go.Figure:
    """
    Render an interactive candlestick chart with all detected patterns overlaid.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        OHLCV data.  Must have columns: open, high, low, close.
        Optional: volume.
    patterns : list[PatternResult]
        Output from ``PatternDetector.detect_all()``.
    title : str
        Chart title.
    show_volume : bool
        If True and volume column exists, draw a volume sub-panel.
    show_targets : bool
        Draw target and stop price horizontal lines for each pattern.
    height : int
        Total figure height in pixels.

    Returns
    -------
    go.Figure
        Fully interactive Plotly figure.  Call ``.show()`` to display.
    """
    df = ohlcv_df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    has_vol = "volume" in df.columns and show_volume

    # Build subplot layout
    if has_vol:
        row_heights = [0.75, 0.25]
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.03,
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    # ---- Candlestick ----
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"] if "open" in df.columns else df["close"],
            high=df["high"] if "high" in df.columns else df["close"],
            low=df["low"] if "low" in df.columns else df["close"],
            close=df["close"],
            name="Price",
            increasing_line_color="rgba(60,179,113,0.9)",
            decreasing_line_color="rgba(205,92,92,0.9)",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # ---- Volume bars ----
    if has_vol:
        vol_colours = [
            "rgba(60,179,113,0.5)" if df["close"].iloc[i] >= (df["open"].iloc[i] if "open" in df.columns else df["close"].iloc[i])
            else "rgba(205,92,92,0.5)"
            for i in range(len(df))
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                marker_color=vol_colours,
                name="Volume",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # ---- Draw each pattern ----
    legend_added: set = set()

    for r in patterns:
        colour = _colour(r.direction)
        draw_fn = _get_draw_fn(r.pattern_name)

        # Draw pattern shape
        draw_fn(fig, df, r, colour)

        # Draw target and stop lines
        if show_targets and r.target_price > 0:
            _draw_horizontal(
                fig, df, r.end_idx, min(r.end_idx + 30, len(df) - 1),
                r.target_price, _COLOURS["target"], f"Target ({r.pattern_name})", dash="dash"
            )
        if show_targets and r.stop_price > 0:
            _draw_horizontal(
                fig, df, r.end_idx, min(r.end_idx + 30, len(df) - 1),
                r.stop_price, _COLOURS["stop"], f"Stop ({r.pattern_name})", dash="dash"
            )

        # Pattern label annotation
        label_idx = min(r.end_idx, len(df) - 1)
        label_price = df["high"].iloc[label_idx] if "high" in df.columns else df["close"].iloc[label_idx]
        annotation_text = (
            f"<b>{r.pattern_name}</b><br>"
            f"{r.direction.upper()} | {r.confidence:.0%}"
        )
        fig.add_annotation(
            x=_x_val(df, label_idx),
            y=label_price,
            text=annotation_text,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=colour,
            ax=0,
            ay=-40,
            font=dict(size=9, color="white"),
            bgcolor=colour.replace("0.85", "0.75"),
            bordercolor=colour,
            borderwidth=1,
        )

        # Legend entry (one per unique pattern name)
        if r.pattern_name not in legend_added:
            legend_added.add(r.pattern_name)
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color=colour, width=2),
                    name=f"{r.pattern_name} ({r.direction[0].upper()})",
                    showlegend=True,
                )
            )

    # ---- Layout ----
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="white")),
        template="plotly_dark",
        height=height,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            font=dict(size=9),
        ),
        margin=dict(l=60, r=180, t=60, b=40),
    )

    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.1)",
        tickfont=dict(size=10),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.05)",
        tickfont=dict(size=10),
    )

    return fig


# ---------------------------------------------------------------------------
# Convenience: pattern summary table (text)
# ---------------------------------------------------------------------------


def pattern_table(patterns: List[PatternResult]) -> str:
    """Return a formatted ASCII table of all detected patterns."""
    if not patterns:
        return "No patterns detected."
    header = f"{'#':>3}  {'Pattern':<28}  {'Type':<12}  {'Dir':<8}  {'Bars':>12}  {'Conf':>6}  {'Target':>10}  {'Stop':>10}"
    sep = "-" * len(header)
    rows = [header, sep]
    for i, r in enumerate(patterns, 1):
        rows.append(
            f"{i:>3}  {r.pattern_name:<28}  {r.pattern_type:<12}  {r.direction:<8}  "
            f"{str(r.start_idx) + '–' + str(r.end_idx):>12}  {r.confidence:>5.0%}  "
            f"{r.target_price:>10.4f}  {r.stop_price:>10.4f}"
        )
    return "\n".join(rows)
