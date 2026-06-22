"""
Report Generator
================
Produces a fully self-contained HTML backtest report from a
:class:`~analytics.PatternAnalytics` instance.

The only external dependency at *runtime* (in the browser) is the
Plotly CDN — no Python packages are required to open the file.

Usage
-----
    from analytics import PatternAnalytics
    from report_generator import generate_html_report

    analytics = PatternAnalytics(results_df)
    generate_html_report(analytics, output_path="backtest_report.html")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

from analytics import PatternAnalytics


# ---------------------------------------------------------------------------
# Plotly chart builders (return HTML div strings)
# ---------------------------------------------------------------------------


def _win_rate_bar(summ: pd.DataFrame) -> str:
    """Horizontal bar chart of win rate by pattern, colour-coded."""
    if summ.empty:
        return "<p>No data.</p>"

    sorted_df = summ.sort_values("win_rate", ascending=True)
    colors = [
        "#e74c3c" if wr < 0.45 else "#f39c12" if wr < 0.60 else "#2ecc71"
        for wr in sorted_df["win_rate"]
    ]

    fig = go.Figure(
        go.Bar(
            x=sorted_df["win_rate"] * 100,
            y=sorted_df["pattern_name"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in sorted_df["win_rate"] * 100],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Win Rate by Pattern",
        xaxis_title="Win Rate (%)",
        yaxis_title="",
        template="plotly_dark",
        height=max(350, len(summ) * 28),
        margin=dict(l=160, r=20, t=50, b=40),
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font_color="#cdd6f4",
    )
    fig.add_vline(x=50, line_dash="dash", line_color="#6c7086", annotation_text="50%")
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


def _expectancy_bar(summ: pd.DataFrame) -> str:
    """Horizontal bar chart of expectancy by pattern."""
    if summ.empty:
        return "<p>No data.</p>"

    sorted_df = summ.sort_values("expectancy", ascending=True)
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in sorted_df["expectancy"]]

    fig = go.Figure(
        go.Bar(
            x=sorted_df["expectancy"],
            y=sorted_df["pattern_name"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}%" for v in sorted_df["expectancy"]],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Expectancy by Pattern (%)",
        xaxis_title="Expectancy (%)",
        yaxis_title="",
        template="plotly_dark",
        height=max(350, len(summ) * 28),
        margin=dict(l=160, r=20, t=50, b=40),
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font_color="#cdd6f4",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#6c7086")
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


def _confidence_scatter(results_df: pd.DataFrame) -> str:
    """Scatter plot of confidence score vs PnL% for all signals."""
    if results_df.empty:
        return "<p>No data.</p>"

    colors = results_df["hit_target"].map({True: "#2ecc71", False: "#e74c3c"})
    pattern_labels = results_df["pattern_name"]

    fig = go.Figure(
        go.Scatter(
            x=results_df["confidence"],
            y=results_df["pnl_pct"],
            mode="markers",
            marker=dict(color=colors, size=5, opacity=0.6),
            text=pattern_labels,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Confidence: %{x:.2f}<br>"
                "PnL: %{y:.2f}%<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Confidence Score vs PnL% (all signals)",
        xaxis_title="Confidence",
        yaxis_title="PnL (%)",
        template="plotly_dark",
        height=450,
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font_color="#cdd6f4",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#6c7086")
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


def _heatmap_chart(analytics: PatternAnalytics) -> str:
    """Heatmap of pattern × ticker win rate."""
    try:
        matrix = analytics.heatmap_matrix()
    except Exception:
        return "<p>Heatmap unavailable.</p>"

    if matrix.empty:
        return "<p>No data.</p>"

    z = matrix.values.tolist()
    patterns = matrix.index.tolist()
    tickers = matrix.columns.tolist()

    # Replace NaN with None for JSON serialisation
    z_clean = [[None if (isinstance(v, float) and np.isnan(v)) else round(v, 3)
                for v in row]
               for row in z]

    text = [[f"{v:.1%}" if v is not None else "n/a" for v in row] for row in z_clean]

    fig = go.Figure(
        go.Heatmap(
            z=z_clean,
            x=tickers,
            y=patterns,
            text=text,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#e74c3c"],
                [0.45, "#e74c3c"],
                [0.5, "#f39c12"],
                [0.60, "#f39c12"],
                [0.65, "#2ecc71"],
                [1.0, "#2ecc71"],
            ],
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar=dict(
                tickvals=[0, 0.45, 0.60, 1],
                ticktext=["0%", "45%", "60%", "100%"],
                tickcolor="#cdd6f4",
            ),
        )
    )
    fig.update_layout(
        title="Win Rate Heatmap: Pattern x Ticker",
        template="plotly_dark",
        height=max(400, len(patterns) * 28),
        margin=dict(l=160, r=20, t=50, b=80),
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font_color="#cdd6f4",
        xaxis_tickangle=-45,
    )
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


def _cumulative_pnl_chart(analytics: PatternAnalytics) -> str:
    """Line chart of cumulative portfolio PnL over all signals."""
    try:
        cum_pnl = analytics.cumulative_pnl()
    except Exception:
        return "<p>Cumulative PnL chart unavailable.</p>"

    if cum_pnl.empty:
        return "<p>No data.</p>"

    x = list(range(1, len(cum_pnl) + 1))
    y = cum_pnl.tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Portfolio",
            line=dict(color="#89b4fa", width=2),
            fill="tozeroy",
            fillcolor="rgba(137,180,250,0.15)",
        )
    )
    fig.add_hline(y=100, line_dash="dash", line_color="#6c7086", annotation_text="Start")
    fig.update_layout(
        title="Cumulative PnL — Trading Every Signal Equally",
        xaxis_title="Signal #",
        yaxis_title="Portfolio Value (start = 100)",
        template="plotly_dark",
        height=400,
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font_color="#cdd6f4",
    )
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


# ---------------------------------------------------------------------------
# HTML table helpers
# ---------------------------------------------------------------------------


def _color_win_rate(wr: float) -> str:
    """Return a CSS background colour string based on win rate."""
    if wr >= 0.60:
        return "background:#1a5c3a; color:#2ecc71;"
    if wr >= 0.45:
        return "background:#5c4a1a; color:#f39c12;"
    return "background:#5c1a1a; color:#e74c3c;"


def _pattern_table_html(summ: pd.DataFrame) -> str:
    """Generate a sortable HTML table for the Pattern Performance section."""
    if summ.empty:
        return "<p>No pattern data available.</p>"

    col_headers = [
        "Pattern", "Signals", "Win Rate", "Avg PnL%",
        "Avg MFE%", "Avg MAE%", "Avg R:R", "Expectancy", "Confidence",
        "Best Ticker", "Best Interval",
    ]

    rows_html = []
    for i, row in summ.iterrows():
        gold_style = "background:#3d3500;" if i < 3 else ""
        wr = row["win_rate"]
        wr_style = _color_win_rate(wr)
        rows_html.append(
            f"<tr style='{gold_style}'>"
            f"<td>{row['pattern_name']}</td>"
            f"<td>{row['total_signals']}</td>"
            f"<td style='{wr_style}'>{wr:.1%}</td>"
            f"<td>{row['avg_pnl_pct']:+.2f}%</td>"
            f"<td>{row['avg_mfe']:+.2f}%</td>"
            f"<td>{row['avg_mae']:+.2f}%</td>"
            f"<td>{row['avg_rr_ratio']:.2f}</td>"
            f"<td>{row['expectancy']:+.2f}%</td>"
            f"<td>{row['avg_confidence']:.2f}</td>"
            f"<td>{row['best_ticker']}</td>"
            f"<td>{row['best_interval']}</td>"
            f"</tr>"
        )

    headers_html = "".join(
        f"<th onclick=\"sortTable(this)\" style='cursor:pointer'>{h} &#8597;</th>"
        for h in col_headers
    )

    return f"""
<div style="overflow-x:auto;">
<table id="patternTable" class="data-table">
  <thead><tr>{headers_html}</tr></thead>
  <tbody>{''.join(rows_html)}</tbody>
</table>
</div>
"""


def _top_cards_html(top_df: pd.DataFrame) -> str:
    """Generate card layout HTML for top N patterns."""
    if top_df.empty:
        return "<p>Not enough data for top patterns (need &ge; 10 signals each).</p>"

    cards = []
    for _, row in top_df.iterrows():
        wr = row["win_rate"]
        wr_color = "#2ecc71" if wr >= 0.60 else "#f39c12" if wr >= 0.45 else "#e74c3c"
        cards.append(f"""
<div class="card">
  <h3>{row['pattern_name']}</h3>
  <div class="metric">
    <span class="label">Win Rate</span>
    <span class="value" style="color:{wr_color}">{wr:.1%}</span>
  </div>
  <div class="metric">
    <span class="label">Expectancy</span>
    <span class="value">{row['expectancy']:+.2f}%</span>
  </div>
  <div class="metric">
    <span class="label">Signals</span>
    <span class="value">{row['total_signals']}</span>
  </div>
  <div class="metric">
    <span class="label">Avg R:R</span>
    <span class="value">{row['avg_rr_ratio']:.2f}</span>
  </div>
  <div class="metric">
    <span class="label">Best Ticker</span>
    <span class="value">{row['best_ticker']}</span>
  </div>
</div>
""")

    return f"<div class='card-row'>{''.join(cards)}</div>"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_html_report(
    analytics: PatternAnalytics,
    output_path: str = "backtest_report.html",
) -> str:
    """
    Generate a fully self-contained HTML backtest report.

    Parameters
    ----------
    analytics : PatternAnalytics
        Populated analytics instance.
    output_path : str
        File path where the HTML report will be written.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    stats = analytics.overall_stats()
    summ = analytics.summary_by_pattern()
    top = analytics.top_patterns(n=5)
    df_raw = analytics._df  # used for scatter chart

    # Build chart divs
    chart_winrate = _win_rate_bar(summ)
    chart_expectancy = _expectancy_bar(summ)
    chart_scatter = _confidence_scatter(df_raw)
    chart_heatmap = _heatmap_chart(analytics)
    chart_cumulative = _cumulative_pnl_chart(analytics)

    table_html = _pattern_table_html(summ)
    cards_html = _top_cards_html(top)

    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    wr_pct = f"{stats['overall_win_rate']:.1%}"
    avg_pnl = f"{stats['overall_avg_pnl_pct']:+.2f}%"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Chart Pattern Backtest Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --bg: #1e1e2e;
    --surface: #313244;
    --surface2: #45475a;
    --text: #cdd6f4;
    --subtext: #a6adc8;
    --green: #2ecc71;
    --yellow: #f39c12;
    --red: #e74c3c;
    --blue: #89b4fa;
    --gold: #f1c40f;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.6;
  }}
  header {{
    background: var(--surface);
    padding: 24px 40px;
    border-bottom: 2px solid var(--blue);
  }}
  header h1 {{ font-size: 1.8rem; color: var(--blue); margin-bottom: 4px; }}
  header p {{ color: var(--subtext); font-size: 0.9rem; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 40px; }}
  .stats-bar {{
    display: flex; gap: 16px; flex-wrap: wrap;
    margin: 24px 0;
  }}
  .stat-chip {{
    background: var(--surface);
    border-radius: 8px;
    padding: 12px 20px;
    min-width: 160px;
    flex: 1;
  }}
  .stat-chip .label {{ color: var(--subtext); font-size: 0.8rem; text-transform: uppercase; }}
  .stat-chip .value {{ font-size: 1.4rem; font-weight: 700; color: var(--blue); }}
  section {{ margin: 36px 0; }}
  section h2 {{
    font-size: 1.2rem; color: var(--blue);
    border-left: 4px solid var(--blue);
    padding-left: 12px; margin-bottom: 16px;
  }}
  .data-table {{
    width: 100%; border-collapse: collapse;
    background: var(--surface); border-radius: 8px; overflow: hidden;
  }}
  .data-table th {{
    background: var(--surface2); padding: 10px 12px;
    text-align: left; font-size: 0.8rem; text-transform: uppercase;
    color: var(--subtext); white-space: nowrap;
  }}
  .data-table td {{ padding: 8px 12px; border-bottom: 1px solid var(--surface2); }}
  .data-table tr:last-child td {{ border-bottom: none; }}
  .data-table tr:hover {{ background: rgba(137,180,250,0.07); }}
  .card-row {{
    display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px;
  }}
  .card {{
    background: var(--surface);
    border-radius: 10px;
    padding: 18px 22px;
    flex: 1; min-width: 180px;
    border-top: 3px solid var(--gold);
  }}
  .card h3 {{ font-size: 1rem; color: var(--gold); margin-bottom: 12px; }}
  .metric {{
    display: flex; justify-content: space-between;
    margin: 6px 0; font-size: 0.85rem;
  }}
  .metric .label {{ color: var(--subtext); }}
  .metric .value {{ font-weight: 600; color: var(--text); }}
  .chart-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
    gap: 20px; margin-top: 8px;
  }}
  .chart-box {{
    background: var(--surface); border-radius: 10px;
    padding: 8px; overflow: hidden;
  }}
  .chart-full {{ margin-top: 8px; }}
  footer {{
    text-align: center; padding: 24px;
    color: var(--subtext); font-size: 0.8rem;
    border-top: 1px solid var(--surface2);
    margin-top: 48px;
  }}
</style>
</head>
<body>

<!-- ===================== HEADER ===================== -->
<header>
  <h1>Chart Pattern Backtest Report</h1>
  <p>
    Generated: {run_date} &nbsp;|&nbsp;
    Tickers: {stats['total_tickers']} &nbsp;|&nbsp;
    Intervals: {stats['total_intervals']} &nbsp;|&nbsp;
    Date range: {stats['date_range_start']} to {stats['date_range_end']}
  </p>
</header>

<div class="container">

<!-- ===================== STATS BAR ===================== -->
<div class="stats-bar">
  <div class="stat-chip">
    <div class="label">Total Signals</div>
    <div class="value">{stats['total_signals']:,}</div>
  </div>
  <div class="stat-chip">
    <div class="label">Overall Win Rate</div>
    <div class="value">{wr_pct}</div>
  </div>
  <div class="stat-chip">
    <div class="label">Avg PnL / Trade</div>
    <div class="value">{avg_pnl}</div>
  </div>
  <div class="stat-chip">
    <div class="label">Best Pattern</div>
    <div class="value" style="font-size:1rem">{stats['best_pattern']}</div>
  </div>
  <div class="stat-chip">
    <div class="label">Worst Pattern</div>
    <div class="value" style="font-size:1rem; color:var(--red)">{stats['worst_pattern']}</div>
  </div>
</div>

<!-- ===================== PERFORMANCE TABLE ===================== -->
<section>
  <h2>Pattern Performance Table</h2>
  <p style="color:var(--subtext); margin-bottom:12px; font-size:0.85rem;">
    Click any column header to sort. Top 3 rows highlighted in gold.
    <span style="background:#1a5c3a; color:#2ecc71; padding:2px 8px; border-radius:4px; margin-left:8px;">Green &ge; 60%</span>
    <span style="background:#5c4a1a; color:#f39c12; padding:2px 8px; border-radius:4px; margin-left:4px;">Yellow 45–60%</span>
    <span style="background:#5c1a1a; color:#e74c3c; padding:2px 8px; border-radius:4px; margin-left:4px;">Red &lt; 45%</span>
  </p>
  {table_html}
</section>

<!-- ===================== TOP PATTERNS ===================== -->
<section>
  <h2>Top 5 Patterns by Expectancy</h2>
  {cards_html}
</section>

<!-- ===================== CHARTS ===================== -->
<section>
  <h2>Performance Charts</h2>
  <div class="chart-grid">
    <div class="chart-box">{chart_winrate}</div>
    <div class="chart-box">{chart_expectancy}</div>
    <div class="chart-box">{chart_scatter}</div>
    <div class="chart-box">{chart_heatmap}</div>
  </div>
  <div class="chart-full chart-box" style="margin-top:20px;">
    {chart_cumulative}
  </div>
</section>

</div><!-- /container -->

<footer>
  Chart Pattern Backtest Report &bull; Generated {run_date}
</footer>

<!-- ===================== TABLE SORT JS ===================== -->
<script>
function sortTable(th) {{
  const table = th.closest('table');
  const tbody = table.querySelector('tbody');
  const colIdx = Array.from(th.parentNode.children).indexOf(th);
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const asc = th.dataset.order !== 'asc';
  th.dataset.order = asc ? 'asc' : 'desc';

  rows.sort((a, b) => {{
    const va = a.children[colIdx].textContent.replace(/[%+,]/g, '').trim();
    const vb = b.children[colIdx].textContent.replace(/[%+,]/g, '').trim();
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});

  rows.forEach(r => tbody.appendChild(r));
}}
</script>

</body>
</html>
"""

    out = Path(output_path).resolve()
    out.write_text(html, encoding="utf-8")
    print(f"Report written to: {out}")
    return str(out)
