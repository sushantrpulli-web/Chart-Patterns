"""
Pattern Backtester
===================
Measures historical forward performance of each detected pattern occurrence
and aggregates statistics across all occurrences of the same pattern type.

Metrics computed per occurrence
---------------------------------
* ``hit_target``   – Did the price reach ``target_price`` before ``stop_price``?
* ``hit_stop``     – Did the price touch ``stop_price`` before ``target_price``?
* ``mfe``          – Maximum Favourable Excursion (best unrealised gain, normalised).
* ``mae``          – Maximum Adverse Excursion (worst unrealised loss, normalised).
* ``return_pct``   – Actual close-to-close return at ``forward_bars`` out.
* ``risk_reward``  – MFE / MAE  (∞ when MAE ≈ 0).

Aggregated per pattern name
----------------------------
* ``win_rate``     – Fraction of occurrences that hit target before stop.
* ``avg_return``   – Mean ``return_pct`` across all occurrences.
* ``avg_rr``       – Mean risk-reward ratio.
* ``count``        – Number of occurrences evaluated.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from patterns.base import PatternResult


# ---------------------------------------------------------------------------
# Per-occurrence result
# ---------------------------------------------------------------------------


@dataclass
class BacktestOccurrence:
    pattern_name: str
    start_idx: int
    end_idx: int
    direction: str
    confidence: float
    entry_price: float
    target_price: float
    stop_price: float
    hit_target: bool
    hit_stop: bool
    mfe: float          # maximum favourable excursion (fraction of entry)
    mae: float          # maximum adverse excursion (fraction of entry)
    return_pct: float   # forward close-to-close return at forward_bars
    risk_reward: float  # mfe / mae


# ---------------------------------------------------------------------------
# Per-pattern aggregate
# ---------------------------------------------------------------------------


@dataclass
class PatternStats:
    pattern_name: str
    count: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    avg_rr: float = 0.0
    occurrences: List[BacktestOccurrence] = field(default_factory=list)

    def _update(self) -> None:
        n = self.count
        if n == 0:
            return
        self.win_rate = self.wins / n
        self.avg_return = float(np.mean([o.return_pct for o in self.occurrences]))
        self.avg_mfe = float(np.mean([o.mfe for o in self.occurrences]))
        self.avg_mae = float(np.mean([o.mae for o in self.occurrences]))
        rrs = [o.risk_reward for o in self.occurrences if np.isfinite(o.risk_reward)]
        self.avg_rr = float(np.mean(rrs)) if rrs else 0.0

    def summary_line(self) -> str:
        return (
            f"{self.pattern_name:<30}  "
            f"n={self.count:>4}  "
            f"WR={self.win_rate:>5.1%}  "
            f"Ret={self.avg_return:>+7.2%}  "
            f"MFE={self.avg_mfe:>6.2%}  "
            f"MAE={self.avg_mae:>6.2%}  "
            f"RR={self.avg_rr:>5.2f}"
        )


# ---------------------------------------------------------------------------
# Core backtest function (single occurrence)
# ---------------------------------------------------------------------------


def backtest_pattern(
    ohlcv_df: pd.DataFrame,
    pattern_result: PatternResult,
    forward_bars: int = 20,
) -> Optional[BacktestOccurrence]:
    """
    Measure forward performance of a single pattern occurrence.

    The pattern is assumed to complete at ``pattern_result.end_idx``.
    Entry price = close[end_idx].
    We then look at the next ``forward_bars`` closes.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        Original OHLCV DataFrame (full history, not windowed).
    pattern_result : PatternResult
        A detected pattern occurrence.
    forward_bars : int
        How many bars ahead to simulate the trade.

    Returns
    -------
    BacktestOccurrence or None if there is insufficient forward data.
    """
    df = ohlcv_df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    end = pattern_result.end_idx
    if end >= len(df) - 1:
        return None  # No forward data available

    forward_end = min(end + forward_bars, len(df) - 1)
    if forward_end <= end:
        return None

    entry_price = float(df["close"].iloc[end])
    if entry_price <= 0:
        return None

    target = pattern_result.target_price
    stop = pattern_result.stop_price
    direction = pattern_result.direction

    forward_high = df["high"].iloc[end + 1 : forward_end + 1] if "high" in df.columns else df["close"].iloc[end + 1 : forward_end + 1]
    forward_low = df["low"].iloc[end + 1 : forward_end + 1] if "low" in df.columns else df["close"].iloc[end + 1 : forward_end + 1]
    forward_close = df["close"].iloc[end + 1 : forward_end + 1]

    if len(forward_close) == 0:
        return None

    # Determine hit sequence bar-by-bar
    hit_target = False
    hit_stop = False
    for h, lo in zip(forward_high, forward_low):
        if direction == "bullish":
            if h >= target and not hit_stop:
                hit_target = True
                break
            if lo <= stop and not hit_target:
                hit_stop = True
                break
        else:  # bearish
            if lo <= target and not hit_stop:
                hit_target = True
                break
            if h >= stop and not hit_target:
                hit_stop = True
                break

    # MFE and MAE from entry
    if direction == "bullish":
        prices_vs_entry = forward_high.values - entry_price
        adverse = entry_price - forward_low.values
    else:
        prices_vs_entry = entry_price - forward_low.values
        adverse = forward_high.values - entry_price

    mfe = float(np.max(prices_vs_entry)) / entry_price if len(prices_vs_entry) > 0 else 0.0
    mae = float(np.max(adverse)) / entry_price if len(adverse) > 0 else 0.0
    mfe = max(0.0, mfe)
    mae = max(0.0, mae)

    # Actual return
    final_close = float(forward_close.iloc[-1])
    if direction == "bullish":
        return_pct = (final_close - entry_price) / entry_price
    else:
        return_pct = (entry_price - final_close) / entry_price

    rr = mfe / mae if mae > 1e-9 else float("inf")

    return BacktestOccurrence(
        pattern_name=pattern_result.pattern_name,
        start_idx=pattern_result.start_idx,
        end_idx=end,
        direction=direction,
        confidence=pattern_result.confidence,
        entry_price=entry_price,
        target_price=target,
        stop_price=stop,
        hit_target=hit_target,
        hit_stop=hit_stop,
        mfe=mfe,
        mae=mae,
        return_pct=float(return_pct),
        risk_reward=float(rr),
    )


# ---------------------------------------------------------------------------
# Aggregate across many occurrences
# ---------------------------------------------------------------------------


def run_backtest(
    ohlcv_df: pd.DataFrame,
    patterns: List[PatternResult],
    forward_bars: int = 20,
) -> Dict[str, PatternStats]:
    """
    Backtest all detected pattern occurrences and aggregate by pattern name.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
    patterns : list[PatternResult]
    forward_bars : int

    Returns
    -------
    dict mapping pattern_name → PatternStats
    """
    stats: Dict[str, PatternStats] = defaultdict(lambda: PatternStats(""))

    for r in patterns:
        occ = backtest_pattern(ohlcv_df, r, forward_bars)
        if occ is None:
            continue

        name = r.pattern_name
        if stats[name].pattern_name == "":
            stats[name].pattern_name = name

        s = stats[name]
        s.count += 1
        s.occurrences.append(occ)
        if occ.hit_target:
            s.wins += 1
        elif occ.hit_stop:
            s.losses += 1

    for s in stats.values():
        s._update()

    return dict(stats)


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def print_backtest_report(stats: Dict[str, PatternStats]) -> None:
    """Print a formatted backtest report to stdout."""
    if not stats:
        print("No backtest results.")
        return

    header = (
        f"{'Pattern':<30}  {'n':>4}  {'Win%':>6}  {'Ret':>8}  "
        f"{'MFE':>7}  {'MAE':>7}  {'RR':>6}"
    )
    print("=" * len(header))
    print("PATTERN BACKTEST REPORT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name in sorted(stats, key=lambda n: -stats[n].win_rate):
        print(stats[name].summary_line())

    print("=" * len(header))


def backtest_to_dataframe(stats: Dict[str, PatternStats]) -> pd.DataFrame:
    """Convert backtest stats dict to a pandas DataFrame for analysis."""
    rows = []
    for name, s in stats.items():
        rows.append(
            {
                "pattern_name": name,
                "count": s.count,
                "wins": s.wins,
                "losses": s.losses,
                "win_rate": s.win_rate,
                "avg_return_pct": s.avg_return * 100,
                "avg_mfe_pct": s.avg_mfe * 100,
                "avg_mae_pct": s.avg_mae * 100,
                "avg_risk_reward": s.avg_rr,
            }
        )
    return pd.DataFrame(rows).sort_values("win_rate", ascending=False).reset_index(drop=True)
