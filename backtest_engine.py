"""
Backtest Engine
================
Multi-asset, multi-timeframe backtesting engine built on top of the
existing ``backtester.backtest_pattern()`` function.

``BacktestEngine.run_all(data_dict)`` iterates over every
(ticker, interval) combination returned by ``data_loader.fetch_multiple()``,
detects all chart patterns, and measures their forward performance.

The output is a flat ``pd.DataFrame`` — the master results table — where
each row represents one detected pattern occurrence with full trade metrics.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtester import backtest_pattern
from detector import PatternDetector
from patterns.base import PatternResult


# ---------------------------------------------------------------------------
# BacktestRecord — one row in the master results table
# ---------------------------------------------------------------------------


@dataclass
class BacktestRecord:
    """Full metrics for one detected pattern occurrence."""

    ticker: str
    interval: str
    pattern_name: str
    direction: str               # "bullish" / "bearish" / "neutral"
    confidence: float            # detector confidence [0, 1]
    start_date: datetime         # date of first bar of the pattern
    end_date: datetime           # date of completion bar (entry date)
    entry_price: float           # close price at pattern completion
    target_price: float          # projected target from PatternResult
    stop_price: float            # projected stop from PatternResult
    risk_reward_ratio: float     # |reward| / |risk| (signed — positive = valid trade)
    outcome: str                 # "win" / "loss" / "open" (neither hit within forward_bars)
    pnl_pct: float               # close-to-close % return over forward_bars
    mfe_pct: float               # max favourable excursion %
    mae_pct: float               # max adverse excursion %
    hit_target: bool             # price reached target before stop
    hit_stop: bool               # price hit stop before target
    bars_to_resolution: int      # bars until first of (target, stop) hit; forward_bars if open


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _calc_rr(entry: float, target: float, stop: float, direction: str) -> float:
    """
    Risk-reward ratio = |target - entry| / |entry - stop|.
    Positive means target is further from entry than stop is.
    Returns 0.0 when risk is negligible (stop == entry).
    """
    if direction == "bullish":
        reward = target - entry
        risk = entry - stop
    else:
        reward = entry - target
        risk = stop - entry

    if abs(risk) < 1e-9:
        return 0.0
    return reward / risk


def _bars_to_resolution(
    ohlcv_df: pd.DataFrame,
    end_idx: int,
    target: float,
    stop: float,
    direction: str,
    forward_bars: int,
) -> int:
    """
    Scan forward bar-by-bar and return how many bars after *end_idx* it
    took to touch *target* or *stop* (whichever comes first).
    Returns *forward_bars* if neither is hit within the window.
    """
    hi = ohlcv_df["high"].values if "high" in ohlcv_df.columns else ohlcv_df["close"].values
    lo = ohlcv_df["low"].values if "low" in ohlcv_df.columns else ohlcv_df["close"].values

    limit = min(end_idx + forward_bars, len(ohlcv_df) - 1)

    for bar in range(end_idx + 1, limit + 1):
        h, l = float(hi[bar]), float(lo[bar])
        if direction == "bullish":
            if h >= target or l <= stop:
                return bar - end_idx
        else:
            if l <= target or h >= stop:
                return bar - end_idx

    return forward_bars


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """
    Multi-asset, multi-timeframe backtesting engine.

    Detects chart patterns on each (ticker, interval) combination and
    measures forward performance using the existing
    ``backtester.backtest_pattern()`` function.

    Parameters
    ----------
    forward_bars : int
        How many bars ahead to simulate each trade.
    min_confidence : float
        Only patterns with ``confidence >= min_confidence`` are backtested.
    max_patterns_per_run : int
        Safety cap on the number of patterns evaluated per (ticker, interval).
        The highest-confidence patterns are evaluated first.

    Examples
    --------
    >>> from data_loader import fetch_multiple
    >>> from backtest_engine import BacktestEngine
    >>> data = fetch_multiple(["AAPL", "SPY"], ["1d"])
    >>> engine = BacktestEngine(forward_bars=20, min_confidence=0.5)
    >>> results_df = engine.run_all(data)
    """

    def __init__(
        self,
        forward_bars: int = 20,
        min_confidence: float = 0.5,
        max_patterns_per_run: int = 500,
    ) -> None:
        self.forward_bars = forward_bars
        self.min_confidence = min_confidence
        self.max_patterns_per_run = max_patterns_per_run

        # PatternDetector is shared and reused across all runs
        self._detector = PatternDetector(
            min_confidence=min_confidence,
            lookback=0,  # use the full dataframe for backtesting
        )

    # ------------------------------------------------------------------

    def run_single(
        self,
        ticker: str,
        interval: str,
        ohlcv_df: pd.DataFrame,
    ) -> List[BacktestRecord]:
        """
        Detect patterns in *ohlcv_df* and backtest each occurrence.

        Parameters
        ----------
        ticker : str
        interval : str
        ohlcv_df : pd.DataFrame
            Full OHLCV history with DatetimeIndex.

        Returns
        -------
        list[BacktestRecord]
        """
        # Detect
        try:
            patterns: List[PatternResult] = self._detector.detect_all(
                ohlcv_df, ticker=ticker, timeframe=interval
            )
        except Exception as exc:
            warnings.warn(
                f"[BacktestEngine] detect_all failed for {ticker}/{interval}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return []

        # Filter and cap
        patterns = [p for p in patterns if p.confidence >= self.min_confidence]
        # Sort highest-confidence first so the cap keeps the best signals
        patterns.sort(key=lambda p: -p.confidence)
        patterns = patterns[: self.max_patterns_per_run]

        # Map integer iloc positions → datetime index values
        idx_to_date = list(ohlcv_df.index)

        records: List[BacktestRecord] = []

        for pat in patterns:
            # Call the existing backtester
            occ = backtest_pattern(ohlcv_df, pat, self.forward_bars)
            if occ is None:
                continue  # insufficient forward data

            # Date lookup
            s_idx, e_idx = pat.start_idx, pat.end_idx
            if s_idx >= len(idx_to_date) or e_idx >= len(idx_to_date):
                continue
            start_date = idx_to_date[s_idx]
            end_date = idx_to_date[e_idx]

            # Outcome label
            if occ.hit_target:
                outcome = "win"
            elif occ.hit_stop:
                outcome = "loss"
            else:
                outcome = "open"

            rr = _calc_rr(occ.entry_price, pat.target_price, pat.stop_price, pat.direction)
            bars_res = _bars_to_resolution(
                ohlcv_df, e_idx,
                pat.target_price, pat.stop_price,
                pat.direction, self.forward_bars,
            )

            records.append(
                BacktestRecord(
                    ticker=ticker,
                    interval=interval,
                    pattern_name=pat.pattern_name,
                    direction=pat.direction,
                    confidence=pat.confidence,
                    start_date=start_date,
                    end_date=end_date,
                    entry_price=occ.entry_price,
                    target_price=pat.target_price,
                    stop_price=pat.stop_price,
                    risk_reward_ratio=round(rr, 4),
                    outcome=outcome,
                    pnl_pct=round(occ.return_pct * 100.0, 4),
                    mfe_pct=round(occ.mfe * 100.0, 4),
                    mae_pct=round(occ.mae * 100.0, 4),
                    hit_target=occ.hit_target,
                    hit_stop=occ.hit_stop,
                    bars_to_resolution=bars_res,
                )
            )

        return records

    # ------------------------------------------------------------------

    def run_all(
        self,
        data_dict: Dict[str, Dict[str, pd.DataFrame]],
    ) -> pd.DataFrame:
        """
        Run backtests across all tickers and intervals in *data_dict*.

        Parameters
        ----------
        data_dict : dict[ticker][interval] → DataFrame
            As returned by ``data_loader.fetch_multiple()``.

        Returns
        -------
        pd.DataFrame
            Master results table — one row per pattern occurrence.
            Columns mirror the fields of :class:`BacktestRecord`.
            Returns an empty DataFrame if no results are produced.
        """
        all_records: List[BacktestRecord] = []

        total_combos = sum(len(iv) for iv in data_dict.values())
        done = 0

        for ticker, intervals in data_dict.items():
            for interval, df in intervals.items():
                done += 1
                print(
                    f"[{done}/{total_combos}] {ticker}/{interval}  ({len(df)} bars) ...",
                    flush=True,
                )
                try:
                    recs = self.run_single(ticker, interval, df)
                    all_records.extend(recs)
                    print(f"  -> {len(recs)} pattern occurrences")
                except Exception as exc:
                    warnings.warn(
                        f"[BacktestEngine] run_single failed for {ticker}/{interval}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        if not all_records:
            return pd.DataFrame()

        rows = [asdict(r) for r in all_records]
        df_out = pd.DataFrame(rows)
        df_out.sort_values(["ticker", "interval", "end_date"], inplace=True)
        df_out.reset_index(drop=True, inplace=True)
        return df_out
