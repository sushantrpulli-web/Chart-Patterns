"""
Pattern Analytics
==================
Statistical analysis layer on top of the master results DataFrame
produced by ``BacktestEngine.run_all()``.

All methods return plain ``pd.DataFrame`` objects for easy downstream
processing (saving to CSV, feeding into report_generator, etc.).

Expectancy formula
------------------
    expectancy = (win_rate * avg_win_pct) - ((1 - win_rate) * avg_loss_pct)

where ``avg_win_pct`` is the mean ``pnl_pct`` of winning trades and
``avg_loss_pct`` is the absolute mean ``pnl_pct`` of losing trades.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class PatternAnalytics:
    """
    Statistical analysis of multi-asset, multi-timeframe backtest results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Master results table from ``BacktestEngine.run_all()``.
        Must not be empty and must contain at least the columns:
        ``[pattern_name, ticker, interval, hit_target, hit_stop,
          pnl_pct, mfe_pct, mae_pct, confidence, risk_reward_ratio]``.

    Raises
    ------
    ValueError
        If *results_df* is empty.
    """

    def __init__(self, results_df: pd.DataFrame) -> None:
        if results_df.empty:
            raise ValueError("results_df is empty — nothing to analyse.")
        self._df = results_df.copy()
        # Boolean win column (hit_target == True)
        self._df["is_win"] = self._df["hit_target"].astype(bool)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expectancy(group: pd.DataFrame) -> float:
        """
        expectancy = (win_rate × avg_win%) − ((1 − win_rate) × avg_loss%)

        Wins  → trades where ``hit_target`` is True.
        Losses → trades where ``hit_stop`` is True.
        ``avg_loss_pct`` is the *absolute* mean pnl of losing trades so it
        is always positive; a negative expectancy means expected net loss.
        """
        wins = group[group["hit_target"].astype(bool)]
        losses = group[group["hit_stop"].astype(bool)]
        n = len(group)
        win_rate = len(wins) / n if n > 0 else 0.0
        avg_win = float(wins["pnl_pct"].mean()) if not wins.empty else 0.0
        avg_loss = abs(float(losses["pnl_pct"].mean())) if not losses.empty else 0.0
        return (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)

    @staticmethod
    def _best_by_winrate(group: pd.DataFrame, col: str) -> str:
        """Return the value of *col* with the highest win rate in *group*."""
        sub = (
            group.groupby(col)["is_win"]
            .mean()
            .sort_values(ascending=False)
        )
        return str(sub.index[0]) if not sub.empty else ""

    # ------------------------------------------------------------------
    # Public summaries
    # ------------------------------------------------------------------

    def summary_by_pattern(self) -> pd.DataFrame:
        """
        Aggregate statistics grouped by ``pattern_name``.

        Returns
        -------
        pd.DataFrame
            Columns:
            ``pattern_name, total_signals, win_rate, avg_pnl_pct,
            avg_mfe, avg_mae, avg_rr_ratio, expectancy,
            avg_confidence, best_ticker, best_interval``

            Sorted by ``expectancy`` descending.
        """
        rows = []
        for name, grp in self._df.groupby("pattern_name"):
            rows.append({
                "pattern_name": name,
                "total_signals": len(grp),
                "win_rate": round(float(grp["is_win"].mean()), 4),
                "avg_pnl_pct": round(float(grp["pnl_pct"].mean()), 4),
                "avg_mfe": round(float(grp["mfe_pct"].mean()), 4),
                "avg_mae": round(float(grp["mae_pct"].mean()), 4),
                "avg_rr_ratio": round(float(grp["risk_reward_ratio"].mean()), 4),
                "expectancy": round(self._expectancy(grp), 4),
                "avg_confidence": round(float(grp["confidence"].mean()), 4),
                "best_ticker": self._best_by_winrate(grp, "ticker"),
                "best_interval": self._best_by_winrate(grp, "interval"),
            })

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values("expectancy", ascending=False).reset_index(drop=True)

    def summary_by_ticker(self) -> pd.DataFrame:
        """
        Aggregate statistics grouped by ``ticker``.

        Returns
        -------
        pd.DataFrame
            Columns: ``ticker, total_signals, win_rate, best_pattern, avg_pnl_pct``
            Sorted by ``win_rate`` descending.
        """
        rows = []
        for ticker, grp in self._df.groupby("ticker"):
            rows.append({
                "ticker": ticker,
                "total_signals": len(grp),
                "win_rate": round(float(grp["is_win"].mean()), 4),
                "best_pattern": self._best_by_winrate(grp, "pattern_name"),
                "avg_pnl_pct": round(float(grp["pnl_pct"].mean()), 4),
            })
        return (
            pd.DataFrame(rows)
            .sort_values("win_rate", ascending=False)
            .reset_index(drop=True)
        )

    def summary_by_interval(self) -> pd.DataFrame:
        """
        Aggregate statistics grouped by ``interval``.

        Returns
        -------
        pd.DataFrame
            Columns: ``interval, total_signals, win_rate, avg_pnl_pct``
            Sorted by ``win_rate`` descending.
        """
        rows = []
        for interval, grp in self._df.groupby("interval"):
            rows.append({
                "interval": interval,
                "total_signals": len(grp),
                "win_rate": round(float(grp["is_win"].mean()), 4),
                "avg_pnl_pct": round(float(grp["pnl_pct"].mean()), 4),
            })
        return (
            pd.DataFrame(rows)
            .sort_values("win_rate", ascending=False)
            .reset_index(drop=True)
        )

    def top_patterns(self, n: int = 5) -> pd.DataFrame:
        """
        Best *n* patterns by expectancy with at least 10 signals.

        Returns
        -------
        pd.DataFrame
            Same columns as :meth:`summary_by_pattern`.
            Empty if fewer than 10 signals exist for every pattern.
        """
        summ = self.summary_by_pattern()
        filtered = summ[summ["total_signals"] >= 10]
        return filtered.head(n).reset_index(drop=True)

    def worst_patterns(self, n: int = 5) -> pd.DataFrame:
        """
        Worst *n* patterns by expectancy with at least 10 signals.

        Returns
        -------
        pd.DataFrame
            Same columns as :meth:`summary_by_pattern`.
        """
        summ = self.summary_by_pattern()
        filtered = summ[summ["total_signals"] >= 10]
        return filtered.tail(n).sort_values("expectancy").reset_index(drop=True)

    def correlation_matrix(self) -> pd.DataFrame:
        """
        Pearson correlation matrix between key numeric columns.

        Columns included (where present):
        ``confidence, pnl_pct, mfe_pct, mae_pct, risk_reward_ratio``.

        Returns
        -------
        pd.DataFrame
            Square correlation matrix rounded to 4 decimal places.
        """
        numeric_cols = [
            "confidence", "pnl_pct", "mfe_pct", "mae_pct", "risk_reward_ratio",
        ]
        available = [c for c in numeric_cols if c in self._df.columns]
        return self._df[available].corr(method="pearson").round(4)

    def monthly_breakdown(self, pattern_name: str) -> pd.DataFrame:
        """
        Win rate and average PnL grouped by year-month for a specific pattern.

        Parameters
        ----------
        pattern_name : str
            Exact pattern name (e.g. ``"Gartley"``).

        Returns
        -------
        pd.DataFrame
            Columns: ``year_month, total_signals, win_rate, avg_pnl_pct``
            Sorted by ``year_month`` ascending.
            Empty DataFrame if *pattern_name* is not found in the results.
        """
        df = self._df[self._df["pattern_name"] == pattern_name].copy()
        if df.empty:
            return pd.DataFrame(
                columns=["year_month", "total_signals", "win_rate", "avg_pnl_pct"]
            )

        df["year_month"] = (
            pd.to_datetime(df["end_date"]).dt.to_period("M").astype(str)
        )

        rows = []
        for ym, grp in df.groupby("year_month"):
            rows.append({
                "year_month": ym,
                "total_signals": len(grp),
                "win_rate": round(float(grp["is_win"].mean()), 4),
                "avg_pnl_pct": round(float(grp["pnl_pct"].mean()), 4),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("year_month")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Convenience: win-rate heatmap matrix
    # ------------------------------------------------------------------

    def heatmap_matrix(self) -> pd.DataFrame:
        """
        Pattern × ticker win-rate matrix (values in [0, 1]).

        Rows = pattern names, columns = tickers.
        NaN where a (pattern, ticker) combination has no signals.

        Returns
        -------
        pd.DataFrame
        """
        pivot = (
            self._df.groupby(["pattern_name", "ticker"])["is_win"]
            .mean()
            .unstack(level="ticker")
        )
        return pivot.round(4)

    # ------------------------------------------------------------------
    # Convenience: cumulative PnL series
    # ------------------------------------------------------------------

    def cumulative_pnl(self) -> pd.Series:
        """
        Cumulative PnL (%) if you traded every signal with equal position size,
        ordered by ``end_date``.

        Each step adds ``pnl_pct`` of the current portfolio value.

        Returns
        -------
        pd.Series
            Index = sequential signal number (1-based).
            Values = cumulative portfolio growth (starts at 100).
        """
        sorted_df = self._df.sort_values("end_date")
        # Compound growth: each trade's return applied multiplicatively
        factors = 1.0 + sorted_df["pnl_pct"].values / 100.0
        cumulative = 100.0 * np.cumprod(factors)
        return pd.Series(cumulative, name="cumulative_pnl")

    # ------------------------------------------------------------------
    # Overall stats helper
    # ------------------------------------------------------------------

    def overall_stats(self) -> dict:
        """
        High-level summary statistics for the report header.

        Returns
        -------
        dict with keys:
            total_signals, overall_win_rate, overall_avg_pnl_pct,
            best_pattern, worst_pattern, total_tickers, total_intervals,
            date_range_start, date_range_end
        """
        df = self._df
        summ = self.summary_by_pattern()

        best_pattern = (
            summ[summ["total_signals"] >= 5]["pattern_name"].iloc[0]
            if not summ[summ["total_signals"] >= 5].empty else "N/A"
        )
        worst_pattern = (
            summ[summ["total_signals"] >= 5]["pattern_name"].iloc[-1]
            if not summ[summ["total_signals"] >= 5].empty else "N/A"
        )

        dates = pd.to_datetime(df["end_date"])
        return {
            "total_signals": len(df),
            "overall_win_rate": round(float(df["is_win"].mean()), 4),
            "overall_avg_pnl_pct": round(float(df["pnl_pct"].mean()), 4),
            "best_pattern": best_pattern,
            "worst_pattern": worst_pattern,
            "total_tickers": df["ticker"].nunique(),
            "total_intervals": df["interval"].nunique(),
            "date_range_start": str(dates.min().date()) if not dates.empty else "N/A",
            "date_range_end": str(dates.max().date()) if not dates.empty else "N/A",
        }
