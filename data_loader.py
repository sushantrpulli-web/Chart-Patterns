"""
Data Loader
===========
Fetches and cleans OHLCV data from Yahoo Finance (yfinance).
Supports single-ticker and multi-ticker, multi-interval batch loading.

Usage
-----
    from data_loader import fetch_historical, fetch_multiple, DEFAULT_TICKERS

    df = fetch_historical("AAPL", period="5y", interval="1d")
    all_data = fetch_multiple(["AAPL", "TSLA"], ["1d", "1wk"])
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TICKERS: List[str] = [
    "SPY", "QQQ", "AAPL", "TSLA", "MSFT",
    "AMZN", "GLD", "BTC-USD", "EUR=X", "GC=F",
]

DEFAULT_INTERVALS: List[str] = ["1h", "1d", "1wk"]

# yfinance free tier: hourly data is only available for ~60 days
_HOURLY_LIMIT_DAYS = 60
_HOURLY_ALIASES = {"1h", "60m", "1hr", "60min"}


# ---------------------------------------------------------------------------
# Single ticker fetch
# ---------------------------------------------------------------------------


def fetch_historical(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for a single ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. ``"AAPL"``, ``"BTC-USD"``).
    period : str
        Time period string accepted by yfinance (e.g. ``"5y"``, ``"60d"``).
        For hourly intervals the free-tier limit is 60 days; this is
        automatically enforced regardless of what *period* is passed.
    interval : str
        Bar size.  Common values: ``"1h"``, ``"1d"``, ``"1wk"``, ``"1mo"``.
        The alias ``"1hr"`` is normalised to ``"1h"`` automatically.

    Returns
    -------
    pd.DataFrame or None
        Columns: ``[open, high, low, close, volume]`` (all lowercase).
        Index: ``DatetimeIndex``.
        The ``df.attrs["metadata"]`` dict contains:
        ``{ticker, interval, start_date, end_date, total_bars}``.
        Returns ``None`` if download fails or result is empty after cleaning.
    """
    # Normalise hourly aliases and enforce free-tier date limit
    if interval in _HOURLY_ALIASES:
        interval = "1h"
        period = f"{_HOURLY_LIMIT_DAYS}d"

    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        warnings.warn(
            f"[data_loader] Failed to download {ticker}/{interval}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if raw is None or raw.empty:
        return None

    df = raw.copy()

    # yfinance sometimes returns a MultiIndex after auto_adjust; flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                      for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    # Keep only the canonical OHLCV columns; ignore extras (e.g. "dividends")
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]

    # Drop rows where any price column is NaN
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    df.dropna(subset=price_cols, inplace=True)

    if df.empty:
        return None

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Attach metadata
    df.attrs["metadata"] = {
        "ticker": ticker,
        "interval": interval,
        "start_date": df.index[0].isoformat(),
        "end_date": df.index[-1].isoformat(),
        "total_bars": len(df),
    }

    return df


# ---------------------------------------------------------------------------
# Multi-ticker, multi-interval batch fetch
# ---------------------------------------------------------------------------


def fetch_multiple(
    tickers: Optional[List[str]] = None,
    intervals: Optional[List[str]] = None,
    period: str = "5y",
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Download OHLCV data for multiple tickers and intervals.

    Parameters
    ----------
    tickers : list[str], optional
        Defaults to :data:`DEFAULT_TICKERS`.
    intervals : list[str], optional
        Defaults to :data:`DEFAULT_INTERVALS`.
    period : str
        Passed to :func:`fetch_historical`.
        Automatically overridden to ``"60d"`` for hourly intervals.

    Returns
    -------
    dict[ticker][interval] → DataFrame
        Nested dict.  Combinations that fail to download (or return empty
        data) are silently omitted.
    """
    tickers = tickers or DEFAULT_TICKERS
    intervals = intervals or DEFAULT_INTERVALS

    data: Dict[str, Dict[str, pd.DataFrame]] = {}

    for ticker in tickers:
        data[ticker] = {}
        for interval in intervals:
            print(f"  Fetching {ticker}/{interval}...", end=" ", flush=True)
            df = fetch_historical(ticker, period=period, interval=interval)
            if df is not None and not df.empty:
                data[ticker][interval] = df
                print(f"{len(df)} bars")
            else:
                print("SKIP (no data)")

    # Drop tickers where every interval failed
    data = {t: iv_map for t, iv_map in data.items() if iv_map}

    return data
