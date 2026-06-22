"""
Chart Pattern Analysis – CLI Demo
===================================
Usage
-----
    python main.py [TICKER] [PERIOD] [INTERVAL] [--backtest] [--no-chart]

Defaults: SPY, 2y, 1d

Examples
--------
    python main.py SPY 2y 1d
    python main.py AAPL 1y 1d --backtest
    python main.py BTC-USD 6mo 1h --no-chart
"""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Technical Chart Pattern Detector & Visualiser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("ticker", nargs="?", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("period", nargs="?", default="2y", help="yfinance period (default: 2y)")
    parser.add_argument("interval", nargs="?", default="1d", help="yfinance interval (default: 1d)")
    parser.add_argument("--lookback", type=int, default=500, help="Max bars to analyse (default: 500)")
    parser.add_argument("--forward-bars", type=int, default=20, help="Forward bars for backtest (default: 20)")
    parser.add_argument("--min-confidence", type=float, default=0.05, help="Min confidence threshold (default: 0.05)")
    parser.add_argument("--backtest", action="store_true", help="Run backtest on detected patterns")
    parser.add_argument("--no-chart", action="store_true", help="Skip the interactive chart")
    parser.add_argument("--output", default="", help="Save chart to HTML file (optional)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance is not installed.  Run:  pip install yfinance")
        sys.exit(1)

    print(f"Downloading {ticker}  period={period}  interval={interval} …")
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)

    if df.empty:
        print(f"No data returned for {ticker}.  Check the ticker symbol and period.")
        sys.exit(1)

    # Normalise column names – recent yfinance returns MultiIndex like ('Close', 'TSLA')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    # Keep only OHLCV
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna()

    print(f"Loaded {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ---------------------------------------------------------------------------
# Synthetic data generator (fallback / testing)
# ---------------------------------------------------------------------------


def generate_synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with embedded patterns for offline testing.
    Uses a random-walk price model with injected cup-and-handle and double-top.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    log_returns = rng.normal(0.0005, 0.012, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    spread = close * 0.005
    high = close + rng.uniform(0, spread)
    low = close - rng.uniform(0, spread)
    open_ = close - rng.uniform(-spread / 2, spread / 2)
    volume = rng.integers(500_000, 5_000_000, size=n).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    # ---- Load data ----
    try:
        df = load_ohlcv(args.ticker, args.period, args.interval)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Download failed ({exc}).  Using synthetic data instead.")
        df = generate_synthetic_ohlcv(300)

    # ---- Detect patterns ----
    from detector import PatternDetector
    from visualizer import draw_patterns, pattern_table

    detector = PatternDetector(
        lookback=args.lookback,
        min_confidence=args.min_confidence,
    )

    print(f"\nRunning pattern detection on {len(df)} bars …")
    patterns = detector.detect_all(df, ticker=args.ticker, timeframe=args.interval)

    print(f"\nDetected {len(patterns)} pattern occurrence(s):\n")
    print(pattern_table(patterns))

    # ---- Per-direction summary ----
    bulls = detector.filter(patterns, direction="bullish")
    bears = detector.filter(patterns, direction="bearish")
    neutrals = detector.filter(patterns, direction="neutral")
    print(f"\nBullish: {len(bulls)}  |  Bearish: {len(bears)}  |  Neutral: {len(neutrals)}")

    # ---- Backtest ----
    if args.backtest:
        from backtester import run_backtest, print_backtest_report, backtest_to_dataframe

        print(f"\nRunning backtest (forward_bars={args.forward_bars}) …")
        stats = run_backtest(df, patterns, forward_bars=args.forward_bars)
        print()
        print_backtest_report(stats)

        bt_df = backtest_to_dataframe(stats)
        print(f"\nBacktest DataFrame shape: {bt_df.shape}")
        print(bt_df.to_string(index=False))

    # ---- Visualise ----
    if not args.no_chart:
        title = f"{args.ticker} – Chart Pattern Analysis ({args.period}, {args.interval})"
        fig = draw_patterns(df, patterns, title=title)

        if args.output:
            fig.write_html(args.output)
            print(f"\nChart saved to {args.output}")
        else:
            print("\nLaunching interactive chart …")
            fig.show()


if __name__ == "__main__":
    main()
