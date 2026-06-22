"""
Pattern Detector Orchestrator
===============================
``PatternDetector`` runs all registered pattern detectors against an OHLCV
DataFrame, deduplicates overlapping findings by confidence score, and returns
a unified sorted list of ``PatternResult`` objects.
"""

from __future__ import annotations

from typing import List, Optional, Type

import pandas as pd

from patterns.base import ChartPattern, PatternResult
from patterns.continuation import CONTINUATION_PATTERNS
from patterns.harmonic import HARMONIC_PATTERNS
from patterns.reversal import REVERSAL_PATTERNS
from patterns.triangles import TRIANGLE_PATTERNS


# ---------------------------------------------------------------------------
# Default registry – all 29 patterns
# ---------------------------------------------------------------------------

_ALL_DETECTORS: List[ChartPattern] = (
    CONTINUATION_PATTERNS
    + TRIANGLE_PATTERNS
    + REVERSAL_PATTERNS
    + HARMONIC_PATTERNS
)


# ---------------------------------------------------------------------------
# Overlap / deduplication helpers
# ---------------------------------------------------------------------------


def _overlap_ratio(r1: PatternResult, r2: PatternResult) -> float:
    """
    Return the Jaccard overlap of two pattern bar ranges [start, end].
    Range overlap / union.
    """
    lo = max(r1.start_idx, r2.start_idx)
    hi = min(r1.end_idx, r2.end_idx)
    intersection = max(0, hi - lo + 1)
    if intersection == 0:
        return 0.0
    union = (r1.end_idx - r1.start_idx + 1) + (r2.end_idx - r2.start_idx + 1) - intersection
    return intersection / union if union > 0 else 0.0


def _deduplicate(
    results: List[PatternResult],
    overlap_threshold: float = 0.60,
) -> List[PatternResult]:
    """
    Remove lower-confidence duplicates when two patterns share >overlap_threshold
    of their bar range.  Greedy algorithm: keep higher-confidence result when
    two patterns of the SAME name overlap; for different-named patterns that
    overlap heavily, keep both (unless one is strictly dominated).
    """
    # Sort descending by confidence
    results = sorted(results, key=lambda r: -r.confidence)
    kept: List[PatternResult] = []

    for candidate in results:
        dominated = False
        for existing in kept:
            if existing.pattern_name != candidate.pattern_name:
                continue
            if _overlap_ratio(candidate, existing) > overlap_threshold:
                # Same pattern type, overlapping range → keep the higher one (already in kept)
                dominated = True
                break
        if not dominated:
            kept.append(candidate)

    return kept


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class PatternDetector:
    """
    Unified pattern detector for OHLCV price data.

    Parameters
    ----------
    detectors : list[ChartPattern], optional
        Custom list of detector instances.  Defaults to all 29 built-in patterns.
    lookback : int
        Maximum number of recent bars to analyse.  Use 0 for the full DataFrame.
    overlap_threshold : float
        Jaccard overlap above which two results of the same pattern type are
        considered duplicates (lower-confidence one is dropped).
    min_confidence : float
        Results with confidence below this value are discarded before returning.
    """

    def __init__(
        self,
        detectors: Optional[List[ChartPattern]] = None,
        lookback: int = 500,
        overlap_threshold: float = 0.60,
        min_confidence: float = 0.30,
    ) -> None:
        self.detectors: List[ChartPattern] = detectors if detectors is not None else list(_ALL_DETECTORS)
        self.lookback = lookback
        self.overlap_threshold = overlap_threshold
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------

    def detect_all(
        self,
        ohlcv_df: pd.DataFrame,
        ticker: str = "",
        timeframe: str = "",
    ) -> List[PatternResult]:
        """
        Run every registered detector on *ohlcv_df* and return deduplicated results.

        Parameters
        ----------
        ohlcv_df : pd.DataFrame
            OHLCV data.  Must contain at least a ``close`` column.
        ticker : str
            Optional ticker symbol stored in each result for reference.
        timeframe : str
            Optional timeframe label (e.g. "1d", "4h") stored in each result.

        Returns
        -------
        list[PatternResult]
            Sorted by end_idx descending (most recent first), then by confidence
            descending.
        """
        if ohlcv_df is None or len(ohlcv_df) == 0:
            return []

        # Apply lookback window
        if self.lookback > 0 and len(ohlcv_df) > self.lookback:
            window = ohlcv_df.iloc[-self.lookback :].reset_index(drop=True)
            offset = len(ohlcv_df) - self.lookback
        else:
            window = ohlcv_df.reset_index(drop=True)
            offset = 0

        all_results: List[PatternResult] = []

        for detector in self.detectors:
            try:
                found = detector.detect(window)
            except Exception as exc:
                # Log but continue – one broken detector must not kill the run
                import warnings
                warnings.warn(
                    f"[{detector.name}] raised {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            for r in found:
                # Adjust indices back to full-DataFrame space
                adjusted = PatternResult(
                    pattern_name=r.pattern_name,
                    start_idx=r.start_idx + offset,
                    end_idx=r.end_idx + offset,
                    key_points={
                        k: {"idx": v["idx"] + offset, "price": v["price"]}
                        for k, v in r.key_points.items()
                    },
                    confidence=r.confidence,
                    direction=r.direction,
                    target_price=r.target_price,
                    stop_price=r.stop_price,
                    pattern_type=r.pattern_type,
                    ticker=ticker,
                    timeframe=timeframe,
                )
                if adjusted.confidence >= self.min_confidence:
                    all_results.append(adjusted)

        # Deduplicate
        deduped = _deduplicate(all_results, self.overlap_threshold)

        # Sort: most recent first, then by confidence
        deduped.sort(key=lambda r: (-r.end_idx, -r.confidence))
        return deduped

    # ------------------------------------------------------------------

    def summary(self, results: List[PatternResult]) -> str:
        """Return a formatted multi-line summary table of all results."""
        if not results:
            return "No patterns detected."
        lines = [
            f"{'Pattern':<30} {'Type':<12} {'Dir':<8} {'Bars':>10} {'Conf':>6} {'Target':>12} {'Stop':>12}",
            "-" * 95,
        ]
        for r in results:
            lines.append(
                f"{r.pattern_name:<30} "
                f"{r.pattern_type:<12} "
                f"{r.direction:<8} "
                f"{str(r.start_idx) + '–' + str(r.end_idx):>10} "
                f"{r.confidence:>5.0%} "
                f"{r.target_price:>12.4f} "
                f"{r.stop_price:>12.4f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def filter(
        self,
        results: List[PatternResult],
        direction: Optional[str] = None,
        pattern_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
        pattern_name: Optional[str] = None,
    ) -> List[PatternResult]:
        """
        Filter a result list by any combination of attributes.

        Parameters
        ----------
        results : list[PatternResult]
        direction : str, optional  – ``"bullish"``, ``"bearish"``, or ``"neutral"``
        pattern_type : str, optional – ``"continuation"``, ``"reversal"``, or ``"harmonic"``
        min_confidence : float, optional
        pattern_name : str, optional – exact name match

        Returns
        -------
        list[PatternResult]
        """
        out = results
        if direction is not None:
            out = [r for r in out if r.direction == direction]
        if pattern_type is not None:
            out = [r for r in out if r.pattern_type == pattern_type]
        if min_confidence is not None:
            out = [r for r in out if r.confidence >= min_confidence]
        if pattern_name is not None:
            out = [r for r in out if r.pattern_name == pattern_name]
        return out
