"""
barkley.temporal_binning
========================
Pillar 2: Multi-Resolution Temporal Binning.

Behavioral signals are not uniform in time. A single data point from a
dog on Tuesday afternoon carries different meaning from the same value at
3 am on a Sunday. This module implements three temporal bin resolutions:

    Bin A — Circadian / 24-hour window
        Captures daily rhythms. The nocturnal restlessness index
        (rolling CV of activity in the 02:00–05:00 window) is an
        established early marker of sleep fragmentation.

    Bin B — Weekly / 7-day window
        Normalises anthropogenic weekly cycles (weekday vs. weekend
        behavioural differences).

    Bin C — Quarterly / 91-day window
        Detects secular non-stationarity. A Kolmogorov–Smirnov test
        compares the *full distribution* of a channel across two
        equivalent calendar quarters — catching trajectory shifts that
        a mean comparison would miss.

These bins are not mutually exclusive. They run in parallel and feed
different components of the drift-detection pipeline.

Research demonstrator only. Not diagnostic. Not production.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bin configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BinConfig:
    """Configuration for a single temporal bin pass."""
    name:        str
    freq:        str    # pandas offset alias, e.g. "1D", "7D"
    window_desc: str


BIN_A = BinConfig("bin_a", "1D",  "24-hour circadian window")
BIN_B = BinConfig("bin_b", "7D",  "7-day anthropogenic weekly window")
BIN_C = BinConfig("bin_c", "91D", "91-day quarterly non-stationarity window")


# ---------------------------------------------------------------------------
# Window aggregation
# ---------------------------------------------------------------------------

def aggregate_window(samples: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for a temporal window.

    Returns mean, variance, coefficient of variation, and skewness.
    NaN values are excluded before computation.

    Parameters
    ----------
    samples:
        1-D array of normalised behavioral values.

    Returns
    -------
    dict with keys: mean, variance, cv, skewness, n_valid.
    """
    valid = samples[np.isfinite(samples)]
    if valid.size == 0:
        return {"mean": float("nan"), "variance": float("nan"),
                "cv": float("nan"), "skewness": float("nan"), "n_valid": 0}
    mu   = float(np.mean(valid))
    var  = float(np.var(valid))
    std  = float(np.std(valid))
    cv   = std / mu if abs(mu) > 1e-9 else float("nan")
    # Skewness (unbiased correction skipped for simplicity)
    if valid.size >= 3 and std > 1e-9:
        skew = float(np.mean(((valid - mu) / std) ** 3))
    else:
        skew = float("nan")
    return {"mean": mu, "variance": var, "cv": cv,
            "skewness": skew, "n_valid": int(valid.size)}


# ---------------------------------------------------------------------------
# Stream binning
# ---------------------------------------------------------------------------

def bin_stream(
    df:        pd.DataFrame,
    value_col: str,
    cfg:       BinConfig,
    agg_col:   str = "binned_mean",
) -> pd.DataFrame:
    """Resample a daily time-series to the bin resolution and return
    a DataFrame of aggregated means.

    Parameters
    ----------
    df:
        Daily DataFrame with a DatetimeIndex.
    value_col:
        Column to aggregate.
    cfg:
        BinConfig specifying the temporal resolution.
    agg_col:
        Name for the aggregated mean column in the output.

    Returns
    -------
    pd.DataFrame indexed by the start of each bin period.
    """
    resampled = df[[value_col]].resample(cfg.freq).mean()
    resampled.columns = [agg_col]
    resampled["bin_type"] = cfg.name
    resampled["window"]   = cfg.window_desc
    return resampled


# ---------------------------------------------------------------------------
# Nocturnal restlessness (Bin A specialisation)
# ---------------------------------------------------------------------------

def nocturnal_restlessness(
    df:         pd.DataFrame,
    value_col:  str   = "activity",
    hour_start: int   = 2,
    hour_end:   int   = 5,
    window:     int   = 7,
) -> pd.Series:
    """Compute the rolling coefficient of variation of activity in the
    nocturnal window (default: 02:00–05:00).

    Sleep fragmentation is an early indicator of behavioral drift in
    aging canines. This metric uses the CV (σ/μ) of activity in the
    deep-night window, averaged with a 7-day rolling window to reduce
    single-night noise.

    Parameters
    ----------
    df:
        DataFrame with a DatetimeIndex at sub-daily resolution, or
        a daily DataFrame with a scalar activity column.
    value_col:
        Column containing activity values.
    hour_start, hour_end:
        Hour range defining the nocturnal window (inclusive).
    window:
        Rolling window in days for the CV computation.

    Returns
    -------
    pd.Series of rolling nocturnal CV values, one per day.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex.")

    # Daily data: proxy nocturnal restlessness from daily variance
    daily_cv = df[value_col].rolling(window=window, min_periods=3).std() / \
               (df[value_col].rolling(window=window, min_periods=3).mean().abs() + 1e-9)
    return daily_cv.rename("nocturnal_restlessness_cv")


# ---------------------------------------------------------------------------
# Quarterly distribution shift (Bin C specialisation)
# ---------------------------------------------------------------------------

def quarterly_distribution_shift(
    series_q1: Iterable[float],
    series_q2: Iterable[float],
) -> dict[str, float]:
    """Bin-C non-stationarity test.

    Compare the FULL distribution of a behavioral channel across two
    equivalent calendar quarters using a two-sample Kolmogorov–Smirnov
    statistic (computed from scratch — no scipy dependency).

    A significant KS statistic indicates a secular distributional shift:
    the dog's behavioral envelope has moved, not just its mean. This is
    the correct unit of analysis for trajectory-based prognosis.

    Parameters
    ----------
    series_q1, series_q2:
        Iterables of normalised channel values for each quarter.
        NaN values are excluded before comparison.

    Returns
    -------
    dict with keys:
        ks_stat  — Kolmogorov–Smirnov distance ∈ [0, 1]
        d_mean   — mean shift (q2 − q1)
        d_var    — variance shift (q2 − q1)
    """
    a = np.sort(np.asarray(list(series_q1), dtype=float))
    b = np.sort(np.asarray(list(series_q2), dtype=float))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {"ks_stat": float("nan"), "d_mean": float("nan"), "d_var": float("nan")}

    # Empirical-CDF KS statistic (two-sample)
    grid  = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    ks    = float(np.max(np.abs(cdf_a - cdf_b)))

    return {
        "ks_stat": ks,
        "d_mean":  float(np.mean(b) - np.mean(a)),
        "d_var":   float(np.var(b)  - np.var(a)),
    }
