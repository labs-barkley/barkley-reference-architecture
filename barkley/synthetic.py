"""
barkley.synthetic
=================
Deterministic synthetic dog data generator.

Produces a daily behavioral time-series for one synthetic individual with:
- A personal baseline (not a breed average)
- Anthropogenic weekly cycle (weekends differ)
- Controllable behavioral decline injected at a specified day
- Configurable missingness rate (blanked channel-days)

This module generates FULLY SYNTHETIC DATA. No real animals, no real sensors,
no proprietary telemetry. The purpose is to provide a reproducible, inspectable
reference implementation of the Barkley behavioral intelligence pipeline.

Research demonstrator only. Not diagnostic. Not production.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


#: The five behavioral channels used in the reference architecture.
CORE_CHANNELS: tuple[str, ...] = (
    "activity_level",
    "sleep_fragmentation",
    "restlessness_index",
    "social_response_latency",
    "routine_variability",
)


def generate_dog(
    days:               int             = 365,
    decline_start_day:  Optional[int]   = 240,
    decline_rate:       float           = 0.0015,
    missing_rate:       float           = 0.05,
    seed:               int             = 42,
) -> pd.DataFrame:
    """Generate one synthetic dog's daily behavioral time-series.

    Parameters
    ----------
    days:
        Length of the time-series in days.
    decline_start_day:
        Day index at which a slow, monotonic behavioral decline begins.
        Set to None for a stable healthy baseline throughout.
    decline_rate:
        Per-day magnitude of the injected decline (dimensionless [0, 1]).
    missing_rate:
        Fraction of channel-days blanked as NaN to exercise the
        missingness pipeline.
    seed:
        Random seed for full reproducibility.

    Returns
    -------
    pd.DataFrame
        Daily rows indexed by date. Columns: CORE_CHANNELS + bookkeeping
        columns (weekday, is_weekend, decline).

    Notes
    -----
    All values are normalised to [0, 1]. The "direction" of each channel
    with respect to a behavioral decline is encoded in the sign dictionary
    used by the drift-detection pipeline, not here.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=days, freq="1D")

    # This dog's own healthy baseline — individual, not breed-average.
    base = {
        "activity_level":          0.55,
        "sleep_fragmentation":     0.20,
        "restlessness_index":      0.25,
        "social_response_latency": 0.30,
        "routine_variability":        0.60,
    }
    # Sign of the decline effect per channel:
    # +1 → rises during decline; -1 → falls during decline.
    direction = {
        "activity_level":          -1,
        "sleep_fragmentation":     +1,
        "restlessness_index":      +1,
        "social_response_latency": +1,
        "routine_variability":        -1,
    }

    rows: list[dict] = []
    for i, d in enumerate(dates):
        weekend = 1.0 if d.weekday() >= 5 else 0.0
        decline = 0.0
        if decline_start_day is not None and i >= decline_start_day:
            decline = decline_rate * (i - decline_start_day)

        rec = {
            "date":       d,
            "weekday":    d.day_name(),
            "is_weekend": bool(weekend),
            "decline":    decline,
        }
        for ch in CORE_CHANNELS:
            val = base[ch] + direction[ch] * decline
            # Weekend modulates activity slightly
            val += 0.03 * weekend * (1 if ch == "activity_level" else 0)
            val += rng.normal(0, 0.04)        # daily observation noise
            val = float(np.clip(val, 0.0, 1.0))
            # Inject random missingness
            if rng.random() < missing_rate:
                val = float("nan")
            rec[ch] = val
        rows.append(rec)

    return pd.DataFrame(rows).set_index("date")


def to_matrix(
    df:       pd.DataFrame,
    channels: tuple[str, ...] = CORE_CHANNELS,
) -> np.ndarray:
    """Extract a (T, C) float64 matrix from a synthetic dog DataFrame.

    Missing values are preserved as NaN.
    """
    return df[list(channels)].to_numpy(dtype=float)
