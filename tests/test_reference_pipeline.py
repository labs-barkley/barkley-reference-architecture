"""
tests/test_reference_pipeline.py
=================================
Reference pipeline test suite.

Runs with either:
    python -m pytest tests/
    python tests/test_reference_pipeline.py

Tests cover the four pillars of the Barkley reference architecture:
    1. Individual Baseline / ICF
    2. Multi-Resolution Temporal Binning
    3. Rate of Drift / Behavioral Drift
    4. Sovereignty of Silence / Missingness

And the DogGraph event schema.

Research demonstrator only. Synthetic data only. Not diagnostic.
"""
import os
import sys
import math
from typing import Callable

# Make the repository importable from a source checkout without installation.
# No-op when the package is installed (e.g. `pip install -e .`).
try:
    import barkley  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ── Test helpers (pure-Python, no pytest dependency required) ──────────────────

_TESTS:   list[tuple[str, Callable[[], None]]] = []
_PASSED:  list[str]                  = []
_FAILED:  list[str]                  = []


def register_test(fn):
    """Decorator to register a test function."""
    _TESTS.append((fn.__name__, fn))
    return fn


def assert_equal(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg or 'assert_equal'}: {a!r} != {b!r}")


def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg or f"Expected True, got {cond!r}")


def assert_false(cond, msg=""):
    if cond:
        raise AssertionError(msg or f"Expected False, got {cond!r}")


def assert_finite(value, msg=""):
    if not math.isfinite(value):
        raise AssertionError(msg or f"Expected finite, got {value!r}")


def assert_in_range(value, lo, hi, msg=""):
    if not (lo <= value <= hi):
        raise AssertionError(msg or f"{value} not in [{lo}, {hi}]")


# ── Pillar 1: Individual Baseline / ICF ───────────────────────────────────────

@register_test
def test_icf_initialises_from_synthetic_data():
    """ICF builds a per-dog median+MAD baseline from synthetic data."""
    from barkley.synthetic import generate_dog, to_matrix, CORE_CHANNELS
    from barkley.individual_baseline import ICF, ICF_INIT_DAYS

    df     = generate_dog(days=70, seed=42)
    matrix = to_matrix(df.iloc[:ICF_INIT_DAYS])
    icf    = ICF.initialise(list(CORE_CHANNELS), matrix)

    assert_equal(len(icf.mean),  len(CORE_CHANNELS), "ICF mean has wrong channel count")
    assert_equal(len(icf.scale), len(CORE_CHANNELS), "ICF scale has wrong channel count")
    assert_equal(icf.n_seen, ICF_INIT_DAYS)

    for ch in CORE_CHANNELS:
        assert_finite(icf.mean[ch],  f"ICF mean[{ch}] is not finite")
        assert_true(icf.scale[ch] > 0, f"ICF scale[{ch}] must be positive")


@register_test
def test_icf_zscore_returns_finite_values():
    """ICF z-scores are finite for in-distribution values."""
    from barkley.synthetic import generate_dog, to_matrix, CORE_CHANNELS
    from barkley.individual_baseline import ICF, ICF_INIT_DAYS

    df   = generate_dog(days=100, seed=0)
    mat  = to_matrix(df.iloc[:ICF_INIT_DAYS])
    icf  = ICF.initialise(list(CORE_CHANNELS), mat)

    for ch in CORE_CHANNELS:
        z = icf.zscore(ch, icf.mean[ch])        # value at baseline mean → z ≈ 0
        assert_finite(z, f"z-score for {ch} at baseline mean is not finite")
        assert_in_range(abs(z), 0.0, 0.1,       # should be very close to 0
                        f"z-score at baseline mean is unexpectedly large: {z}")


@register_test
def test_icf_zscore_nan_for_missing_value():
    """ICF returns NaN z-score for NaN input (missing data handling)."""
    from barkley.synthetic import generate_dog, to_matrix, CORE_CHANNELS
    from barkley.individual_baseline import ICF, ICF_INIT_DAYS

    df  = generate_dog(days=60, seed=1)
    mat = to_matrix(df.iloc[:ICF_INIT_DAYS])
    icf = ICF.initialise(list(CORE_CHANNELS), mat)

    z = icf.zscore(CORE_CHANNELS[0], float("nan"))
    assert_true(math.isnan(z), "NaN input should produce NaN z-score")


@register_test
def test_icf_update_shifts_baseline():
    """Slow baseline update shifts the mean in the correct direction."""
    from barkley.individual_baseline import ICF, ICF_INIT_DAYS
    import numpy as np

    channels = ["ch_a"]
    mat = np.full((ICF_INIT_DAYS, 1), 0.5)
    icf = ICF.initialise(channels, mat)

    original_mean = icf.mean["ch_a"]
    for _ in range(100):
        icf.update("ch_a", 0.9, alpha=0.01)   # push mean upward

    assert_true(icf.mean["ch_a"] > original_mean,
                "Repeated high updates should shift baseline mean up")


# ── Pillar 2: Multi-Resolution Temporal Binning ───────────────────────────────

@register_test
def test_bin_a_resamples_to_daily():
    """Bin A (circadian) resamples a daily series back to daily."""
    from barkley.synthetic import generate_dog
    from barkley.temporal_binning import bin_stream, BIN_A

    df     = generate_dog(days=30, missing_rate=0.0, seed=3)
    result = bin_stream(df, "activity_level", BIN_A)

    assert_true(len(result) == 30, f"Expected 30 daily rows, got {len(result)}")


@register_test
def test_quarterly_ks_detects_shift():
    """Bin-C KS test returns higher statistic for a shifted distribution."""
    from barkley.temporal_binning import quarterly_distribution_shift
    import numpy as np

    rng    = np.random.default_rng(7)
    stable = rng.normal(0.3, 0.04, 91).tolist()
    drifted = rng.normal(0.5, 0.04, 91).tolist()   # clear mean shift

    ks_shift   = quarterly_distribution_shift(stable, drifted)
    ks_stable  = quarterly_distribution_shift(stable, stable)

    assert_true(ks_shift["ks_stat"] > ks_stable["ks_stat"],
                "Shifted distributions should produce higher KS statistic")
    assert_true(ks_shift["ks_stat"] > 0.3,
                f"KS stat for large shift too low: {ks_shift['ks_stat']:.3f}")


@register_test
def test_aggregate_window_handles_all_nan():
    """Aggregate window returns NaN gracefully when all values are missing."""
    from barkley.temporal_binning import aggregate_window
    import numpy as np

    result = aggregate_window(np.array([float("nan"), float("nan")]))
    assert_true(math.isnan(result["mean"]),   "mean should be NaN for all-NaN window")
    assert_equal(result["n_valid"], 0)


# ── Pillar 3: Rate of Drift / Behavioral Drift ────────────────────────────────

@register_test
def test_cusum_fires_after_sustained_drift():
    """CUSUM detects a sustained positive drift before day 365."""
    from barkley.synthetic import generate_dog, to_matrix, CORE_CHANNELS
    from barkley.individual_baseline import ICF, ICF_INIT_DAYS
    from barkley.rate_of_drift import CUSUM
    import numpy as np

    df     = generate_dog(days=365, decline_start_day=200, decline_rate=0.002, seed=99)
    matrix = to_matrix(df, channels=CORE_CHANNELS)
    icf    = ICF.initialise(list(CORE_CHANNELS), matrix[:ICF_INIT_DAYS])

    SIGN = {"activity_level": -1, "sleep_fragmentation": +1,
            "restlessness_index": +1, "social_response_latency": +1,
            "routine_variability": -1}

    cusum         = CUSUM(k=0.5, h=5.0)
    alarm_fired   = False

    for i in range(ICF_INIT_DAYS, matrix.shape[0]):
        zs = [SIGN[ch] * icf.zscore(ch, matrix[i, j])
              for j, ch in enumerate(CORE_CHANNELS)
              if np.isfinite(matrix[i, j])]
        comp = float(np.mean(zs)) if zs else float("nan")
        det  = cusum.update(comp if np.isfinite(comp) else float("nan"))
        if det["alarm"]:
            alarm_fired = True
            break

    assert_true(alarm_fired, "CUSUM should fire for a strong injected decline")


@register_test
def test_cusum_no_alarm_on_stable_data():
    """CUSUM should not fire on a purely stable baseline (no injected decline)."""
    from barkley.synthetic import generate_dog, to_matrix, CORE_CHANNELS
    from barkley.individual_baseline import ICF, ICF_INIT_DAYS
    from barkley.rate_of_drift import CUSUM
    import numpy as np

    df     = generate_dog(days=200, decline_start_day=None, seed=77)
    matrix = to_matrix(df, channels=CORE_CHANNELS)
    icf    = ICF.initialise(list(CORE_CHANNELS), matrix[:ICF_INIT_DAYS])

    SIGN = {"activity_level": -1, "sleep_fragmentation": +1,
            "restlessness_index": +1, "social_response_latency": +1,
            "routine_variability": -1}

    cusum = CUSUM(k=0.5, h=5.0)
    alarm_count = 0

    for i in range(ICF_INIT_DAYS, matrix.shape[0]):
        zs = [SIGN[ch] * icf.zscore(ch, matrix[i, j])
              for j, ch in enumerate(CORE_CHANNELS)
              if np.isfinite(matrix[i, j])]
        comp = float(np.mean(zs)) if zs else float("nan")
        det  = cusum.update(comp if np.isfinite(comp) else float("nan"))
        if det["alarm"]:
            alarm_count += 1

    # With h=5.0 on stable data, the alarm rate should be very low.
    assert_true(alarm_count <= 1,
                f"CUSUM fired {alarm_count} times on stable data (expected ≤ 1)")


@register_test
def test_rate_of_drift_positive_for_declining_trend():
    """Rate of Drift OLS slope is positive when the composite z trends up."""
    from barkley.rate_of_drift import rate_of_drift
    import numpy as np

    trend = np.linspace(0.0, 3.0, 91) + np.random.default_rng(5).normal(0, 0.1, 91)
    rod   = rate_of_drift(trend)
    assert_true(rod > 0, f"Upward trend should yield positive Rate of Drift: {rod:.4f}")


# ── Pillar 4: Sovereignty of Silence / Missingness ────────────────────────────

@register_test
def test_missingness_classifies_hardware_fault():
    """A dead battery is classified as ARTEFACTUAL with high confidence."""
    from barkley.missingness import classify_missingness, SensorContext
    from barkley.schema import MissingnessType

    ctx = SensorContext(battery_ok=False)
    kind, conf = classify_missingness("restlessness_index", ctx)

    assert_equal(kind, MissingnessType.ARTEFACTUAL)
    assert_true(conf >= 0.90, f"Confidence for hardware fault too low: {conf}")


@register_test
def test_missingness_classifies_known_event():
    """A known contextual event (e.g. 'charging') is CONTEXTUAL."""
    from barkley.missingness import classify_missingness, SensorContext
    from barkley.schema import MissingnessType

    ctx = SensorContext(known_event="charging")
    kind, _ = classify_missingness("activity_level", ctx)
    assert_equal(kind, MissingnessType.CONTEXTUAL)


@register_test
def test_missingness_classifies_informative_silence():
    """On-body + corroborating context present = INFORMATIVE (silence = stability signal)."""
    from barkley.missingness import classify_missingness, SensorContext
    from barkley.schema import MissingnessType

    ctx = SensorContext(device_on_body=True, presence_confirmed=True)
    kind, conf = classify_missingness("routine_variability", ctx)

    assert_equal(kind, MissingnessType.INFORMATIVE,
                 "On-body + corroborating context should be INFORMATIVE")
    assert_true(conf >= 0.70)


# ── DogGraph event schema ──────────────────────────────────────────────────────

@register_test
def test_doggraph_event_serialises_to_json():
    """A DogGraph event round-trips through JSON without error."""
    import json
    from barkley.schema import (
        DogGraphEvent, BinSpec, BinType, Outputs, SignalLevel, utc_now_iso,
    )
    from barkley.synthetic import CORE_CHANNELS

    event = DogGraphEvent(
        dog_id    = "test_dog",
        timestamp = utc_now_iso(),
        bin       = BinSpec(BinType.BIN_A, "1D", "24h"),
        features  = {ch: 0.5 for ch in CORE_CHANNELS},
        outputs   = Outputs(signal_level=SignalLevel.NONE, research_only=True),
    )
    j = event.to_json()
    d = json.loads(j)
    assert_equal(d["dog_id"],  "test_dog")
    assert_equal(d["outputs"]["research_only"], True)


@register_test
def test_doggraph_event_validates_correctly():
    """Valid event produces no validation errors."""
    from barkley.schema import (
        DogGraphEvent, BinSpec, BinType, utc_now_iso,
    )
    from barkley.synthetic import CORE_CHANNELS

    event = DogGraphEvent(
        dog_id    = "valid_dog",
        timestamp = utc_now_iso(),
        bin       = BinSpec(BinType.BIN_C, "91D", "quarterly"),
        features  = {ch: 0.42 for ch in CORE_CHANNELS},
    )
    errors = event.validate()
    assert_equal(errors, [], f"Expected no validation errors, got: {errors}")


@register_test
def test_full_pipeline_output_matches_spec():
    """End-to-end pipeline returns the expected summary structure."""
    from examples.run_reference_pipeline import run

    out = run()
    assert_equal(out["events_generated"],       309)
    assert_equal(out["baseline_window_days"],    56)
    assert_true(out["drift_detected"],          "Drift should be detected")
    assert_true(out["missingness_classified"],  "Missingness should be classified")
    assert_true(out["research_demonstrator_only"])


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all_tests() -> None:
    print(f"\n{'─'*60}")
    print("Barkley Reference Architecture — test suite")
    print(f"{'─'*60}")

    for name, fn in _TESTS:
        try:
            fn()
            _PASSED.append(name)
            print(f"  ✓  {name}")
        except Exception as exc:
            _FAILED.append(name)
            print(f"  ✗  {name}")
            print(f"       {type(exc).__name__}: {exc}")

    print(f"{'─'*60}")
    print(f"  {len(_PASSED)} passed / {len(_FAILED)} failed / {len(_TESTS)} total")
    print(f"{'─'*60}\n")

    if _FAILED:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
