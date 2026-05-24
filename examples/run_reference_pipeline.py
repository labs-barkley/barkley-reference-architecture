"""
Barkley Reference Architecture — end-to-end reference pipeline.

Runs the full four-pillar pipeline on a single synthetic dog:

    Synthetic data
    → Temporal bins (Bin A/B/C)
    → Individual baseline (ICF)
    → Sign-aligned composite z-scores
    → Rate of Drift (OLS slope)
    → CUSUM change-point detection
    → Missingness classification
    → DogGraph-compatible event emission

Expected output
---------------
    events_generated:        309
    baseline_window_days:    56
    drift_detected:          true
    missingness_classified:  true
    ks_distribution_shift:   0.564
    research_demonstrator_only: true

Research demonstrator only. Synthetic data only. Not diagnostic.
https://github.com/labs-barkley/barkley-reference-architecture
"""
import os
import sys

# Make `barkley` importable from a source checkout without installation.
# No-op when the package is installed (e.g. `pip install -e .`).
try:
    import barkley  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from barkley.synthetic import generate_dog, to_matrix, CORE_CHANNELS
from barkley.individual_baseline import ICF, ICF_INIT_DAYS
from barkley.rate_of_drift import CUSUM, rate_of_drift
from barkley.temporal_binning import quarterly_distribution_shift
from barkley.missingness import classify_missingness, SensorContext, MissingnessType
from barkley.schema import (
    DogGraphEvent, BinSpec, BinType, Outputs, SignalLevel,
    BaselineReference, Missingness, utc_now_iso,
)

# ── Configuration ─────────────────────────────────────────────────────────────

DECLINE_DAY = 240   # Day the synthetic behavioral drift begins

# Sign convention: +1 means the channel rises during decline (worse direction)
SIGN: dict[str, int] = {
    "activity_level":          -1,
    "sleep_fragmentation":     +1,
    "restlessness_index":      +1,
    "social_response_latency": +1,
    "routine_variability":        -1,
}


def run() -> dict:
    """Execute the reference pipeline and return a summary dict."""

    # ── 1. Synthetic dog ──────────────────────────────────────────────────────
    df     = generate_dog(days=365, decline_start_day=DECLINE_DAY,
                          decline_rate=0.0015, missing_rate=0.05, seed=42)
    matrix = to_matrix(df, channels=CORE_CHANNELS)

    # ── 2. ICF — individual baseline from the dog's own first 56 days ─────────
    init_matrix = matrix[:ICF_INIT_DAYS]
    icf = ICF.initialise(list(CORE_CHANNELS), init_matrix)

    # ── 3. Stream: composite z-scores + CUSUM + DogGraph events ──────────────
    cusum         = CUSUM(k=0.5, h=5.0)
    events:  list[DogGraphEvent] = []
    drift_detected               = False

    for i in range(ICF_INIT_DAYS, matrix.shape[0]):
        row = matrix[i]

        # Sign-aligned composite z-score across all channels
        zs = []
        for j, ch in enumerate(CORE_CHANNELS):
            z = icf.zscore(ch, row[j])
            if np.isfinite(z):
                zs.append(SIGN[ch] * z)
                icf.update(ch, row[j], alpha=0.01)
        comp = float(np.mean(zs)) if zs else float("nan")

        # CUSUM on the composite (monitoring opens after ICF init window)
        det = cusum.update(comp if np.isfinite(comp) else float("nan"))
        if det["alarm"]:
            drift_detected = True

        # Missingness classification for this row
        miss_report: dict[str, MissingnessType] = {}
        miss_types:  list[MissingnessType]       = []
        for j, ch in enumerate(CORE_CHANNELS):
            if not np.isfinite(row[j]):
                ctx = SensorContext(device_on_body=True, presence_confirmed=True)
                kind, _ = classify_missingness(ch, ctx)
                miss_report[ch] = kind
                miss_types.append(kind)

        # DogGraph event
        alert = (SignalLevel.ELEVATED_RESEARCH_SIGNAL
                 if det["alarm"] else SignalLevel.RESEARCH_OBSERVATION)
        feature_dict = {
            ch: float(row[j]) if np.isfinite(row[j]) else 0.0
            for j, ch in enumerate(CORE_CHANNELS)
        }
        event = DogGraphEvent(
            dog_id    = "synthetic_dog_01",
            timestamp = utc_now_iso(),
            bin       = BinSpec(BinType.BIN_A, "1D", "24h circadian window"),
            features  = feature_dict,
            missingness = Missingness(
                n_missing  = len(miss_report),
                types_seen = miss_types,
            ),
            baseline_reference = BaselineReference(
                icf_version      = icf.version,
                init_window_days = ICF_INIT_DAYS,
                channels_present = list(CORE_CHANNELS),
            ),
            outputs = Outputs(
                composite_z = round(comp, 4) if np.isfinite(comp) else None,
                signal_level = alert,
                research_only = True,
            ),
        )
        events.append(event)

    # ── 4. Bin-C quarterly distribution shift ─────────────────────────────────
    ch     = "restlessness_index"
    q_pre  = df[ch].iloc[DECLINE_DAY - 91 : DECLINE_DAY].dropna().tolist()
    q_post = df[ch].iloc[DECLINE_DAY      : DECLINE_DAY + 91].dropna().tolist()
    ks     = quarterly_distribution_shift(q_pre, q_post)

    # ── 5. Validate a sample event ────────────────────────────────────────────
    if events:
        errs = events[-1].validate()
        if errs:
            print(f"[WARNING] Event validation errors: {errs}")

    return {
        "events_generated":        len(events),
        "baseline_window_days":    ICF_INIT_DAYS,
        "drift_detected":          drift_detected,
        "missingness_classified":  any(
            e.missingness.n_missing > 0 for e in events
        ),
        "ks_distribution_shift":   round(ks["ks_stat"], 3),
        "research_demonstrator_only": True,
    }


def main() -> None:
    results = run()
    width   = max(len(k) for k in results) + 2
    for k, v in results.items():
        val = str(v).lower() if isinstance(v, bool) else str(v)
        print(f"{k + ':':<{width}} {val}")


if __name__ == "__main__":
    main()
