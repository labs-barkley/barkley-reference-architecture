"""
Barkley Reference Architecture
================================
A source-available research demonstrator for individual-referenced
behavioral intelligence in dogs.

Four pillars:

    1. Individual Baseline / ICF
       barkley.individual_baseline — learn each dog's own normal

    2. Multi-Resolution Temporal Binning
       barkley.temporal_binning — read behavior at multiple time scales

    3. Rate of Drift / Behavioral Drift
       barkley.rate_of_drift — detect trajectory, not snapshot

    4. Sovereignty of Silence / Missingness
       barkley.missingness — classify absence instead of erasing it

DogGraph (barkley.schema) is the memory and structuring layer that ties
these four pillars into a coherent behavioral event representation.

Research demonstrator only. Synthetic data only. Not diagnostic.
https://github.com/labs-barkley/barkley-reference-architecture
"""
from .schema import (
    DogGraphEvent, BinSpec, BinType, Context, Missingness, MissingnessType,
    BaselineReference, Outputs, SignalLevel, FEATURE_CHANNELS, utc_now_iso,
)
from .temporal_binning import (
    BinConfig, BIN_A, BIN_B, BIN_C,
    aggregate_window, nocturnal_restlessness, quarterly_distribution_shift,
)
from .missingness import (
    SensorContext, classify_missingness, encode_missingness,
)
from .individual_baseline import ICF, ICF_INIT_DAYS, baseline_loss
from .rate_of_drift import CUSUM, BOCPD_BLS, rate_of_drift

__version__ = "0.1.0"
__author__  = "Elodie Aishwarya P. Remoissenet — Barkley AI"
__license__ = "Source-Available Research License"
__email__   = "labs@getbarkley.com"

__all__ = [
    # Schema / DogGraph memory layer
    "DogGraphEvent", "BinSpec", "BinType", "Context", "Missingness",
    "MissingnessType", "BaselineReference", "Outputs", "SignalLevel",
    "FEATURE_CHANNELS", "utc_now_iso",
    # Temporal binning
    "BinConfig", "BIN_A", "BIN_B", "BIN_C",
    "aggregate_window", "nocturnal_restlessness", "quarterly_distribution_shift",
    # Missingness
    "SensorContext", "classify_missingness", "encode_missingness",
    # Individual baseline / ICF
    "ICF", "ICF_INIT_DAYS", "baseline_loss",
    # Rate of Drift
    "CUSUM", "BOCPD_BLS", "rate_of_drift",
]
