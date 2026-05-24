"""
barkley.schema
==============
DogGraph-compatible event schema for individual-referenced behavioral intelligence.

The DogGraph is the memory and structuring layer of the Barkley reference
architecture. It is not the full system — it is the substrate that ties the
four research pillars together:

  1. Individual Baseline / ICF
  2. Multi-Resolution Temporal Binning
  3. Rate of Drift / Behavioral Drift
  4. Sovereignty of Silence / Missingness

A DogGraph event captures one temporal slice of a dog's behavioral signal:
the input features, the bin context, the baseline reference, the missingness
encoding, and the research output. It is a structured, machine-readable unit
of behavioral memory — a research data structure, nothing more.

Research demonstrator only. Not diagnostic. Not production.
labs@getbarkley.com | https://github.com/labs-barkley/barkley-reference-architecture
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _enum_default(obj):
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Not serialisable: {type(obj)}")


# ---------------------------------------------------------------------------
# Feature channels
# ---------------------------------------------------------------------------

#: The five behavioral channels in the reference architecture.
FEATURE_CHANNELS: tuple[str, ...] = (
    "activity_level",           # IMU-derived rest / activity
    "sleep_fragmentation",      # nocturnal restlessness index
    "restlessness_index",       # circadian deviation from own norm
    "social_response_latency",  # owner-interaction engagement signal
    "routine_variability",      # variability of daily routine / context
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BinType(str, Enum):
    """Multi-resolution temporal bin level."""
    BIN_A = "bin_a"   # Circadian / 24-hour window
    BIN_B = "bin_b"   # Weekly / 7-day window
    BIN_C = "bin_c"   # Quarterly / 91-day window


class MissingnessType(str, Enum):
    """Three-way taxonomy for behavioral data absence.

    See: Sovereignty of Silence — barkley/missingness.py
    """
    ARTEFACTUAL  = "artefactual"   # Hardware fault; conservative default
    CONTEXTUAL   = "contextual"    # Known external cause (device off, swim, groom)
    INFORMATIVE  = "informative"   # Dog present, no anomaly — silence IS the signal


class SignalLevel(str, Enum):
    """Research signal level.

    These are *behavioral research signals* on synthetic data, not diagnoses
    and not triage. A qualified professional remains in the loop for any
    real-world action.
    """
    NONE                     = "none"
    RESEARCH_OBSERVATION     = "research_observation"      # Routine research signal
    ELEVATED_RESEARCH_SIGNAL = "elevated_research_signal"  # Sustained drift in the synthetic series


# ---------------------------------------------------------------------------
# Sub-dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BinSpec:
    """Temporal bin specification."""
    bin_type:    BinType
    resolution:  str   # e.g. "1D", "7D", "91D"
    window_desc: str   # human-readable, e.g. "24h circadian window"


@dataclass
class Context:
    """Household and environmental context modifiers."""
    weekday:           Optional[str]  = None   # e.g. "Monday"
    is_weekend:        bool           = False
    known_event:       Optional[str]  = None   # "travel", "vet_visit", "new_pet", …
    owner_present:     Optional[bool] = None
    environmental_tag: Optional[str]  = None   # "heat_wave", "fireworks", …


@dataclass
class Missingness:
    """Per-event missingness encoding."""
    n_missing:  int                    = 0
    types_seen: list[MissingnessType]  = field(default_factory=list)
    note:       Optional[str]          = None


@dataclass
class BaselineReference:
    """ICF baseline snapshot at event time."""
    icf_version:       str            = "icf_v0.1"
    init_window_days:  int            = 56
    channels_present:  list[str]      = field(default_factory=list)
    baseline_means:    dict[str, float] = field(default_factory=dict)
    baseline_scales:   dict[str, float] = field(default_factory=dict)


@dataclass
class Outputs:
    """Research outputs — behavioral research signals only.

    NOT a diagnosis. NOT a research finding about any real animal. A qualified
    professional remains in the loop for any real-world action.
    """
    behavioral_state:   Optional[str]   = None     # e.g. "resting", "pacing"
    composite_z:        Optional[float] = None     # sign-aligned composite z-score
    signal_level:        SignalLevel      = SignalLevel.NONE
    research_only:      bool            = True     # always True — research demonstrator


# ---------------------------------------------------------------------------
# Primary event dataclass
# ---------------------------------------------------------------------------

@dataclass
class DogGraphEvent:
    """One temporal slice of a dog's behavioral signal.

    The DogGraph event is the unit of behavioral memory in the Barkley
    reference architecture. It encodes features, context, baseline reference,
    missingness, and research output in a single structured record.

    All values derive from fully synthetic data in this demonstrator.
    """
    dog_id:             str
    timestamp:          str                      # ISO-8601 UTC
    bin:                BinSpec
    features:           dict[str, float]         # channel name → normalised [0, 1]
    context:            Context            = field(default_factory=Context)
    missingness:        Missingness        = field(default_factory=Missingness)
    baseline_reference: BaselineReference  = field(default_factory=BaselineReference)
    outputs:            Outputs            = field(default_factory=Outputs)

    # ----------------------------------------------------------------------- #
    # Serialisation helpers
    # ----------------------------------------------------------------------- #

    def to_json(self, **kw) -> str:
        """Serialise to JSON string."""
        return json.dumps(asdict(self), default=_enum_default, **kw)

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty → valid)."""
        errors: list[str] = []

        # timestamp must parse as ISO-8601
        try:
            datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError:
            errors.append(f"timestamp is not ISO-8601: {self.timestamp!r}")

        # feature values must be finite floats
        import math
        for k, v in self.features.items():
            if not isinstance(v, (int, float)):
                errors.append(f"feature {k!r} is not numeric")
            elif not math.isfinite(v):
                errors.append(f"feature {k!r} is not finite: {v}")

        # dog_id must be non-empty
        if not self.dog_id.strip():
            errors.append("dog_id is empty")

        return errors
