"""
barkley.missingness
===================
Pillar 4: Sovereignty of Silence / Missingness Classification.

Standard data pipelines treat gaps as noise to be deleted or imputed. This is
wrong for behavioral monitoring systems, where absence is often *informative*.

The Missing Data Paradox:
    In behavioral monitoring, a prolonged absence of anomaly signals is itself a
    meaningful observation. Deleting it is not neutral — it is an active loss
    of information.

This module implements a three-category taxonomy of behavioral data absence:

    ARTEFACTUAL — Hardware fault, battery dead, corrupt stream.
        Conservative default when no context explains the gap.
        → Handle defensively; do not infer behavioral state.

    CONTEXTUAL — A known external cause explains the absence.
        Device removed for swimming, grooming, charging; travel.
        → Contextual interpolation is appropriate.

    INFORMATIVE — Device on-body, context confirms presence, no anomaly.
        The dog is present and still. Silence IS the signal.
        A sustained period here is a *positive stability signal*.
        → Encode as a structured feature; do not impute away.

This taxonomy transforms a data-quality problem into a modeling resource.

Research demonstrator only. Not diagnostic. Not production.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schema import MissingnessType


# ---------------------------------------------------------------------------
# Sensor context
# ---------------------------------------------------------------------------

_CONTEXTUAL_EVENTS: frozenset[str] = frozenset({
    "swimming", "charging", "grooming", "bath", "vet_visit",
    "travel", "boarding", "device_removal",
})


@dataclass
class SensorContext:
    """Side-channel evidence available when a primary value is absent.

    Attributes
    ----------
    battery_ok:
        False if a low-battery or hardware fault was logged.
    device_on_body:
        Whether a capacitive / thermal on-body sensor confirms contact.
    presence_confirmed:
        True if available context confirms the dog is present and still.
        None if no corroborating context is available.
    known_event:
        A named external event explaining the absence, e.g. "charging".
    """
    battery_ok:         bool          = True
    device_on_body:     bool          = True
    presence_confirmed: Optional[bool] = None
    known_event:        Optional[str]  = None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_missingness(
    channel: str,
    ctx:     SensorContext,
) -> tuple[MissingnessType, float]:
    """Classify a missing behavioral value into the three-way taxonomy.

    Decision order (most certain evidence first):
        1. Hardware fault evidence  → ARTEFACTUAL
        2. Known external context   → CONTEXTUAL
        3. Off-body removal         → CONTEXTUAL
        4. On-body + presence confirmed → INFORMATIVE (silence = stability signal)
        5. Otherwise               → ARTEFACTUAL (conservative default)

    Parameters
    ----------
    channel:
        Name of the channel with the missing value (informational only).
    ctx:
        SensorContext with available side-channel evidence.

    Returns
    -------
    (MissingnessType, confidence)
        confidence ∈ [0, 1] reflects confidence in the *classification of
        the absence* — NOT in any behavioral inference.
    """
    if not ctx.battery_ok:
        return MissingnessType.ARTEFACTUAL, 0.95

    if ctx.known_event in _CONTEXTUAL_EVENTS:
        return MissingnessType.CONTEXTUAL, 0.90

    if not ctx.device_on_body:
        return MissingnessType.CONTEXTUAL, 0.80

    if ctx.device_on_body and ctx.presence_confirmed is True:
        # On-body, presence confirmed and still → meaningful stillness.
        return MissingnessType.INFORMATIVE, 0.78

    # Conservative default when context is insufficient.
    return MissingnessType.ARTEFACTUAL, 0.55


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode_missingness(
    feature_vector:     dict[str, float],
    missingness_report: dict[str, MissingnessType],
) -> dict[str, float]:
    """Add missingness-type indicator features to a feature vector.

    Rather than imputing, we encode *why* each channel is missing as a
    structured binary feature. This allows downstream models to condition
    on the type of absence.

    Parameters
    ----------
    feature_vector:
        Existing feature dictionary (modified in-place and returned).
    missingness_report:
        Mapping from channel name to its classified MissingnessType.

    Returns
    -------
    Updated feature vector with added missingness indicator features.
    """
    for ch, mtype in missingness_report.items():
        feature_vector[f"{ch}__missing_artefactual"]  = float(
            mtype == MissingnessType.ARTEFACTUAL)
        feature_vector[f"{ch}__missing_contextual"]   = float(
            mtype == MissingnessType.CONTEXTUAL)
        feature_vector[f"{ch}__missing_informative"]  = float(
            mtype == MissingnessType.INFORMATIVE)
    return feature_vector
