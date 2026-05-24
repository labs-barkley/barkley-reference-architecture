"""
barkley.rate_of_drift
=====================
Pillar 3: Rate of Drift / Behavioral Drift.

The key insight: behavioral decline in companion animals is not an event —
it is a trajectory. A single low-activity Tuesday means nothing. A sustained,
monotonic drift away from the individual's own baseline across multiple channels
over multiple weeks is a different signal entirely.

This module implements three drift-detection components:

1. rate_of_drift()
   OLS slope of the composite z-score trajectory over a trailing window.
   Answers: how fast is this dog's behavior moving away from its own normal?

2. CUSUM (Page, 1954)
   Two-sided cumulative sum. Accumulates small, sustained deviations until
   the evidence crosses a decision threshold. Useful for detecting slow,
   persistent drifts that are invisible in daily snapshots.

3. BOCPD_BLS — Bayesian Online Change-Point Detection with Baseline-Shift
   Standard BOCPD compares forever against the original normal. Canine aging
   is an *irreversibly shifting baseline*: after a true change, the new normal
   is not the old normal. BOCPD-BLS re-centres after a confirmed change,
   so detection remains sensitive during ongoing longitudinal tracking.

All thresholds (k, h, hazard) in this module are research defaults for the
synthetic demonstrator. They have not been validated against real cohort data.
A real deployment would require independent validation against appropriate
reference labels and formal operating-point analysis.

Research demonstrator only. Not diagnostic. Not production.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Rate of Drift (trajectory slope)
# ---------------------------------------------------------------------------

def rate_of_drift(quarterly_index: np.ndarray) -> float:
    """Estimate the Rate of Drift as the OLS slope of the z-score trajectory.

    Parameters
    ----------
    quarterly_index:
        1-D array of composite z-scores over a trailing window
        (typically 91 days for Bin-C analysis).
        NaN values are excluded before fitting.

    Returns
    -------
    OLS slope (σ / day). Positive → increasing drift, negative → stable/improving.
    """
    z = np.asarray(quarterly_index, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < 2:
        return float("nan")
    t    = np.arange(z.size, dtype=float)
    t_c  = t - t.mean()
    z_c  = z - z.mean()
    denom = float(np.dot(t_c, t_c))
    return float(np.dot(t_c, z_c) / denom) if denom > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# CUSUM — cumulative sum detector
# ---------------------------------------------------------------------------

@dataclass
class CUSUM:
    """Two-sided CUSUM on individual-referenced z-scores.

        S⁺ₜ = max(0, S⁺ₜ₋₁ + (zₜ − k))
        S⁻ₜ = max(0, S⁻ₜ₋₁ − (zₜ + k))

    Fires when either accumulator exceeds the decision threshold h.
    After firing, both accumulators reset to zero.

    Parameters
    ----------
    k:
        Slack (reference value). Half the smallest drift worth detecting,
        in units of σ. Default 0.5σ.
    h:
        Decision threshold, in units of σ. Larger h → fewer false positives,
        longer detection lag. Default 5.0σ.

    Note
    ----
    k=0.5, h=5.0 are illustrative research defaults. Not validated against
    real canine cohorts.
    """
    k:       float = 0.5
    h:       float = 5.0
    s_pos:   float = 0.0
    s_neg:   float = 0.0
    history: list[tuple[float, float]] = field(default_factory=list)

    def update(self, z: float) -> dict:
        """Feed one z-score observation.

        Parameters
        ----------
        z:
            Individual-referenced z-score for the current time step.
            Non-finite values are skipped (accumulators unchanged).

        Returns
        -------
        dict with keys:
            alarm     — bool, True if a threshold was crossed this step
            direction — "up" | "down" | None
            s_pos     — current S⁺ value (after potential reset)
            s_neg     — current S⁻ value (after potential reset)
        """
        if not np.isfinite(z):
            return {"alarm": False, "direction": None,
                    "s_pos": self.s_pos, "s_neg": self.s_neg}

        self.s_pos = max(0.0, self.s_pos + (z - self.k))
        self.s_neg = max(0.0, self.s_neg - (z + self.k))
        self.history.append((self.s_pos, self.s_neg))

        alarm = (self.s_pos > self.h) or (self.s_neg > self.h)
        direction = (
            "up"   if self.s_pos > self.h else
            "down" if self.s_neg > self.h else
            None
        )
        if alarm:
            self.s_pos = self.s_neg = 0.0

        return {"alarm": alarm, "direction": direction,
                "s_pos": self.s_pos, "s_neg": self.s_neg}


# ---------------------------------------------------------------------------
# BOCPD-BLS — Bayesian Online Change-Point with Baseline-Level Shift
# ---------------------------------------------------------------------------

@dataclass
class BOCPD_BLS:
    """Lightweight Bayesian Online Change-Point Detection with baseline-level shift.

    Canine aging is characterised by an *irreversibly shifting baseline* —
    after a genuine change, the new behavioural normal is not the pre-change
    normal. Standard BOCPD compares all future observations against the
    original baseline; BOCPD-BLS collapses the run-length posterior when a
    change is confirmed, re-centering detection on the new level.

    This implementation uses a Student-t predictive model with conjugate
    Normal-Inverse-Gamma priors.

    Parameters
    ----------
    hazard:
        Prior probability of a change point at any time step (1/expected_run_length).
        Default: 1/250 ≈ change every ~8 months on average.

    Note
    ----
    All default parameters are research illustrative values, not validated
    operational thresholds.
    """
    hazard:   float = 1.0 / 250.0
    mu0:      float = 0.0
    kappa0:   float = 1.0
    alpha0:   float = 1.0
    beta0:    float = 1.0

    def __post_init__(self) -> None:
        self._R     = np.array([1.0])
        self._mu    = np.array([self.mu0])
        self._kappa = np.array([self.kappa0])
        self._alpha = np.array([self.alpha0])
        self._beta  = np.array([self.beta0])

    def _student_t_pdf(
        self,
        x:     float,
        mu:    np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta:  np.ndarray,
    ) -> np.ndarray:
        """Vectorised Student-t predictive log-density (numerically stable)."""
        nu    = 2.0 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (kappa * alpha))
        t     = (x - mu) / (scale + 1e-12)
        log_p = (
            - 0.5 * np.log(nu * np.pi + 1e-12)
            - 0.5 * np.log((scale ** 2) + 1e-12)
            + np.log(alpha + 1e-12)
            - (nu / 2 + 0.5) * np.log(1.0 + t ** 2 / (nu + 1e-12))
        )
        return np.exp(log_p - log_p.max())   # relative; not normalised

    def update(self, x: float) -> dict:
        """Feed one observation.

        Returns
        -------
        dict with keys:
            cp_prob        — probability of a change point at this step
            map_run_length — MAP estimate of the current run length (days)
        """
        if not np.isfinite(x):
            out = {"cp_prob": 0.0, "map_run_length": int(np.argmax(self._R))}
            return out

        pred  = self._student_t_pdf(x, self._mu, self._kappa, self._alpha, self._beta)
        cp    = float(np.sum(self._R * pred * self.hazard))
        growth = self._R * pred * (1 - self.hazard)

        new_R = np.concatenate([[cp], growth])
        norm  = new_R.sum() + 1e-12
        new_R = new_R / norm

        mu    = np.concatenate([[self.mu0],
                                (self._kappa * self._mu + x) / (self._kappa + 1)])
        kappa = np.concatenate([[self.kappa0], self._kappa + 1])
        alpha = np.concatenate([[self.alpha0], self._alpha + 0.5])
        beta  = np.concatenate([[self.beta0],
                                self._beta + (self._kappa * (x - self._mu) ** 2)
                                / (2 * (self._kappa + 1))])

        self._R, self._mu = new_R, mu
        self._kappa, self._alpha, self._beta = kappa, alpha, beta

        map_rl = int(np.argmax(self._R))

        # Baseline-shift: when MAP run length collapses → re-centre
        if map_rl == 0 and self._R[0] > 0.5:
            self._mu    = np.array([x])
            self._kappa = np.array([self.kappa0])
            self._alpha = np.array([self.alpha0])
            self._beta  = np.array([self.beta0])
            self._R     = np.array([1.0])

        return {"cp_prob": float(new_R[0]), "map_run_length": map_rl}
