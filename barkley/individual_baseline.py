"""
barkley.individual_baseline
============================
Pillar 1: Individual Cognitive Fingerprint (ICF).

The central architectural decision of the Barkley reference implementation
is the unit of comparison:

    NOT: "Is this dog's activity lower than average for a Border Collie?"
    YES: "Is this dog's activity lower than *its own* established normal?"

The Individual Cognitive Fingerprint (ICF) is a per-dog baseline built from
the animal's own first 56 days of observation. It uses the median and scaled
MAD (Median Absolute Deviation) for robustness to early outliers — the same
statistics used in robust behavioral phenotyping.

Once initialised, the ICF is updated slowly (small α) so that genuine
long-term maturation is tracked while acute episodes are not absorbed into
the baseline.

This framing is grounded in the distinction between:
- Population-level normative models (breed average → erases individual)
- Individual-referenced trajectory tracking (each dog's own history)

Research demonstrator only. Not diagnostic. Not production.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


#: Number of days used to initialise the ICF baseline.
#: Grounded in published canine longitudinal behavioural research.
ICF_INIT_DAYS: int = 56


@dataclass
class ICF:
    """Per-channel individual behavioral baseline for one dog.

    Attributes
    ----------
    channels:
        List of behavioral channel names.
    mean:
        Per-channel baseline mean (median of the init window).
    scale:
        Per-channel robust scale (1.4826 × MAD ≈ σ for Gaussian data).
    n_seen:
        Total number of observations used to build / maintain the baseline.
    version:
        Schema version tag.
    """
    channels: list[str]
    mean:     dict[str, float] = field(default_factory=dict)
    scale:    dict[str, float] = field(default_factory=dict)
    n_seen:   int              = 0
    version:  str              = "icf_v0.1"

    # ----------------------------------------------------------------------- #
    # Construction
    # ----------------------------------------------------------------------- #

    @classmethod
    def initialise(cls, channels: list[str], init_matrix: np.ndarray) -> "ICF":
        """Build an ICF from the first ICF_INIT_DAYS rows of observation.

        Parameters
        ----------
        channels:
            Ordered list of channel names (length must equal init_matrix.shape[1]).
        init_matrix:
            Shape (T_init, C) array of the dog's own first ~56 days.
            NaN values are excluded per channel.

        Returns
        -------
        ICF instance with per-channel median and scaled MAD.
        """
        icf = cls(channels=list(channels))
        for j, ch in enumerate(channels):
            col = init_matrix[:, j]
            col = col[np.isfinite(col)]
            if col.size == 0:
                icf.mean[ch]  = 0.0
                icf.scale[ch] = 1.0
                continue
            med  = float(np.median(col))
            mad  = float(np.median(np.abs(col - med)))
            icf.mean[ch]  = med
            icf.scale[ch] = max(1.4826 * mad, 1e-6)   # 1.4826: MAD → σ scaling
        icf.n_seen = int(init_matrix.shape[0])
        return icf

    # ----------------------------------------------------------------------- #
    # Inference
    # ----------------------------------------------------------------------- #

    def zscore(self, channel: str, value: float) -> float:
        """Individual-referenced z-score for one observation.

        Answers: how far is this value from *this dog's own* established normal?

        Returns NaN if the channel is not in the ICF or the value is not finite.
        """
        if channel not in self.mean or not np.isfinite(value):
            return float("nan")
        return (value - self.mean[channel]) / self.scale[channel]

    # ----------------------------------------------------------------------- #
    # Slow baseline maintenance
    # ----------------------------------------------------------------------- #

    def update(self, channel: str, value: float, alpha: float = 0.01) -> None:
        """Exponential moving update of the baseline mean.

        A small α (default 0.01) ensures the baseline tracks genuine long-term
        maturation without absorbing an acute episode.

        Parameters
        ----------
        channel:
            Channel name to update.
        value:
            New observation. Non-finite values are skipped.
        alpha:
            Exponential smoothing coefficient ∈ (0, 1). Smaller → more stable.
        """
        if channel not in self.mean or not np.isfinite(value):
            return
        self.mean[channel] = (1 - alpha) * self.mean[channel] + alpha * value
        self.n_seen += 1


# ---------------------------------------------------------------------------
# Baseline loss (for research evaluation)
# ---------------------------------------------------------------------------

def baseline_loss(
    pred:       np.ndarray,
    y_baseline: np.ndarray,
    lambda_reg: float = 1e-3,
    weights:    np.ndarray | None = None,
) -> float:
    """Mean squared error against the individual baseline, with L2 regularisation.

    Used for evaluating how well a learned representation tracks individual
    behavioral trajectory — not for any real-world prediction.

    J(W, b) = (1/m) Σ L(f(Xᵢ), y_baseline_i) + (λ/2m) Σ wⱼ²

    Parameters
    ----------
    pred:
        Predicted values, shape (m,).
    y_baseline:
        Individual baseline targets, shape (m,).
    lambda_reg:
        L2 regularisation coefficient.
    weights:
        Optional parameter vector for regularisation. If None, regularisation
        term is zero.

    Returns
    -------
    Scalar loss value.
    """
    m    = len(pred)
    mse  = float(np.mean((pred - y_baseline) ** 2))
    reg  = 0.0
    if weights is not None and lambda_reg > 0:
        reg = (lambda_reg / (2 * m)) * float(np.sum(weights ** 2))
    return mse + reg
