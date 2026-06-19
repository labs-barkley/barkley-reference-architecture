# v0.2.0 - Head-to-head: individual vs population reference

Adds a reproducible synthetic head-to-head that addresses the open empirical question
in the framework paper (Section 6, Q1): does the individual baseline actually
outperform the population reference for welfare-relevant change?

This is a proof-of-concept on fully synthetic data - not validation on real dogs.

Added:
- examples/head_to_head_validation.py - individual- vs population-referenced detection
  on a heterogeneous synthetic cohort, built on the existing barkley package (same ICF
  baseline, same CUSUM detector; only the reference frame differs).
- results/barkley_head_to_head.png - 4-panel figure (mean ROC +/- SD, detection-lag
  distributions, heterogeneity sweep with error bars, calm-dog false-reassurance).
- results/HEAD_TO_HEAD_RESULTS.md - method, fairness contract, and full results.
- README "Worked example" section.

Method: both arms use the same detector (CUSUM), the same robust statistic
(median + MAD), and the same 56-day calibration window; the population arm gets the
pooled-cohort advantage. Healthy controls are included and false alarms are measured
(threshold-independent ROC + matched operating point). Every condition is averaged over
30 independent synthetic seeds (mean +/- SD).

Result (synthetic, illustrative): as between-dog heterogeneity grows, the population
"normal" band widens until a calm dog's genuine decline stays inside it - the
false-reassurance mechanism the paper names the reference-class trap. The individual
reference holds 100% detection at a matched 10% false-alarm rate across all
heterogeneity levels tested; the population reference falls from ~100% to ~56% as
heterogeneity grows. At low heterogeneity the two are comparable. The advantage is
conditional, not universal. The real-data head-to-head remains the open validation step.

Zenodo (this version): 10.5281/zenodo.20754351

---
# Release Notes

## v0.1.0 — Initial public release (2026-05-24)

First public, source-available release of the **Barkley Reference Architecture** —
the minimal, runnable computational backbone behind the Barkley research thesis:

> A dog can remain normal for its breed while drifting away from its own
> behavioral baseline.

### What's included

This release demonstrates four research pillars plus the structuring schema,
on fully synthetic data:

- **Individual Baseline / ICF** (`barkley/individual_baseline.py`) — per-dog
  median + scaled-MAD baseline with slow exponential maintenance.
- **Multi-Resolution Temporal Binning** (`barkley/temporal_binning.py`) —
  circadian (Bin A), weekly (Bin B), and quarterly (Bin C) windows, including a
  from-scratch two-sample Kolmogorov–Smirnov distribution-shift test.
- **Rate of Drift / Behavioral Drift** (`barkley/rate_of_drift.py`) — OLS drift
  slope, two-sided CUSUM, and a lightweight BOCPD-BLS baseline-shift detector.
- **Sovereignty of Silence / Missingness** (`barkley/missingness.py`) — the
  Artefactual / Contextual / Informative taxonomy and feature encoding.
- **DogGraph event schema** (`barkley/schema.py`) — the memory and structuring
  layer that ties the pillars into a typed, serialisable behavioral event.

### Runnable reference pipeline

`examples/run_reference_pipeline.py` runs the full pipeline end-to-end on one
synthetic dog and prints a reproducible summary:

```
events_generated:            309
baseline_window_days:        56
drift_detected:              true
missingness_classified:      true
ks_distribution_shift:       0.564
research_demonstrator_only:  true
```

### Tests

16 tests covering all four pillars and the schema. Runnable two ways:

```
python tests/test_reference_pipeline.py     # no external dependency
python -m pytest tests/ -v                  # pytest
```

### Requirements

Python 3.10+, `numpy`, `pandas`. No GPU, no internet, no external services.

### Scope and limitations

- Fully synthetic data only; no real animal data.
- All numerical parameters are illustrative research defaults, **not** validated
  operational thresholds.
- Research demonstrator only; **not** diagnostic, **not** production.
- This is the minimal backbone; it does not reproduce or expose the full
  Barkley system.

### Citation

This release accompanies the Barkley framework paper
(DOI: 10.5281/zenodo.20060327). A separate software DOI will be minted once this
release is archived on Zenodo; `CITATION.cff` will be updated at that point.

---

*Research demonstrator only. Synthetic data only. Not diagnostic.
labs@getbarkley.com*
