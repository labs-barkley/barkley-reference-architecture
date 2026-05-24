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
