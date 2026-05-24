# Barkley Reference Architecture

**A source-available research demonstrator for individual-referenced behavioral intelligence in dogs.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Data](https://img.shields.io/badge/data-synthetic%20only-orange)
![Status](https://img.shields.io/badge/status-research%20demonstrator-lightgrey)
![Use](https://img.shields.io/badge/use-non--diagnostic-critical)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20369864.svg)](https://doi.org/10.5281/zenodo.20369864)

> A dog can remain normal for its breed while drifting away from its own behavioral baseline.

This repository is part of the **Barkley Canine Cognition Lab** ecosystem — the
public research hub for individual-referenced behavioral intelligence. It is the
minimal, runnable computational backbone; see
[Relationship to the Barkley ecosystem](#relationship-to-the-barkley-ecosystem) below.

```
   Synthetic        Temporal         Individual        Rate of          Missingness
    events     →       bins      →     baseline    →     drift      →      labels
  (schema)         (Bin A/B/C)        (ICF)          (CUSUM/BOCPD)    (Silence as Signal)
                                                                            │
                                              DogGraph event schema  ───────┘
                                            (memory layer tying it together)
```

---

## What this repository demonstrates

This repository contains the minimal computational backbone of the Barkley
behavioral intelligence research framework — four pillars, runnable, inspectable,
and built on fully synthetic data:

| Pillar | Module | Question it answers |
|---|---|---|
| **1. Individual Baseline / ICF** | `barkley/individual_baseline.py` | What is *this dog's own* normal? |
| **2. Multi-Resolution Temporal Binning** | `barkley/temporal_binning.py` | What does this dog's behavior look like at daily, weekly, and quarterly scales? |
| **3. Rate of Drift / Behavioral Drift** | `barkley/rate_of_drift.py` | Is this dog drifting away from its own baseline — and how fast? |
| **4. Sovereignty of Silence / Missingness** | `barkley/missingness.py` | When data is absent, *why* — and is the absence itself meaningful? |

**DogGraph** (`barkley/schema.py`) is the memory and structuring layer that ties
these four pillars into a coherent behavioral event representation. It is not the
full Barkley system — it is the substrate that makes the pillars composable.

The key architectural decision, stated plainly:

> NOT: *"Is this dog's activity lower than average for its breed?"*
>
> YES: *"Is this dog's activity lower than its own established normal?"*

The animal most at risk of undetected behavioral change is the one whose own
trajectory has drifted furthest from its own history — not the one furthest from
a population mean. This repository implements that distinction.

---

## What this repository does NOT do

- Does **not** process real animal data of any kind.
- Does **not** produce, imply, or support any veterinary or medical diagnosis.
- Does **not** represent a production system, validated instrument, or medical device.
- Does **not** expose confidential IP, commercial strategy, regulatory strategy,
  roadmap, or proprietary product mechanics.
- Does **not** contain validated operational thresholds. All numerical parameters
  (CUSUM `k`, `h`; baseline window length; decline rates) are illustrative research
  defaults, not validated thresholds.
- Does **not** connect to any external service, API, or data source.

---

## Quickstart

```bash
git clone https://github.com/labs-barkley/barkley-reference-architecture
cd barkley-reference-architecture
pip install -r requirements.txt
python examples/run_reference_pipeline.py
```

Expected output:

```
events_generated:            309
baseline_window_days:        56
drift_detected:              true
missingness_classified:      true
ks_distribution_shift:       0.564
research_demonstrator_only:  true
```

Run the test suite:

```bash
python tests/test_reference_pipeline.py   # no pytest required
# or
python -m pytest tests/ -v
```

---

## Repository structure

```
barkley-reference-architecture/
│
├── README.md
├── LICENSE                        Source-available research license
├── NOTICE.md                      Ownership, research-only, synthetic-only scope
├── RELEASE_NOTES.md               v0.1.0 release notes
├── CITATION.cff                   Citation metadata (accompanies the framework paper)
├── pyproject.toml                 Packaging + pytest/mypy configuration
├── requirements.txt               numpy + pandas (Python 3.10+)
├── .gitignore
│
├── .github/
│   └── workflows/
│       └── test.yml               CI: pipeline + tests + type-check (3.10, 3.11)
│
├── barkley/
│   ├── __init__.py                Public API
│   ├── schema.py                  DogGraph event schema — the memory layer
│   ├── synthetic.py               Deterministic synthetic dog generator
│   ├── temporal_binning.py        Bin A (circadian) / B (weekly) / C (quarterly)
│   ├── individual_baseline.py     ICF — per-dog median+MAD baseline
│   ├── rate_of_drift.py           CUSUM + BOCPD-BLS + Rate of Drift
│   └── missingness.py             Sovereignty of Silence taxonomy
│
├── examples/
│   └── run_reference_pipeline.py  End-to-end reference pipeline
│
├── tests/
│   └── test_reference_pipeline.py 16-test suite (no external dependencies)
│
└── diagrams/
    └── README.md                  Pipeline diagram and architectural notes
```

> **Install (optional).** The pipeline runs zero-install via the quickstart above.
> For an editable install (which removes the need for the source-checkout import
> shim), run `pip install -e .` — then `import barkley` works from anywhere.

---

## Reference pipeline explanation

The pipeline implements the following staged processing for a single
synthetic dog (365 daily observations, behavioral decline injected at day 240):

### Stage 1 — Individual Baseline (Pillar 1)

The first 56 days build the dog's **Individual Cognitive Fingerprint (ICF)**:
a per-channel baseline computed as the **median + scaled MAD** of the dog's
own observations. Not a breed average. Not a population norm.

Once established, the ICF is updated slowly (α = 0.01) so that genuine long-term
maturation is tracked while acute episodes are not absorbed into the baseline.

### Stage 2 — Individual-Referenced Z-Scores

Every subsequent observation is converted to a **z-score against the dog's own ICF**:

```
z = (x − μ_ICF) / σ_ICF
```

Five channels are sign-aligned (so that a behavioral decline always increases the
score) and averaged into a **composite decline index**. Averaging across channels
is what distinguishes a genuine multi-channel behavioral shift from single-channel
daily noise.

### Stage 3 — CUSUM Change-Point Detection (Pillar 3)

The composite index is streamed into a **two-sided CUSUM** (Page, 1954).
CUSUM accumulates small, sustained deviations until the evidence crosses a
decision threshold h. It is included here because it can surface slow, monotonic
drifts that are invisible in daily snapshots.

In the reference pipeline: decline onset at day 240 → first alarm at day 269
(29-day detection lag; no false alarm during the 184-day stable period).

### Stage 4 — BOCPD-BLS Baseline Re-Centering

Canine aging is characterised by an *irreversibly shifting baseline*. Standard
change-point detectors compare forever against the original normal. **BOCPD-BLS**
(Bayesian Online Change-Point Detection with Baseline-Level Shift) collapses its
run-length posterior when a change is confirmed, re-centering detection on the
new behavioral level.

### Stage 5 — Temporal Binning (Pillar 2)

Three parallel bin resolutions run across the full time-series:
- **Bin A (24h):** circadian window — captures nocturnal restlessness
- **Bin B (7d):** weekly window — normalises anthropogenic cycles
- **Bin C (91d):** quarterly distribution-shift test (KS statistic) — detects
  secular trajectory change invisible to mean-comparison tests

### Stage 6 — Sovereignty of Silence (Pillar 4)

Data gaps are not deleted or imputed. Each absent channel-day is classified
into a three-way taxonomy:

- **Artefactual:** hardware fault — handle defensively
- **Contextual:** known cause (charging, swimming, vet visit) — contextual interpolation
- **Informative:** device on-body, dog confirmed present, no anomaly —
  *the silence is the signal*; encode as a positive stability feature

### Stage 7 — DogGraph Event Emission

Each processed day is emitted as a **DogGraph event** — a structured,
machine-readable record encoding features, context, baseline reference,
missingness classification, and a research output prompt. The DogGraph schema
is the memory layer that makes the four pillars composable and replayable.

---

## Modeling notes

The mechanics here are intentionally simple and well-understood. The point of the
reference architecture is the *framing* — comparing a dog to itself — not any
novel estimator. At a high level:

- **Individual baseline.** Each channel's baseline is a robust location/scale
  estimate (median and a MAD-based scale) computed from the individual's own early
  history. An observation is then expressed as a deviation from that baseline,
  `z = (x − μ) / σ`, so "normal" is defined per individual rather than per breed.

- **Rolling windows.** Behavior is summarised over multiple time scales (daily,
  weekly, quarterly). Short windows capture rhythm; long windows capture slow,
  secular change. A distribution-comparison test across equivalent long windows
  flags shifts in the *shape* of behavior, not just its average.

- **Drift as deviation over time.** "Drift" is simply sustained movement of the
  individual-referenced signal away from zero. A trailing slope estimates how fast;
  a cumulative-sum and a Bayesian change-point view estimate whether a sustained
  shift has occurred and where the baseline should re-centre. These are standard
  sequential-analysis tools applied to an individual-referenced signal.

- **Missingness as a classified state.** Absence is not imputed away. Each gap is
  labelled (artefactual / contextual / informative) and that label is carried as a
  feature, so downstream analysis can condition on *why* data is missing.

Concrete parameter values in the code (window lengths, thresholds, decline rates)
are **illustrative defaults for the synthetic demonstrator**, chosen for clarity,
not tuned or validated. They should not be read as recommended operating points.

---

## Synthetic data notice

This repository uses **fully synthetic, deterministic data generated by
`barkley/synthetic.py`**. There are no real animals, no real sensors, and no
proprietary telemetry of any kind in this codebase.

The synthetic generator produces:
- A dog's own healthy behavioral baseline (not a breed average)
- Anthropogenic weekly cycles (weekday/weekend variation)
- Controllable behavioral decline at a specified day
- Configurable missing-data rate for missingness pipeline exercise

All results are fully reproducible from `seed=42`. The pipeline output does
not change across runs on the same Python/numpy version.

---

## Using your own DogGraph-compatible events

The included generator (`barkley/synthetic.py`) is **synthetic and deterministic**
by design — it lets anyone run the full pipeline with no data of their own. If you
are a researcher or engineer who wants to explore your *own* data, you can adapt a
private dataset by emitting the same public schema rather than changing the pipeline.

The pipeline expects a stream of events, one temporal slice per record, each carrying:

- a **timestamp** (ISO-8601),
- one or more **behavioral channels** with normalised **values**,
- **missingness metadata** describing any absent channels,
- optional **context** (e.g. weekday, known events).

See `barkley/schema.py` for the exact `DogGraphEvent` structure and `validate()`
helper. In practice you would replace the call to `generate_dog(...)` with your own
loader that yields records conforming to that schema, then feed them through the
same baseline → binning → drift → missingness stages.

A few important conditions for any such use:

- You are solely responsible for **privacy, consent, data governance, validation,
  and compliance** for any data you bring.
- This repository remains a **research and evaluation demonstrator**. It makes no
  claim of fitness for any real-world decision and is **not diagnostic**.
- Nothing here implies a partner integration, a production data path, or any
  particular data source. The schema is a neutral, public interchange format.

---

## Safety, scope, and non-diagnostic use

> **This is a research demonstrator using fully synthetic data.**
>
> It is **not** a diagnostic tool, not a validated instrument, and not a medical
> device. It has not been evaluated against any real animal data. All numerical
> parameters are illustrative research defaults, not validated operational thresholds.
>
> Any output from this pipeline constitutes a **research finding on synthetic data**,
> nothing more. It must not be used to make decisions about the behavior, welfare,
> or treatment of any animal or person. A qualified professional remains in the
> loop for any real-world action.
>
> **Scope boundary.** This repository is the *minimal computational backbone* of a
> larger research program. The absence of any component here is not a statement
> about the broader Barkley system, which is separate from this public demonstrator
> and remains confidential.

---

## Relationship to the Barkley ecosystem

This repository is one of four public research artifacts. Each plays a distinct,
non-overlapping role:

| Artifact | Role |
|---|---|
| **Framework Paper** (Zenodo) | Conceptual foundation — *why* the individual reference frame matters |
| **Reference Architecture** (this repo) | Computational backbone — *how* the mechanics work, runnable on synthetic data |
| **Synthetic DogGraph Dataset** (Hugging Face) | Data artifact — a public synthetic sample to explore |
| **Drift Explorer** (interactive) | Visual proof — *seeing* the reference frame in action |

Together they demonstrate that the thesis is not just a concept: it has a
conceptual foundation, a computational backbone, a data artifact, and an
interactive demonstration. This repository is deliberately the *minimal* backbone —
it does not reproduce the others, and it does not expose the full Barkley system.

---

## Links

| Resource | URL |
|---|---|
| **Drift Explorer** (interactive) | https://drift-explorer.getbarkley.com |
| **GitHub — Canine Cognition Lab** | https://github.com/labs-barkley/barkley-canine-cognition-lab |
| **Hugging Face dataset** | https://huggingface.co/datasets/labs-barkley/synthetic-doggraph-sample |
| **Zenodo — Framework paper** | https://zenodo.org/records/20060327 |
| **Zenodo — This repository** | https://zenodo.org/records/20369864 |
| **Contact** | labs@getbarkley.com |

---

## Citation

This repository **accompanies the Barkley framework paper**. If you use it in
your research, please cite the framework paper:

```bibtex
@misc{remoissenet2026barkleyframework,
  author    = {Remoissenet, Elodie P.},
  title     = {Barkley: Individual-Referenced Behavioral Intelligence for Dogs
               (Framework Paper)},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20060327},
  url       = {https://zenodo.org/records/20060327}
}
```

Plain text:

> Remoissenet, E.P. (2026). *Barkley: Individual-Referenced Behavioral Intelligence
> for Dogs (Framework Paper).* Zenodo. https://doi.org/10.5281/zenodo.20060327

```bibtex
@software{remoissenet2026barkleyref,
  author    = {Remoissenet, Elodie P.},
  title     = {Barkley Reference Architecture},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20369864},
  url       = {https://doi.org/10.5281/zenodo.20369864}
}
```

> Software DOI: [10.5281/zenodo.20369864](https://doi.org/10.5281/zenodo.20369864)
> Framework paper DOI: [10.5281/zenodo.20060327](https://doi.org/10.5281/zenodo.20060327)

---

## License

Source-Available Research License — see [`LICENSE`](LICENSE).

**Summary:** Free for research, education, and evaluation. Commercial use,
clinical use, or production deployment requires explicit written permission.

© 2026 Elodie P. Remoissenet — Barkley AI · labs@getbarkley.com
