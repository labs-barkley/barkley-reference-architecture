# Diagrams

## Reference Architecture Pipeline

```
Synthetic Events  →  Temporal Bins  →  Individual Baseline  →  Rate of Drift  →  Missingness Labels
   (schema)            (Bin A/B/C)          (ICF)              (CUSUM/BOCPD)     (Sovereignty of Silence)
```

### Pipeline summary

| Stage | Module | Core concept |
|---|---|---|
| **Schema / DogGraph** | `barkley/schema.py` | Typed behavioral event — the memory layer tying all pillars together |
| **Synthetic data** | `barkley/synthetic.py` | Deterministic 365-day individual with injectable behavioral drift |
| **Temporal binning** | `barkley/temporal_binning.py` | Circadian (24h), weekly (7d), quarterly (91d) distribution windows |
| **Individual baseline** | `barkley/individual_baseline.py` | ICF: per-dog median+MAD — not a breed average |
| **Drift detection** | `barkley/rate_of_drift.py` | OLS Rate of Drift + CUSUM + BOCPD-BLS on individual-referenced z-scores |
| **Missingness** | `barkley/missingness.py` | Artefactual / Contextual / Informative taxonomy |

The pipeline diagram SVG is available at:

- **Interactive Drift Explorer**: https://drift-explorer.getbarkley.com
- **Hugging Face dataset**: https://huggingface.co/datasets/labs-barkley/synthetic-doggraph-sample
- **Zenodo framework paper**: https://zenodo.org/records/20060327

---

## Key architectural decisions

### Individual referencing
The primary decision: each dog is compared to its own 56-day behavioral history,
not to a breed average. The animal most at risk is the one whose own profile
has drifted furthest from its own norm — not the one furthest from the mean of its breed.

### Multi-resolution temporal windows
Behavior operates at multiple time scales simultaneously:
- **Bin A (24h):** nocturnal restlessness — early sleep fragmentation marker
- **Bin B (7d):** anthropogenic cycle normalisation (weekday vs. weekend)
- **Bin C (91d):** full distribution shift test (KS statistic) — detects secular trajectory change

### Trajectory, not snapshot
A single anomalous day is noise. A sustained, monotonic drift across multiple channels
over multiple weeks is a trajectory signal. CUSUM accumulates small deviations;
BOCPD-BLS re-centres when the baseline itself shifts irreversibly (e.g. healthy aging).

### Silence as data
A 6-hour gap in activity, with the device on-body and context confirming presence,
is not missing data — it is the absence of anomaly. This module transforms a
data-quality problem into a modeling resource via the Sovereignty of Silence taxonomy.

---

*Research demonstrator only. Not diagnostic. labs@getbarkley.com*
