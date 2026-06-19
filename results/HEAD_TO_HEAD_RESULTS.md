# Head-to-Head: Individual-Referenced vs Population-Referenced Detection

**A proof-of-concept on fully synthetic data — addressing §6 Q1 of the working paper
*The Reference-Class Trap in Animal-Computer Interaction*.**

> **This does not validate Barkley on real dogs.** It demonstrates why the
> reference-class trap is *mechanically plausible*: as individual heterogeneity
> grows, population normality becomes too wide to protect the individual.
>
> Controlled synthetic demonstration of a mechanism. The decline it detects is
> injected and known. No real animals, no real sensors. Not diagnostic. Not production.

Reproduce: `PYTHONPATH=. python examples/head_to_head_validation.py`
Figure: `barkley_head_to_head.png`

---

## The question

The working paper states, as its central open empirical question (§6, Q1):

> Does the individual baseline actually outperform the population reference, on real
> animals, for real welfare-relevant events? … not yet established … it requires a
> head-to-head comparison of self-referenced versus population-referenced detection.

This note runs exactly that head-to-head — on synthetic data, as a first proof-of-concept.

## What is compared, and why it is a fair fight

Both arms are **identical except for the reference frame**:

|                     | Individual arm                          | Population arm                              |
| ------------------- | --------------------------------------- | ------------------------------------------- |
| Reference built from| each dog's own first 56 days            | the **pooled** first 56 days of all dogs    |
| Statistic           | median + 1.4826·MAD (`ICF`)             | median + 1.4826·MAD (`ICF`, same code)      |
| Detector            | Barkley `CUSUM` (k = 0.5)               | Barkley `CUSUM` (k = 0.5), same             |
| Channels            | 5, sign-aligned, composite z            | 5, sign-aligned, composite z, same          |

The **only** difference is whether each day's z-score is measured against the dog's own
history or against the cohort. This isolates the reference-class effect — the variable
the paper is actually about — and nothing else.

Fairness guarantees built into the experiment:

- **Both arms use the same 56-day calibration window.** The population arm receives the
  usual advantage of pooled cohort data; the individual arm receives only the dog's own
  history — so any individual-arm win comes *despite* handing the population arm the larger,
  pooled sample.
- **Healthy control dogs are included**, so false alarms are measured. The comparison is
  reported as a full **ROC** (threshold-independent), not at one hand-picked operating point.
- A fraction (40%) of the healthy dogs undergo a **benign regime change** — a one-time,
  sustained, random-direction baseline shift (new home, new routine), *not* a welfare
  decline. The individual detector, comparing to the old baseline, may flag these as a
  false alarm. This deliberately surfaces the individual approach's **own failure mode**
  (working paper §6 Q3) rather than hiding it.
- The result is reported across a **range of cohort heterogeneity levels**, so the effect
  is shown to emerge from a realistic mechanism rather than a tuned parameter. On data we
  generate, this is the safeguard against a rigged win.

## Results

**Primary cohort** (between-dog baseline spread σ = 0.10; 100 declining + 100 healthy dogs;
**30 independent seeds, mean ± SD**):

| Metric                                          | Individual        | Population    |
| ----------------------------------------------- | ----------------- | ------------- |
| ROC AUC (declining vs healthy)                  | **0.988 ± 0.010** | 0.935 ± 0.020 |
| Declines caught at matched ~10% false-alarm rate| **100% ± 0%**     | 81% ± 20%     |
| Median detection lag at that operating point    | **50 ± 14 days**  | 84 ± 16 days  |

At the same false-alarm budget and this heterogeneity level, the population reference
**misses ~19% of the genuine declines the individual reference catches** (with wide
seed-to-seed variation), and is ~34 days slower on the ones it does catch. As heterogeneity
rises, the miss rate grows sharply — at σ = 0.14 the population reference catches only
56% (missing ~44%), while the individual reference stays at 100%. See the sweep below.

**Heterogeneity sweep** (detection at matched 10% false-alarm rate; 30 seeds, mean ± SD):

| σ_pop (between-dog spread) | AUC ind       | AUC pop       | Caught ind | Caught pop |
| -------------------------- | ------------- | ------------- | ---------- | ---------- |
| 0.02                       | 0.989 ± 0.010 | 0.991 ± 0.009 | 100% ± 0%  | 100% ± 0%  |
| 0.05                       | 0.988 ± 0.011 | 0.981 ± 0.013 | 100% ± 0%  | 100% ± 0%  |
| 0.08                       | 0.990 ± 0.010 | 0.957 ± 0.017 | 100% ± 0%  | 95% ± 5%   |
| 0.11                       | 0.985 ± 0.011 | 0.925 ± 0.024 | 100% ± 0%  | 76% ± 21%  |
| 0.14                       | 0.984 ± 0.012 | 0.893 ± 0.025 | 100% ± 0%  | 56% ± 22%  |

## The honest reading

This is **not** "the individual reference always wins."

- **When the cohort is nearly homogeneous (σ = 0.02–0.05), the population reference is
  every bit as good — even marginally better by AUC.** With little between-dog variation,
  the population band is tight, so it detects declines about as well *and* it is less
  fooled by the benign individual regime-changes that cost the individual arm a few false
  alarms. When individuals barely differ, individualizing the reference buys little and
  carries its own risk.
- **The individual reference's advantage appears, and grows, with heterogeneity.** As the
  population becomes more varied, its "normal" band must widen to contain everyone — and a
  calm dog's genuine decline disappears inside that wide band (the paper's *false
  reassurance*, §3). Panel D shows one such dog: its activity falls out of its own tight
  band while never leaving the population band.
- **The individual arm has a real cost**, visible in its sub-1.0 AUC: it can flag benign
  individual changes as if they were decline. That is precisely the failure mode the
  paper flags in §6 Q3, and managing it (context, calibration, conformal confidence) is
  part of the actual engineering — not a footnote.

The takeaway is conditional and, I think, harder to dismiss for being so: **the more the
individuals in a population genuinely differ from one another, the more a population
reference misreads them — and dogs, where breed explains only ~9% of behavioral variation
between individuals (Morrill et al. 2022), are squarely in the regime where the individual
reference matters.**

## What this does NOT show

- It does not validate anything on real dogs. The decline is synthetic and known; this
  measures detection of a controlled signal, not discovery of real pathology.
- The synthetic generator is deliberately simple. Real behavioral data carries confounds
  this demonstrator does not model; both arms would perform worse in reality, and the
  individual arm would face many more sources of false alarm than the single benign-shift
  confound included here.
- The absolute numbers (AUC, lag) are properties of this synthetic setup and its
  parameters. The **relative** result — population degrades as heterogeneity grows — is
  the mechanistic claim, and it is robust to the specific numbers.
- Every condition here is averaged over **30 independent synthetic seeds** (mean ± SD,
  with ROC bands and error bars). The relative effect is stable across seeds; the absolute
  numbers remain properties of this synthetic setup. The claim is the *relative* trend,
  not the exact values.

§6 Q1 therefore remains open *for real data*. What this establishes is narrower and still
useful: the mechanism the paper describes is real, reproducible, and points in the
predicted direction under controlled conditions.

---

*Research demonstrator only. Synthetic data only. Not diagnostic. Not production.
© 2026 Élodie P. Remoissenet — Barkley AI · labs@getbarkley.com*
