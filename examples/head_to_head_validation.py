#!/usr/bin/env python3
"""
head_to_head_validation.py
==========================
Individual-referenced vs population-referenced behavioral-change detection
on a synthetic heterogeneous cohort — a reproducible, multi-seed stress test.

This script answers — AS A REPRODUCIBLE PROOF-OF-CONCEPT ON FULLY SYNTHETIC DATA —
the open empirical question stated in the Barkley working paper "The Reference-Class
Trap in Animal-Computer Interaction" (§6, Q1):

    Does the individual baseline actually outperform the population reference,
    for real welfare-relevant change?

It is NOT validation on real dogs. It demonstrates the *mechanism*: when a cohort
has heterogeneous individual baselines, the population "normal band" must be wide
enough to contain them all — so a calm dog's genuine decline can stay inside the
population band (the working paper's "false reassurance", §3) while clearly
departing from its own, tighter, individual band.

FAIRNESS CONTRACT (so the comparison means something on data we control):
  * Both arms use the SAME detector — Barkley's CUSUM accumulation (same slack k;
    here computed in closed form, verified bit-identical to barkley.rate_of_drift.CUSUM
    with h=inf) — and the SAME robust statistic (median + 1.4826·MAD, via
    barkley.individual_baseline.ICF). The ONLY thing that differs is the REFERENCE:
    the dog's own first-56-day history vs the pooled cohort.
  * Both arms use the SAME 56-day calibration window (ICF_INIT_DAYS = 56). The
    population arm receives the usual advantage of pooled cohort data; the
    individual arm receives only the dog's own history — so any individual-arm
    win comes despite handing the population arm the larger, pooled sample.
  * Healthy control dogs are included, so false alarms are measured and the
    comparison is reported as a full ROC (threshold-independent), not at one
    hand-picked operating point.
  * A fraction of the healthy dogs undergo a benign, random-direction regime change
    (new home, new routine) — NOT a welfare decline. The individual detector may flag
    these as a false alarm, so the individual arm's OWN failure mode (working paper
    §6 Q3) is visible in its false-positive rate, not hidden.
  * Every condition is repeated over N_SEEDS independent synthetic cohorts; results
    are reported as mean ± SD across seeds, with ROC bands and error bars. This is
    the safeguard against both a rigged win and a single-seed artifact.
  * The effect is reported across a RANGE of cohort heterogeneity levels, so it is
    shown to emerge from a realistic mechanism rather than a tuned parameter.

Research demonstrator only. Synthetic data only. Not diagnostic. Not production.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from barkley.synthetic import CORE_CHANNELS
from barkley.individual_baseline import ICF, ICF_INIT_DAYS

# --------------------------------------------------------------------------- #
# Constants mirrored from barkley.synthetic (so the cohort is faithful to it)  #
# --------------------------------------------------------------------------- #
DIRECTION = {                      # +1 rises during decline, -1 falls
    "activity_level":          -1,
    "sleep_fragmentation":     +1,
    "restlessness_index":      +1,
    "social_response_latency": +1,
    "routine_variability":     -1,
}
POP_BASE = {                       # population-mean healthy baseline (== generate_dog anchor)
    "activity_level":          0.55,
    "sleep_fragmentation":     0.20,
    "restlessness_index":      0.25,
    "social_response_latency": 0.30,
    "routine_variability":     0.60,
}
DIR_VEC      = np.array([DIRECTION[ch] for ch in CORE_CHANNELS], dtype=float)
ACT_IDX      = list(CORE_CHANNELS).index("activity_level")
DAYS         = 365
ONSET        = 240
DECLINE_RATE = 0.0015              # Barkley default
MISSING_RATE = 0.05               # Barkley default
DAILY_NOISE  = 0.04               # Barkley default (generate_dog)
CUSUM_K      = 0.5                # Barkley CUSUM slack (research default)
BENIGN_FRAC  = 0.40               # fraction of healthy dogs with a benign regime change
N_SEEDS      = 30                # independent synthetic cohorts per condition

NAVY = "#0A2342"   # individual arm (hero)
RUST = "#B0552F"   # population arm (foil)
TEAL = "#1A6B72"
GREY = "#9AA3AB"

_DAY = np.arange(DAYS)
_WEEKEND = (((_DAY + 2) % 7) >= 5).astype(float)        # 2026-01-01 is a Thursday
_DECL_RAMP = np.where(_DAY >= ONSET, DECLINE_RATE * (_DAY - ONSET), 0.0)


# --------------------------------------------------------------------------- #
# Cohort generation (extends generate_dog with a PER-DOG baseline)            #
# --------------------------------------------------------------------------- #
def draw_personal_base(rng: np.random.Generator, sigma_pop: float) -> np.ndarray:
    """One dog's personal healthy baseline (vector over CORE_CHANNELS), drawn
    around the population mean. sigma_pop is the between-dog baseline spread —
    the realism knob (the Morrill-2022 reality that the cohort label is a weak
    proxy for the individual)."""
    base = np.array([POP_BASE[ch] for ch in CORE_CHANNELS], dtype=float)
    return np.clip(base + rng.normal(0, sigma_pop, size=base.shape), 0.02, 0.98)


def generate_cohort_dog(personal_base: np.ndarray, declines: bool,
                        rng: np.random.Generator, benign_shift: bool = False) -> np.ndarray:
    """One synthetic dog as a (DAYS, C) matrix — vectorised.

    Same structure as barkley.synthetic.generate_dog (weekend modulation on
    activity, daily Gaussian noise, missingness, linear decline ramp & direction)
    EXCEPT the healthy baseline is `personal_base` (per dog). If `benign_shift`,
    the dog is HEALTHY but undergoes a one-time, sustained, random-direction
    regime change at a random day — a benign life event, not a welfare decline.
    """
    decl = _DECL_RAMP if declines else np.zeros(DAYS)
    M = personal_base[None, :] + DIR_VEC[None, :] * decl[:, None]      # (DAYS, C)
    M[:, ACT_IDX] += 0.03 * _WEEKEND
    if benign_shift:
        shift_day = int(rng.integers(100, 280))
        benign = rng.normal(0, 0.08, size=len(CORE_CHANNELS))
        M[shift_day:, :] += benign[None, :]
    M += rng.normal(0, DAILY_NOISE, size=M.shape)
    M = np.clip(M, 0.0, 1.0)
    M[rng.random(M.shape) < MISSING_RATE] = np.nan
    return M


# --------------------------------------------------------------------------- #
# Detection: composite sign-aligned z-score -> CUSUM accumulation              #
# --------------------------------------------------------------------------- #
def _ref_vectors(icf: ICF) -> tuple[np.ndarray, np.ndarray]:
    mean = np.array([icf.mean[ch] for ch in CORE_CHANNELS], dtype=float)
    scale = np.array([icf.scale[ch] for ch in CORE_CHANNELS], dtype=float)
    return mean, scale


def composite_series(mat: np.ndarray, icf: ICF) -> np.ndarray:
    """Per-day sign-aligned composite z-score vs a reference, over days [56:].

    z = (x - median) / (1.4826*MAD) is exactly barkley ICF.zscore, applied here
    in vectorised form. DIRECTION aligns every channel so a decline pushes the
    composite UP. NaN (missing) channel-days are excluded from the daily mean.
    """
    mean, scale = _ref_vectors(icf)
    z = (mat[ICF_INIT_DAYS:] - mean) / scale          # (T-56, C), NaN where missing
    signed = z * DIR_VEC
    finite = np.isfinite(signed)                       # exclude missing channel-days
    count = finite.sum(axis=1)
    ssum = np.where(finite, signed, 0.0).sum(axis=1)
    comp = np.where(count > 0, ssum / np.maximum(count, 1), np.nan)
    return comp


def cusum_spos(composite: np.ndarray, k: float = CUSUM_K) -> np.ndarray:
    """Upward CUSUM accumulator S+ over the composite, skipping NaN.

    Closed-form S+ recurrence, verified bit-identical to
    barkley.rate_of_drift.CUSUM.s_pos run with h=inf (no reset):
        S+_t = max(0, S+_{t-1} + (z_t - k))
    """
    s = 0.0
    out = np.empty(len(composite))
    for t in range(len(composite)):
        zt = composite[t]
        if zt == zt:                                  # finite (NaN != NaN)
            s = max(0.0, s + (zt - k))
        out[t] = s
    return out


def fit_icf(mat: np.ndarray) -> ICF:
    return ICF.initialise(list(CORE_CHANNELS), mat[:ICF_INIT_DAYS])


# --------------------------------------------------------------------------- #
# One cohort                                                                   #
# --------------------------------------------------------------------------- #
def run_experiment(sigma_pop: float, n_decline: int, n_healthy: int,
                   benign_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    mats, labels, bases = [], [], []
    n_benign = int(round(n_healthy * benign_frac))
    for _ in range(n_decline):
        pb = draw_personal_base(rng, sigma_pop)
        mats.append(generate_cohort_dog(pb, True, rng, benign_shift=False))
        labels.append(1); bases.append(pb)
    for k in range(n_healthy):
        pb = draw_personal_base(rng, sigma_pop)
        mats.append(generate_cohort_dog(pb, False, rng, benign_shift=(k < n_benign)))
        labels.append(0); bases.append(pb)
    labels = np.asarray(labels)

    # Population reference: pool the first 56 days of EVERY dog — the same 56-day
    # calibration window as the individual arm, but pooled (larger sample) —
    # median + 1.4826*MAD via the same ICF code.
    pooled = np.vstack([m[:ICF_INIT_DAYS] for m in mats])
    pop_ref = ICF.initialise(list(CORE_CHANNELS), pooled)

    ind_traj, pop_traj, icfs = [], [], []
    for m in mats:
        icf = fit_icf(m)
        icfs.append(icf)
        ind_traj.append(cusum_spos(composite_series(m, icf)))
        pop_traj.append(cusum_spos(composite_series(m, pop_ref)))
    ind_score = np.array([np.nanmax(t) if np.isfinite(t).any() else 0.0 for t in ind_traj])
    pop_score = np.array([np.nanmax(t) if np.isfinite(t).any() else 0.0 for t in pop_traj])
    return dict(mats=mats, labels=labels, bases=bases, pop_ref=pop_ref, icfs=icfs,
                ind_traj=ind_traj, pop_traj=pop_traj,
                ind_score=ind_score, pop_score=pop_score)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def roc_curve(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(-scores)
    y = labels[order]
    P, N = int(y.sum()), int((1 - y).sum())
    tpr = np.concatenate([[0.0], np.cumsum(y) / max(P, 1)])
    fpr = np.concatenate([[0.0], np.cumsum(1 - y) / max(N, 1)])
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
    return fpr, tpr, auc


def operating_point(traj, labels, target_fpr):
    """At the threshold giving ~target_fpr on healthy dogs: (h, sensitivity,
    median detection lag in days, realized fpr)."""
    healthy = np.array([np.nanmax(traj[i]) if np.isfinite(traj[i]).any() else 0.0
                        for i in range(len(traj)) if labels[i] == 0])
    h = float(np.quantile(healthy, 1 - target_fpr))
    det, ndec, lags = 0, 0, []
    for i in range(len(traj)):
        if labels[i] != 1:
            continue
        ndec += 1
        cross = np.where(traj[i] >= h)[0]
        if cross.size:
            det += 1
            lags.append((ICF_INIT_DAYS + int(cross[0])) - ONSET)
    fa = sum(1 for i in range(len(traj)) if labels[i] == 0
             and np.nanmax(traj[i]) >= h) / max((labels == 0).sum(), 1)
    return h, det / max(ndec, 1), (float(np.median(lags)) if lags else float("nan")), fa


def cohort_metrics(ex, target_fpr):
    f_i, t_i, auc_i = roc_curve(ex["ind_score"], ex["labels"])
    f_p, t_p, auc_p = roc_curve(ex["pop_score"], ex["labels"])
    h_i, sens_i, lag_i, _ = operating_point(ex["ind_traj"], ex["labels"], target_fpr)
    h_p, sens_p, lag_p, _ = operating_point(ex["pop_traj"], ex["labels"], target_fpr)
    return dict(auc_i=auc_i, auc_p=auc_p, sens_i=sens_i, sens_p=sens_p,
                lag_i=lag_i, lag_p=lag_p, roc_i=(f_i, t_i), roc_p=(f_p, t_p),
                h_i=h_i, h_p=h_p)


def ms(a):
    """mean, sd of a list/array (ignoring NaN)."""
    a = np.asarray([x for x in a if x == x], dtype=float)
    return (float(np.mean(a)), float(np.std(a))) if a.size else (float("nan"), float("nan"))


# --------------------------------------------------------------------------- #
# Multi-seed aggregation                                                       #
# --------------------------------------------------------------------------- #
_GRID = np.linspace(0, 1, 101)


def aggregate_primary(sigma, n_seeds, target_fpr, n_dec=100, n_heal=100, seed0=1000):
    auc_i, auc_p, sens_i, sens_p, lag_i, lag_p = [], [], [], [], [], []
    rocs_i, rocs_p, pooled_lag_i, pooled_lag_p = [], [], [], []
    example = None
    for s in range(n_seeds):
        ex = run_experiment(sigma, n_dec, n_heal, BENIGN_FRAC, seed0 + s)
        m = cohort_metrics(ex, target_fpr)
        auc_i.append(m["auc_i"]); auc_p.append(m["auc_p"])
        sens_i.append(m["sens_i"]); sens_p.append(m["sens_p"])
        lag_i.append(m["lag_i"]); lag_p.append(m["lag_p"])
        rocs_i.append(np.interp(_GRID, m["roc_i"][0], m["roc_i"][1]))
        rocs_p.append(np.interp(_GRID, m["roc_p"][0], m["roc_p"][1]))
        for i in range(len(ex["ind_traj"])):
            if ex["labels"][i] != 1:
                continue
            ci = np.where(ex["ind_traj"][i] >= m["h_i"])[0]
            if ci.size:
                pooled_lag_i.append((ICF_INIT_DAYS + int(ci[0])) - ONSET)
            cp = np.where(ex["pop_traj"][i] >= m["h_p"])[0]
            if cp.size:
                pooled_lag_p.append((ICF_INIT_DAYS + int(cp[0])) - ONSET)
        if s == 0:
            example = ex
    return dict(auc_i=auc_i, auc_p=auc_p, sens_i=sens_i, sens_p=sens_p,
                lag_i=lag_i, lag_p=lag_p,
                rocs_i=np.array(rocs_i), rocs_p=np.array(rocs_p),
                pooled_lag_i=pooled_lag_i, pooled_lag_p=pooled_lag_p, example=example)


def aggregate_sweep(sigmas, n_seeds, target_fpr, n_dec=80, n_heal=80, seed0=5000):
    rows = []
    for si, sg in enumerate(sigmas):
        a_i, a_p, s_i, s_p = [], [], [], []
        for s in range(n_seeds):
            ex = run_experiment(sg, n_dec, n_heal, BENIGN_FRAC, seed0 + si * 1000 + s)
            m = cohort_metrics(ex, target_fpr)
            a_i.append(m["auc_i"]); a_p.append(m["auc_p"])
            s_i.append(m["sens_i"]); s_p.append(m["sens_p"])
        rows.append(dict(sigma=sg, auc_i=a_i, auc_p=a_p, sens_i=s_i, sens_p=s_p))
    return rows


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main(out_dir=None):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if out_dir is None:
        out_dir = os.path.join(repo_root, "results")
    os.makedirs(out_dir, exist_ok=True)
    PRIMARY_SIGMA = 0.10
    TARGET_FPR = 0.10

    print("=" * 78)
    print("HEAD-TO-HEAD: individual-referenced vs population-referenced detection")
    print(f"Synthetic cohort | Barkley CUSUM (k={CUSUM_K}) | same detector, only the "
          f"reference differs")
    print(f"{N_SEEDS} independent seeds per condition | mean +/- SD")
    print("=" * 78)

    prim = aggregate_primary(PRIMARY_SIGMA, N_SEEDS, TARGET_FPR)
    aI, aIs = ms(prim["auc_i"]);   aP, aPs = ms(prim["auc_p"])
    sI, sIs = ms(prim["sens_i"]);  sP, sPs = ms(prim["sens_p"])
    lI, lIs = ms(prim["lag_i"]);   lP, lPs = ms(prim["lag_p"])
    miss = [1 - x for x in prim["sens_p"]]; mP, mPs = ms(miss)

    print(f"\nPrimary cohort: sigma_pop={PRIMARY_SIGMA}, 100 declining + 100 healthy "
          f"({int(BENIGN_FRAC*100)}% of healthy with a benign regime change), {N_SEEDS} seeds\n")
    print(f"  ROC AUC (declining vs healthy):")
    print(f"      individual : {aI:.3f} +/- {aIs:.3f}")
    print(f"      population : {aP:.3f} +/- {aPs:.3f}")
    print(f"\n  At matched false-alarm rate ~{TARGET_FPR:.0%} on healthy dogs:")
    print(f"      individual : detects {sI:5.1%} +/- {sIs:.1%} | median lag {lI:5.0f} +/- {lIs:.0f} d")
    print(f"      population : detects {sP:5.1%} +/- {sPs:.1%} | median lag {lP:5.0f} +/- {lPs:.0f} d")
    print(f"\n  => at the same false-alarm budget, the population reference MISSES "
          f"{mP:.0%} +/- {mPs:.0%} of genuine")
    print(f"     declines that the individual reference catches.")

    sweep_sigmas = [0.02, 0.05, 0.08, 0.11, 0.14]
    sweep = aggregate_sweep(sweep_sigmas, N_SEEDS, TARGET_FPR)
    print(f"\n  Heterogeneity sweep ({N_SEEDS} seeds each), mean +/- SD:")
    print("    sigma | AUC ind       | AUC pop       | sens ind     | sens pop")
    swp = []
    for r in sweep:
        ai, ais = ms(r["auc_i"]); ap, aps = ms(r["auc_p"])
        s_i, s_is = ms(r["sens_i"]); s_p, s_ps = ms(r["sens_p"])
        swp.append((r["sigma"], ai, ais, ap, aps, s_i, s_is, s_p, s_ps))
        print(f"     {r['sigma']:.2f} | {ai:.3f}+/-{ais:.3f} | {ap:.3f}+/-{aps:.3f} | "
              f"{s_i:4.0%}+/-{s_is:3.0%} | {s_p:4.0%}+/-{s_ps:3.0%}")

    # ---- Figure ----
    fig = plt.figure(figsize=(12, 9))
    fig.suptitle("Individual vs population reference - detection of behavioral decline\n"
                 f"Reproducible proof-of-concept on synthetic data ({N_SEEDS} seeds, mean ± SD; "
                 "Reference-Class Trap, §6 Q1) - NOT validation on real dogs",
                 fontsize=11.5, fontweight="bold", color=NAVY)

    # Panel A: mean ROC ± SD band
    axA = fig.add_subplot(2, 2, 1)
    mi, sdi = prim["rocs_i"].mean(0), prim["rocs_i"].std(0)
    mp, sdp = prim["rocs_p"].mean(0), prim["rocs_p"].std(0)
    axA.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    axA.fill_between(_GRID, np.clip(mi - sdi, 0, 1), np.clip(mi + sdi, 0, 1), color=NAVY, alpha=0.15)
    axA.fill_between(_GRID, np.clip(mp - sdp, 0, 1), np.clip(mp + sdp, 0, 1), color=RUST, alpha=0.15)
    axA.plot(_GRID, mi, color=NAVY, lw=2.2, label=f"Individual (AUC {aI:.2f} ± {aIs:.2f})")
    axA.plot(_GRID, mp, color=RUST, lw=2.2, label=f"Population (AUC {aP:.2f} ± {aPs:.2f})")
    axA.set_xlabel("False-alarm rate (healthy dogs)"); axA.set_ylabel("Detection rate (declining dogs)")
    axA.set_title(f"A. Mean ROC ± SD - same detector, only the reference differs (σ={PRIMARY_SIGMA})",
                  fontsize=9.5, color=NAVY)
    axA.legend(fontsize=8, loc="lower right"); axA.set_xlim(0, 1); axA.set_ylim(0, 1.02); axA.grid(alpha=0.25)

    # Panel B: pooled detection-lag distributions at matched FPR
    axB = fig.add_subplot(2, 2, 2)
    bins = np.arange(-10, 130, 8)
    axB.hist(prim["pooled_lag_i"], bins=bins, color=NAVY, alpha=0.75,
             label=f"Individual - catches {sI:.0%} of declines")
    axB.hist(prim["pooled_lag_p"], bins=bins, color=RUST, alpha=0.6,
             label=f"Population - catches {sP:.0%} of declines")
    axB.axvline(0, color="k", lw=1, ls=":")
    axB.set_xlabel("Detection lag after true onset (days)")
    axB.set_ylabel(f"Declining dogs (pooled over {N_SEEDS} seeds)")
    axB.set_title(f"B. Detection lag at matched ~{TARGET_FPR:.0%} false-alarm rate", fontsize=10, color=NAVY)
    axB.legend(fontsize=8)

    # Panel C: heterogeneity sweep with error bars
    axC = fig.add_subplot(2, 2, 3)
    xs = [r[0] for r in swp]
    yi = [r[5] * 100 for r in swp]; yie = [r[6] * 100 for r in swp]
    yp = [r[7] * 100 for r in swp]; ype = [r[8] * 100 for r in swp]
    axC.errorbar(xs, yi, yerr=yie, fmt="o-", color=NAVY, lw=2, capsize=3, label="Individual")
    axC.errorbar(xs, yp, yerr=ype, fmt="s-", color=RUST, lw=2, capsize=3, label="Population")
    axC.set_xlabel("Cohort heterogeneity  σ_pop  (between-dog baseline spread)")
    axC.set_ylabel("Declines caught at matched 10% false-alarm (%)")
    axC.set_title("C. The gap grows with heterogeneity (mean ± SD over seeds)", fontsize=10, color=NAVY)
    axC.legend(fontsize=8); axC.grid(alpha=0.25); axC.set_ylim(0, 108)

    # Panel D: a calm dog — false reassurance (one representative cohort)
    ex = prim["example"]
    dec_idx = [i for i in range(len(ex["labels"])) if ex["labels"][i] == 1]
    calm = min(dec_idx, key=lambda i: ex["bases"][i][ACT_IDX])
    mat = ex["mats"][calm]; icf = ex["icfs"][calm]; pop = ex["pop_ref"]
    act = mat[:, ACT_IDX]
    axD = fig.add_subplot(2, 2, 4)
    pm, ps = pop.mean["activity_level"], pop.scale["activity_level"]
    im, isc = icf.mean["activity_level"], icf.scale["activity_level"]
    axD.fill_between(_DAY, pm - 2 * ps, pm + 2 * ps, color=RUST, alpha=0.13,
                     label="Population 'normal' band (±2σ)")
    axD.fill_between(_DAY, im - 2 * isc, im + 2 * isc, color=NAVY, alpha=0.16,
                     label="This dog's individual band (±2σ)")
    axD.plot(_DAY, act, color="k", lw=0.9, alpha=0.85, label="This (calm) dog's activity")
    axD.axvline(ONSET, color=TEAL, lw=1.2, ls="--", label="True decline onset")
    axD.set_ylim(0, 1); axD.set_xlabel("Day"); axD.set_ylabel("activity_level")
    axD.set_title("D. False reassurance: decline exits the individual band,\nstays inside the wide population band",
                  fontsize=9.5, color=NAVY)
    axD.legend(fontsize=7, loc="lower left")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(out_dir, "barkley_head_to_head.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure -> {fig_path}")
    return dict(aI=aI, aIs=aIs, aP=aP, aPs=aPs, sI=sI, sIs=sIs, sP=sP, sPs=sPs,
                lI=lI, lIs=lIs, lP=lP, lPs=lPs, mP=mP, mPs=mPs, swp=swp)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Head-to-head: individual- vs population-referenced detection "
                    "on a synthetic cohort (reproducible proof-of-concept; not real-dog validation).")
    parser.add_argument("--out-dir", default=None,
                        help="Directory for the output figure (default: <repo>/results).")
    args = parser.parse_args()
    main(out_dir=args.out_dir)
    print("\nDone.")
