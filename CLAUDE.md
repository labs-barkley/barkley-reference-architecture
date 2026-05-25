# CLAUDE.md — Barkley Reference Architecture

> This file is read automatically by Claude Code when working in this repository.

## What this repo is

A **source-available research demonstrator** for individual-referenced behavioral
intelligence in dogs. It is not production software, not a validated medical device,
and not a clinical tool of any kind.

---

## Before touching any code

```bash
python -m pytest tests/ -q        # all tests must pass
python -m mypy barkley             # strict mode (settings in pyproject.toml) — zero errors
python examples/run_reference_pipeline.py  # pipeline runs end-to-end
```

Do not suggest a change that breaks any of these three checks.

---

## Non-negotiable framing — read this first

Every edit — code, comment, docstring, README, notebook — must preserve this framing:

**Do not use these terms as capability claims:**
`clinical`, `diagnostic`, `validated threshold`, `early warning`,
`detects disease`, `cognitive decline`, `medical device`, `clinically meaningful`

These terms may appear **only in negative / disclaimer contexts** — e.g.
"not a clinical tool", "no clinical claims", "not a medical device". Never strip
them from such contexts.

**Prefer, for capability language:**
`behavioral signal`, `rate of drift`, `behaviorally informative`,
`research demonstrator`, `population-normative`, `individual-referenced`

**Ontology strictness.** Do not infer, invent, or generate new behavioral categories
or biological states outside the defined Barkley ontology. The representation of the
animal must remain exactly as structurally defined in the reference layer — do not
"enrich", extend, or smooth it.

**Preserve all existing disclaimer comments.** Do not remove, soften, or rephrase
the `# research-only` and `# no clinical claims` annotations already in the code.
They are load-bearing for IP and regulatory positioning.

---

## What belongs here — and what doesn't

This repo implements the **reference architecture layer only**:
- Synthetic data generation and loading
- Baseline formation and rolling window logic
- Rate-of-drift computation
- Silence / informative absence classification

**Do not add:**
- Proprietary DogGraph internals or ICF scoring logic
- Spatial, relational, co-regulatory, or production context-fusion logic
- Feature engineering specifics from Barkley's production pipeline
- Hardcoded thresholds presented as empirically derived from real data
- Any dependency that would change the license posture (check `LICENSE` first)

---

## Data

**Synthetic only.** The pipeline generates or loads synthetic DogGraph data.
Never add code paths that reference, import, or assume real dog telemetry.
If you need realistic-looking data, extend the synthetic generator in `barkley/`.

---

## Code style

- Type-annotate all new functions and methods — mypy must stay clean.
- New public functions get a one-line docstring + a `Note: research demonstrator only.`
  reminder where the function's output could be misread as an operational recommendation.
- Keep dependencies minimal. New imports need a comment explaining why they're necessary.
- This is a Python package (`pyproject.toml`). Editable install: `pip install -e .`

---

## Repo structure reminder

```
barkley/          # core library — type-annotated, mypy-clean
examples/         # runnable reference pipeline
tests/            # pytest suite — must pass at all times
notebooks/        # exploratory only, not part of the library
docs/             # ethics, limitations, scientific positioning
```

---

## License

Source-available, not open source. See `LICENSE`.
Do not introduce dependencies or contributions that conflict with the license terms.
Do not reproduce or adapt this architecture in a way that misrepresents it as
Barkley's production system.

---

*Barkley AI · labs@getbarkley.com · getbarkley.com*
