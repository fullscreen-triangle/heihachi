# Validation suite

Numerical validation of the claims in *Searching Continuous Audio as a Whole
Item: A Sequence-Individuation Algorithm for Radio, Mixes, and Sets*.

Every object in the paper is a finite weighted graph or a construction on one, so
every theorem is checkable by direct computation. This suite implements one
experiment per structural result, using **exact** minimum cuts (Ford–Fulkerson
via `networkx`) and **exact** edit-distance sequence alignment. All randomness is
seeded (master seed `20260710`), so results are reproducible.

## Run

```bash
python run_all.py
```

Writes one JSON file per category to `results/`, plus `results/summary.json`
with per-experiment pass/fail. Each experiment can also be run on its own
(`python exp_floor.py`, etc.).

## What is checked

| Exp | Paper claim | File |
|-----|-------------|------|
| E1  | Thm 4.3 — resolution floor: σ(v) ≥ β, no sharp cut | `exp_floor.py` |
| E2  | Cor 4.5 — identity is region-valued (min-cut side non-singleton) | `exp_floor.py` |
| E3  | Thm 4.7 — sufficiency stops resolution; insufficiency needs acoustics | `exp_floor.py` |
| E4  | Cor 4.8 — acoustic recognition invoked exactly on the residual | `exp_floor.py` |
| E5  | Thm 5.4 — signature pattern invariant under re-encoding; content non-identity; pattern ⇔ re-encoding | `exp_signature.py` |
| E5b | Thm 5.4(ii) — content set conflates items the signature separates | `exp_signature.py` |
| E6  | Prop 5.7 — ambiguity set shrinks (non-increasing) with context length | `exp_signature.py` |
| E7  | Thm 6.6 — convergence-only admissibility (global, not per-position) | `exp_matching.py` |
| E8  | Cor 6.7 — partial-recognition bound: measured == predicted admissibility | `exp_matching.py` |
| E9  | Prop 6.10 — alignment residual == mis-placed positions | `exp_matching.py` |
| E10 | Thm 7.3 — search–match duality: one symmetric alignment objective | `exp_duality_complexity.py` |
| E11 | Cor 8.5 — whole-item cost < per-track cost when nameable fraction > c_sym/c_ac | `exp_duality_complexity.py` |
| E11b| Thm 8.4 — alignment DP scales ~O(n·N) | `exp_duality_complexity.py` |
| E12 | Sec 9 — the worked example, end-to-end | `exp_worked_example.py` |

## Files

- `core.py` — shared primitives: contact-graph construction, exact min-cut,
  signatures / content sets / re-encodings, sequence alignment (global + local)
  with traceback, least-sufficient-identifier resolution, JSON output.
- `exp_*.py` — the experiments grouped by paper section.
- `run_all.py` — master runner; writes `results/summary.json`.
- `results/` — JSON outputs.

## Scope

The suite verifies each theorem on constructed and randomly drawn instances; it
is an internal-consistency and instance-verification check, not a substitute for
the proofs (which hold for every finite weighted graph and handle signature
satisfying the stated hypotheses). It uses no real audio: `Name` and `Acoustic`
recognisers are modelled abstractly, exactly as the paper specifies them as
implementation inputs the calculus organises but does not itself supply.
