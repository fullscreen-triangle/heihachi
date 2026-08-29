# micro-kernel

A runtime and two languages for making records, organised as an accumulating
record of authored decisions rather than as a pipeline of processing steps.

The runtime is `heihachi`; `mishima` computes over the record it accumulates,
and `sangoma` constructs new material against declared targets.

---

## The idea in one paragraph

A conventional production tool stores state (which devices, what parameters) and
output (what they produced), and stores nothing about the relation between them
— which is what a later question is about. Worse, the relation cannot be
recovered: a bounce is regenerable from a session, a session is not regenerable
from a bounce. So the relation has to be recorded while it is being made. The
runtime does that recording, and it does so without a scheduler, an orchestrator,
or an exit code: nodes are individuated by *subtask*, so two analyses reaching the
same question converge on one node and there is no ordering decision to make;
relevance is constituted by *being read*, so an emission nothing consumes needs no
rejecting; and the graph holds no expectation, so the runtime cannot compare an
achieved value to an intended one and therefore cannot fail in the sense an exit
code encodes. Anomalies are recorded as values and every run reaches completion —
which is the right discipline for a domain where the unexpected reading is
frequently the point.

---

## The three papers

| Paper | Directory | Subject |
|---|---|---|
| I | [`docs/heihachi-runtime-graph/`](docs/heihachi-runtime-graph/) | the runtime: nodes, execution, the record, trajectories, accountability, resolution |
| II | [`docs/mishima-propagator/`](docs/mishima-propagator/) | the computing language (`.mma`): negation, ladders, closure or honest decline |
| III | [`docs/sangoma-instrument-synthesizer/`](docs/sangoma-instrument-synthesizer/) | the constructor language (`.sgn`): stages, targets, reachability |

Each is self-contained: every notion it uses is defined in it, and no other paper
is required to read it. Citations are to external literature only.

### Some results worth knowing about

- **No exit code** (I, Thm 4.2). A verdict presupposes a comparison against an
  expectation; the graph provides no vocabulary in which one can be written, so
  no quantity the runtime computes can carry that meaning.
- **Run to completion** (I, Cor. 4.4). Nothing branches on emission content. Over
  48 chain-length × anomaly-density cells the runtime completes every node while
  a halt-on-error runtime falls to 9.4% and abandons up to 29 nodes.
- **The trajectory is not a schedule** (I, Thm 6.3). Over one fixed six-node
  catalogue, 36 runs induce 6 distinct causal edge relations.
- **A search must say what it excludes** (II, Prop. 3.2). A region individuated
  against nothing is not a region, so a `seek` without a `not` clause is a parse
  error — not a type error.
- **Nesting is forced** (II, Thm 4.2; III, Thm 4.2). Refinement cost diverges
  while return is bounded, so fine resolution is reached by composing ordinary
  distinctions, never by sharpening one.
- **Closure beats confidence** (II, Thm 6.3). Over five registries a fixed
  threshold was met after a *single* probe in every case while closure required
  the whole registry — and every registry terminated contested.
- **Unreachability is a diagnostic** (III, Prop. 5.2). A target that cannot be
  reached is refused before rendering, and the refusal names the depth required.
- **Two claims withdrawn** (II, Rem. 4.9; III, Rem. 10.1). The natural strong form
  of the diversification result is false — repeating the *strongest* rung beats a
  mixed ladder — and the ladder-versus-sharpening comparison is exactly a tie at
  a target of 0.95 under a reciprocal cost law. Both counterexamples are recorded
  by the experiments rather than suppressed.

---

## Validation

```bash
cd validation
python run_all.py       # 40 experiments; writes results/*.json
python check_claims.py  # every figure quoted in a paper, against results/
python check_tex.py     # LaTeX pre-flight: citations, labels, environments
```

No third-party dependencies; standard library only. Every seed is fixed, and
re-running reproduces each experiment record bit-identically.

**Current status: 40 run, 40 passed, 0 failed. 39 quoted figures, 39 confirmed.**

| Suite | Experiments | Covers |
|---|---|---|
| `kernel` | 11 | floor positivity, min-cut against brute force, run to completion, no exit code, monotone record, convergence, total chunk execution, trajectory emergence, non-propagation, blast radius, floor-as-scale |
| `mishima` | 16 | lexer, parser, mandatory `not`, composition law, diversification, saturation, depth, refinement divergence, triangle, coherence, closure vs threshold, decline, water-filling, knapsack, rung-kind invisibility, floor negotiation |
| `sangoma` | 13 | parser, target resolution, over-claim, reachability and depth, detectability, thickness additivity, headroom, latency, floor negotiation, nesting crossover, diagnostics |

Notable checks:

- `kernel_mincut_brute_force` verifies max-flow against a **brute-force** minimum
  over all source–sink bipartitions across 300 random graphs, so the
  accountability theorem is checked against an independent computation rather
  than against itself.
- `kernel_floor_is_scale` sweeps the floor over a sixteen-fold range and finds
  ϱ/β constant to 0 in double precision — the operational test that an
  implementation treats the floor as a scale, not a clamp.
- `kernel_monotone_record` re-runs an identical measurement three times and
  obtains records 401, 402, 403: no memoisation.
- `check_claims.py` cross-checks each figure quoted in a manuscript against the
  JSON record of the experiment it is attributed to, which catches a number
  edited in prose but never re-derived.

---

## Layout

```
micro-kernel/
├── docs/
│   ├── heihachi-runtime-graph/            Paper I   + references.bib
│   ├── mishima-propagator/                Paper II  + references.bib
│   └── sangoma-instrument-synthesizer/    Paper III + references.bib
└── validation/
    ├── hkcore.py          contact graphs, max-flow/min-cut, the runtime,
    │                      ladders and closure, separators, both front ends
    ├── exp_kernel.py      11 experiments
    ├── exp_mishima.py     16 experiments
    ├── exp_sangoma.py     13 experiments
    ├── run_all.py         runner; writes results/ and _summary.json
    ├── check_claims.py    manuscript figures against results/
    ├── check_tex.py       LaTeX structural pre-flight
    └── results/           one JSON per experiment, plus _summary.json
```

## Building the papers

```bash
cd docs/heihachi-runtime-graph
pdflatex heihachi-runtime-graph && bibtex heihachi-runtime-graph \
  && pdflatex heihachi-runtime-graph && pdflatex heihachi-runtime-graph
```

The same three-pass sequence applies to the other two.

## Status

The prototype is a **reference implementation of the specifications**, not a
production runtime. It establishes that the definitions are coherent and
implementable and that an implementation behaves as the theorems predict. It does
not establish that the model is adequate for any particular studio, that the
floors chosen for audio measurements are right, or that the runtime performs at
session scale. There is no audio processing here: the runtime carries values, not
samples, and hosting real plugins is not attempted. Paper I §13, Paper II §15 and
Paper III §12 each state where the design must not be used.
