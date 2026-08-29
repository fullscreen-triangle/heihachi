# Implementation

Two halves, and the split is the point.

**`daemon/`** — Rust. The runtime graph, both language front ends, the
analysis rungs, and the integrations. It holds the compute and it holds
your files, and it binds to loopback.

**`web/`** — TypeScript. The bench: three collapsable columns, an editor,
and the views over a run. It computes nothing. It sends source to be
checked and renders what the graph says.

That division is why the page can be served from anywhere while everything
that touches your work stays on your machine.

---

## Running it

```bash
# once
cd daemon && cargo build --release
cd ../web && npm install

# every session
cd daemon
cargo run --release -- serve --watch "D:\FL Renders" --workspace ../workspace

# in another terminal
cd web && npm run dev
```

The daemon prints a pairing token. Paste it into the page and press Pair.

```
  heihachi daemon
  ───────────────────────────────────────────────
  listening   http://127.0.0.1:7749
  token       hk_9f3a2b7c4e1d8a6b5c3f2e19
  workspace   ../workspace
  watching    D:\FL Renders
  ollama      http://127.0.0.1:11434 [llama3.2] reachable
  ───────────────────────────────────────────────
```

### CLI, without the browser

```bash
heihachi check workspace/mishima/recall.mma   # diagnostics; exits 1 if refused
heihachi observe "D:\FL Renders\bounce_v17.wav"
```

---

## What FL Studio integration actually is

FL exposes no general remote-control API. Anything claiming to "connect to
FL" is either a MIDI controller script (narrow, needs installing into FL,
reads transport and mixer only) or it is watching the filesystem. This
build watches the filesystem, and that is a deliberate choice rather than a
shortfall: nothing is installed into FL, nothing is controlled remotely,
and the integration cannot break when FL updates.

You render as normal. The daemon notices the file, measures it, reads the
`.flp` beside it for the device chain, asks the model for distinctions, and
commits a node to the runtime graph. Watching is debounced by two seconds
because a DAW writes a render in bursts.

Measured on a synthesised bass, the floors are derived from the instrument
rather than asserted:

```
level.peak                 -3.075 dBFS  floor 0.00038   (quantisation step)
level.crest                 6.874 dB    floor 0.00076   (two levels differenced)
spectrum.centroid         221.006 Hz    floor 10.76660  (44100/4096, the bin width)
spectrum.low_ratio          0.806       floor 0.00049   (one bin of the band)
stereo.correlation          1.000       floor 0.00389   (1/sqrt(n))
```

**What is parsed from `.flp`**: the outer event stream — plugin names,
channel names, tempo. The format is undocumented; anything unrecognised is
skipped rather than guessed at, and an unreadable project yields a note
rather than a failure.

**Only WAV is decoded.** Other formats yield a summary carrying a note,
which is a legal value: the rung ran and drew no measurements.

---

## What Ollama is allowed to do

Two jobs, both narrow.

**A constructor rung.** Given the measurements of a render it returns the
distinctions the material draws — `sub_heavy`, `short_decay`,
`wide_stereo` — and nothing else. It does not rank, score, or decide, and
there is no path by which its output overrides another rung. Quality words
are filtered at the boundary rather than argued with: a reply containing
`good`, `muddy`, `professional` has those terms dropped.

**A composer.** It turns "find the dark rolling bass from last winter" into
`.mma` source, which lands in the editor **for review and is not run**. The
composed source is checked before it is returned, so a model that produced
something ill-formed is caught here rather than at the point you press Run.
In testing, llama3.2 wrote `toward model(transient)` without braces and the
checker refused it — which is the boundary working.

Why this confinement is safe rather than merely cautious: a hallucinated
distinction adds a contact to the graph, so the cut computed over it is
still correct and the cell merely comes out coarser. Extraction error
coarsens; it does not corrupt. During testing the model called a signal
with correlation `1.000` "wide_stereo", which is wrong — and it cost
resolution, not validity.

---

## The IDE

```
┌──────────┬────────────────────────┬─────────────────────┐
│ files    │ editor                 │ output              │
│          │                        │                     │
│ mishima/ │  floor 0.02            │ [mishima] [sangoma] │
│  recall  │                        │ [graph] [studio]    │
│ sangoma/ │  seek reese_growl      │                     │
│  reese   │    not { thin, mono }  │  record   20        │
│          │    ...                 │  emissions 20       │
│          │                        │  anomalies  2       │
│          ├────────────────────────┤                     │
│          │ describe what you want │                     │
└──────────┴────────────────────────┴─────────────────────┘
```

Both side columns collapse. The gutter marks lines carrying diagnostics.

**The output tabs are four views over one run**, not four tools: mishima
output, sangoma output, the runtime graph, and the studio. The graph is
laid out on a fixed circle rather than force-directed, deliberately — a
layout that moves while you read it makes it impossible to tell whether a
node appeared or merely drifted, and watching nodes appear is the point.

**There is no pass/fail anywhere in the output.** The stats row shows
record, emissions, anomalies, edges. That absence is load-bearing: the
runtime holds no expectation, so it has nothing to compare an achieved
value against, and an interface that synthesised a verdict would be adding
a claim the system cannot support.

A contested closure renders as a `decline` box carrying the classes reached
and, where one exists, the probe that would separate them. It is styled as
information, not as an error.

---

## Security

- The daemon binds `127.0.0.1` by default. **Leave it there.** The bind
  address is the actual boundary; the token is what lets a page served from
  elsewhere reach a machine that is already only reachable locally.
- Tokens are compared in constant time.
- File paths are resolved against the workspace and refused if they escape
  it — `../`, absolute paths, and directories other than `mishima/` and
  `sangoma/` are all rejected.
- CORS is open because the token authorises and the bind confines. If you
  move the bind off loopback, that reasoning no longer holds.

---

## Layout

```
daemon/
├── src/
│   ├── graph/
│   │   ├── mod.rs        the kernel: nodes, chunks, values, the record
│   │   └── cut.rs        contact graphs, max-flow, accountability
│   ├── lang/
│   │   ├── lexer.rs      one lexer; no zero-resolution literal is writable
│   │   ├── mishima.rs    .mma: the mandatory `not`, ladders, closure
│   │   └── sangoma.rs    .sgn: targets, reachability, headroom, latency
│   ├── integrations/
│   │   ├── analysis.rs   WAV decoding and measurement with derived floors
│   │   ├── fl.rs         export watching and .flp parsing
│   │   └── ollama.rs     the constructor rung
│   ├── studio.rs         render → node
│   ├── compose.rs        English → source, checked before it is returned
│   ├── server.rs         HTTP + WebSocket on loopback
│   └── main.rs           the CLI
└── examples/             the worked programs from the papers

web/
├── src/
│   ├── lib/
│   │   ├── protocol.ts   mirrors the Rust types
│   │   ├── daemon.ts     pairing, HTTP, socket with backoff
│   │   └── dom.ts        the whole rendering layer
│   ├── panels/output.ts  the four views
│   ├── main.ts           the bench
│   └── styles.css
└── index.html
```

## Status

**48 Rust tests pass; the web IDE typechecks under `strict` and builds to
17.5 KB of JavaScript with no runtime dependencies.**

Verified end to end on this machine: a render dropped into a watched folder
was measured, described by llama3.2, and committed to the graph; both
worked examples check clean and run; every refusal fires with a remedy.

**Not built yet**, and each is a real gap rather than a detail:

- **No audio is processed.** `sangoma` checks a chain and reports whether a
  target is reachable; it does not render one. Plugin hosting (CLAP or
  VST3) is the wall this eventually meets.
- **Stage and rung powers are declared, not calibrated.** The reachability
  check is only as good as those numbers, and establishing what a
  particular saturator actually closes on particular material is a
  measurement problem nothing here solves.
- **`seek` reaches classes by name matching**, not by the full
  reachability-then-necessity computation the papers specify. Contested
  closure is detected; the `nec ∘ seek` ordering is not implemented.
- **The graph is in memory.** It does not survive a daemon restart, which
  makes the accumulating record — the entire point — not yet accumulating.
  Persistence is the next thing that matters.
