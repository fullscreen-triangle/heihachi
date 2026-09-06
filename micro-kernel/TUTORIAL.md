# Tutorial: a first project

This walks through one session end to end — bringing in one-shots, rendering
versions, and asking questions of what you recorded.

## Read this first

**The system does not process audio.** It cannot apply a saturator to a
snare, cannot render an effect chain, and cannot master a track. FL Studio
remains your processor and nothing here replaces it.

What the system does is **measure what you rendered, record the decisions
around it, and let you ask questions later**. `sangoma` declares what a
sound must satisfy and tells you whether your chain can get there; it does
not build the chain. If you came looking for an effects processor, this is
not one, and no amount of configuration will make it one.

Two further limits, both of which you will hit in this tutorial:

- **The graph is held in memory.** Stop the daemon and everything recorded
  is gone. For a trial that is fine; for real accumulation it is not, and
  it is the next thing to be built.
- **The daemon only sees files that appear while it is running.** Renders
  already sitting in the folder when it starts are not picked up.

---

## 1. Setup

### Build once

```bash
cd micro-kernel/daemon
cargo build --release

cd ../web
npm install
```

The release build takes several minutes the first time — it fetches around
250 crates and links with LTO. Let it finish; if you interrupt it you get no
binary. `cargo build` (debug) completes in well under a minute and behaves
identically, and is fast enough for one-shots. Everything below works with
either; substitute `cargo run --` for `cargo run --release --` if you use
the debug build.

### Put `heihachi` on your path

The build leaves the binary at `micro-kernel/daemon/target/release/heihachi.exe`.
It is not installed anywhere, so `heihachi observe ...` will not resolve
until you say where it is. For this session:

```powershell
$env:Path += ";$PWD\daemon\target\release"     # from micro-kernel/
```

Or install it properly, once:

```powershell
cargo install --path micro-kernel\daemon
```

Everywhere below writes `heihachi` for brevity. Without one of the above,
substitute the full path, or run it through cargo from `micro-kernel/daemon`:

```powershell
cargo run --release -- observe "C:\path\to\render.wav"
```

### Make a project folder

```
my-project/
├── renders/            where FL exports to
└── workspace/
    ├── mishima/        .mma files
    └── sangoma/        .sgn files
```

The daemon creates `workspace/mishima` and `workspace/sangoma` for you and
seeds each with a worked example on first run.

### Point FL Studio at it

In FL: **Options → File settings → Audio export path**, or simply choose
`my-project/renders` in the export dialogue each time. Nothing is installed
into FL and nothing is configured inside it. FL has no general remote-control
API, so the integration is: you render, the daemon notices.

---

## 2. Start the daemon

**Start it before you render anything.** A file already present when the
daemon starts will not be picked up.

```powershell
# from the repository root -- cargo must be run from this directory
cd micro-kernel\daemon
cargo run --release -- serve `
  --watch "C:\path\to\my-project\renders" `
  --workspace "C:\path\to\my-project\workspace"
```

Those are backticks at the ends of the lines, which is how PowerShell
continues a command; a backslash there is a parse error. On a shell that
uses `\`, substitute it. Either way the command is fine on one line.

```
  heihachi daemon
  ───────────────────────────────────────────────
  listening   http://127.0.0.1:7749
  token       hk_9f3a2b7c4e1d8a6b5c3f2e19
  workspace   C:\path\to\my-project\workspace
  watching    C:\path\to\my-project\renders
  ollama      http://127.0.0.1:11434 [llama3.2] reachable
  ───────────────────────────────────────────────
```

Copy the token. If Ollama reads `UNREACHABLE`, run `ollama serve` in another
terminal — everything else still works without it, and renders will simply
carry a note saying the model rung did not run.

In a second terminal:

```bash
cd micro-kernel/web
npm run dev
```

Open the printed address, paste the token, press **Pair**.

---

## 3. Measure your one-shots

Before writing any target, find out what your samples actually are. This is
the step people skip, and it is the one that makes every later number mean
something.

```bash
heihachi observe "my-project/renders/snare_raw.wav"
```

```
snare_raw
  level.peak                 -1.355 dBFS   resolved to 0.00031
  level.rms                 -20.760 dBFS   resolved to 0.00031
  level.crest                19.404 dB     resolved to 0.00062
  spectrum.centroid       10523.578 Hz     resolved to 10.76660
  spectrum.low_ratio          0.058        resolved to 0.00049
  stereo.correlation          1.000        resolved to 0.00805
  note: project: no .flp found beside this render
```

**Read the right-hand column.** Every measurement states the resolution it
was obtained at, and those are derived from the instrument rather than
asserted:

| Measurement | Floor | Where it comes from |
|---|---|---|
| `level.*` | 0.00031 dB | the quantisation step of a 16-bit source |
| `spectrum.centroid` | 10.77 Hz | the FFT bin width, 44100/4096 |
| `spectrum.low_ratio` | 0.00049 | one bin out of the summed band |
| `stereo.correlation` | 0.00805 | 1/√n for n samples |

A centroid of 10523 Hz means 10523 ± 10.77. Any claim you make finer than
that is a claim about an artefact.

### What the measurements mean

- **`level.crest`** — peak minus RMS, in dB. High means transient-heavy
  (19.4 dB is a raw, unprocessed snare); low means flattened. This is the
  single most useful number for drum work.
- **`spectrum.centroid`** — the spectrum's centre of mass. A hat sits near
  11 kHz, a reese near 200 Hz.
- **`spectrum.low_ratio`** — proportion of energy below 200 Hz. Around 0.8
  for a bass, under 0.1 for a snare.
- **`stereo.correlation`** — 1.0 is mono-compatible, 0 is decorrelated,
  negative is out of phase.

> **A caution about `stereo.correlation`.** It measures phase relationship,
> not level difference. A file whose right channel is the left channel at a
> different gain reads **1.000**, however wide it sounds. Only genuinely
> decorrelated content moves it — independent noise per channel measures
> about −0.07. Most one-shots read 1.000 and that is correct.

### Note the note

`note: project: no .flp found beside this render` — the project rung ran and
found nothing. It is reported rather than left blank, because "no project
found" and "project unreadable" and "project parsed fine" are three different
facts.

Save the `.flp` beside your render and the device chain is recovered too.

---

## 4. Compare two versions

This is the workflow the tool is actually for. Render your raw snare, then
render it again through a saturator, and compare.

```bash
heihachi observe renders/snare_raw.wav
heihachi observe renders/snare_v2_saturated.wav
```

| | raw | saturated | change |
|---|---|---|---|
| `level.peak` | −1.355 dBFS | −4.016 dBFS | −2.7 dB |
| `level.rms` | −20.760 dBFS | −18.000 dBFS | **+2.8 dB** |
| `level.crest` | 19.404 dB | 13.984 dB | **−5.4 dB** |
| `spectrum.centroid` | 10523 Hz | 10620 Hz | +97 Hz (≈9 bins) |

The saturator flattened 5.4 dB of transient and raised RMS by 2.8 dB while
leaving the spectral balance essentially alone. That is a specific,
checkable statement about what the processing did — and it is the kind of
thing that is very hard to hear reliably and very easy to measure.

Run this on every version you bounce. The numbers accumulate into an answer
to "what does this plugin actually do to my drums", which is a better
question than "is this plugin good".

---

## 5. Write a target with `sangoma`

Now that you know the raw snare has 19.4 dB of crest, you can state what you
want the processed one to satisfy.

`workspace/sangoma/snare.sgn`:

```
-- snare.sgn
floor 0.02

medium air { ceiling: -1.0 }

construct snare {
  stage saturate
  stage transient_shape
  stage width

  target {
    crest    >= 12.0#0.5      -- keep some transient; raw was 19.4
    crest    <= 16.0#0.5      -- but flatten it some
    centroid >= 8000.0#10.77  -- the bin width; see the caution below
  }

  via { rung saturate        at 0.40
     >> rung transient_shape at 0.35
     >> rung width           at 0.30 }
}
```

```bash
heihachi check workspace/sangoma/snare.sgn
```

```
snare.sgn:9: error [rule:reachability]
  construct `snare` attains 0.7270 of a required 0.8
  remedy: 4 rungs at the strongest available power (0.4) would reach it
```

Three rungs compose to 1−(0.60)(0.65)(0.70) = 0.727, short of the 0.8 the
checker requires. It refuses **before** you render, and names how far short
you are. Add a fourth stage, or raise a stage's declared power if you have
reason to believe it does more work than you claimed.

### The `#` is not decoration

`crest >= 12.0#0.5` reads "at least 12 dB of crest, asserted to half a
decibel". Writing `12.0#0.001` is refused:

```
error [rule:over-claim]
  target `crest` claims resolution 0.001, finer than the ambient floor 0.02
  remedy: coarsen the target, or declare a finer floor your tools support
```

> **What the over-claim rule does not catch.** It compares your target's
> resolution against the single ambient `floor`, and nothing else. It has no
> idea that a *spectral* target needs the FFT bin width. Writing
> `centroid >= 8000.0#0.5` under `floor 0.02` **passes the checker** even
> though no analysis here resolves half a hertz. You have to know your
> instrument's floor and write it. Run `heihachi observe` first and copy the
> numbers from the right-hand column.

### Declare the floor as your *finest* target

Because every target is checked against the one ambient floor, that floor
has to be at least as fine as your sharpest claim. A file wanting
`correlation >= 0.75#0.0003` cannot declare `floor 0.02` — the checker
refuses it as an over-claim. Declare the finest resolution you actually use:

```
floor 0.0001        -- the level floor of a 32-bit float render
```

Then tell the checker what your analysis path resolves, or it will refuse
the file the other way round:

```bash
heihachi check --backend-resolution 0.00001 workspace/sangoma/master.sgn
```

Without the flag it assumes 0.001 and reports `rule:floor-negotiation`,
which is correct: a program cannot assert distinctions finer than the
machinery computing them.

### What `sangoma` will not do

It will not tell you to put the compressor before the saturator. Stage
ordering and processor choice are matters of taste, and a checker that
refused on those grounds would be enforcing somebody's opinions. It refuses
on physics — headroom, latency, resolution — and on reachability, and is
silent above that.

Declare stage gains and it will check headroom:

```
construct master {
  stage glue     gain 2.0
  stage limiter  gain 3.0
  target { crest >= 6.0#0.5 }
}
```

```
error [rule:headroom]
  declared gains sum to +5.00 dB against a ceiling of -1.00 dB
  remedy: reduce total gain by at least 6.00 dB, or raise the ceiling
```

---

## 6. Render and watch it land

With the daemon running, export from FL into the watched folder. Within a
few seconds:

```
INFO committed snare_v2 (6 measurements, 2 distinctions)
```

In the IDE, the **studio** tab shows a card per render: the measurements,
the devices recovered from the `.flp`, and the distinctions the model drew.
The **runtime graph** tab shows `render.snare_v2` appear as a node.

### About the model's distinctions

Ollama is confined to one job: naming the distinctions the material draws.
It returns things like `short_decay`, `sub_heavy`, `gritty_saturation`. It
does not rank, score, or judge — quality words are filtered out at the
boundary, so a reply containing "muddy" or "professional" has those dropped.

**It will sometimes be wrong.** In testing, llama3.2 twice labelled a signal
with `stereo.correlation = 1.000` as `wide_stereo`, which is simply
incorrect. This is expected and it is survivable: a wrong distinction adds a
contact to the graph, so the cut computed over it is still correct and the
resulting cell is coarser rather than wrong. The model's error costs
resolution, never validity — which is exactly why it is allowed to
participate at all.

Do not treat its output as an assessment. Treat it as one rung among
several, which is all the system treats it as.

---

## 7. Ask a question with `mishima`

`workspace/mishima/find_snare.mma`:

```
-- find_snare.mma
floor 0.02

seek punchy_snare
  not    { flat, dull, mono }
  toward { region(snare_v2) }
  via    { rung spectral   at 0.45
        >> rung annotation at 0.30
        >> rung model      at 0.55 }
  until  closure
  otherwise decline
  yield  found
```

Press **Run**. The `not` clause is mandatory — omit it and the parser
refuses:

```
error [rule:mandatory-not]
  a seek without a `not` clause does not specify a region
  remedy: state what the search excludes, e.g. not { thin, undistorted }
```

That rule exists because the exclusions *are* the specification. When you
reject fifteen versions of a snare and keep the sixteenth, what you rejected
them for describes the survivor far more precisely than any positive
adjective. Sessions throw that away; this does not.

### Three rungs, deliberately unlike

`spectral` measures the material, `annotation` recalls what you wrote at the
time, `model` reads the graph. They fail in unrelated ways, which is the
point. A ladder of one or two rungs is refused under `until closure`:

```
error [rule:coherence]
  ladder of 2 rung(s) cannot close: a support structure of fewer than three
  is not robust to the loss of one
  remedy: add rungs of differing kind until at least three support the result
```

Three prompts to the same model is one rung, not three.

### Declines are results

If the rungs reach several irreconcilable classes, the run returns a
**decline** carrying them and, where one exists, the probe that would
separate them:

```
declined — the evidence does not single one out
reached: render.snare_v2 · render.snare_v5
the probe that would separate them: spectral
```

This is not an error. It means your record genuinely supports two answers
and says which measurement would break the tie. A search that picked one and
hid the other would be claiming more than the record supports.

---

## 8. Composing from English

The bar under the editor sends a request to the model, which writes source
into the editor **for you to read**. It is never run automatically.

Type: *"find the dark rolling bass I made last winter, not the bright ones"*

The composed source is checked before it comes back. In testing, llama3.2
produced `toward model(transient)` — missing braces — and the checker caught
it:

```
error [syntax] expected "{", found "model"
```

Which is the boundary working. Treat composed source as a draft to edit, not
as an answer.

---

## 9. A worked project, with real stems

A complete project set up against the Neonlight *Sprech Funk* remix stems is
in [`projects/sprechfunk-remix/`](../projects/sprechfunk-remix/). It carries
all 44 stems measured, the released master as a reference target, and three
`.sgn` files whose every number was taken from those measurements rather than
guessed. Read its README alongside this one — it shows the same workflow on
material you can actually load into FL.

---

## 10. A realistic first session

1. Start the daemon and the web tool. **Daemon first, before any rendering.**
2. Bring your one-shots into FL. Set the export path to `renders/`.
3. `heihachi observe` each raw sample. Write the numbers down — they are your
   baseline and the source of every floor you will later declare.
4. Build a chain in FL. Render. Watch it land in the studio tab.
5. `heihachi observe` the processed version. Compare crest, RMS, centroid.
6. Write a `.sgn` target for what you want, using resolutions copied from
   step 3. Let the checker tell you if your chain can reach it.
7. Iterate in FL. Every render lands in the graph.
8. When you have a few versions, write a `.mma` seek and see whether the
   record singles one out or declines.

**Before you stop for the day**, remember the graph does not persist. Export
what you want to keep:

```powershell
curl.exe -s -H "Authorization: Bearer YOUR_TOKEN" `
  http://127.0.0.1:7749/api/graph > session-2026-08-30.json
curl.exe -s -H "Authorization: Bearer YOUR_TOKEN" `
  http://127.0.0.1:7749/api/exports > renders-2026-08-30.json
```

`curl.exe` rather than `curl`: in PowerShell the bare name is an alias for
`Invoke-WebRequest`, which takes neither `-s` nor `-H`.

---

## 11. Troubleshooting

**Nothing appears when I render.** The daemon must be started *before* the
file appears; it watches for changes, not for existing files. Re-render, or
touch the file. Check the log says `watching <your path>`.

**Re-rendering the same filename.** This commits again, and is meant to: a
render is identified by path *and* modification time, so bouncing
`bass_v1.wav`, changing something, and bouncing it again records both. The
studio tab will show two cards with the same name and different numbers,
which is the revision history you want.

**`Ausdruck fehlt nach dem unären Operator "--"`** (or, in English,
`Missing expression after unary operator '--'`). PowerShell read a `\` at
the end of a line as an ordinary character rather than a continuation, so
the next line began with a bare `--`. Use a backtick to continue a line, or
put the command on one line.

**`heihachi : Die Benennung "heihachi" wurde nicht erkannt`.** The binary is
not on your path -- see the top of §1. Either add
`micro-kernel\daemon\target\release` to `$env:Path`, `cargo install --path
micro-kernel\daemon`, or run it via `cargo run --release --` from
`micro-kernel/daemon`.

**`can't find library heihachi, rename file to src/lib.rs`.** You ran cargo
from somewhere other than `micro-kernel\daemon`. Cargo searches *parent*
directories for a manifest, and the repository root used to hold a stub one
left over from the Python-era project. That file is now
`Cargo.toml.disabled`; if you see this error, you are on an older checkout.
Either way the fix is the same -- run cargo from `micro-kernel\daemon`, or
call the built binary by its full path from anywhere:

```powershell
C:\...\heihachi\micro-kernel\daemon\target\release\heihachi.exe serve --addr 127.0.0.1:7750
```

**`could not find Cargo.toml in ... or any parent directory`.** Same cause,
clearer message: there is no Rust project where you are standing. `cd` into
`micro-kernel\daemon`.

**`port 7749 is already held by another process`.** Something else is
listening there. The daemon refuses rather than starting -- it will not
print a banner for an address it does not own. Start it elsewhere with
`--addr 127.0.0.1:7750`, or find the holder:

```powershell
Get-NetTCPConnection -LocalPort 7749 | Select-Object OwningProcess
```

**`ollama UNREACHABLE`.** Run `ollama serve`. Everything else works without
it; renders will carry `model: unreachable` as a note.

**`token rejected`.** The token changes on every restart unless you pass
`--token`. Pair again with the current one, or start with
`--token hk_something_stable`.

**`heihachi check` exits 1.** That is a refused program, and it is meant to.
Read the remedy line; every refusal carries one.

**Only WAV is measured.** MP3, FLAC and AIFF are recognised as audio but not
decoded, and the render will carry a note saying so. Render WAV.

**My `.flp` isn't parsed.** Save it beside the render, ideally with the same
stem. The format is undocumented; what is recovered is plugin names, channel
names and tempo. Anything unrecognised is skipped rather than guessed at.

---

## What is not here

Stated plainly so you do not go looking:

- **No audio processing.** No effect chains are applied, nothing is
  rendered, nothing is mastered. `sangoma` declares and checks; FL processes.
- **No persistence.** The graph dies with the daemon.
- **No calibrated rung powers.** The numbers in `via { rung x at 0.40 }` are
  declared by you, not measured. The reachability check is only as good as
  your estimates, and establishing what a particular saturator actually
  closes is a measurement problem nothing here solves.
- **No plugin hosting.** CLAP and VST3 are not attempted.
