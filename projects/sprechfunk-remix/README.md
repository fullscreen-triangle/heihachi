# Sprech Funk remix — a worked project

A real project set up against the Neonlight *Sprech Funk* remix stems, so
the tutorial has something to bite on. Every number in the `.sgn` files here
was measured off the actual stems with `heihachi observe` — none of them is
a guess.

## Read this first

**The system does not process audio.** It does not apply effects, render
chains, or master anything. FL Studio does all of that. What this does is
measure what you render, record it, and check your declared targets against
what your stages can actually reach.

So the workflow is: **build in FL → render → the daemon measures and records
→ you compare against the reference and your targets → iterate.**

## Layout

```
sprechfunk-remix/
├── reference/
│   ├── DSCOPE004_4_...2020Remaster.wav   the release, as a target
│   └── stem-profile.txt                  all 44 stems, measured
├── renders/                              FL exports here
└── workspace/
    ├── sangoma/  master.sgn  bass.sgn  drums.sgn
    └── mishima/  find_bass.mma  compare_master.mma
```

## The reference, measured

```
DSCOPE004_4_Neonlight-SprechFunk_2020Remaster
  level.peak                 -0.200 dBFS   resolved to 0.00010
  level.rms                  -8.998 dBFS   resolved to 0.00010
  level.crest                 8.798 dB     resolved to 0.00020
  level.sounding              0.965        resolved to 0.00000
  spectrum.centroid        5264.797 Hz     resolved to 10.76660
  spectrum.low_ratio          0.130        resolved to 0.00049
  stereo.correlation          0.853        resolved to 0.00029
```

This is what a released neurofunk master actually measures. `master.sgn`
targets a band around it rather than trying to hit it exactly.

## What the stems tell you

From `reference/stem-profile.txt`, the parts that matter most:

| Stem | Peak | Crest | Sounding | Centroid | Corr |
|---|---:|---:|---:|---:|---:|
| Kick Drum (clipped) | **+2.70** | 11.21 | 0.283 | 2095.9 | 1.000 |
| Snare Drum | −4.20 | 18.39 | 0.377 | 5938.7 | 0.967 |
| sticky Hi Hat | −14.55 | 21.92 | 0.410 | 10368.2 | 0.770 |
| Sub Bass | −4.15 | 7.44 | 0.414 | **59.5** | **1.000** |
| Sub Bass FÄT HOOK | −2.95 | 10.02 | 0.149 | 59.4 | 1.000 |
| High Voltage Bass | −8.30 | 18.27 | 0.132 | 1066.9 | 0.847 |
| Mid + Top FÄT HOOK | −10.66 | 17.07 | 0.135 | 1615.5 | 0.807 |

Three things worth reading off this table:

**The kick peaks at +2.70 dBFS.** It is genuinely over full scale, and
Neonlight named the file `(clipped)` — the measurement agrees with the
producer's own label. If you use it as shipped you are working with a
clipped kick deliberately, which is a normal choice in this genre and now
an informed one.

**The bass is split, and the split is the design.** Sub sits at 59.5 Hz with
correlation 1.000 — perfectly mono, as sub must be to translate on a club
system. Mid+Top sits at 1615 Hz with correlation 0.807 and *zero* low-band
energy. Everything wide and moving is above; everything below is mono. That
is most of what makes a neurofunk bass work, and it is visible in two
numbers.

**`level.sounding` explains the levels.** Most stems play for 13–41% of the
track. That is why RMS is measured over the sounding part only — averaging
the silence in would report a level the material never has.

## Working through it

### 1. Start the daemon, then render

```powershell
cd micro-kernel\daemon
cargo run --release -- serve `
  --watch "C:\Users\kunda\Documents\audio\heihachi\projects\sprechfunk-remix\renders" `
  --workspace "C:\Users\kunda\Documents\audio\heihachi\projects\sprechfunk-remix\workspace"
```

The line continuation in PowerShell is a backtick, not a backslash. If a
paste loses them you get `Ausdruck fehlt nach dem unären Operator "--"` --
put the whole command on one line instead.

**Start the daemon before rendering.** It watches for changes, so a file
already sitting in the folder is not picked up.

In FL, set the export path to `renders/`. Import the stems you want from
`samples/NEONLIGHT-SPRECH FUNK_Remix Stems/`.

### 2. Measure before you target

`heihachi` is not installed on your path by the build. From `micro-kernel/`:

```powershell
$env:Path += ";$PWD\daemon\target\release"
```



```powershell
heihachi observe "renders\bass_v1.wav"
```

Copy the resolutions from the right-hand column into your `.sgn` targets.
That is where the numbers in `bass.sgn` came from.

### 3. Check a target

The checker needs to know your analysis resolution:

```powershell
heihachi check --backend-resolution 0.00001 workspace\sangoma\bass.sgn
```

Without the flag it assumes 0.001, and a file declaring `floor 0.0001` is
refused with `rule:floor-negotiation` — correctly, since that is finer than
the assumed path can resolve.

### 4. Iterate

Re-render to the same filename as often as you like. Each render is recorded
separately, so the record holds the whole revision history:

```
bass_v1    corr 0.999  low 0.899  centroid  231.7
bass_v1    corr 0.986  low 0.683  centroid  591.9
```

That is one filename, two bounces, with the sub pulled back and the mid
pushed up between them. Against `bass.sgn`'s `mid_layer` target — correlation
0.70–0.90, low_ratio ≤ 0.10 — both still fail, and you can see how far.

## About the floors

Every `.sgn` here declares `floor 0.0001` rather than the `0.02` in the
worked examples. The reason matters:

**The over-claim rule checks every target against the single ambient floor.**
So the floor has to be at least as fine as your sharpest target. Level
measurements on a 32-bit float render resolve to about 0.0001 dB, and
`correlation >= 0.75#0.0003` needs a floor at or below 0.0003. Declaring
`floor 0.02` refuses both.

**The rule does not know which instrument a target uses.** Writing
`centroid >= 4200.0#0.5` under `floor 0.0001` *passes*, even though no FFT
here resolves half a hertz. The checker cannot catch that. Run
`heihachi observe` first and copy the real figure — 10.77 Hz, the bin width
— which is what these files do.

## About the model

Ollama draws distinctions and nothing else — it does not rank or judge.
It will sometimes be wrong: on these very bounces, llama3.2 returned
`wide_stereo` for a signal measuring `stereo.correlation = 0.999`, which is
as mono as a file gets. It has done this on every test so far.

That is survivable by design. A wrong distinction adds a contact to the
graph, so the cut computed over it is still correct and the cell comes out
coarser rather than wrong. The model's error costs resolution, never
validity — which is why it is allowed to participate at all, and why you
should read its output as one rung among several rather than as an
assessment.

## Known limits, in this project specifically

- **The graph does not survive a daemon restart.** Dump it before you stop:
  ```powershell
  curl.exe -s -H "Authorization: Bearer YOUR_TOKEN" `
    http://127.0.0.1:7749/api/graph > session.json
  ```
  Use `curl.exe`, not `curl` -- in PowerShell the bare name is an alias for
  `Invoke-WebRequest`, which takes neither `-s` nor `-H`.
- **Rung powers are declared, not calibrated.** The `at 0.45` figures in
  these files are estimates. The reachability check is exactly as reliable
  as they are.
- **No `.flp` is parsed here** because these renders have no project beside
  them. Save your FL project next to the render and the device chain is
  recovered too.
- **Only WAV is measured.** Render WAV; the contest spec asked for 24-bit
  44.1 or 48 kHz anyway.
