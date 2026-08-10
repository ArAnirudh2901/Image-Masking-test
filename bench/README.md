# bench — the mask lane's measurement harness

Everything here measures the **refinement lane** (`ai/mask-refine.js`) against
real DSLR frames. `bench/orig-mask-refine.js` is the pristine pre-optimization
upstream (`../seglab/js/mask-refine.js`), kept as the A/B baseline.

## Setup

```bash
bun run bench/make-corpus.mjs ~/Downloads   # copies RAW/JPEG + builds RGBA proxies
```

Needs `sips` (macOS) and `ffmpeg`. `corpus/` is gitignored — it holds ~200 MB of
camera files.

## Node-side (the filter itself)

| script | what it answers |
|---|---|
| `validate.mjs` | **the suite.** Accuracy vs the exact filter, stability under a benign bbox nudge, tiling exactness at fixed `s`, and timings. Exits non-zero on regression. Accuracy/stability bars allow 0.002, which is inside the method's own measured phase noise. |
| `ab.mjs` | `bandAlpha` bit-identity + `guidedFilter*` numeric agreement with upstream |
| `budget.mjs` | sweeps `CELL_BUDGET`, the accuracy/speed dial. This is how the shipped 100k was chosen; 40k was the first guess and lost at every size. |
| `stride.mjs` | **a negative result, kept on purpose.** A 1.57 Mpx proxy refining slower than a 1.80 Mpx one looks like cache aliasing on a power-of-two row stride. It is not — strides are flat, and the cause is the `s=3`/`s=4` boundary. |
| `diag.mjs` | sweeps the subsample factor `s` → the table that showed `s` is the sole source of edge non-determinism |
| `phase.mjs` | the control: how far upstream's own output moves when the bbox grows 1 px |
| `stability.mjs` | tiling soundness and phase stability, isolated |
| `baseline.mjs` | raw upstream timings per stage |

```bash
bun run bench/validate.mjs
```

## Browser-side (the real app)

`cdp.mjs` is a small Chrome DevTools-Protocol driver (`dev-browser` ships no
darwin-arm64 binary). It launches Chrome, evaluates in the page, and can put real
files on the app's hidden file input.

```bash
PORT=8811 bun run serve.mjs &
bun run bench/e2e.mjs                       # every AI tool; 3 representative images
bun run bench/e2e.mjs a.arw b.arw c.jpeg    # ...or name them, to cover the rest
bun run bench/inpage.mjs 2680558334.nef     # A/B the filter on V8, in the page
```

Where a user's time actually goes, and why:

| script | what it answers |
|---|---|
| `workflow.mjs` | **start here.** open → encode → subject → box → click → export, per image, with the lane's own stage breakdown (`encode`/`decode`/`post`, and inside `post`: upsample, bandWidth, bandAlpha, refine, re-threshold) plus the decomposition `refineField` chose. This is what showed `post` outweighing the GPU decode. |
| `latency.mjs` | pointer→paint latency and frame times during paced 120 Hz drags, plus **draws per frame**. The draw counter is what proved the render effect was already ≤1 draw/frame, so rAF-coalescing it was pure added latency. |
| `overlay.mjs` | cost of the selection outline when a grade changes — the path where a texture mask's boundary is re-traced per frame. |
| `cpuprof.mjs` | V8 sampling profile of a drag, self-time by function. |
| `profpost.mjs` | same, for a run of selections, plus the renderer's program-cache hit rate. |
| `interact.mjs` | frame/long-task distribution across drags. Superseded by `latency.mjs` for pacing, kept for its long-task view. |
| `uiprof.mjs` | per-effect attribution. Note its 33 ms floor (double-rAF) — it cannot resolve sub-frame work; use `cpuprof.mjs` for that. |

Two measurement traps these hit, both of which produce confident nonsense:

- **`window.__studio` is rebuilt by an effect whose deps include the chain**, so a
  captured reference goes stale the moment a layer is added. Go through a live
  `Proxy`, or `add()` silently applies to an old snapshot.
- **`aiState().busy` is not set when a tool call returns.** Polling straight for
  "not busy" succeeds instantly and measures nothing; wait for busy to *appear*
  first. This is what made box/click selections look like 1 ms.

`e2e.mjs` drives `window.__studio`. Two traps it documents, both of which make a
naive harness report false failures:

- `runAi()` returns `null` on the spot while `ai.busy` is set, and loading an
  image kicks off an eager encode — idle has to be awaited first.
- `commitAiMask` **replaces** a same-kind layer, so chain length is not a
  completion signal; read the status line instead.

## What the current numbers are

`refineField` + `bandAlpha`, 45 MP Nikon NEF proxy. Accuracy is IoU against the
**exact `s=1` filter** — not against upstream, which is a different approximation.

| band coverage | upstream | now | |
|---|---|---|---|
| 0.1 % | 1.0 ms · 0.810 | 1.9 ms · **1.0000** | exact filter, was a 0.19-IoU approximation |
| 0.5 % | 1.6 ms · 0.957 | 3.2 ms · **1.0000** | |
| 2 % | 3.1 ms · 0.964 | 7.7 ms · **1.0000** | |
| 10 % | 10.0 ms · 0.987 | 11.0 ms · **0.9976** | |
| 35 % | 25.2 ms · 0.990 | 17.9 ms · **0.9954** | 1.4× |
| 70 % | 30.4 ms · 0.997 | 17.5 ms · 0.9958 | 1.7× |
| 35 % @ export | 95.2 ms · 0.998 | 39.0 ms · 0.9971 | 2.4× |
| 70 % @ export | 120.3 ms · 0.997 | 38.6 ms · **0.9989** | **3.1×** |

Small masks are *deliberately slower*: they now run the exact filter instead of a
0.81–0.96-IoU approximation, and 8 ms is nothing against a ~250 ms selection.
Stability under a benign 1 px bbox nudge: **0.862 → 1.00000**, never below 0.992.

Steady-state selection on a 1756×1024 proxy, warm embedding:
`encode ~8 ms · decode ~95 ms · post ~134 ms`. `post` is the largest stage, and
`refine` is ~95 ms of it — so it is bound by full-resolution passes over the frame,
not by the subsampled solve.
