# Perf plan — WASM/WebGPU split for the mask lane

Backend rule: GPU-suited → WebGPU, CPU-suited → WASM SIMD. Two facts decide
every assignment below.

## The two constraints

1. **megashader is WebGL1** (`../phosmith/src/lib/megashader/megashader-renderer.js:147`,
   `getContext('webgl')`). WebGL cannot read a WebGPU buffer, so any compute
   result round-trips through CPU memory before the renderer draws it. Rules out
   GPU-resident mask across decode → post → render.
2. **`post` runs on the main thread.** `sam-client.js:21` → `sam21-adapter.js:17`
   → `mask-refine.js`, no worker in the chain. ~134 ms of blocked UI per selection.

Also: ORT owns the WebGPU device and `decode` is ~95 ms of queued GPU work. The
GPU is the contended unit; the CPU is idle during decode.

## Assignment

| stage | backend | why |
|---|---|---|
| `refine` full-res passes | WASM SIMD | result must reach CPU for the WebGL renderer regardless |
| morphology, CCL, boundary trace | WASM SIMD | branchy; best C-over-JS multiplier (~3–6×) |
| exact small-mask filter | WASM SIMD | sub-10 ms; dispatch overhead would dominate |
| export-HD | WebGPU | 45 MP, round-trip amortized, no contention (decode done) |
| encode / decode | WebGPU (ORT, existing) | leave alone |

## Measured ceiling — refine is only ~58 % recoverable

Stub run, 2680558334.nef 1756×1024, warm, n=2 per arm. `refineRect` skipped;
tile scan, decomposition and union rect left intact.

| | baseline | refine stubbed |
|---|---|---|
| refine | 92.7 / 93.2 | 15.7 / 17.5 |
| rethreshold | 6.3 / 6.1 | 24.0 / 24.1 |
| bandAlpha | 5.6 / 5.8 | 20.7 / 24.3 |
| upsample | 19.5 / 19.6 | 12.8 / 14.6 |
| **post** | **131.7 / 132.3** | **83.6 / 91.3** |
| total | 269 / 249 | 241 / 242 |

Removing 77 ms of refine work cuts `post` by only ~44 ms. **~36 ms of refine's
cost is cache warming that `rethreshold` and `bandAlpha` otherwise pay
themselves** — both roughly quadruple when refine stops touching the frame first.
Reproducible across both pairs.

So a 2–3× C+SIMD port (refine 93 → ~35–45) nets ~30 ms off a ~260 ms selection,
about **1.12×**, not the 1.6× the stage timing suggests. Re-run this stub before
spending on the port.

## Order

1. ~~**Move `post` into a worker.**~~ **Done.** `mask-post-core.js` (shared
   pipeline, no DOM) + `mask-post-worker.js` (owns the guide, keyed by imageKey)
   + `mask-post-client.js` (falls back in-process, so a broken worker costs jank
   and never a failed selection). `sam21Segment` awaits it; `sam21Cycle` stays on
   the in-process path, which its docblock requires.

   **Long tasks during a selection: ≥132 ms → none observed.** A synchronous
   132 ms `post` necessarily tripped the 50 ms longtask threshold; with the
   worker, a selection carrying a 204 ms `post` recorded zero.

   Wall-clock A/B, same file, 1756×1024 only (the arm is one file, and `ai/` is
   unbundled, so swapping it needs no rebuild). Baseline n=2, worker n=3:

   | | baseline | worker |
   |---|---|---|
   | box | 313 | 328 |
   | click1 | 229 | **195** |
   | click2 | 248 | 263 |
   | post (wall) | 139 | 158 |

   **Wall is a wash, not a win** — ~+20 ms of round-trip and guide transfer on
   box/click2, ~−34 ms on click1 where the freed main thread overlaps React's
   commit with post. The change moves work rather than removing it; the jank is
   the deliverable. Match proxy sizes before comparing: `proxy-plan.js` re-plans
   dimensions per pressure level, and a 1536×1024 run is a different `s` regime.

   Known cost: cycling no longer shares a guide with the worker path, so the
   first cycle after a selection pays one getImageData (~10–20 ms).
2. ~~**WebGPU compute for export-HD.**~~ **Dropped — aimed at the wrong code.**
   `bench/profexport.mjs` (new) profiles the real export. `app.jsx`'s `exportHd`
   never calls `export-hd.js` / `buildCutout` at all: it decodes the original to
   a full-res canvas, runs `renderMegashader` (once, twice with a base grade),
   and `toBlob`s the result. **`refineField`, `bandAlpha` and the guided filter
   do not appear in the profile.** The `39 ms @ export` figures from
   `bench/README.md` belong to a path this app's export button does not execute.

   Export wall 1279 ms, one main-thread long task of **560 ms**, self-time:

   | | ms | |
   |---|---|---|
   | `(program)` native | 881 | unattributed: original decode, GPU driver, GC |
   | `readPixels` | 125 | megashader's WebGL1 framebuffer readback |
   | `toBlob` | 45 | PNG encode, on the main thread |
   | `putImageData` | 26 | |

   Real targets, in order: encode off-thread (OffscreenCanvas `convertToBlob`);
   drop the `readPixels` → `putImageData` → `toBlob` round-trip by encoding the
   WebGL canvas directly; then attribute `(program)` with Chrome tracing, since
   at 69 % it dominates everything else and a sampling profiler cannot split it.

   Export wall varies 267–1399 ms across runs — treat 1279 as one sample, not a
   number to optimize against until the variance is explained.
3. **C+SIMD port of refine** — deprioritized to ~1.12×. Weigh against the
   permanent cost of a second implementation and the bench-bar rewrite below.
   Toolchain exists (`ai/cv-refine-worker.js`, `public/wasm/cv-refine.wasm`);
   C sources are not in this repo.

## Export variance explained — the memory governor, not the export

   `PROF=0 profexport.mjs 2680558334.nef`, four consecutive runs:

   | run | wall ms | long tasks ms | export resolution | proxy | bounded |
   |---|---|---|---|---|---|
   | 1 | 2362 | 1045 | 4243×2828 (12.0 MP) | 1536×1024 | yes |
   | 2 | 2623 | 1397, 54 | 4243×2828 (12.0 MP) | 1536×1024 | yes |
   | 3 | **291** | **none** | **2034×1186** (2.4 MP) | 1756×1024 | **no** |
   | 4 | 918 | 435 | 4243×2828 (12.0 MP) | 1536×1024 | yes |

   The 5× wall-time spread tracks the ~5× pixel-count ratio exactly.
   `proxy-plan.js` re-plans the proxy under memory pressure, and `exportSource`
   bounds the original by the governor's budget. When the budget is tight the
   export resolution drops to ~2.4 MP and the entire export finishes in under
   300 ms with zero long tasks. When it gets 12 MP, it takes 900–2600 ms with
   a 400–1400 ms main-thread block.

   **The variance is not in the export code. It is in which original resolution
   the memory governor chose.** Any export-path optimisation must be measured at
   a FIXED resolution — pin the budget or measure at a known `out.width`.

## `(program)` attribution — GPU driver wait, not decode or GC

   `TRACE=1 profexport.mjs` (devtools.timeline + gpu categories), 4243×2828:

   | ms | event | where |
   |---|---|---|
   | 134 | `CommandBufferHelper::Finish` | renderer → GPU flush (WebGL drawArrays at 12 MP) |
   | 39 | `RasterImplementation::ReadbackImagePixels` | `readPixels` |
   | 29 | `CommandBufferHelper::Finish` | second GPU flush (the `hasGrade` bake pass) |
   | 19 | `DoReadbackARGBImagePixelsINTERNAL` | actual pixel transfer |

   No `ImageDecodeTask`, no `MajorGC`, no `DecodeImage`. The sampling
   profiler's 881 ms `(program)` is almost entirely **GPU command-buffer wait**:
   the main thread blocks in `WaitForGetOffset` while Metal executes the shader
   and the readback completes.

   The readPixels round-trip (`ReadbackImagePixels` + `DoReadbackARGBImagePixelsINTERNAL`)
   is ~58 ms of the ~460 ms long task. Encoding the WebGL canvas directly would
   skip this, but the larger win is keeping the GPU work off the main thread's
   critical path entirely (OffscreenCanvas).

## Caveat

SIMD reassociation of box-filter sums changes float rounding. `validate.mjs`
(IoU 1.0000) and `ab.mjs` (bit-identity) must keep the JS path canonical and
apply a tolerance bar to the WASM path.
