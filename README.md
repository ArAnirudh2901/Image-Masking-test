# 🎭 Mask Studio

Load an image, stack masks of every kind — geometric, parametric, freehand and
**AI** — and colour-grade each masked region in real time with the production
mask card UI.

Two engines, spliced:

| | |
|---|---|
| **phosmith megashader** | the mask compositing + 13-parameter per-mask grade, tone curves, colour wheels, and the real editor UI (`MaskChainCard`, `LayerGradeEditor`, `ProRulerSlider`) |
| **seglab lane** (`ai/`) | on-device segmentation: **SAM 2.1 small, fp16, WebGPU**, bounded decode workers, C++/wasm mask refinement, a one-job-at-a-time heavy queue and a live memory governor |

**Everything runs in this browser.** There is no upload endpoint, no Python
service and — once `bun run models` has vendored the weights — no network call
at all. Images, masks and every inference stay on the device.

---

## ✨ Tools

### Masks
| Tool | What it does |
|---|---|
| **Radial** | Ellipse / circle gradient — drag handles to resize & rotate |
| **Linear** | Linear gradient — drag endpoints to aim |
| **Brush** | Paint a freehand region; edge-aware (smart brush) optional |
| **Pen / Lasso** | Click points to build a path, close to commit |
| **Luminance** | Tonal range selection (highlights / shadows / midtones) |
| **Color** | Colour range — click the image to sample the target hue |
| **AI Subject** | The main subject, from three SAM prompts over one cached encode |
| **AI Sky / Bg** | The same mask, inverted — the only reliable "whole background" selection |
| **AI Click-Select** | Click any object; click again to **add**, Alt+click to **remove** |
| **AI Box-Select** | Drag a box around an object |
| **AI Lasso** | Draw a rough loop — it snaps to the object and can never bleed outside it |
| **AI Text** | Describe an object ("the red car"); an on-device open-vocab detector finds every instance and SAM segments them |

Every AI mask is an ordinary mask afterwards: grade it, invert it, grow/shrink
its **Boundary**, or **Brush-refine** its coverage by hand.

### Per-mask grading (13 parameters)
Exposure · Contrast · Highlights · Shadows · Whites · Blacks · Saturation ·
Vibrance · Hue Shift · Tone Curves (per-channel RGB + composite) · Colour
Wheels (Shadows / Midtones / Highlights) · Gamma

### Everything else
- Multi-mask stacking with per-layer blend ops (`add`, `subtract`, `replace`)
- Full undo / redo **including** brush strokes and AI mask textures
- Overlay mode + clean preview (hide all handles)
- Global invert, fill colour per mask
- **HD export** — re-decodes the original at export resolution and renders the
  whole chain onto it
- Drop or paste an image anywhere; **camera RAW** (`.nef/.cr2/.cr3/.arw/.dng/…`)
  opens through its embedded preview, or an on-device LibRaw develop

---

## 🚀 Quick start

Prerequisites: [Bun](https://bun.sh) 1.0+ and `git`. Everything else — the
phosmith checkout, both dependency trees, the runtime and the weights — is what
`setup` fetches.

```bash
git clone https://github.com/ArAnirudh2901/Image-Masking-test.git
cd Image-Masking-test
bun run setup:all      # ~227 MB, one-time — or `bun run setup` for 111 MB without text search
bun run dev            # build.mjs → app.js, then serve.mjs on :8810
```

Open **http://127.0.0.1:8810**. Drop in any photo, or add a `test.png` to have
one auto-load.

`bun run setup` is idempotent — re-run it any time; anything already in place is
skipped. It ends with a real build, so if it exits 0 the app builds.

<details>
<summary>What setup does, and doing it by hand</summary>

```bash
git clone https://github.com/ArAnirudh2901/Phosmith.git ../phosmith
git -C ../phosmith checkout $(bun -e 'console.log(require("./package.json").phosmith.commit)')
bun install                 # here
bun install --cwd ../phosmith
bun run models:all          # ORT + weights
bun run build
```

**phosmith is a build input, not a runtime one.** `app.jsx` imports the
megashader engine and the real editor UI (`MaskChainCard`, `LayerGradeEditor`,
`ProRulerSlider`) over the `@/` alias, which `build.mjs` resolves into a sibling
`../phosmith/src/`. Set `PHOSMITH_DIR` if your checkout lives elsewhere. It
needs its own `bun install` too: `build.mjs` pins every `react` / `react-dom` /
`scheduler` import there so exactly **one** React copy is bundled — two copies
crash at runtime with `Cannot read properties of null (reading 'useState')`.

The commit is pinned in `package.json` → `phosmith.commit`. Setup checks out the
pin on a fresh clone (detached, on purpose) and only *warns* if an existing
checkout differs, so it never rewrites work in progress.

</details>

### Where the models come from

`bun run models` is what makes the app work **with no internet**. It vendors
onnxruntime-web under `lib/` and the weights under `models/` (both gitignored).
Skip it and the app falls back to the pinned CDN on first use, caching into
Cache Storage via `sw.js` — which survives reloads but is evictable.

The weights come from this repo's [`weights-v1`](../../releases/tag/weights-v1)
release, SHA-256 pinned in `scripts/download-models.mjs`, so every machine runs
the same bytes — ONNX export is not reproducible across torch versions, and the
fp16 encoder is sensitive enough that a re-export is a different model.

### Requirements

**WebGPU with `shader-f16` is required, not preferred.** The mask lane is
fp16-only; there is no WASM fallback and no second segmentation model, because a
second one would break the bounded interaction-memory contract. Chrome/Edge 121+
and Safari 18+ on Apple Silicon qualify.

Serving is **127.0.0.1 only**, and that is not arbitrary: the page must be
cross-origin isolated *and* a secure context for WebGPU, threaded WASM and
`measureUserAgentSpecificMemory()`. A plain-HTTP LAN address is neither, so the
mask lane degrades there. To view it from another device, tunnel to
`127.0.0.1:8810` over HTTPS rather than binding a LAN interface. `PORT=9000 bun
run serve` moves the port.

### Troubleshooting

| Symptom | Cause |
|---|---|
| `BUILD FAILED — phosmith not found at …` | no sibling checkout — `bun run setup`, or set `PHOSMITH_DIR` |
| `Cannot read properties of null (reading 'useState')` | two React copies — `bun install --cwd ../phosmith` |
| `mask lane incomplete — N core asset(s) missing` | weights never landed — re-run `bun run models` |
| `digest mismatch for weights-*.tar.gz` | truncated download — re-run; the bad file is discarded, never extracted |
| AI tools greyed out, "AI Text" absent | no WebGPU/`shader-f16`, or detector not installed (`bun run models:all`) |

---

## 🏗 How the two engines meet

`ai/studio-bridge.js` is the **only** module the React bundle talks to. It is a
translation layer, not a second implementation:

```
seglab                                   Mask Studio
──────────────────────────────────────   ────────────────────────────────────
white-on-black RGBA ImageData        →   the opaque mask canvas the megashader
  (R=G=B=coverage, A=255)                  semantic shader already samples
click / box / lasso prompt sets      →   one select() per mask tool
asset-store blob custody + proxy     →   the working canvas React grades
policy budget + memory governor      →   the status chip in the panel
```

The mask formats are already identical, so a mask crosses between engines as a
`putImageData` — there is no resample and no quality loss anywhere in the path.

### `ai/` is deliberately **not bundled**

The lane resolves its workers, wasm and weights with
`new URL(…, import.meta.url)`. Bundling would rewrite those to point at
`app.js`'s directory and break every worker spawn and model fetch. So `app.jsx`
reaches the bridge through a runtime `import(<computed URL>)`, which Bun leaves
alone, and `serve.mjs` serves `ai/*.js` as real ES modules. `index.html`
`modulepreload`s the bridge so the graph is warm before the first click.

---

## ⚡ Why it stays smooth

- **One heavy job at a time** (`ai/heavy-job-queue.js`) — proxy decode, model
  warm, encode, detector runs, wasm refinement and export re-decodes are
  serialised at concurrency 1, so their peak allocations can never stack.
  Import outranks model work; user interaction outranks speculative prewarm; a
  new image invalidates stale queued jobs.
- **Encode once, decode per click.** The image embedding is content-keyed and
  cached, so the expensive ViT pass runs once per photo and every later click,
  box, lasso or refine is a decoder pass. Measured on the sample: 3.2 s cold
  encode, then **270–470 ms** per selection.
- **The original is bytes, never RGBA.** `asset-store` keeps the upload as a
  compressed Blob and decodes straight to a bounded proxy; crops and exports
  re-decode only the region they need. A 45 MP frame never materialises.
- **Per-axis proxy sizing.** SAM 2.1 encodes a 1024×1024 square, so a long-edge
  cap starves the short axis. The proxy is sized to put the **short** edge at
  1024 (bounded by a 2048 long-edge stop and a ~2.1 MP total cap): the sample
  lands at 1756×1024 instead of a flat 1400×816.
- **Cross-origin isolated.** `serve.mjs` sets COOP/COEP, which unlocks threaded
  WASM and `measureUserAgentSpecificMemory()` — the only real byte signal the
  governor has. Everything is vendored, so isolation costs nothing.
- **A live memory governor** (`ai/memory-governor.js`) watches measured bytes,
  an allocation ledger and timer drift, and sheds — detector → refine →
  embedding → sessions — the moment real pressure appears. It is a one-way
  ratchet; it never re-enables a feature behind your back.
- **Idle hibernate.** The resident cost between edits is the ORT session arena,
  not the 8 MB embedding, so after an idle window the arena goes back to the OS.
  The next selection rebuilds from cached weights; nothing on screen is lost.
- **Long-cached weights, no-cache build output.** Model blobs are served
  `immutable` with Range support; `app.js` and `index.html` are always fresh.

---

## 🗂 Layout

```
Image-Masking-test/
├── app.jsx           # the React app (bundled → app.js)
├── ai/               # the seglab engine, verbatim + studio-bridge.js
│   ├── studio-bridge.js    ← the only entry point app.jsx uses
│   ├── sam-client.js · sam21-{lane,host,client,adapter,store}.js
│   ├── sam-core.js · mask-select.js · mask-refine.js · cv-refine-*.js
│   ├── decode-{client,core,worker}.js · image-io.js · image-raw.js · raw-develop-*.js
│   ├── asset-store.js · export-hd.js · proxy-plan.js
│   ├── policy.js · capability.js · memory-governor.js · heavy-job-queue.js
│   └── text-*.js · yoloe-detect.js · clip-tokenizer.js · detect-worker.js
├── lib/ort-web/      # vendored onnxruntime-web  (bun run models)
├── models/           # SAM 2.1 + CLIP text + YOLOE weights  (bun run models)
├── public/wasm/      # cv-refine + LibRaw develop, compiled from C++
├── sw.js             # model cache + CORP re-tag for the CDN fallback
├── build.mjs · serve.mjs · styles.css · index.html
```

## 🔧 Development

```bash
bun run build.mjs   # rebuild app.js after editing app.jsx / styles.css
# ai/*.js needs NO rebuild — it is served straight from disk. Just refresh.
```

`window.__studio` exposes every action for scripted testing: `clickSelect`,
`samBox`, `aiLasso`, `runSubject`, `background`, `findText`, `exportHd`,
`engine()`, `budget()`, `shed(level)`, `undo`, `redo`, `pixels()`.

### URL parameters

Safety precedence is **hard device limit > memory pressure > URL parameter >
feature request**; parameters may only ever lower a limit.

| | |
|---|---|
| `?proxy=768` | smaller interaction frame |
| `?text=0` | disable the text lane entirely |
| `?escalate=0` | no native re-decode escalation |
| `?debug=1` | governor telemetry in the console |

---

## 📦 Not in the repo

| Item | Why | How to get it |
|---|---|---|
| `../phosmith/` | separate repo — the megashader engine + editor UI | `bun run setup` |
| `lib/ort-web/`, `models/` | ~227 MB of runtime + weights | `bun run models:all` |
| `app.js` | build output | `bun run build.mjs` |
| `node_modules/` | dependencies | `bun install` |
| `test.png` | large sample image | drop in any photo |
| `bench/corpus/` | ~200 MB of camera files | `bun bench/make-corpus.mjs` |

## 🔗 Related

- **phosmith** — the Next.js photo editor this masking engine comes from
- **seglab** — the on-device segmentation app `ai/` is lifted from
- [SAM 2.1](https://github.com/facebookresearch/sam2) · [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) · [LibRaw](https://github.com/libraw/libraw)
