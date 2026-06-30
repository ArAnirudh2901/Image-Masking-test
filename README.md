# 🎭 Mask Studio

A standalone, no-auth testbed for the **phosmith megashader masking engine** — load an image, build a stack of masks of every kind, and colour-grade each masked region in real-time using the production mask card UI.

This is the real editor engine running in isolation, outside of the full app, for rapid development and testing.

---

## ✨ Features

### Mask Types
| Tool | Description |
|---|---|
| **Radial** | Ellipse / circle gradient — drag handles to resize & rotate |
| **Linear** | Linear gradient — drag endpoints to aim |
| **Brush** | Paint a freehand region; edge-aware (smart brush) optional |
| **Pen / Lasso** | Click points to build a path, close to commit |
| **Luminance** | Tonal range selection (highlights / shadows / midtones) |
| **Color** | Colour range — click the image to sample the target hue |
| **AI Subject** | Auto-masks the main subject (on-device RMBG or SAM 3.1 via service) |
| **AI Sky / Bg** | Detects the subject and inverts — selects sky / background |
| **AI Click-Select** | Click any object for a precise mask (SlimSAM, on-device) |
| **AI Depth** | Near / far depth-based selection (Depth-Anything, on-device) |

### Per-Mask Grading (13 parameters)
- Exposure, Contrast, Highlights, Shadows, Whites, Blacks
- Saturation, Vibrance, Hue Shift
- Tone Curves (per-channel RGB + composite)
- Colour Wheels (Shadows / Midtones / Highlights)
- Gamma

### AI Backends
- **On-device** (WebGPU → WASM fallback) — no server needed, fully private
- **Local Python service** — SAM 3.1 / SAM 2 / Depth Anything / CLIPSeg via FastAPI (preferred when running, for higher quality)

### Other
- Multi-mask stacking with per-layer blend ops (`add`, `subtract`, `replace`)
- Full undo / redo including brush strokes and AI mask textures
- Boundary grow / shrink on any texture mask
- Overlay mode + clean preview (hide all handles)
- Global invert, fill colour per-mask

---

## 🗂️ Project Structure

```
mask-studio/
├── app.jsx          # Main React application (single file)
├── styles.css       # All UI styles
├── index.html       # Entry point
├── build.mjs        # Bun bundler script (resolves @/ alias from phosmith/src)
├── serve.mjs        # Tiny static server with correct MIME types for .wasm
└── ort/             # ONNX Runtime WASM files (NOT committed — see Setup)
```

> **Note:** This project lives alongside and imports directly from the `phosmith/` repo (its parent directory). The bundler resolves `@/` → `../phosmith/src/` at build time.

---

## 🚀 Quick Start

### Prerequisites
- [Bun](https://bun.sh) (v1.0+)
- The `phosmith/` repo checked out at `../phosmith` (sibling directory)

### 1. Install dependencies

```bash
# In the phosmith repo (installs React, framer-motion, transformers, etc.)
cd ../phosmith
bun install
```

### 2. Copy ONNX Runtime WASM files

The on-device AI models need the ONNX Runtime WASM runtime files served locally. Copy them from the `onnxruntime-web` package:

```bash
cd /path/to/mask-studio

# Copy the 4 WASM + 4 .mjs files from node_modules
cp ../phosmith/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded*.{mjs,wasm} ort/
```

### 3. Build the app

```bash
bun build.mjs
```

This bundles `app.jsx` → `app.js` (+ chunk files) using the phosmith source alias.

### 4. Start the frontend server

```bash
bun serve.mjs
```

Open **http://127.0.0.1:8810** in your browser.

---

## 🤖 Optional: Python AI Service (Higher Quality)

The Python backend enables SAM 3.1 / SAM 2 / Depth Anything / CLIPSeg — significantly better results than the on-device models for subject masking and click-select. The frontend falls back to on-device automatically when the service is not running.

### Setup (one-time)

```bash
cd ../phosmith/services/segment

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies (torch, transformers, rembg, fastapi, etc.)
pip install -r requirements.txt
```

### Start the service

```bash
cd ../phosmith/services/segment
source .venv/bin/activate
uvicorn main:app --reload --port 8001
```

The service will be available at **http://127.0.0.1:8001**.

> **First run:** Models are downloaded lazily on first use:
> - `isnet-general-use` background removal (~179 MB) — downloaded on first *AI Subject* call
> - `facebook/sam2-hiera-small` (~180 MB) — downloaded on first *AI Click-Select* call
> - `depth-anything/Depth-Anything-V2-Small-hf` (~50 MB) — downloaded on first *AI Depth* call

### Hardware acceleration

| Machine | Detected provider |
|---|---|
| macOS Apple Silicon | CoreML + MPS (GPU) |
| Linux + NVIDIA GPU | CUDA |
| Anything else | CPU |

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service status + loaded models |
| `POST` | `/segment` | Background removal (RGBA PNG) |
| `POST` | `/segment/instances` | Subject concept segmentation (SAM 3.1 / saliency) |
| `POST` | `/sam2/click` | Click-to-select mask (SAM 2) |
| `POST` | `/depth` | Depth map estimation |
| `POST` | `/ground/text` | Text-grounded masking (CLIPSeg + SAM 2) |

---

## 🔧 Development Workflow

```bash
# 1. Edit app.jsx or styles.css
# 2. Rebuild
bun build.mjs

# 3. The dev server (bun serve.mjs) serves the new build immediately — just refresh the browser
```

---

## 📦 What's NOT in the repo

| Item | Why | How to get it |
|---|---|---|
| `ort/` | ~77 MB WASM runtime | Copy from `onnxruntime-web` package (see Setup) |
| `app.js`, `chunk-*.js` | Build output | Run `bun build.mjs` |
| `node_modules/` | Dependencies | Run `bun install` in `../phosmith` |
| `test.png` | Large sample image | Add any image locally and rename to `test.png` |

---

## 🔗 Related

- **phosmith** — the main Next.js photo editing application this testbed is extracted from
- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
- [Hugging Face Transformers.js](https://huggingface.co/docs/transformers.js)
- [SAM 2 (facebook/sam2-hiera-small)](https://huggingface.co/facebook/sam2-hiera-small)
- [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
