/**
 * studio-bridge — the ONE module Mask Studio's React bundle talks to.
 *
 * Everything below it is the seglab engine verbatim (SAM 2.1 fp16 on WebGPU,
 * bounded decode workers, wasm mask refinement, heavy-job queue, memory
 * governor). Nothing here re-implements it; this file is the translation layer:
 *
 *   seglab side                          Mask Studio side
 *   ───────────────────────────────────  ─────────────────────────────────────
 *   white-on-black RGBA ImageData        opaque mask canvas the megashader
 *                                          semantic/depth shaders sample
 *   click / box / lasso prompt sets       one `select()` call per mask tool
 *   asset-store blob custody + proxy      the working canvas React grades
 *   policy budget + governor              a status object the panel renders
 *
 * Loaded through a RUNTIME import (see app.jsx) so Bun never bundles it: the
 * lane resolves its workers and weights with `new URL(..., import.meta.url)`,
 * which only works while these files are served from their real paths.
 */

import { applyMemoryPressure, resolveBudget } from './policy.js'
import { observeBandFraction, observePost, savePostFit } from './hardware-fit.js'
import { probeCapability } from './capability.js'
import { createMemoryGovernor } from './memory-governor.js'
import { clearHeavyQueue, getHeavyQueueState } from './heavy-job-queue.js'
import {
    cancelBefore, clientState, encodeImage, encoderReady, forgetEncoder, relievePressure,
    releaseDocument, releaseEmbeddings, segment, subscribe, warmEncoder, warmUp,
} from './sam-client.js'
import {
    getOriginalForExport, getTransform, hasOriginal, importOriginal, releaseAsset,
} from './asset-store.js'
import { extractRawPreview, isRawFile } from './image-raw.js'
import { developRaw } from './raw-develop-client.js'
import { disposeCvRefine } from './cv-refine-client.js'
import { dropMaskPostGuide } from './mask-post-client.js'
import { clearGuideCache } from './sam21-adapter.js'
import { composeChannels, dilateChannel, lassoToPrompts, maskToChannel } from './sam-core.js'
import { modelRegistry } from './model-registry.js'

const trace = (event, detail) => console.log(`[studio][ai] ${event}`, detail ?? '')

/* ─── session state ──────────────────────────────────────────────────────── */

let BUDGET = resolveBudget(typeof location !== 'undefined' ? location.search : '', null, null)
let capability = null
let governor = null
let measureHook = () => ({})
const listeners = new Set()
const emit = (event) => { for (const cb of listeners) { try { cb(event) } catch { /* listener bug */ } } }

const state = {
    revision: 0,
    importEpoch: 0,
    booted: false,
    hasImage: false,
    proxyCanvas: null,
    lastPrompts: null,   // proxy-space prompts behind the live mask (HD export)
}

/** Subscribe to engine events: {type:'progress'|'state'|'waiting'|'pressure'}. */
export const onEngineEvent = (cb) => { listeners.add(cb); return () => listeners.delete(cb) }

/** Any new prompt or document obsoletes every queued job that predates it. */
export const bumpRevision = () => {
    state.revision += 1
    cancelBefore(state.revision)
    return state.revision
}
export const revision = () => state.revision
export const budget = () => ({ ...BUDGET })

/** One flat object the React panel renders — no engine internals leak out. */
export const status = () => ({
    ready: clientState.ready,
    device: clientState.device,
    lane: clientState.lane,
    mode: clientState.mode,
    lastRun: clientState.lastRun,
    encoderReady: encoderReady(),
    pressure: BUDGET.pressureLevel || 0,
    proxyMax: BUDGET.proxyMax,
    gpuTier: capability?.gpuTier || 'unknown',
    queue: getHeavyQueueState(),
    models: modelRegistry(),
})

/* ─── mask plumbing ──────────────────────────────────────────────────────── */

const makeCanvas = (w, h) => {
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    return c
}

/**
 * The lane's mask is already the exact layout the megashader wants — opaque
 * RGBA with coverage duplicated into R=G=B — so this is a putImageData, not a
 * conversion. Kept as a named step because that equivalence is the whole reason
 * the two engines splice together without a resample.
 */
export const fieldToCanvas = (imageData) => {
    const c = makeCanvas(imageData.width, imageData.height)
    c.getContext('2d').putImageData(imageData, 0, 0)
    return c
}

/** Read a mask canvas back as a 1-channel coverage field at w×h. */
const canvasToChannel = (canvas, w, h) => {
    const c = makeCanvas(w, h)
    const ctx = c.getContext('2d', { willReadFrequently: true })
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, w, h)
    if (canvas && canvas.width) ctx.drawImage(canvas, 0, 0, w, h)
    return maskToChannel(ctx.getImageData(0, 0, w, h))
}

/**
 * Union / subtract a freshly segmented region into an existing mask, through
 * sam-core's op model rather than canvas blend modes. `add` takes a per-pixel
 * max so the refined boundary keeps its soft 8-bit falloff (canvas `lighten`
 * does too, but `sub` via multiply-inverse hardens the edge and leaves a
 * one-pixel residue ring where two decodes of the same object disagree — hence
 * the dilate on the subtract channel).
 */
export const composeMask = (baseCanvas, patchCanvas, op, w, h) => {
    const base = baseCanvas ? canvasToChannel(baseCanvas, w, h) : null
    let chan = canvasToChannel(patchCanvas, w, h)
    if (op === 'sub') chan = dilateChannel(chan, w, h, 2)
    const out = composeChannels([{ op, chan }], w, h, base)
    if (!out) return makeCanvas(w, h) // everything removed — an empty mask is valid
    const c = makeCanvas(w, h)
    c.getContext('2d').putImageData(new ImageData(out.rgba, w, h), 0, 0)
    return c
}

/**
 * Clamp a mask to the lasso stroke ∪ `margin`. The lasso is a prompt generator,
 * so SAM is free to snap outside the loop; this is what makes "it can never
 * bleed onto the neighbouring object" true rather than usually true.
 */
const clampToPolygon = (imageData, poly, margin) => {
    const { width: w, height: h } = imageData
    const gate = makeCanvas(w, h)
    const g = gate.getContext('2d', { willReadFrequently: true })
    g.fillStyle = '#fff'
    g.strokeStyle = '#fff'
    g.lineWidth = Math.max(1, margin * 2)
    g.lineJoin = 'round'
    g.lineCap = 'round'
    g.beginPath()
    g.moveTo(poly[0][0], poly[0][1])
    for (let i = 1; i < poly.length; i += 1) g.lineTo(poly[i][0], poly[i][1])
    g.closePath()
    g.fill()
    g.stroke()
    const allow = g.getImageData(0, 0, w, h).data
    const out = new Uint8ClampedArray(imageData.data)
    for (let i = 0; i < out.length; i += 4) {
        if (allow[i + 3] >= 128) continue
        out[i] = 0; out[i + 1] = 0; out[i + 2] = 0
    }
    return new ImageData(out, w, h)
}

const coverageOf = (imageData) => {
    const d = imageData.data
    let count = 0
    for (let i = 0; i < d.length; i += 4) if (d[i] >= 128) count += 1
    return count / (imageData.width * imageData.height)
}

const unionInto = (acc, next) => {
    const a = acc.data
    const b = next.data
    for (let i = 0; i < a.length; i += 4) {
        if (b[i] > a[i]) { a[i] = b[i]; a[i + 1] = b[i + 1]; a[i + 2] = b[i + 2] }
    }
    return acc
}

/* ─── boot ───────────────────────────────────────────────────────────────── */

/**
 * Probe the device, size the budget, start the memory governor, then warm the
 * lane speculatively. `measure` reports the pixel buffers the React side owns
 * (working canvas, overlay, mask textures) so the governor's ledger — the only
 * memory signal that exists on WebKit — accounts for them.
 */
export const boot = async ({ measure } = {}) => {
    if (state.booted) return { budget: budget(), capability }
    state.booted = true
    if (typeof measure === 'function') measureHook = measure

    subscribe((event) => emit(event))

    // Model cache (sw.js): cache-first for weights and the ORT runtime ONLY —
    // app.js and index.html pass straight through, so a rebuild is never served
    // stale. Fire-and-forget; nothing waits on it.
    if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {
        navigator.serviceWorker.register('./sw.js', { scope: './' }).catch(() => { /* private mode */ })
    }

    capability = await probeCapability()
    BUDGET = applyMemoryPressure(
        resolveBudget(location.search, capability, null),
        BUDGET.pressureLevel || 0,
    )
    trace('capability', { gpu: capability.gpuTier, proxyMax: BUDGET.proxyMax, webgpu: capability.webgpu })

    // Ledger anchors measured on the lane itself (sam21-lane releaseIdle):
    // encoder live 2096 MB → releaseAll 1120 MB → worker exit 446 MB.
    const LEDGER_MB = { base: 260, workerWarm: 674, devicePool: 976, encoderSession: 37 }
    let laneStatus = () => null
    import('./sam21-client.js').then((m) => { laneStatus = m.hostStatus }).catch(() => { /* lane not up */ })
    const estimateFootprintMB = () => {
        const s = laneStatus() || null
        const lane = s?.lane || null
        let mb = LEDGER_MB.base
        if (s) {
            if (s.builtEncoder) mb += LEDGER_MB.workerWarm
            if (lane?.encoder || lane?.decoder) mb += LEDGER_MB.devicePool
            if (lane?.encoder) mb += LEDGER_MB.encoderSession
        }
        const m = measureHook() || {}
        mb += (Number(m.pixelBytes) || 0) / (1024 * 1024)
        return mb
    }

    governor = createMemoryGovernor({
        getBudget: () => BUDGET,
        getEstimateMB: estimateFootprintMB,
        isActive: () => state.hasImage && !document.hidden,
        onPressure: (level) => shed(level),
        onHeadroom: () => { /* the single-config build has no tier to climb to */ },
    })
    governor.start()

    document.addEventListener('visibilitychange', () => {
        if (document.hidden && state.hasImage) shed(1, { announce: false })
    })
    for (const ev of ['pointerdown', 'pointerup', 'keydown', 'wheel']) {
        window.addEventListener(ev, noteInteraction, { passive: true })
    }

    return { budget: budget(), capability }
}

/** Build the lane (and, once an image exists, its encoder) off the hot path. */
export const warm = async ({ withEncoder = false } = {}) => {
    try {
        await warmUp({ speculative: true })
        if (withEncoder) await warmEncoder({ speculative: true })
    } catch (err) {
        console.warn('[studio][ai] warm failed (first selection retries):', err?.message)
    }
    emit({ type: 'state' })
}

/* ─── memory ─────────────────────────────────────────────────────────────── */

export const shed = (level, { announce = true } = {}) => {
    const previous = BUDGET.pressureLevel || 0
    BUDGET = applyMemoryPressure(BUDGET, level)
    if ((BUDGET.pressureLevel || 0) >= 2) {
        disposeCvRefine()
        // Two ~7 MB guide copies exist once post runs off-thread: the worker's
        // and the adapter's (cycling/fallback). Shed both.
        dropMaskPostGuide()
        clearGuideCache()
    }
    if (level >= 1) forgetEncoder()
    if (announce && BUDGET.pressureLevel > previous) {
        emit({ type: 'pressure', level: BUDGET.pressureLevel })
    }
    return relievePressure(level).catch(() => [])
}

/* Idle hibernate — the resident cost between edits is the ORT session arena,
 * not the 8 MB embedding. After `samIdleMs` of no input, hand the arena back;
 * the next selection rebuilds from cached weights and re-encodes. Committed
 * masks are canvases on the React side, so nothing on screen is lost. */
let idleTimer = null
const hibernate = () => {
    idleTimer = null
    if (!state.hasImage) return
    relievePressure(3).catch(() => {})
    trace('idle-hibernate')
}
function noteInteraction() {
    if (idleTimer) { clearTimeout(idleTimer); idleTimer = null }
    const base = BUDGET.samIdleMs || 0
    if (!base || !state.hasImage) return
    const ms = (BUDGET.pressureLevel || 0) >= 2 ? Math.min(base, 45_000) : base
    idleTimer = setTimeout(hibernate, ms)
}

/* ─── image import ───────────────────────────────────────────────────────── */

/**
 * Take custody of `source` and paint its bounded interaction frame into
 * `proxyCanvas`. The original is retained as compressed bytes only — never as
 * full-resolution RGBA — so a 45 MP DSLR frame costs the proxy, not the sensor.
 * RAW containers are opened by lifting the camera's own embedded JPEG preview
 * (no demosaic); the rare preview-less RAW falls back to the LibRaw wasm
 * develop, whose worker is terminated straight after.
 *
 * @param {Blob|File} source
 * @param {{ proxyCanvas: HTMLCanvasElement, onStage?: (msg: string) => void }} opts
 * @returns {Promise<object|null>} the asset transform, or null when superseded
 */
export const importImage = async (source, { proxyCanvas, onStage = () => {} }) => {
    bumpRevision()
    clearHeavyQueue()
    // An image SWAP keeps the sessions and the GPU device: releaseAll drops the
    // last session, which destroys the device, and the next click would then pay
    // a device rebuild + a ~1 GB session build before it could encode anything.
    if (state.hasImage) {
        if (BUDGET.pressureLevel || 0) { releaseDocument(); forgetEncoder() } else releaseEmbeddings()
    }
    const epoch = ++state.importEpoch
    const isCurrent = () => epoch === state.importEpoch

    let blob = source
    let proxyBlob = null
    let orientation = null
    const raw = typeof source?.name === 'string' && isRawFile(source)

    if (raw) {
        onStage('Reading the camera preview — the sensor data stays untouched…')
        const preview = await extractRawPreview(source, {
            proxyMinEdge: Math.max(768, BUDGET.proxyMax || 1024, BUDGET.proxyLongMax || 0),
        })
        if (!isCurrent()) return null
        if (preview) {
            blob = preview.blob
            proxyBlob = preview.proxyBlob
            orientation = preview.orientation
        } else {
            onStage('No embedded preview — developing the RAW on-device…')
            const developed = await developRaw(source, { budget: BUDGET })
            if (!isCurrent()) return null
            if (!developed) throw new Error('No readable preview is embedded in this RAW and on-device develop was unavailable — export a JPEG/TIFF and try again.')
            blob = developed.blob
            orientation = 1 // LibRaw already applied the camera flip
        }
    }

    onStage('Building a bounded interaction frame…')
    const transform = await importOriginal(blob, {
        budget: BUDGET,
        proxyCanvas,
        proxyBlob,
        orientation,
        sourceWasRaw: raw,
        revision: state.revision,
        isCurrent,
    })
    if (!transform || !isCurrent()) { releaseAsset(); return null }

    state.hasImage = true
    state.proxyCanvas = proxyCanvas
    state.lastPrompts = null
    noteInteraction()
    trace('import', { proxy: `${transform.proxyW}×${transform.proxyH}`, original: `${transform.originalW}×${transform.originalH}`, raw })

    // The frame is on screen — ONLY NOW does the lane warm, queued behind the
    // decode so their peaks can never stack.
    warm({ withEncoder: false }).then(() => {
        if (!BUDGET.eagerEncode || !isCurrent()) return
        return encodeImage(proxyCanvas, { revision: state.revision })
    }).catch(() => null)

    return transform
}

/** Drop the held original and every embedding for it. */
export const releaseImage = () => {
    state.hasImage = false
    state.proxyCanvas = null
    state.lastPrompts = null
    clearHeavyQueue()
    releaseAsset()
    releaseDocument()
    forgetEncoder()
}

/* ─── selection ──────────────────────────────────────────────────────────── */

/**
 * Feed one real click back into hardware-fit, which sizes the proxy.
 *
 * Two figures, because postMs has two causes: the DEVICE's rate and the SCENE's
 * band. A click whose refined area is unknown updates neither — dividing it by
 * the proxy alone would credit a compact selection to the machine and report it
 * ~4.5x slower than it is. An encoding click is excluded: its postMs carries
 * the encode.
 */
const noteClickCost = (imageData) => {
    const run = clientState.lastRun
    if (!run || run.encoded || !(run.postMs > 0)) return
    const mp = (imageData.width * imageData.height) / 1e6
    if (!(mp > 0) || !(run.bandPixels > 0)) return
    const fraction = run.bandPixels / (mp * 1e6)
    const next = observePost(BUDGET.postMsPerMP, run.postMs, mp, fraction)
    if (!next) return
    BUDGET.postMsPerMP = next
    BUDGET.postMsPerMPSource = 'measured'
    BUDGET.postBandFraction = observeBandFraction(BUDGET.postBandFraction, fraction)
    savePostFit({ msPerMP: next, bandFraction: BUDGET.postBandFraction })
}

const finish = (res, imageData, extra = {}) => {
    noteClickCost(imageData)
    return {
        stale: false,
        usable: res.usable,
        reason: res.reason,
        canvas: fieldToCanvas(imageData),
        imageData,
        coverage: coverageOf(imageData),
        score: res.score,
        lane: res.lane,
        device: res.device,
        encoded: res.encoded,
        ms: res.ms,
        // Stage breakdown. `ms` alone cannot tell a slow encode from a slow refine,
        // which is the only question worth asking when a selection feels sluggish.
        stages: res.stages || null,
        ...extra,
    }
}

/**
 * One selection against the ≤proxy working canvas. Coordinates are canvas
 * pixels — the same space the React app draws its handles in — so no mapping
 * happens above this line.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {{ clicks?: Array<[number,number,0|1]>, box?: number[], lasso?: Array<[number,number]> }} prompts
 */
export const select = async (canvas, prompts = {}) => {
    let clicks = prompts.clicks ? prompts.clicks.slice() : []
    let box = prompts.box || null
    let clamp = null

    if (prompts.lasso) {
        const p = lassoToPrompts(prompts.lasso)
        if (!p) return { stale: false, usable: false, reason: 'the lasso stroke is too small to select with' }
        clicks = [p.point, ...clicks]
        box = p.box
        clamp = { poly: prompts.lasso, margin: p.margin }
    }
    if (!clicks.length && !box) return { stale: false, usable: false, reason: 'nothing to select with' }

    const rev = state.revision
    const res = await segment(canvas, { clicks, box, revision: rev })
    if (res.stale || rev !== state.revision) return { stale: true }
    if (!res.usable) return { stale: false, usable: false, reason: res.reason, score: res.score }

    const field = clamp ? clampToPolygon(res.imageData, clamp.poly, clamp.margin) : res.imageData
    state.lastPrompts = { clicks, box, clampPoly: clamp?.poly || null, clampMargin: clamp?.margin || 0 }
    return finish(res, field)
}

/**
 * "The subject", with no saliency model in the lane. SAM 2.1 emits three
 * granularity candidates and mask-select already rejects whole-scene runaways;
 * what it cannot do is guess which prompt means "the main thing". So probe with
 * the three prompts that disagree usefully — a frame-bounding box (extent), a
 * centre click (position), and both together — and keep the best usable result
 * that still leaves a background behind. The encoder runs ONCE (the embedding is
 * content-keyed and cached), so probes two and three are decoder passes only.
 */
export const selectSubject = async (canvas) => {
    const W = canvas.width
    const H = canvas.height
    const inset = [W * 0.04, H * 0.04, W * 0.96, H * 0.96]
    const probes = [
        { name: 'centre+frame', clicks: [[W / 2, H / 2, 1]], box: inset },
        { name: 'frame', box: inset },
        { name: 'centre', clicks: [[W / 2, H / 2, 1]] },
    ]
    let best = null
    for (const probe of probes) {
        const r = await select(canvas, probe)
        if (r.stale) return r
        if (!r.usable) continue
        // Below 1% is a speck, above 92% is the whole photo — neither is "the
        // subject". Near-runaways still count, but rank below a scoped answer.
        if (r.coverage < 0.01 || r.coverage > 0.92) continue
        const rank = (r.score || 0) - (r.coverage > 0.8 ? 0.25 : 0)
        if (!best || rank > best.rank) best = Object.assign(r, { rank, probe: probe.name })
    }
    return best || { stale: false, usable: false, reason: 'no subject stood out — try Click-Select or Box-Select' }
}

/** Proxy-space prompts behind the live mask — what export-hd maps to the crop. */
export const currentPrompts = () => (state.lastPrompts ? { ...state.lastPrompts } : { clicks: [], box: null })

/* ─── HD export source ───────────────────────────────────────────────────── */

/**
 * The original at export resolution, re-decoded from the retained bytes and
 * bounded by the budget. The caller grades it with the megashader (mask
 * textures scale up cleanly — they are continuous coverage, not bitmasks) and
 * must `close()` the source when `owned`.
 */
export const exportSource = async () => {
    if (!hasOriginal()) return null
    const t = getTransform()
    const { source, owned } = await getOriginalForExport({
        maxSide: BUDGET.exportMaxSide || 4096,
        maxMP: BUDGET.exportMaxMP || 12,
    })
    const width = source.width || source.naturalWidth
    const height = source.height || source.naturalHeight
    return { source, owned, width, height, transform: t, bounded: width < (t?.originalW || width) }
}

export const hasImage = () => state.hasImage
export const engineBudget = budget
