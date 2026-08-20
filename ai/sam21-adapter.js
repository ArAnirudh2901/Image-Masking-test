/**
 * sam21-adapter — bridges the SAM 2.1 lane to sam-client's segment() contract.
 *
 * The lane speaks logits at 256²; the app wants an RGBA mask at proxy size plus
 * a score, and runs its own post pipeline (lasso clamp → hygiene → cv-refine)
 * on top. This converts between the two and nothing else.
 *
 * SAM 2.1 is the only mask lane and there is no switch. The `?lane=slimsam`
 * escape hatch that used to reach a second engine is gone along with that
 * engine: a URL parameter that silently changes which model produced a mask is
 * not a debugging aid, it is a way to misread every measurement taken after it.
 */

import { SIDE, MASK_SIDE, LANE } from './sam21-lane.js'
import { decodeMask, encodeImage, hello, hostMode } from './sam21-client.js'
import { noteModel } from './model-registry.js'
import { bandAlpha, refineField } from './mask-refine.js'
import { bandWidth, postCompute, upsampleLogits as upsampleField } from './mask-post-core.js'
import { postAsync, unpackPost } from './mask-post-client.js'
import { fieldArea } from './mask-select.js'

let greeted = false

const upsampleLogits = (logits, w, h) => upsampleField(logits, w, h, MASK_SIDE)


/**
 * Run one selection. `canvas` is the proxy the app already holds; `clicks` are
 * [x, y, label] triples in ITS pixel space. Returns the shape sam-client's tail
 * (summarize → validate → emit) consumes.
 */
export const sam21Segment = async ({ canvas, imageKey, clicks, box, onWait }) => {
    if (!greeted) { await hello('seglab'); greeted = true }
    const t0 = performance.now()

    // SAM 2.1 squashes the frame to 1024² (it does not letterbox), so mapping
    // is a plain per-axis scale in both directions.
    const kx = SIDE / canvas.width
    const ky = SIDE / canvas.height
    const pts = (clicks || []).map(([x, y, label]) => ({ x: x * kx, y: y * ky, label: label ?? 1 }))
    // The same prompts in PROXY space, for the hygiene pass that runs after the
    // post pipeline: an include click protects the component it landed on, and
    // that component only exists once the field has been upsampled.
    const proxyPts = (clicks || []).map(([x, y, label]) => ({ x, y, label: label ?? 1 }))
    // A box IS expressible: SAM 2.1 encodes it as its two corners with labels
    // 2 (top-left) and 3 (bottom-right). The decoder carries
    // prompt_encoder.point_embeddings.2/.3 and branches on those label values —
    // verified against models/sam21/decoder.fp16.onnx, not assumed.
    //
    // This previously collapsed a box to a single centre click, which threw away
    // the extent the detector had already localised and left SAM to re-guess it.
    // Text search selects by box, so that degraded every text-driven selection.
    if (box) {
        pts.push(
            { x: box[0] * kx, y: box[1] * ky, label: 2 },
            { x: box[2] * kx, y: box[3] * ky, label: 3 },
        )
    }
    if (!pts.length) throw new Error('sam21: no clicks')

    // Transient, self-healing states — a deep-idle shutdown, a lost device, an
    // LRU eviction by another tab, an allocation that lost a race under memory
    // pressure. All are retryable; anything else is a real bug and must surface.
    // Deliberately NOT timeouts: an encode waits 10 minutes, so retrying one
    // buys a 20-minute hang. A dead host is caught in 2 s by the client's ping
    // probe and reported as "host restarting", which is in this set.
    // `device` alone matched any message merely CONTAINING the word — including
    // programming errors like a TypeError naming a `device` property, which were
    // then retried and reported as 'encode failed' instead of surfacing as bugs.
    const RECOVERABLE = /no embedding|host restarting|device lost|lost device|no webgpu device|out of memory|failed to allocate|createbuffer/i

    const encodeOnce = async () => {
        const bitmap = await createImageBitmap(canvas)
        try {
            return await encodeImage(bitmap, imageKey, { onWait })
        } catch (err) {
            try { bitmap.close?.() } catch { /* already transferred */ }
            throw err
        }
    }

    // Same recoverable set as the decode retry below, plus the states only an
    // encode can hit: it is the call that BUILDS the session, so a host that
    // died or a device that went away surfaces here first. One retry — the
    // client reconnects to a fresh host on its own — turns a dead click into a
    // slow one. A bad prompt or a missing model still fails immediately.
    let enc
    try {
        enc = await encodeOnce()
    } catch (err) {
        if (!RECOVERABLE.test(String(err?.message))) throw err
        console.warn('[seglab][sam21] encode failed; retrying once:', err?.message)
        enc = await encodeOnce()
    }
    const encodeMs = performance.now() - t0
    noteModel('sam21', { device: 'webgpu', scale: 'small', release: 'darktable-5.6.0' })

    const t1 = performance.now()
    let dec
    try {
        dec = await decodeMask(pts, { key: imageKey, onWait })
    } catch (err) {
        // The embedding can disappear between encode and decode for reasons
        // that are all NORMAL: deep-idle worker shutdown, a lost GPU device,
        // LRU eviction by another tab. Re-encode once and retry rather than
        // surfacing "selection failed" for a recoverable state.
        if (!RECOVERABLE.test(String(err?.message))) throw err
        console.warn('[seglab][sam21] embedding gone; re-encoding:', err?.message)
        await encodeOnce()
        dec = await decodeMask(pts, { key: imageKey, onWait })
    }
    const decodeMs = performance.now() - t1

    const t2 = performance.now()
    const w = canvas.width
    const h = canvas.height
    const { rgba, rawRgba, bandPixels, regions, maskRect } =
        await postProcessAsync(canvas, imageKey, dec.logits, w, h, proxyPts)

    // Park every candidate SAM already computed, ordered small → large, so the
    // app can answer "you took the wrong part of it" with a repaint. The
    // inference is already paid for; the only cost is 256 KB per plane.
    candidates = orderCandidates({
        imageKey, canvas, w, h, clicks: proxyPts,
        planes: [dec.logits, ...(dec.alternates || [])],
        scores: [dec.iou, ...(dec.altScores || [])],
    })

    return {
        rgba,
        rawRgba,
        width: w,
        height: h,
        score: dec.iou,
        encoded: !enc.cached,
        encodeMs: +encodeMs.toFixed(1),
        decodeMs: +decodeMs.toFixed(1),
        postMs: +(performance.now() - t2).toFixed(1),
        postStages: lastPostStages,
        // hardware-fit's normaliser: postMs scales with the band the filter
        // walked, not with the proxy's own area.
        bandPixels,
        maskRect,
        device: 'webgpu',
        lane: `${LANE} (${hostMode() === 'shared' ? 'shared' : 'per-tab'})`,
        candidates: candidateInfo(),
        // Why this candidate won, and what region hygiene removed. Debug-only,
        // but it is the difference between "the mask is wrong" and knowing which
        // of six mechanisms produced it.
        pick: {
            reason: dec.reason,
            stability: dec.stability,
            refined: dec.refined,
            regions: dec.regions,
            // Hygiene at proxy scale, which the 256² pass cannot see: the
            // upsample and the guided filter both manufacture speckle.
            postRegions: regions,
        },
    }
}

/**
 * 256² logits → the two RGBA masks the app wants, and the continuous field the
 * export path reads. Shared by a fresh decode and by candidate cycling, because
 * cycling has to produce a mask indistinguishable from a decoded one — same
 * band, same guided filter, same parked field.
 */
// Per-stage cost of the last postProcess. `postMs` is the largest stage of a
// steady-state click (larger than the GPU decode), and one number cannot say
// which of four passes owns it.
export let lastPostStages = null

// The guide image is the SAME for every click on a photo, but getImageData
// forces a GPU→CPU readback of the whole proxy (~2.8 MB). Read it once per
// image, not once per click. Only the in-process path uses this — the worker
// keeps its own copy, and mask-post-client tracks which image that copy is for.
let guideCache = null
/** Release the in-process guide copy (memory pressure). */
export const clearGuideCache = () => { guideCache = null }
const guidePixels = (canvas, imageKey, w, h) => {
    if (guideCache && guideCache.key === imageKey && guideCache.w === w && guideCache.h === h) {
        return guideCache.px
    }
    const px = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h).data
    guideCache = { key: imageKey, w, h, px }
    return px
}

/** Record what postCompute produced. Shared by the worker and in-process paths. */
const adoptPost = (imageKey, w, h, out) => {
    lastPostStages = out.stages
    // Keep the continuous field for export: upscaling a THRESHOLDED mask to
    // 8256×5504 is what makes edges stair-step, and no amount of export-time
    // filtering recovers from it (§10).
    lastField = { imageKey, w, h, field: out.field }
    return {
        rgba: out.rgba,
        rawRgba: out.rawRgba,
        bandPixels: out.bandPixels || 0,
        regions: out.regions || null,
        maskRect: out.maskRect || null,
    }
}

/** In-process post — the fallback path. */
const postProcess = (canvas, imageKey, logits, w, h, clicks = [], tight = false) => {
    let guide = null
    try { guide = guidePixels(canvas, imageKey, w, h) } catch { /* tainted canvas */ }
    return adoptPost(imageKey, w, h,
        postCompute({ logits, guide, w, h, maskSide: MASK_SIDE, clicks, tight }))
}

/** Off-thread post, falling back in-process so a broken worker costs jank only. */
const postProcessAsync = async (canvas, imageKey, logits, w, h, clicks = [], tight = false) => {
    const reply = await postAsync({ canvas, imageKey, logits, w, h, maskSide: MASK_SIDE, clicks, tight })
    if (!reply) return postProcess(canvas, imageKey, logits, w, h, clicks, tight)
    return adoptPost(imageKey, w, h, unpackPost(reply))
}

/* ─── Candidate cycling ───────────────────────────────────────────────────────
 * SAM's three masks are a HIERARCHY (subpart / part / whole), and no scoring
 * rule resolves a first click that is genuinely ambiguous — "the petal" and
 * "the bloom" are both correct answers to one point on a rose. Arbitration
 * picks the best default; this is the escape hatch when the default is not what
 * the user meant, and it must not cost a decode, because the answer is already
 * in memory.
 *
 * Ordered by AREA, not by model index, so the control is monotonic: forward
 * always grows the selection, back always shrinks it. That is a promise a user
 * can hold in their head; "the next one the model emitted" is not.
 */
let candidates = null

const orderCandidates = ({ imageKey, canvas, w, h, clicks, planes, scores }) => {
    const rows = planes
        .map((p, i) => ({ p, score: scores[i] ?? 0, area: fieldArea(p), first: i === 0 }))
        .filter((r) => r.area > 0)
        .sort((a, b) => a.area - b.area)
    if (!rows.length) return null
    const index = Math.max(0, rows.findIndex((r) => r.first))
    return { imageKey, canvas, w, h, clicks: clicks || [], rows, index }
}

const candidateInfo = () =>
    (candidates ? { count: candidates.rows.length, index: candidates.index } : null)

/**
 * Move to the next/previous candidate and re-run the post pipeline on it.
 * Async: dispatches to the mask-post worker so the ~90–134 ms of upsample +
 * guided filter runs off the main thread. Falls back in-process when the
 * worker is broken, same as sam21Segment.
 *
 * The original docblock said "synchronous on purpose — awaiting it would put
 * it back on the same footing as the decode it exists to avoid." That
 * reasoning predated the worker existing. Now that the worker is warm, the
 * round-trip adds ~20 ms of latency (invisible for a ↑/↓ keypress) and
 * removes ~130 ms of UI jank.
 *
 * Returns null when there is nothing parked for `imageKey`, or only one
 * distinct candidate.
 */
export const sam21Cycle = async (delta = 1, imageKey = null) => {
    if (!candidates || candidates.rows.length < 2) return null
    if (imageKey && candidates.imageKey !== imageKey) return null
    const t0 = performance.now()
    const n = candidates.rows.length
    candidates.index = (candidates.index + (delta < 0 ? n - 1 : 1)) % n
    const row = candidates.rows[candidates.index]
    const { canvas, w, h, clicks } = candidates
    // Tight hygiene, which a first click does not get: cycling means "the thing
    // I pointed at, at another scope", so a plane that answers with other
    // objects is answering a question nobody asked (mask-select, tight mode).
    const { rgba, rawRgba, bandPixels, maskRect } =
        await postProcessAsync(canvas, candidates.imageKey, row.p, w, h, clicks, true)
    return {
        rgba,
        rawRgba,
        width: w,
        height: h,
        score: row.score,
        encoded: false,
        encodeMs: 0,
        decodeMs: 0,
        postMs: +(performance.now() - t0).toFixed(1),
        bandPixels,
        maskRect,
        device: 'webgpu',
        lane: `${LANE} (${hostMode() === 'shared' ? 'shared' : 'per-tab'})`,
        candidates: candidateInfo(),
        cycled: true,
    }
}

/** How many candidates are parked, and which one is showing. */
export const sam21Candidates = (imageKey = null) =>
    (candidates && (!imageKey || candidates.imageKey === imageKey) ? candidateInfo() : null)

let lastField = null
/** The refined score field behind the current mask, for the export path. */
export const currentField = (imageKey) =>
    (lastField && (!imageKey || lastField.imageKey === imageKey) ? lastField : null)

/**
 * Export-resolution alpha, derived from the CONTINUOUS field.
 *
 * The export path's default is to upscale the thresholded proxy mask with canvas
 * `drawImage` and refine that — the exact failure §10 opens with. A 256² mask
 * reaching an 8256×5504 frame is a ~32× enlargement, and no amount of
 * export-time filtering recovers detail that thresholding already discarded.
 *
 * Instead: crop the field to the export rect, bicubic-upsample the SCORE FIELD
 * to native resolution, guided-filter it against the native crop's own luma, and
 * matte at that resolution. Returns RGBA at (cropW, cropH), or null when no
 * field is resident (caller keeps its own path).
 */
export const sam21HdAlpha = ({ bitmap, imageKey, subrect, cropW, cropH }) => {
    const held = currentField(imageKey)
    if (!held || !cropW || !cropH) return null
    const { field, w: fw, h: fh } = held
    const { sx, sy, sw, sh } = subrect
    if (!(sw > 0) || !(sh > 0)) return null

    // Separable bicubic straight from the field's crop to native size. Same
    // Catmull-Rom taps as the proxy path; x-weights are shared across rows.
    const xi = new Int32Array(cropW * 4)
    const xw = new Float32Array(cropW * 4)
    const clampX = (i) => (i < 0 ? 0 : (i > fw - 1 ? fw - 1 : i))
    const clampY = (i) => (i < 0 ? 0 : (i > fh - 1 ? fh - 1 : i))
    for (let x = 0; x < cropW; x += 1) {
        const fx = sx + (x + 0.5) * (sw / cropW) - 0.5
        const x0 = Math.floor(fx)
        const k = cubic(fx - x0)
        for (let t = 0; t < 4; t += 1) { xi[x * 4 + t] = clampX(x0 - 1 + t); xw[x * 4 + t] = k[t] }
    }
    const out = new Float32Array(cropW * cropH)
    let minX = cropW; let minY = cropH; let maxX = -1; let maxY = -1
    const row4 = new Float32Array(cropW * 4)
    for (let y = 0; y < cropH; y += 1) {
        const fy = sy + (y + 0.5) * (sh / cropH) - 0.5
        const y0 = Math.floor(fy)
        const ky = cubic(fy - y0)
        const rows = [clampY(y0 - 1) * fw, clampY(y0) * fw, clampY(y0 + 1) * fw, clampY(y0 + 2) * fw]
        // Horizontal pass for the four source rows this output row needs.
        for (let r = 0; r < 4; r += 1) {
            const base = rows[r]
            for (let x = 0; x < cropW; x += 1) {
                const b = x * 4
                row4[r * cropW + x] = field[base + xi[b]] * xw[b] + field[base + xi[b + 1]] * xw[b + 1]
                    + field[base + xi[b + 2]] * xw[b + 2] + field[base + xi[b + 3]] * xw[b + 3]
            }
        }
        const dst = y * cropW
        for (let x = 0; x < cropW; x += 1) {
            const v = row4[x] * ky[0] + row4[cropW + x] * ky[1]
                + row4[2 * cropW + x] * ky[2] + row4[3 * cropW + x] * ky[3]
            out[dst + x] = v
            if (v > -6 && v < 6) {
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            }
        }
    }

    // Snap to the native crop's own edges — this is where the detail the proxy
    // never had actually comes from.
    let px = null
    try {
        const c = new OffscreenCanvas(cropW, cropH)
        const ctx = c.getContext('2d', { willReadFrequently: true })
        ctx.drawImage(bitmap, 0, 0, cropW, cropH)
        px = ctx.getImageData(0, 0, cropW, cropH).data
    } catch { /* tainted — matte the unrefined field instead */ }
    const bbox = maxX < 0 ? null : [minX, minY, maxX, maxY]
    // Radius must bracket the upsampled transition (~`up` px wide) and no more.
    // Scaling it BY the enlargement is wrong: radius 80 on a 307 px crop smears
    // the field flat, the gradient collapses, and the matting band — which is
    // derived from that gradient — explodes to a third of the frame (measured).
    const up = Math.max(1, cropW / Math.max(1, sw))
    const radius = Math.round(Math.min(32, Math.max(4, up * 1.5)))
    if (px && bbox) {
        refineField(out, px, cropW, cropH, bbox,
            { radius, eps: 1e-4, scale: Math.max(1, Math.min(8, Math.round(radius / 4))) })
    }
    return { rgba: bandAlpha(out, cropW, cropH, bandWidth(out, cropW, bbox)), width: cropW, height: cropH }
}

/**
 * Encode the CROP itself and decode against it — the real native re-decode.
 * The document embedding saw this object at proxy scale (often ~30 px); here the
 * model sees it at crop resolution. Transient: the crop embedding is keyed
 * separately and never persisted, since it is worthless once the crop changes.
 */
const cropDecode = async ({ bitmap, cropKey, cropW, cropH, prompts }) => {
    const key = `crop:${cropKey}`
    const kx = SIDE / cropW
    const ky = SIDE / cropH
    const pts = (prompts.clicks || []).map(([x, y, label]) => ({ x: x * kx, y: y * ky, label: label ?? 1 }))
    // Same box contract as sam21Segment: corners at labels 2/3. This used to
    // collapse the box to its centre point, which is the trap §5 of
    // DESIGN-TEXT-LANE documents as fixed — it was, but only on the document
    // path. A text pick escalates with a BOX and nothing else, so on this path
    // the extent the detector localised was thrown away and re-guessed from one
    // point, at exactly the moment the crop finally had the resolution to use it.
    if (prompts.box) {
        const [x0, y0, x1, y1] = prompts.box
        pts.push({ x: x0 * kx, y: y0 * ky, label: 2 }, { x: x1 * kx, y: y1 * ky, label: 3 })
    }
    if (!pts.length) return null

    // createImageBitmap so the caller keeps its own handle for the composite.
    const copy = await createImageBitmap(bitmap)
    await encodeImage(copy, key, { warmDecoder: false, persist: false })
    const dec = await decodeMask(pts, { key })

    const { field, bbox } = upsampleLogits(dec.logits, cropW, cropH)
    try {
        const c = new OffscreenCanvas(cropW, cropH)
        const ctx = c.getContext('2d', { willReadFrequently: true })
        ctx.drawImage(bitmap, 0, 0, cropW, cropH)
        const px = ctx.getImageData(0, 0, cropW, cropH).data
        refineField(field, px, cropW, cropH, bbox, { radius: 8, eps: 1e-4, scale: 4 })
        c.width = c.height = 0
    } catch { /* tainted — matte the unrefined field */ }
    return { rgba: bandAlpha(field, cropW, cropH, bandWidth(field, cropW, bbox)), width: cropW, height: cropH, iou: dec.iou }
}

/**
 * Compose the export cutout entirely on this lane — no worker, no model.
 *
 * `hdExport` used to route through the retired SlimSAM worker purely for this
 * step, which meant constructing a whole transformers + ORT stack to multiply
 * pixels by an alpha channel. It also crashed doing it (observed in verify's
 * export phase). The alpha is already correct at native resolution, so the
 * remaining work is straight canvas: fuse RGB with alpha, optionally encode.
 *
 * Returns the same shape sam-engine's hdRefineOnce did, so export-hd.js is
 * unchanged. Null means "not ours" — the caller keeps its own path.
 */
export const sam21HdCompose = async ({
    bitmap, imageKey, subrect, compose, emitBlob, doDecode, cropKey, prompts,
}) => {
    const cropW = bitmap?.width
    const cropH = bitmap?.height

    // Native re-decode (escalation). §11 keeps this OFF automatically but it is
    // still an explicit user action: run the model on the crop at full
    // resolution, where a 30 px object in the proxy becomes hundreds of pixels.
    // Costs one extra encode, which is why it is never automatic.
    let decoded = false
    let hd = null
    if (doDecode && cropKey && (prompts?.clicks?.length || prompts?.box)) {
        try {
            hd = await cropDecode({ bitmap, cropKey, cropW, cropH, prompts })
            decoded = !!hd
        } catch (err) {
            console.warn('[seglab][sam21] crop re-decode failed; using the proxy field:', err?.message)
        }
    }
    // Fall back to upsampling the document's field for this rect.
    if (!hd) hd = sam21HdAlpha({ bitmap, imageKey, subrect, cropW, cropH })
    if (!hd) return null

    // Escalation path wants the mask alone.
    if (!compose) {
        return { alpha: hd.rgba, width: cropW, height: cropH, decoded, iou: hd.iou || 0, lane: LANE }
    }

    const canvas = new OffscreenCanvas(cropW, cropH)
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    ctx.drawImage(bitmap, 0, 0, cropW, cropH)
    const img = ctx.getImageData(0, 0, cropW, cropH)
    const src = img.data
    // Straight alpha, in place — the mask's R channel IS the coverage ramp.
    for (let i = 0, j = 3; i < hd.rgba.length; i += 4, j += 4) src[j] = hd.rgba[i]

    if (emitBlob) {
        ctx.putImageData(img, 0, 0)
        const blob = await canvas.convertToBlob({ type: 'image/png' })
        canvas.width = canvas.height = 0   // drop the backing store promptly
        return { blob, width: cropW, height: cropH, decoded, lane: LANE }
    }
    canvas.width = canvas.height = 0
    return { cutout: src, width: cropW, height: cropH, decoded, iou: hd.iou || 0, lane: LANE }
}
