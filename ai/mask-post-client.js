/**
 * mask-post-client — main-thread API over mask-post-worker.
 *
 * Returns null whenever the worker cannot serve a request (unavailable, timed
 * out, crashed). The caller then runs `postCompute` in-process, so a broken
 * worker costs jank, never a failed selection. Failure is sticky: no retry loop.
 */

import { postCompute } from './mask-post-core.js'

const POST_TIMEOUT_MS = 20_000

let worker = null
let broken = false
let seq = 0
const pending = new Map()

// Which guide the WORKER holds, as { key, w, h }. Tracked here so the ~7 MB
// getImageData readback happens once per photo instead of once per click.
//
// The SIZE is part of the identity, not just the key: proxy dimensions are
// re-planned per pressure level (proxy-plan.js), so the same photo can come
// back at 1536×1024 after being 1756×1024. Keying on imageKey alone leaves the
// worker holding a guide it must reject, and that ships one unrefined mask.
let workerGuide = null

const fail = (err) => {
    console.warn('[seglab][post] worker unavailable; posting on the main thread:', err)
    for (const [, entry] of pending) entry.resolve(null)
    pending.clear()
    try { worker?.terminate() } catch { /* dead */ }
    worker = null
    workerGuide = null
    broken = true
}

const getWorker = () => {
    if (broken) return null
    if (worker) return worker
    try {
        worker = new Worker(new URL('./mask-post-worker.js', import.meta.url), { type: 'module' })
        worker.onmessage = (event) => {
            const data = event.data || {}
            const entry = pending.get(data.requestId)
            if (!entry) return
            pending.delete(data.requestId)
            if (data.type === 'result') {
                workerGuide = data.guide || null
                entry.resolve(data)
            } else {
                console.warn('[seglab][post] worker error; posting on the main thread:', data.error)
                entry.resolve(null)
            }
        }
        worker.onerror = (event) => fail(event?.message || 'worker crashed')
        return worker
    } catch (err) {
        fail(err?.message)
        return null
    }
}

/**
 * Drop the worker's guide copy without killing the worker. Post is not optional
 * the way cv-refine is, so pressure sheds the ~7 MB buffer, not the thread; the
 * next call re-sends it and pays one getImageData.
 */
export const dropMaskPostGuide = () => {
    workerGuide = null
    try { worker?.postMessage({ type: 'drop-guide' }) } catch { /* gone */ }
}

export const disposeMaskPost = () => {
    if (!worker) return
    try { worker.postMessage({ type: 'dispose' }) } catch { /* gone */ }
    try { worker.terminate() } catch { /* gone */ }
    worker = null
    workerGuide = null
    for (const [, entry] of pending) entry.resolve(null)
    pending.clear()
}

const readGuide = (canvas, w, h) => {
    try {
        return canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h).data
    } catch { return null }   // tainted canvas — the raw mask still ships
}

/**
 * Run the post pipeline off-thread. `logits` is COPIED, not transferred: the
 * candidate parking in sam21-adapter keeps the caller's plane for cycling.
 * Resolves null when the worker could not serve it — run postCompute yourself.
 */
export const postAsync = ({ canvas, imageKey, logits, w, h, maskSide, clicks = [], tight = false }) => {
    const wk = getWorker()
    if (!wk) return Promise.resolve(null)

    // Only pay the readback when this worker does not already hold THIS guide
    // at THIS size.
    const held = workerGuide
    const guide = (held && held.key === imageKey && held.w === w && held.h === h)
        ? null
        : readGuide(canvas, w, h)

    seq += 1
    const requestId = `post${seq}`
    return new Promise((resolve) => {
        const timer = setTimeout(() => {
            pending.delete(requestId)
            console.warn('[seglab][post] worker timed out; posting on the main thread')
            resolve(null)
        }, POST_TIMEOUT_MS)
        pending.set(requestId, { resolve: (data) => { clearTimeout(timer); resolve(data) } })

        const plane = logits.slice()
        const transfer = [plane.buffer]
        if (guide) transfer.push(guide.buffer)
        try {
            wk.postMessage({
                type: 'post',
                requestId,
                imageKey,
                logits: plane.buffer,
                guide: guide ? guide.buffer : null,
                w,
                h,
                maskSide,
                clicks,
                tight,
            }, transfer)
        } catch (err) {
            clearTimeout(timer)
            pending.delete(requestId)
            console.warn('[seglab][post] dispatch failed; posting on the main thread:', err?.message)
            resolve(null)
        }
    })
}

/** Unpack a worker reply into the shape postCompute returns. */
export const unpackPost = (data) => {
    const rawRgba = new Uint8ClampedArray(data.rawRgba)
    return {
        rgba: data.aliased ? rawRgba : new Uint8ClampedArray(data.rgba),
        rawRgba,
        field: new Float32Array(data.field),
        stages: data.stages,
        bandPixels: data.bandPixels || 0,
        regions: data.regions || null,
        maskRect: data.maskRect || null,
    }
}

export { postCompute }
