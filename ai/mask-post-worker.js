/**
 * mask-post-worker — runs the post pipeline off the main thread. `post` is the
 * largest stage of a steady-state selection (~132 ms, more than the GPU decode),
 * and on the main thread every millisecond of it is UI jank.
 *
 * Holds the guide image across calls, keyed by imageKey: it is the same for
 * every click on a photo and costs ~7 MB to ship. The client only re-sends it
 * when `guide` in the reply shows this worker lacks it at that size.
 *
 * in : { type:'post', requestId, imageKey, logits: ArrayBuffer (transfer),
 *        w, h, maskSide, clicks, tight, guide: ArrayBuffer|null (transfer) }
 * in : { type:'dispose' }
 * out: { type:'result', requestId, rgba, rawRgba, field (all transfer),
 *        stages, bandPixels, regions, maskRect, guide: {key,w,h}|null }
 * out: { type:'error', requestId, error }
 */

import { postCompute } from './mask-post-core.js'

let guide = null   // { key, w, h, px }

self.onmessage = (event) => {
    const msg = event.data || {}
    // drop-guide, not dispose: under pressure the ~7 MB guide is the cost, and
    // the worker itself has to stay — post is not optional the way refine is.
    if (msg.type === 'dispose' || msg.type === 'drop-guide') { guide = null; return }
    if (msg.type !== 'post') return

    const { requestId, imageKey, w, h, maskSide, clicks, tight } = msg
    try {
        if (msg.guide) guide = { key: imageKey, w, h, px: new Uint8ClampedArray(msg.guide) }
        // A stale guide is worse than none: it would refine against the wrong
        // photo. Drop it and ship the raw mask; the client re-sends next call.
        else if (!guide || guide.key !== imageKey || guide.w !== w || guide.h !== h) guide = null

        const { rgba, rawRgba, field, stages, bandPixels, regions, maskRect } = postCompute({
            logits: new Float32Array(msg.logits), guide: guide?.px || null, w, h, maskSide,
            clicks: clicks || [], tight: !!tight,
        })
        // Unrefined masks come back as the SAME array (postCompute starts with
        // `rgba = rawRgba`). Transferring one buffer twice throws, so send it
        // once and let the client re-alias, which is what the caller already saw.
        const aliased = rgba.buffer === rawRgba.buffer
        const transfer = aliased
            ? [rawRgba.buffer, field.buffer]
            : [rgba.buffer, rawRgba.buffer, field.buffer]
        self.postMessage({
            type: 'result',
            requestId,
            rgba: aliased ? null : rgba.buffer,
            rawRgba: rawRgba.buffer,
            aliased,
            field: field.buffer,
            stages,
            bandPixels,
            regions,
            maskRect,
            // Identity, not just the key — the client re-sends on a size change.
            guide: guide ? { key: guide.key, w: guide.w, h: guide.h } : null,
        }, transfer)
    } catch (err) {
        self.postMessage({ type: 'error', requestId, error: err?.message || String(err) })
    }
}
