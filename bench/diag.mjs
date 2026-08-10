/**
 * Where does the instability actually come from?
 *
 * Sweep the subsample factor s and, for each, measure how much a benign 1 px
 * bbox nudge moves the thresholded mask. If s is the cause, stability should
 * improve monotonically as s → 1, and s=1 should be exactly stable (no
 * subsampled grid exists to shift).
 *
 * Run against the SHIPPED implementation so the conclusion is about the shipped
 * algorithm, not about my rewrite.
 */
import * as base from '../ai/mask-refine.js'
import { loadRgba, makeLogits, upsampleLogits, maskDiff, time, PROXIES } from './harness.mjs'

const BAND = 0.9
const p = PROXIES[0]
const px = loadRgba(p.name, p.w, p.h)

for (const cover of [0.001, 0.02, 0.35]) {
    const { field: f0, bbox } = upsampleLogits(makeLogits(cover), p.w, p.h)
    const run = (bb, scale) => {
        const f = f0.slice()
        base.refineField(f, px, p.w, p.h, bb, { radius: 8, eps: 1e-4, scale })
        return base.bandAlpha(f, p.w, p.h, BAND)
    }
    const grow = (d) => [Math.max(0, bbox[0] - d), Math.max(0, bbox[1] - d),
        Math.min(p.w - 1, bbox[2] + d), Math.min(p.h - 1, bbox[3] + d)]
    const raw = base.bandAlpha(f0, p.w, p.h, BAND)

    console.log(`\n=== cover=${cover}  (band bbox ${bbox.join(',')}) ===`)
    console.log('scale   IoU(s,s+1px)  IoU(s,s+2px)   IoU(s, scale=1)   IoU(s, raw)   refine_ms')
    const ref1 = run(bbox, 1)
    for (const scale of [1, 2, 3, 4, 6, 8]) {
        const m = run(bbox, scale)
        const s1 = maskDiff(m, run(grow(1), scale)).iou
        const s2 = maskDiff(m, run(grow(2), scale)).iou
        const vs1 = maskDiff(m, ref1).iou
        const vsRaw = maskDiff(m, raw).iou
        const t = time(() => { const f = f0.slice(); base.refineField(f, px, p.w, p.h, bbox, { radius: 8, eps: 1e-4, scale }) }, 7, 2)
        console.log([String(scale).padEnd(5), s1.toFixed(5).padStart(12), s2.toFixed(5).padStart(13),
            vs1.toFixed(5).padStart(16), vsRaw.toFixed(5).padStart(13), t.med.toFixed(2).padStart(10)].join(''))
    }
}
