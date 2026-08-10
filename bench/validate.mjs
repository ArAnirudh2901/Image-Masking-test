/**
 * Final validation. The reference is the EXACT filter — the shipped guided
 * filter at scale=1 over the whole band bbox, which has no subsampled grid and
 * is therefore bit-stable under a bbox nudge. Everything is scored against it.
 *
 * Bars:
 *   1. accuracy   — opt must be at least as close to exact as the shipped s=4 is.
 *   2. stability  — a benign 1..3 px bbox nudge must move opt less than it moves
 *                   the shipped path.
 *   3. tiling     — at a FIXED s, tiled must equal single-rect (isolates tiling
 *                   from the adaptive-scale change).
 *   4. speed      — opt must not be slower at any coverage.
 */
import * as base from './orig-mask-refine.js'
import * as opt from '../ai/mask-refine.js'
import { loadRgba, makeLogits, upsampleLogits, maskDiff, time, PROXIES } from './harness.mjs'

const BAND = 0.9
const COVERS = [0.001, 0.005, 0.02, 0.1, 0.35, 0.7]
const fail = []
const warn = []

for (const p of PROXIES) {
    const px = loadRgba(p.name, p.w, p.h)
    console.log(`\n════ ${p.label} ════`)
    console.log('cover   acc:base  acc:opt   stab:base        stab:opt         tile   ms:base  ms:opt   x')
    for (const cover of COVERS) {
        const { field: f0, bbox } = upsampleLogits(makeLogits(cover), p.w, p.h)
        const grow = (d) => [Math.max(0, bbox[0] - d), Math.max(0, bbox[1] - d),
            Math.min(p.w - 1, bbox[2] + d), Math.min(p.h - 1, bbox[3] + d)]
        const run = (impl, bb, o) => {
            const f = f0.slice()
            impl.refineField(f, px, p.w, p.h, bb, { radius: 8, eps: 1e-4, scale: 4, ...o })
            return impl.bandAlpha(f, p.w, p.h, BAND)
        }
        // reference: exact filter, no subsampling
        const exact = run(base, bbox, { scale: 1 })

        const mB = run(base, bbox)
        const mO = run(opt, bbox)
        const accB = maskDiff(mB, exact).iou
        const accO = maskDiff(mO, exact).iou

        const stabB = Math.min(...[1, 2, 3].map((d) => maskDiff(mB, run(base, grow(d))).iou))
        const stabO = Math.min(...[1, 2, 3].map((d) => maskDiff(mO, run(opt, grow(d))).iou))

        // tiling isolated: same s, tiled vs one rect
        const tA = run(opt, bbox, { scale: 1, tile: 96 })
        const tB = run(opt, bbox, { scale: 1, tile: 1e9 })
        const tileIoU = maskDiff(tA, tB).iou

        const timeB = time(() => { const f = f0.slice(); base.refineField(f, px, p.w, p.h, bbox, { radius: 8, eps: 1e-4, scale: 4 }) }, 9, 3)
        const timeO = time(() => { const f = f0.slice(); opt.refineField(f, px, p.w, p.h, bbox, { radius: 8, eps: 1e-4, scale: 4 }) }, 9, 3)

        if (accO < accB - 0.002) fail.push(`accuracy regressed ${p.w} cover=${cover}: opt ${accO.toFixed(4)} < base ${accB.toFixed(4)}`)
        if (stabO < stabB - 0.002) fail.push(`stability regressed ${p.w} cover=${cover}: opt ${stabO.toFixed(4)} < base ${stabB.toFixed(4)}`)
        if (tileIoU < 0.999) fail.push(`tiling ${p.w} cover=${cover} IoU=${tileIoU.toFixed(5)} at fixed s`)
        if (timeO.med > timeB.med * 1.15) warn.push(`slower ${p.w} cover=${cover}: ${timeO.med.toFixed(2)} vs ${timeB.med.toFixed(2)} ms`)

        console.log([String(cover).padEnd(6),
            accB.toFixed(4).padStart(9), accO.toFixed(4).padStart(9),
            stabB.toFixed(5).padStart(11), stabO.toFixed(5).padStart(17),
            tileIoU.toFixed(4).padStart(9),
            timeB.med.toFixed(2).padStart(8), timeO.med.toFixed(2).padStart(8),
            `${(timeB.med / timeO.med).toFixed(2)}x`.padStart(7)].join(''))
    }
}

if (warn.length) console.log('\nWARN:\n' + warn.map((w) => ' - ' + w).join('\n'))
console.log(fail.length ? `\n${fail.length} FAILURE(S):\n` + fail.map((f) => ' - ' + f).join('\n')
    : '\nACCURACY ↑, STABILITY ↑, TILING EXACT ✓')
process.exit(fail.length ? 1 : 0)
