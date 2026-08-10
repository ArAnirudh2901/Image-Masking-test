/**
 * Two properties, isolated.
 *
 *  A. TILING IS SOUND — tiled refinement must agree with the same implementation
 *     run as ONE rect (tile=1e9). Same code, same grid, only the decomposition
 *     differs, so this is a clean apples-to-apples test of the tiling itself.
 *
 *  B. THE PHASE FIX WORKS — nudging the bbox by 1..3 px must no longer move the
 *     output. This is the control that exposed the shipped path's instability.
 */
import * as base from '../ai/mask-refine.js'
import * as opt from './mask-refine-opt.js'
import { loadRgba, makeLogits, upsampleLogits, maskDiff, PROXIES } from './harness.mjs'

const BAND = 0.9
const COVERS = [0.001, 0.005, 0.02, 0.1, 0.35, 0.7]
const fail = []

for (const p of PROXIES) {
    const px = loadRgba(p.name, p.w, p.h)
    console.log(`\n=== ${p.label} ===`)
    console.log('cover   A:tiled-vs-1rect   B:opt±1px  ±2px  ±3px    (control) base±1px  ±2px')
    for (const cover of COVERS) {
        const { field: f0, bbox } = upsampleLogits(makeLogits(cover), p.w, p.h)
        const run = (impl, bb, o = {}) => {
            const f = f0.slice()
            impl.refineField(f, px, p.w, p.h, bb, { radius: 8, eps: 1e-4, scale: 4, ...o })
            return impl.bandAlpha(f, p.w, p.h, BAND)
        }
        const grow = (d) => [Math.max(0, bbox[0] - d), Math.max(0, bbox[1] - d),
            Math.min(p.w - 1, bbox[2] + d), Math.min(p.h - 1, bbox[3] + d)]

        // A. tiling soundness
        const tiled = run(opt, bbox)
        const oneRect = run(opt, bbox, { tile: 1e9 })
        const aTile = maskDiff(tiled, oneRect).iou

        // B. phase stability of opt, and the base control
        const oShift = [1, 2, 3].map((d) => maskDiff(tiled, run(opt, grow(d))).iou)
        const b0 = run(base, bbox)
        const bShift = [1, 2].map((d) => maskDiff(b0, run(base, grow(d))).iou)

        if (aTile < 0.998) fail.push(`tiling ${p.w} cover=${cover} IoU=${aTile.toFixed(5)}`)
        if (Math.min(...oShift) < 0.998) fail.push(`phase ${p.w} cover=${cover} IoU=${Math.min(...oShift).toFixed(5)}`)

        console.log([String(cover).padEnd(6),
            aTile.toFixed(5).padStart(14),
            ...oShift.map((v) => v.toFixed(5)),
            '  ', ...bShift.map((v) => v.toFixed(5))].join('  '))
    }
}

console.log(fail.length ? `\n${fail.length} FAILURE(S):\n` + fail.map((f) => ' - ' + f).join('\n')
    : '\nTILING SOUND + PHASE STABLE ✓')
process.exit(fail.length ? 1 : 0)
