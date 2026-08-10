/**
 * Sweep CELL_BUDGET — the refinement's accuracy/speed dial.
 *
 * s > 1 is the only thing that makes the refined edge non-deterministic, and
 * CELL_BUDGET is what decides when s leaves 1. Picking it by feel is how it first
 * ended up at 40k, which silently dropped mid-size masks to s=3 where accuracy
 * against the exact filter is FLAT in s (so the finer grid bought nothing) and
 * sits around 0.98. 100k holds s<=2 there and beat 40k at every size.
 *
 * Reference is always the exact s=1 filter, never upstream — upstream is a
 * different approximation, not ground truth.
 *
 *   bun run bench/budget.mjs
 */
import { loadRgba, makeLogits, upsampleLogits, maskDiff, PROXIES, time } from './harness.mjs'
import * as opt from '../ai/mask-refine.js'
import * as base from './orig-mask-refine.js'

const COVERS = [0.001, 0.005, 0.02, 0.1, 0.35, 0.7]
const BUDGETS = [40_000, 100_000, 200_000]
const SHIPPED = 100_000

for (const P of PROXIES) {
    const px = loadRgba(P.name, P.w, P.h)
    const w = P.w, h = P.h
    console.log(`\n== ${P.label} ==`)
    let hdr = 'cover     base_acc base_ms '
    for (const B of BUDGETS) hdr += `| B=${B / 1000}k acc    ms  s `
    console.log(hdr)
    for (const cover of COVERS) {
        const { field: f0, bbox } = upsampleLogits(makeLogits(cover), w, h)
        const O = { radius: 8, eps: 1e-4, scale: 4 }
        const fe = f0.slice(); base.refineField(fe, px, w, h, bbox, { ...O, scale: 1 })
        const mE = base.bandAlpha(fe, w, h, 0.9)
        const fb = f0.slice(); base.refineField(fb, px, w, h, bbox, O)
        let line = String(cover).padEnd(10)
            + maskDiff(base.bandAlpha(fb, w, h, 0.9), mE).iou.toFixed(4).padStart(8)
            + time(() => { const f = f0.slice(); base.refineField(f, px, w, h, bbox, O) }).med.toFixed(2).padStart(8)
        for (const B of BUDGETS) {
            opt.__setCellBudget(B)
            const f = f0.slice(); opt.refineField(f, px, w, h, bbox, O)
            const st = opt.lastRefineStats
            line += ' |' + maskDiff(opt.bandAlpha(f, w, h, 0.9), mE).iou.toFixed(4).padStart(11)
                + time(() => { const g = f0.slice(); opt.refineField(g, px, w, h, bbox, O) }).med.toFixed(2).padStart(6)
                + String(st.s).padStart(3)
        }
        console.log(line)
    }
    opt.__setCellBudget(SHIPPED)
}
console.log(`\n(shipped CELL_BUDGET = ${SHIPPED / 1000}k)`)
