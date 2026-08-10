/**
 * Is the base↔opt refineField gap a tiling bug, or the method's own sensitivity
 * to where the s=4 coefficient grid lands?
 *
 * Control: run the SHIPPED refineField twice, with the bbox grown by 1..3 px —
 * a benign change that shifts the rect origin and nothing else. Whatever IoU
 * spread that produces is the method's intrinsic phase noise. The optimized
 * version only has to land inside it.
 */
import * as base from '../ai/mask-refine.js'
import * as opt from './mask-refine-opt.js'
import { loadRgba, makeLogits, upsampleLogits, maskDiff, PROXIES } from './harness.mjs'

const BAND = 0.9
const p = PROXIES[0]
const px = loadRgba(p.name, p.w, p.h)

console.log('cover, IoU(base,base+1px), IoU(base,base+2px), IoU(base,base+3px), IoU(base,opt), IoU(base,raw)')
for (const cover of [0.001, 0.02, 0.1, 0.35, 0.7]) {
    const { field: f0, bbox } = upsampleLogits(makeLogits(cover), p.w, p.h)

    const run = (impl, bb) => {
        const f = f0.slice()
        impl.refineField(f, px, p.w, p.h, bb, { radius: 8, eps: 1e-4, scale: 4 })
        return impl.bandAlpha(f, p.w, p.h, BAND)
    }
    const grow = (d) => [Math.max(0, bbox[0] - d), Math.max(0, bbox[1] - d),
        Math.min(p.w - 1, bbox[2] + d), Math.min(p.h - 1, bbox[3] + d)]

    const m0 = run(base, bbox)
    const shifts = [1, 2, 3].map((d) => maskDiff(m0, run(base, grow(d))).iou)
    const mo = maskDiff(m0, run(opt, bbox)).iou
    const raw = maskDiff(m0, base.bandAlpha(f0, p.w, p.h, BAND)).iou

    console.log([String(cover).padEnd(5),
        ...shifts.map((v) => v.toFixed(5)), mo.toFixed(5), raw.toFixed(4)].join('  '))
}
