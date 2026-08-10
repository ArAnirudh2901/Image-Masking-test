/** Baseline: time the real refine path on real DSLR pixels, across coverages. */
import { bandAlpha, bandAlphaRect, refineField } from '../ai/mask-refine.js'
import { loadRgba, makeLogits, upsampleLogits, time, PROXIES } from './harness.mjs'

const COVERS = [0.001, 0.02, 0.1, 0.35, 0.7]
const BAND = 0.9

console.log('impl,proxy,cover,bandpx_frac,upsample_ms,refine_ms,bandalpha_ms,total_ms')

for (const p of PROXIES) {
    const px = loadRgba(p.name, p.w, p.h)
    for (const cover of COVERS) {
        const logits = makeLogits(cover)
        const { field: field0, bbox } = upsampleLogits(logits, p.w, p.h)

        const tUp = time(() => upsampleLogits(logits, p.w, p.h), 9, 3)
        // refineField mutates `field`, so hand it a fresh copy each run
        const tRef = time(() => {
            const f = field0.slice()
            refineField(f, px, p.w, p.h, bbox, { radius: 8, eps: 1e-4, scale: 4 })
        }, 9, 3)
        const tBand = time(() => bandAlpha(field0, p.w, p.h, BAND), 9, 3)

        // how much of the frame the refine band actually covers
        const pad = 8 * 2 + 4 * 2
        const x0 = Math.max(0, bbox[0] - pad), y0 = Math.max(0, bbox[1] - pad)
        const x1 = Math.min(p.w, bbox[2] + pad + 1), y1 = Math.min(p.h, bbox[3] + pad + 1)
        const frac = ((x1 - x0) * (y1 - y0)) / (p.w * p.h)

        console.log([
            'baseline', `${p.w}x${p.h}`, cover, frac.toFixed(3),
            tUp.med.toFixed(2), tRef.med.toFixed(2), tBand.med.toFixed(2),
            (tUp.med + tRef.med + tBand.med).toFixed(2),
        ].join(','))
    }
}
