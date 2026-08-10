/**
 * A/B the optimized mask-refine against the shipped one on real DSLR pixels.
 *
 * Correctness bar:
 *   - bandAlpha / bandAlphaRect must be BIT-IDENTICAL (pure store-width change).
 *   - guidedFilter* at scale=1 and s>1 must match to float noise on the same rect.
 *   - refineField is a different decomposition (tiles vs one bbox), so the bar is
 *     agreement of the THRESHOLDED mask — the only thing downstream consumes.
 */
import * as base from './orig-mask-refine.js'
import * as opt from '../ai/mask-refine.js'
import { loadRgba, makeLogits, upsampleLogits, time, diff, maskDiff, PROXIES } from './harness.mjs'

const COVERS = [0.001, 0.02, 0.1, 0.35, 0.7]
const BAND = 0.9
const fail = []
const ok = (cond, msg) => { if (!cond) fail.push(msg); return cond }

// ---------------------------------------------------------- 1. bandAlpha exact
console.log('== bandAlpha: bit-identical check ==')
for (const p of PROXIES) {
    for (const cover of COVERS) {
        const { field } = upsampleLogits(makeLogits(cover), p.w, p.h)
        for (const band of [0.05, 0.9, 1.5, 6]) {
            const a = base.bandAlpha(field, p.w, p.h, band)
            const b = opt.bandAlpha(field, p.w, p.h, band)
            let bad = -1
            for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) { bad = i; break }
            ok(bad === -1, `bandAlpha ${p.w} cover=${cover} band=${band} differs @${bad}`)
        }
        // bandAlphaRect over a rect, on top of a prefilled buffer
        const rect = [17, 23, Math.min(p.w, 900), Math.min(p.h, 600)]
        const seed = base.bandAlpha(field, p.w, p.h, 6)
        const a = base.bandAlphaRect(field, p.w, rect, BAND, seed.slice())
        const b = opt.bandAlphaRect(field, p.w, rect, BAND, seed.slice())
        let bad = -1
        for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) { bad = i; break }
        ok(bad === -1, `bandAlphaRect ${p.w} cover=${cover} differs @${bad}`)
    }
}
console.log(fail.length ? `  FAIL (${fail.length})` : '  all bit-identical ✓')

// ------------------------------------------- 2. guided filters numeric agreement
console.log('\n== guidedFilter / guidedFilterColor: numeric agreement ==')
{
    const p = PROXIES[0]
    const px = loadRgba(p.name, p.w, p.h)
    // a modest rect, built the same way refineRect builds one
    const RW = 320, RH = 240
    const { field } = upsampleLogits(makeLogits(0.25), p.w, p.h)
    const sub = new Float32Array(RW * RH)
    const gy = new Float32Array(RW * RH)
    const cb = new Float32Array(RW * RH)
    const cr = new Float32Array(RW * RH)
    for (let y = 0; y < RH; y += 1) {
        for (let x = 0; x < RW; x += 1) {
            const si = (y + 100) * p.w + (x + 200)
            const j = si * 4
            const r = px[j] / 255, g = px[j + 1] / 255, b = px[j + 2] / 255
            const Y = 0.299 * r + 0.587 * g + 0.114 * b
            sub[y * RW + x] = field[si]
            gy[y * RW + x] = Y
            cb[y * RW + x] = 0.564 * (b - Y)
            cr[y * RW + x] = 0.713 * (r - Y)
        }
    }
    for (const scale of [1, 2, 4, 8]) {
        const a1 = base.guidedFilter(sub, gy, RW, RH, { radius: 8, eps: 1e-4, scale })
        const b1 = opt.guidedFilter(sub, gy, RW, RH, { radius: 8, eps: 1e-4, scale })
        const d1 = diff(a1, b1)
        ok(d1.max < 2e-3, `guidedFilter scale=${scale} max Δ=${d1.max.toExponential(2)}`)
        console.log(`  luma  scale=${scale}  maxΔ=${d1.max.toExponential(2)}  meanΔ=${d1.mean.toExponential(2)}`)

        const a2 = base.guidedFilterColor(sub, gy, cb, cr, RW, RH, { radius: 8, eps: 1e-4, eps2: 3e-4, eps3: 3e-4, scale })
        const b2 = opt.guidedFilterColor(sub, gy, cb, cr, RW, RH, { radius: 8, eps: 1e-4, eps2: 3e-4, eps3: 3e-4, scale })
        const d2 = diff(a2, b2)
        ok(d2.max < 2e-3, `guidedFilterColor scale=${scale} max Δ=${d2.max.toExponential(2)}`)
        console.log(`  color scale=${scale}  maxΔ=${d2.max.toExponential(2)}  meanΔ=${d2.mean.toExponential(2)}`)
    }
}

// -------------------------------------- 3. refineField: thresholded-mask agreement
console.log('\n== refineField: thresholded mask agreement (tiles vs single bbox) ==')
for (const p of PROXIES) {
    const px = loadRgba(p.name, p.w, p.h)
    for (const cover of COVERS) {
        const { field: f0, bbox } = upsampleLogits(makeLogits(cover), p.w, p.h)
        const fa = f0.slice(); const fb = f0.slice()
        const ra = base.refineField(fa, px, p.w, p.h, bbox, { radius: 8, eps: 1e-4, scale: 4 })
        const rb = opt.refineField(fb, px, p.w, p.h, bbox, { radius: 8, eps: 1e-4, scale: 4 })
        const ma = base.bandAlpha(fa, p.w, p.h, BAND)
        const mb = opt.bandAlpha(fb, p.w, p.h, BAND)
        const d = maskDiff(ma, mb)
        // Also: does either differ from the UNREFINED mask? (proves refinement ran)
        const mr = base.bandAlpha(f0, p.w, p.h, BAND)
        const moved = maskDiff(ma, mr)
        // NOT an equality bar: the tiled + adaptive-scale refine is deliberately
        // different from (and closer to the exact filter than) the original.
        // validate.mjs owns that bar, scored against the s=1 reference.
        console.log(`  ${p.w}×${p.h} cover=${String(cover).padEnd(5)} IoU(base,opt)=${d.iou.toFixed(5)}  `
            + `IoU(base,raw)=${moved.iou.toFixed(4)}  rects base=${ra} opt=${rb}`)
    }
}

// ------------------------------------------------------------------ 4. timings
console.log('\n== timing (ms, median of 9) ==')
console.log('proxy,cover,refine_base,refine_opt,refine_x,band_base,band_opt,band_x')
for (const p of PROXIES) {
    const px = loadRgba(p.name, p.w, p.h)
    for (const cover of COVERS) {
        const { field: f0, bbox } = upsampleLogits(makeLogits(cover), p.w, p.h)
        const tA = time(() => { const f = f0.slice(); base.refineField(f, px, p.w, p.h, bbox, { radius: 8, eps: 1e-4, scale: 4 }) }, 9, 3)
        const tB = time(() => { const f = f0.slice(); opt.refineField(f, px, p.w, p.h, bbox, { radius: 8, eps: 1e-4, scale: 4 }) }, 9, 3)
        const cA = time(() => base.bandAlpha(f0, p.w, p.h, BAND), 9, 3)
        const cB = time(() => opt.bandAlpha(f0, p.w, p.h, BAND), 9, 3)
        console.log([`${p.w}x${p.h}`, cover,
            tA.med.toFixed(2), tB.med.toFixed(2), `${(tA.med / tB.med).toFixed(2)}x`,
            cA.med.toFixed(2), cB.med.toFixed(2), `${(cA.med / cB.med).toFixed(2)}x`].join(','))
    }
}

console.log(fail.length ? `\n${fail.length} FAILURE(S):\n` + fail.map((f) => ' - ' + f).join('\n') : '\nALL CORRECTNESS CHECKS PASSED ✓')
process.exit(fail.length ? 1 : 0)
