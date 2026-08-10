/**
 * Is refinement cost sensitive to the row STRIDE rather than the pixel count?
 *
 * The workflow bench kept reporting a 1536×1024 proxy (1.57 M px) refining SLOWER
 * than a 1756×1024 one (1.80 M px). Fewer pixels, more time, reproducibly — which
 * is the signature of cache-set aliasing on a power-of-two row stride, not of any
 * work the code is doing. Every plane here is w floats per row, and the filter
 * walks 4–8 of them in lockstep, so a bad stride collides them all in the same sets.
 *
 * Sweeps widths around the powers of two, holding the pixel COUNT roughly fixed by
 * cropping the same photo, and prints ms per megapixel — flat means stride is
 * irrelevant, spikes at 1024/1536/2048 mean it is not.
 *
 * ANSWER: stride is irrelevant. 4096 B, 6144 B and 8192 B strides all cost the
 * same as their ±2 px neighbours. The variance was CELL_BUDGET: widths up to ~1542
 * land at s=3 and widths from ~1756 at s=4, and 1/s² is a 1.78x difference in
 * filter work. A 1.57 Mpx proxy refining slower than a 1.80 Mpx one is that
 * boundary, not a cache effect. Kept so the question is not re-opened.
 *
 *   bun run bench/stride.mjs
 */
import { loadRgba, makeLogits, upsampleLogits, time } from './harness.mjs'
import * as opt from '../ai/mask-refine.js'

const SRC_W = 2048, SRC_H = 1365
const src = loadRgba('nef_proxy_2048.rgba', SRC_W, SRC_H)

/** Crop the RGBA proxy to w×h, producing a buffer whose row stride IS w. */
const crop = (w, h) => {
    const out = new Uint8ClampedArray(w * h * 4)
    for (let y = 0; y < h; y += 1) {
        const s = (y * SRC_W) * 4
        out.set(src.subarray(s, s + w * 4), y * w * 4)
    }
    return out
}

const H = 1024
const WIDTHS = [1020, 1022, 1024, 1026, 1030, 1276, 1278, 1280, 1282,
    1532, 1534, 1536, 1538, 1542, 1756, 2044, 2046, 2048]

console.log(`height ${H}, cover 0.35, colour guided filter\n`)
console.log('width  stride(B)  pow2?   px(M)     ms    ms/Mpx   cells(M)  s')
for (const w of WIDTHS) {
    if (w > SRC_W) continue
    const px = crop(w, H)
    const { field: f0, bbox } = upsampleLogits(makeLogits(0.35), w, H)
    const O = { radius: 8, eps: 1e-4, scale: 4 }
    const f = f0.slice(); opt.refineField(f, px, w, H, bbox, O)
    const st = opt.lastRefineStats
    const t = time(() => { const g = f0.slice(); opt.refineField(g, px, w, H, bbox, O) }, 11, 4)
    const mpx = (w * H) / 1e6
    // A stride is "bad" when it is a large power of two times a small factor —
    // that is what puts consecutive rows in the same cache sets.
    const bytes = w * 4
    const pow2 = (bytes & (bytes - 1)) === 0 ? 'yes'
        : (bytes % 4096 === 0 ? '4k|' : (bytes % 2048 === 0 ? '2k|' : ''))
    console.log([String(w).padEnd(7), String(bytes).padStart(9), pow2.padStart(7),
        mpx.toFixed(2).padStart(8), t.med.toFixed(2).padStart(7),
        (t.med / mpx).toFixed(2).padStart(9),
        (st.cells / 1e6).toFixed(2).padStart(10), String(st.s).padStart(3)].join(''))
}
