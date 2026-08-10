/**
 * mask-post-core — the post pipeline (upsample → band → guided filter →
 * threshold) with no DOM in it, so the worker and the main thread run the SAME
 * code. Split out of sam21-adapter when post moved off-thread; keeping one
 * implementation is what stops the two paths drifting apart numerically.
 *
 * `maskSide` is a parameter rather than an import: pulling it from sam21-lane
 * would drag the whole lane (and ORT) into the worker bundle for one constant.
 */

import { bandAlpha, bandAlphaRect, refineField, lastRefineStats } from './mask-refine.js'

// Catmull-Rom (a = -0.5) cubic kernel, evaluated directly. Four taps per axis,
// no matrices, no library — the whole upsample is ~16 multiply-adds per pixel.
const cubic = (t) => {
    const t2 = t * t
    const t3 = t2 * t
    // Returns the four tap weights for offsets -1, 0, +1, +2.
    return [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1,
        -1.5 * t3 + 2 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/**
 * Upsample the continuous logit field, THEN threshold (DESIGN-MASK-LANE §10).
 * Thresholding at 256² and scaling the binary mask is what produces blocky,
 * stair-stepped edges, and no encoder upgrade fixes that. Bicubic on the score
 * field puts the zero crossing at sub-pixel position for free.
 *
 * Separable: rows are resampled once into a scratch buffer, then columns — so
 * the cost is 4 taps per axis rather than 16 per pixel. Row weights repeat for
 * every output row, so they are computed once up front.
 */
export const upsampleLogits = (logits, w, h, maskSide) => {
    const clampIdx = (i) => (i < 0 ? 0 : (i > maskSide - 1 ? maskSide - 1 : i))
    const out = new Float32Array(w * h)
    const sx = maskSide / w
    const sy = maskSide / h

    // Precompute the x taps: identical for every row.
    const xi = new Int32Array(w * 4)
    const xw = new Float32Array(w * 4)
    for (let x = 0; x < w; x += 1) {
        const fx = (x + 0.5) * sx - 0.5
        const x0 = Math.floor(fx)
        const k = cubic(fx - x0)
        for (let t = 0; t < 4; t += 1) {
            xi[x * 4 + t] = clampIdx(x0 - 1 + t)
            xw[x * 4 + t] = k[t]
        }
    }

    // Pass 1: resample rows → [maskSide][w]
    const tmp = new Float32Array(maskSide * w)
    for (let r = 0; r < maskSide; r += 1) {
        const src = r * maskSide
        const dst = r * w
        for (let x = 0; x < w; x += 1) {
            const b = x * 4
            tmp[dst + x] = logits[src + xi[b]] * xw[b]
                + logits[src + xi[b + 1]] * xw[b + 1]
                + logits[src + xi[b + 2]] * xw[b + 2]
                + logits[src + xi[b + 3]] * xw[b + 3]
        }
    }

    // Pass 2: resample columns into the output field (still continuous) and
    // track the transition band's bbox while the values are already in hand —
    // refineField would otherwise need its own full-frame scan to find it.
    const BAND = 6
    let minX = w; let minY = h; let maxX = -1; let maxY = -1
    for (let y = 0; y < h; y += 1) {
        const fy = (y + 0.5) * sy - 0.5
        const y0 = Math.floor(fy)
        const k = cubic(fy - y0)
        const r0 = clampIdx(y0 - 1) * w
        const r1 = clampIdx(y0) * w
        const r2 = clampIdx(y0 + 1) * w
        const r3 = clampIdx(y0 + 2) * w
        const row = y * w
        for (let x = 0; x < w; x += 1) {
            const v = tmp[r0 + x] * k[0] + tmp[r1 + x] * k[1]
                + tmp[r2 + x] * k[2] + tmp[r3 + x] * k[3]
            out[row + x] = v
            if (v > -BAND && v < BAND) {
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            }
        }
    }
    return { field: out, bbox: maxX < 0 ? null : [minX, minY, maxX, maxY] }
}

// The band is specified in PIXELS and converted to logit units per image. A
// fixed logit width is wrong: the field's slope at the boundary varies with the
// object and with the upsample factor, so the same constant produced a 1 px
// ramp on one frame and a 10 px smear on another. |∇field| at the crossing is
// exactly the conversion factor.
const BAND_PX = 1.5

/** Mean |∇field| across the transition band → logit units per pixel. */
export const bandWidth = (field, w, bbox) => {
    if (!bbox) return 1
    const [x0, y0, x1, y1] = bbox
    let acc = 0
    let n = 0
    for (let y = Math.max(1, y0); y < Math.min(y1 + 1, (field.length / w) - 1); y += 1) {
        const row = y * w
        for (let x = Math.max(1, x0); x < Math.min(x1 + 1, w - 1); x += 1) {
            const v = field[row + x]
            if (v < -1 || v > 1) continue          // only right at the crossing
            const gx = field[row + x + 1] - field[row + x - 1]
            const gy = field[row + w + x] - field[row - w + x]
            acc += Math.hypot(gx, gy) * 0.5
            n += 1
        }
    }
    const grad = n ? acc / n : 1
    // bandAlpha ramps across 2*band logit units, so the pixel width is
    // 2*band/|grad| — solve for band. NO absolute floor: at export scale the
    // field has been upsampled ~10x, so |grad| is legitimately ~10x smaller, and
    // a fixed floor (0.25 was tried) forces a ~35 px smear instead of 1.5 px.
    // The epsilon exists only to keep a degenerate flat field finite.
    return Math.max(1e-3, (grad * BAND_PX) / 2)
}

/**
 * 256² logits + the guide image → the two RGBA masks the app wants, plus the
 * continuous field the export path reads. `guide` may be null (tainted canvas):
 * the raw mask still ships, unrefined.
 */
export const postCompute = ({ logits, guide, w, h, maskSide }) => {
    const tU = performance.now()
    const { field, bbox } = upsampleLogits(logits, w, h, maskSide)
    // Raw first: the refinement rewrites `field` in place, and the app's
    // raw/refined toggle needs both.
    const tB = performance.now()
    const BAND = bandWidth(field, w, bbox)
    const tA = performance.now()
    const rawRgba = bandAlpha(field, w, h, BAND)
    const tR = performance.now()

    // Guided filter against the photo's own luma: pulls the boundary onto the
    // real object edge instead of wherever the 256² grid put it.
    let rgba = rawRgba
    const tG = performance.now()
    let tF = tG
    try {
        const rect = guide && refineField(field, guide, w, h, bbox, { radius: 8, eps: 1e-4, scale: 4 })
        tF = performance.now()
        if (rect) {
            // Refinement only rewrote `rect`; the rest of the field is
            // untouched, so copy the raw mask and re-threshold just that band
            // instead of paying a second full-frame pass.
            rgba = bandAlphaRect(field, w, rect, BAND, new Uint8ClampedArray(rawRgba))
        }
    } catch { /* refinement failed — the raw mask still ships */ }

    const stages = {
        upsampleMs: +(tB - tU).toFixed(1),
        bandWidthMs: +(tA - tB).toFixed(1),
        bandAlphaMs: +(tR - tA).toFixed(1),
        guideMs: +(tG - tR).toFixed(1),
        refineMs: +(tF - tG).toFixed(1),
        rethresholdMs: +(performance.now() - tF).toFixed(1),
        // Refinement cost is linear in the cells the filter walked; without the
        // decomposition a slow refine is indistinguishable from a slow filter.
        refineShape: lastRefineStats,
    }
    return { rgba, rawRgba, field, stages }
}
