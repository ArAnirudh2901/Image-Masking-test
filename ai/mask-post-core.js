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
import { cleanRegions } from './mask-select.js'

/** Smallest rect covering both, either of which may be absent. Ends exclusive. */
const unionRect = (a, b) => {
    if (!a) return b || null
    if (!b) return a
    return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])]
}

// Catmull-Rom (a = -0.5) cubic kernel, evaluated directly. Four taps per axis,
// no matrices, no library — the whole upsample is ~16 multiply-adds per pixel.
// Weights land in a shared 4-slot buffer: the x-tap loop calls this once per
// output column, and a fresh array each time is pure collector pressure. Every
// caller consumes the taps before the next call.
const taps = new Float32Array(4)
const cubic = (t) => {
    const t2 = t * t
    const t3 = t2 * t
    // The four tap weights for offsets -1, 0, +1, +2.
    taps[0] = -0.5 * t3 + t2 - 0.5 * t
    taps[1] = 1.5 * t3 - 2.5 * t2 + 1
    taps[2] = -1.5 * t3 + 2 * t2 + 0.5 * t
    taps[3] = 0.5 * t3 - 0.5 * t2
    return taps
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
    let bx0 = w; let by0 = h; let bx1 = -1; let by1 = -1
    for (let y = 0; y < h; y += 1) {
        const fy = (y + 0.5) * sy - 0.5
        const y0 = Math.floor(fy)
        // Inline cubic weights — avoids allocating a 4-element array per row
        const t = fy - y0, t2 = t * t, t3 = t2 * t
        const k0 = -0.5 * t3 + t2 - 0.5 * t
        const k1 =  1.5 * t3 - 2.5 * t2 + 1
        const k2 = -1.5 * t3 + 2 * t2 + 0.5 * t
        const k3 =  0.5 * t3 - 0.5 * t2
        const r0 = clampIdx(y0 - 1) * w
        const r1 = clampIdx(y0) * w
        const r2 = clampIdx(y0 + 1) * w
        const r3 = clampIdx(y0 + 2) * w
        const row = y * w
        // Row-local extents: x only ever increases, so the running max is a
        // store rather than a compare, and the y bounds move twice per row
        // instead of once per pixel.
        let mLo = -1; let mHi = -1
        let bLo = -1; let bHi = -1
        for (let x = 0; x < w; x += 1) {
            const v = tmp[r0 + x] * k0 + tmp[r1 + x] * k1
                + tmp[r2 + x] * k2 + tmp[r3 + x] * k3
            out[row + x] = v
            // Two boxes out of one test. The band box (|v| < BAND) is what the
            // guided filter needs. The mask box (v > -BAND) is its superset and
            // is the window region hygiene scans: it has to contain the mask's
            // INTERIOR, because a subject running off-frame has foreground with
            // no zero crossing beside it, and a band-only window would read that
            // as a component of its own.
            if (v > -BAND) {
                if (mLo < 0) mLo = x
                mHi = x
                if (v < BAND) { if (bLo < 0) bLo = x; bHi = x }
            }
        }
        if (mHi >= 0) {
            if (mLo < bx0) bx0 = mLo
            if (mHi > bx1) bx1 = mHi
            if (by1 < 0) by0 = y
            by1 = y
        }
        if (bHi >= 0) {
            if (bLo < minX) minX = bLo
            if (bHi > maxX) maxX = bHi
            if (maxY < 0) minY = y
            maxY = y
        }
    }
    return {
        field: out,
        bbox: maxX < 0 ? null : [minX, minY, maxX, maxY],
        // Ends exclusive — a scan window, not an inclusive bbox.
        box: bx1 < 0 ? null : [bx0, by0, bx1 + 1, by1 + 1],
    }
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
export const postCompute = ({ logits, guide, w, h, maskSide, clicks = [], tight = false }) => {
    const tU = performance.now()
    const { field, bbox, box } = upsampleLogits(logits, w, h, maskSide)
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
    // Pixels the refinement actually walked — the quantity post cost scales
    // with, and hardware-fit's normaliser. The proxy's own area is not: one
    // 1.376 MP proxy cost 12.2 ms for a 9.7 kpx band and 52.6 ms for a
    // frame-spanning one.
    let bandPixels = 0
    // Refinement rewrites only its own rect and hygiene only its own runs, so
    // the mask is re-thresholded over the union of the two rather than paying a
    // second full-frame pass.
    let paint = null
    const tG = performance.now()
    let tF = tG
    try {
        const rect = guide && refineField(field, guide, w, h, bbox, { radius: 8, eps: 1e-4, scale: 4 })
        tF = performance.now()
        if (rect) {
            bandPixels = (rect[2] - rect[0]) * (rect[3] - rect[1])
            paint = rect
        }
    } catch { /* refinement failed — the raw mask still ships */ }

    // Region hygiene at PROXY resolution — after the upsample and the guided
    // filter, the two stages that manufacture speckle out of a field that
    // reached them clean (mask-select). Fill is 2·BAND, not the 256² default:
    // this field is matted in BAND units, and on a steep field a fixed 2.5 lands
    // inside the ramp and leaves a half-transparent patch behind.
    //
    // Scanned over the union of the mask box and the rect the filter rewrote.
    // The filter works on the BAND box plus its own pad (radius·2 + scale·2 =
    // 24 px), which reaches outside the mask box, and a pixel it lifts over zero
    // out there is a component no pass would otherwise see.
    const scan = unionRect(box, paint)
    const regions = scan
        ? cleanRegions(field, w, h, { clicks, rect: scan, fill: Math.max(2.5, 2 * BAND), tight })
        : { islands: 0, holes: 0, dirty: null }
    const tH = performance.now()
    if (regions.dirty) paint = unionRect(paint, regions.dirty)
    if (paint) rgba = bandAlphaRect(field, w, paint, BAND, new Uint8ClampedArray(rawRgba))

    const stages = {
        upsampleMs: +(tB - tU).toFixed(1),
        bandWidthMs: +(tA - tB).toFixed(1),
        bandAlphaMs: +(tR - tA).toFixed(1),
        guideMs: +(tG - tR).toFixed(1),
        refineMs: +(tF - tG).toFixed(1),
        hygieneMs: +(tH - tF).toFixed(1),
        rethresholdMs: +(performance.now() - tH).toFixed(1),
        // Refinement cost is linear in the cells the filter walked; without the
        // decomposition a slow refine is indistinguishable from a slow filter.
        refineShape: lastRefineStats,
    }
    // Every non-zero alpha is inside maskRect: `box` is where the field clears
    // -BAND (what bandAlpha paints), and `paint` is the only place anything was
    // rewritten after. Saves the caller a full-frame scan to summarise the mask.
    return { rgba, rawRgba, field, stages, bandPixels, regions, maskRect: unionRect(box, paint) }
}
