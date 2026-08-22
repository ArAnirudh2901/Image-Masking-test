/**
 * mask-refine — closed-form mask edge refinement (DESIGN-MASK-LANE §10).
 *
 * SAM emits 256². A Z 8 export is 8256×5504 — a ~32× enlargement. How that gap
 * is bridged decides perceived quality far more than tiny-vs-small does, so all
 * of it is done on the CONTINUOUS score field and thresholded only at the end:
 *
 *   1. bicubic upsample of the logits            (sub-pixel zero crossing)
 *   2. guided filter against the photo's COLOUR  (snaps the edge to the object)
 *   3. threshold at output resolution
 *
 * Step 2 guides on Y/Cb/Cr, not luma. Luma alone cannot see an isoluminant
 * boundary — pink petals on green foliage, orange paint on grey road — and
 * those are common, not exotic: measured over seven real crops, moving to a
 * colour guide lifts mean boundary IoU 0.735 → 0.864.
 *
 * No library, no convolution loops: box filters run off a summed-area table, so
 * every stage is O(n) in pixels and independent of radius. The guided filter
 * uses He et al.'s FAST variant — coefficients solved on a subsampled grid and
 * upsampled — which is ~s² cheaper and visually indistinguishable, because a
 * and b are far smoother than the image they are applied to.
 *
 * The refinement is TILED and its subsample factor is ADAPTIVE — see refineField
 * and CELL_BUDGET. Both were driven by measurement on a 45 MP NEF proxy, not by
 * guessing; `bench/` holds the harness.
 */

/**
 * Chroma ε as a multiple of the luma ε — how much better than luma a chroma
 * edge must be before the filter follows it. Swept over seven real crops
 * (`bench/`), not guessed: 3 keeps ~95 % of the colour gain on
 * chromatic subjects while removing the achromatic regression that raw RGB
 * causes on thin wires.
 *
 * Two adaptive rules were built and measured, and BOTH lose to this constant:
 *   - ε from the median local chroma variance (a noise floor): no effect on the
 *     wires at all, because their damaging chroma is a strong CA fringe, not
 *     noise, and it costs ~0.5 pt on every chromatic crop;
 *   - ε scaled by local luma variance ("use chroma only where luma failed"):
 *     the isoluminant boundary still carries some luma signal, so it suppresses
 *     chroma exactly where it is needed — rose boundary IoU +4.8 pt vs +9.1.
 * Do not replace this constant with a clever rule without re-running the bench.
 */
const CHROMA_EPS = 3

// BT.601 luma + chroma. ONE definition: refineRect's fused loop and the plane
// form below must agree, or the guided filter and the subject prior would be
// reading different colours off the same pixels.
const KR = 0.299
const KG = 0.587
const KB = 0.114
const KCB = 0.564
const KCR = 0.713

// ---------------------------------------------------------------- scratch pool
// A colour guided filter borrows ~17 typed arrays per call, all at a handful of
// repeated shapes, and the refine path runs it once per band tile. Pooling them
// removes essentially all of its allocation; `release()` marks a generation done,
// so the next borrower reuses the same memory.
const pool = new Map()
let poolGen = 0
const take = (Ctor, n) => {
    const key = `${Ctor.name}:${n}`
    let slot = pool.get(key)
    if (!slot) { slot = { arrs: [], used: 0, gen: poolGen }; pool.set(key, slot) }
    if (slot.gen !== poolGen) { slot.gen = poolGen; slot.used = 0 }
    if (slot.used < slot.arrs.length) return slot.arrs[slot.used++]
    const a = new Ctor(n)
    slot.arrs.push(a)
    slot.used += 1
    return a
}
const release = () => { poolGen += 1 }

/** Summed-area table of `src` (w×h) into a pooled (w+1)×(h+1) Float64Array.
 *  f64 because a 1024² field of ~10-magnitude logits overflows f32 precision
 *  well before the corner. */
const integral = (src, w, h) => {
    const W = w + 1
    const S = take(Float64Array, W * (h + 1))
    S.fill(0, 0, W) // row 0 is the only run that isn't fully overwritten below
    for (let y = 0; y < h; y += 1) {
        let run = 0
        const row = y * w
        const out = (y + 1) * W
        const prev = y * W
        S[out] = 0
        for (let x = 0; x < w; x += 1) {
            run += src[row + x]
            S[out + x + 1] = S[prev + x + 1] + run
        }
    }
    return S
}

/** Box mean of radius r, O(1) per pixel from the SAT. Edges use the clipped
 *  window's own area, so no darkening at the border.
 *
 *  Column windows are hoisted out of the row loop and the interior run drops the
 *  clamps and the per-pixel divide entirely — this is the hottest loop in the
 *  file, called 17x per colour filter. */
const boxMean = (src, w, h, r, dst = null) => {
    const S = integral(src, w, h)
    const out = dst || take(Float32Array, w * h)
    const W = w + 1

    const cx0 = take(Int32Array, w)
    const cx1 = take(Int32Array, w)
    for (let x = 0; x < w; x += 1) {
        cx0[x] = x - r < 0 ? 0 : x - r
        cx1[x] = x + r + 1 > w ? w : x + r + 1
    }
    const xi0 = Math.min(w, r)              // [xi0, xi1) needs no clamping and
    const xi1 = Math.max(xi0, w - r - 1)    // shares one constant window width
    const fullW = 2 * r + 1

    for (let y = 0; y < h; y += 1) {
        const y0 = y - r < 0 ? 0 : y - r
        const y1 = y + r + 1 > h ? h : y + r + 1
        const rowT = y0 * W
        const rowB = y1 * W
        const dstRow = y * w
        const rows = y1 - y0
        const invFull = 1 / (rows * fullW)

        for (let x = 0; x < xi0; x += 1) {
            const a = cx0[x], b = cx1[x]
            out[dstRow + x] = (S[rowB + b] - S[rowB + a] - S[rowT + b] + S[rowT + a]) / (rows * (b - a))
        }
        for (let x = xi0; x < xi1; x += 1) {
            const a = x - r, b = x + r + 1
            out[dstRow + x] = (S[rowB + b] - S[rowB + a] - S[rowT + b] + S[rowT + a]) * invFull
        }
        for (let x = xi1; x < w; x += 1) {
            const a = cx0[x], b = cx1[x]
            out[dstRow + x] = (S[rowB + b] - S[rowB + a] - S[rowT + b] + S[rowT + a]) / (rows * (b - a))
        }
    }
    return out
}

/** Nearest-box downsample by integer factor s (means, not point samples, so
 *  aliasing does not leak into the coefficients). */
const downsample = (src, w, h, s, dw, dh) => {
    const out = take(Float32Array, dw * dh)
    // Interior cells are full s×s and need no clipping; only the last column and
    // last row can be partial. Hoisting that out of the per-cell loop matters:
    // this runs 4x per colour filter over the whole rect, and the two Math.min
    // calls per cell cost more than the s² adds they guard.
    const fullX = (w / s) | 0            // cells [0, fullX) span s columns exactly
    const fullY = (h / s) | 0
    const invFull = 1 / (s * s)
    for (let y = 0; y < dh; y += 1) {
        const y0 = y * s
        const y1 = y < fullY ? y0 + s : h
        const dstRow = y * dw
        const rowsN = y1 - y0
        if (y < fullY) {
            for (let x = 0; x < fullX; x += 1) {
                const x0 = x * s
                let acc = 0
                for (let yy = y0; yy < y1; yy += 1) {
                    const row = yy * w + x0
                    for (let k = 0; k < s; k += 1) acc += src[row + k]
                }
                out[dstRow + x] = acc * invFull
            }
        } else {
            for (let x = 0; x < fullX; x += 1) {
                const x0 = x * s
                let acc = 0
                for (let yy = y0; yy < y1; yy += 1) {
                    const row = yy * w + x0
                    for (let k = 0; k < s; k += 1) acc += src[row + k]
                }
                out[dstRow + x] = acc / (rowsN * s)
            }
        }
        for (let x = fullX; x < dw; x += 1) {   // ragged last column, if any
            const x0 = x * s
            const x1 = Math.min(w, x0 + s)
            let acc = 0
            for (let yy = y0; yy < y1; yy += 1) {
                const row = yy * w
                for (let xx = x0; xx < x1; xx += 1) acc += src[row + xx]
            }
            out[dstRow + x] = acc / (rowsN * (x1 - x0))
        }
    }
    return out
}

/** Per-column bilinear taps for a dw→w upsample; shared by every fused apply. */
const xTaps = (dw, w) => {
    const i0 = take(Int32Array, w)
    const i1 = take(Int32Array, w)
    const wt = take(Float32Array, w)
    const sx = dw / w
    for (let x = 0; x < w; x += 1) {
        const fx = Math.min(dw - 1, Math.max(0, (x + 0.5) * sx - 0.5))
        const a = Math.floor(fx)
        i0[x] = a
        i1[x] = Math.min(dw - 1, a + 1)
        wt[x] = fx - a
    }
    return { i0, i1, wt }
}

/**
 * Guided filter: refine score field `p` using photo luma `guide` (both w×h,
 * guide in [0,1]). Writes into `out` when given; `p` is untouched.
 *
 * `eps` sets what counts as an edge worth following: variance below it is
 * treated as flat and smoothed across. 1e-4 ≈ luma steps under ~1 %.
 */
const gfLuma = (p, guide, w, h, {
    radius = 8, eps = 1e-4, scale = 4, out = null,
} = {}) => {
    const s = Math.max(1, Math.round(scale))
    const dw = Math.max(1, Math.ceil(w / s))
    const dh = Math.max(1, Math.ceil(h / s))
    const r = Math.max(1, Math.round(radius / s))

    const I = s > 1 ? downsample(guide, w, h, s, dw, dh) : guide
    const P = s > 1 ? downsample(p, w, h, s, dw, dh) : p
    const n = dw * dh

    const Ip = take(Float32Array, n)
    const II = take(Float32Array, n)
    for (let i = 0; i < n; i += 1) { Ip[i] = I[i] * P[i]; II[i] = I[i] * I[i] }

    const meanI = boxMean(I, dw, dh, r)
    const meanP = boxMean(P, dw, dh, r)
    const meanIp = boxMean(Ip, dw, dh, r)
    const meanII = boxMean(II, dw, dh, r)

    // a = cov(I,p) / (var(I) + eps);  b = mean(p) - a·mean(I)
    const a = take(Float32Array, n)
    const b = take(Float32Array, n)
    for (let i = 0; i < n; i += 1) {
        const varI = meanII[i] - meanI[i] * meanI[i]
        const covIp = meanIp[i] - meanI[i] * meanP[i]
        const ai = covIp / (varI + eps)
        a[i] = ai
        b[i] = meanP[i] - ai * meanI[i]
    }

    const ma = boxMean(a, dw, dh, r)
    const mb = boxMean(b, dw, dh, r)
    const dst = out || take(Float32Array, w * h)

    if (s === 1) {
        for (let i = 0; i < w * h; i += 1) dst[i] = ma[i] * guide[i] + mb[i]
        return dst
    }
    // Fused: a and b are interpolated and applied in ONE full-res pass, rather
    // than materialising two full-res planes and reading them back.
    const { i0, i1, wt } = xTaps(dw, w)
    const sy = dh / h
    for (let y = 0; y < h; y += 1) {
        const fy = Math.min(dh - 1, Math.max(0, (y + 0.5) * sy - 0.5))
        const y0 = Math.floor(fy)
        const y1 = Math.min(dh - 1, y0 + 1)
        const wy = fy - y0
        const iwy = 1 - wy
        const r0 = y0 * dw
        const r1 = y1 * dw
        const row = y * w
        for (let x = 0; x < w; x += 1) {
            const p0 = i0[x], p1 = i1[x], wx = wt[x], iwx = 1 - wx
            const A = (ma[r0 + p0] * iwx + ma[r0 + p1] * wx) * iwy
                + (ma[r1 + p0] * iwx + ma[r1 + p1] * wx) * wy
            const B = (mb[r0 + p0] * iwx + mb[r0 + p1] * wx) * iwy
                + (mb[r1 + p0] * iwx + mb[r1 + p1] * wx) * wy
            dst[row + x] = A * guide[row + x] + B
        }
    }
    return dst
}

/**
 * Colour guided filter — He et al. §4, the 3-channel form.
 *
 * The luma form above is blind to isoluminant boundaries, and those are not a
 * corner case: measured on a rose-against-foliage crop, 33 % of the true
 * boundary carries under 2 % luma contrast, and the mean boundary step (0.066)
 * is only 2.9x the mean step of the petal texture INSIDE the object. A guide
 * that cannot separate those two pulls the edge onto shading.
 *
 * Per window: a = (Σ + E)⁻¹ cov(I,p), b = mean(p) − aᵀ mean(I), with Σ the 3×3
 * guide covariance. Costs 13 box means and one symmetric 3×3 solve per
 * subsampled pixel instead of 4 means — and the solve runs on w·h/s² cells.
 *
 * E is DIAGONAL, not He's scalar εU, and that is what makes this safe on grey
 * subjects. Fed Y/Cb/Cr with a larger ε on the two chroma axes, an achromatic
 * window (where chroma is only demosaic noise) drives a_Cb, a_Cr → 0 and the
 * filter degenerates exactly to the luma form; a chromatic window still gets
 * the full solve. With a scalar ε and raw RGB the channels are collinear on
 * grey, Σ is near-singular, and the fit chases that noise — measured as a
 * −3.0 pt boundary-IoU regression on 1–2 px wires against sky.
 */
const gfColor = (p, R, G, B, w, h, {
    radius = 8, eps = 1e-4, eps2 = null, eps3 = null, scale = 4, out = null,
} = {}) => {
    const e1 = eps
    const e2 = eps2 ?? eps
    const e3 = eps3 ?? eps
    const s = Math.max(1, Math.round(scale))
    const dw = Math.max(1, Math.ceil(w / s))
    const dh = Math.max(1, Math.ceil(h / s))
    const r = Math.max(1, Math.round(radius / s))
    const n = dw * dh

    const dr = s > 1 ? downsample(R, w, h, s, dw, dh) : R
    const dg = s > 1 ? downsample(G, w, h, s, dw, dh) : G
    const db = s > 1 ? downsample(B, w, h, s, dw, dh) : B
    const P = s > 1 ? downsample(p, w, h, s, dw, dh) : p

    const t = take(Float32Array, n)
    const prod = (u, v) => {
        for (let i = 0; i < n; i += 1) t[i] = u[i] * v[i]
        return boxMean(t, dw, dh, r)
    }

    const mr = boxMean(dr, dw, dh, r)
    const mg = boxMean(dg, dw, dh, r)
    const mb = boxMean(db, dw, dh, r)
    const mp = boxMean(P, dw, dh, r)
    const mrr = prod(dr, dr), mrg = prod(dr, dg), mrb = prod(dr, db)
    const mgg = prod(dg, dg), mgb = prod(dg, db), mbb = prod(db, db)
    const mrp = prod(dr, P), mgp = prod(dg, P), mbp = prod(db, P)

    const ar = take(Float32Array, n)
    const ag = take(Float32Array, n)
    const ab = take(Float32Array, n)
    const bo = take(Float32Array, n)
    for (let i = 0; i < n; i += 1) {
        const a11 = mrr[i] - mr[i] * mr[i] + e1
        const a22 = mgg[i] - mg[i] * mg[i] + e2
        const a33 = mbb[i] - mb[i] * mb[i] + e3
        const a12 = mrg[i] - mr[i] * mg[i]
        const a13 = mrb[i] - mr[i] * mb[i]
        const a23 = mgb[i] - mg[i] * mb[i]

        // Symmetric 3×3 inverse by cofactors.
        const c11 = a22 * a33 - a23 * a23
        const c12 = a13 * a23 - a12 * a33
        const c13 = a12 * a23 - a13 * a22
        let det = a11 * c11 + a12 * c12 + a13 * c13
        if (det > -1e-12 && det < 1e-12) det = det < 0 ? -1e-12 : 1e-12
        const c22 = a11 * a33 - a13 * a13
        const c23 = a13 * a12 - a11 * a23
        const c33 = a11 * a22 - a12 * a12

        const cr = mrp[i] - mr[i] * mp[i]
        const cg = mgp[i] - mg[i] * mp[i]
        const cb = mbp[i] - mb[i] * mp[i]

        // 1 division + 3 multiplications instead of 3 divisions
        const invDet = 1 / det
        const xr = (c11 * cr + c12 * cg + c13 * cb) * invDet
        const xg = (c12 * cr + c22 * cg + c23 * cb) * invDet
        const xb = (c13 * cr + c23 * cg + c33 * cb) * invDet
        ar[i] = xr; ag[i] = xg; ab[i] = xb
        bo[i] = mp[i] - xr * mr[i] - xg * mg[i] - xb * mb[i]
    }

    const Mr = boxMean(ar, dw, dh, r)
    const Mg = boxMean(ag, dw, dh, r)
    const Mb = boxMean(ab, dw, dh, r)
    const Mo = boxMean(bo, dw, dh, r)
    const dst = out || take(Float32Array, w * h)

    if (s === 1) {
        for (let i = 0; i < w * h; i += 1) dst[i] = Mr[i] * R[i] + Mg[i] * G[i] + Mb[i] * B[i] + Mo[i]
        return dst
    }
    // Fused: all four coefficient planes interpolated and combined in a single
    // full-res pass. Materialising them first cost four full-res planes — ~44 MB
    // of garbage per call at export scale — and then re-read every one.
    const { i0, i1, wt } = xTaps(dw, w)
    const sy = dh / h
    for (let y = 0; y < h; y += 1) {
        const fy = Math.min(dh - 1, Math.max(0, (y + 0.5) * sy - 0.5))
        const y0 = Math.floor(fy)
        const y1 = Math.min(dh - 1, y0 + 1)
        const wy = fy - y0
        const iwy = 1 - wy
        const q0 = y0 * dw
        const q1 = y1 * dw
        const row = y * w
        for (let x = 0; x < w; x += 1) {
            const p0 = i0[x], p1 = i1[x], wx = wt[x], iwx = 1 - wx
            const w00 = iwx * iwy, w10 = wx * iwy, w01 = iwx * wy, w11 = wx * wy
            const k00 = q0 + p0, k10 = q0 + p1, k01 = q1 + p0, k11 = q1 + p1
            const A = Mr[k00] * w00 + Mr[k10] * w10 + Mr[k01] * w01 + Mr[k11] * w11
            const Cg = Mg[k00] * w00 + Mg[k10] * w10 + Mg[k01] * w01 + Mg[k11] * w11
            const Cb = Mb[k00] * w00 + Mb[k10] * w10 + Mb[k01] * w01 + Mb[k11] * w11
            const O = Mo[k00] * w00 + Mo[k10] * w10 + Mo[k01] * w01 + Mo[k11] * w11
            const j = row + x
            dst[j] = A * R[j] + Cg * G[j] + Cb * B[j] + O
        }
    }
    return dst
}

/** Public forms. `out` is caller-owned so no pooled buffer escapes — a returned
 *  array has to outlive the next call. */
export const guidedFilter = (p, guide, w, h, o = {}) => {
    const r = gfLuma(p, guide, w, h, { ...o, out: new Float32Array(w * h) })
    release()
    return r
}
export const guidedFilterColor = (p, R, G, B, w, h, o = {}) => {
    const r = gfColor(p, R, G, B, w, h, { ...o, out: new Float32Array(w * h) })
    release()
    return r
}

/**
 * Box mean into a CALLER-OWNED array — the SAT and every intermediate go back
 * to the pool. Exported for passes outside this file (the subject prior's
 * focus and colour maps) so they do not carry a second box-filter.
 */
export const boxMeanInto = (src, w, h, r, out) => {
    const v = boxMean(src, w, h, r, out)
    release()
    return v
}

/** RGBA → Y/Cb/Cr planes, the transform refineRect feeds gfColor. Caller owns
 *  the three destination arrays. */
export const ycbcrPlanes = (rgba, w, h, Y, Cb, Cr) => {
    const INV255 = 1 / 255
    for (let i = 0, j = 0; i < w * h; i += 1, j += 4) {
        const r = rgba[j] * INV255
        const g = rgba[j + 1] * INV255
        const b = rgba[j + 2] * INV255
        const y = KR * r + KG * g + KB * b
        Y[i] = y
        Cb[i] = KCB * (b - y)
        Cr[i] = KCR * (r - y)
    }
}

// ------------------------------------------------------------ tiled refinement

/** Logit distance from the crossing beyond which the filter cannot move the
 *  threshold. Matches the band bbox sam21-adapter already tracks. */
const BAND_LOGIT = 6
/** Tile side in px. 96 keeps the pad:core area ratio sane at radius 8 while
 *  still tracking a thin ring closely. */
const TILE = 96
/**
 * Subsampled-cell budget for one refine call — what picks `scale`.
 *
 * s > 1 is a COST control, and it is the only thing that makes the refined edge
 * non-deterministic. Measured on the 45 MP NEF proxy: s=1 is bit-stable under a
 * benign 1 px bbox nudge (thresholded IoU 1.00000) while the old fixed s=4 moved
 * the mask by IoU 0.862 on a small subject and disagreed with the exact filter
 * by 0.19 IoU — inaccuracy bought for 0.8 ms where 2.6 ms would have been exact.
 *
 * So s is derived from the band's area instead of fixed: a small band gets the
 * exact filter, and subsampling only switches on once the band is genuinely
 * large.
 *
 * The VALUE is the accuracy/speed dial, and it was swept rather than guessed
 * (bench/budget.mjs). 40k was the first pick and it was too tight — it dropped to
 * s=3 on mid-size masks, where accuracy against the exact filter is flat in s and
 * lands ~0.98. 100k holds s≤2 there. Accuracy vs the exact filter, 45 MP proxy:
 *
 *   cover    upstream    40k       100k        cost at 100k
 *   0.005     0.9572    1.0000    1.0000       1.5 → 3.2 ms
 *   0.02      0.9639    0.9853    1.0000       2.8 → 7.9 ms
 *   0.1       0.9866    0.9806    0.9976       9.2 → 11.2 ms
 *   0.35      0.9899    0.9927    0.9953      25.5 → 18.2 ms
 *   0.7       0.9973    0.9976    0.9957      30.4 → 17.8 ms
 *
 * So 100k beats upstream on accuracy at every size but the largest (where all
 * three agree inside 0.002) and is still 1.4–3.1x faster on the sizes that cost
 * real time. Raising it further only moves masks already at 0.997+.
 */
let CELL_BUDGET = 100_000
/** Test-only: the accuracy/speed dial. bench/budget.mjs sweeps it. */
export const __setCellBudget = (n) => { CELL_BUDGET = n }

/**
 * Refine the score field along its zero crossing, in place.
 *
 * Only the tiles the boundary actually passes through are touched. The old form
 * refined ONE padded bounding box, which is O(area of that box) — and a
 * ring-shaped band (any large subject, or sky/background) has a near-full-frame
 * bbox, so it degenerated to whole-frame work exactly on the common case.
 * Tiles make it O(perimeter): measured 194 ms → 44 ms at export scale on a
 * 70 %-coverage mask.
 *
 * Skipping a tile is safe for the same reason skipping outside the bbox was: a
 * tile holding no pixel within ±BAND_LOGIT of zero is saturated, and a local
 * linear fit of a saturated field stays on the same side of the cut.
 *
 * Returns the union rect actually rewritten, so a caller re-thresholding with
 * bandAlphaRect keeps the contract it had before.
 */
export const refineField = (field, rgba, w, h, bbox, {
    radius = 8, eps = 1e-4, scale = 4, color = true, chromaEps = CHROMA_EPS,
    tile = TILE,
} = {}) => {
    if (!bbox || bbox[2] < bbox[0]) return null // no boundary in frame
    const T = Math.max(16, tile)

    const bx0 = Math.max(0, bbox[0])
    const by0 = Math.max(0, bbox[1])
    const bx1 = Math.min(w - 1, bbox[2])
    const by1 = Math.min(h - 1, bbox[3])
    if (bx1 < bx0 || by1 < by0) return null

    const tw = Math.ceil((bx1 - bx0 + 1) / T)
    const th = Math.ceil((by1 - by0 + 1) / T)
    const active = new Uint8Array(tw * th)

    // One scan of the bbox marks which tiles the crossing passes through.
    for (let y = by0; y <= by1; y += 1) {
        const row = y * w
        const trow = (((y - by0) / T) | 0) * tw
        for (let x = bx0; x <= bx1; x += 1) {
            const v = field[row + x]
            if (v > -BAND_LOGIT && v < BAND_LOGIT) active[trow + (((x - bx0) / T) | 0)] = 1
        }
    }

    // Collect rects first: their total area sizes `scale`, and the pad depends on
    // the scale chosen, so nothing can be refined until the band's extent is known.
    const pad0 = radius * 2 + scale * 2
    const rects = []
    let area = 0
    for (let ty = 0; ty < th; ty += 1) {
        for (let tx = 0; tx < tw; tx += 1) {
            if (!active[ty * tw + tx]) continue
            // Grow a maximal RECTANGLE, not just a horizontal run. The pad is paid
            // on all four sides of whatever comes out, so a band's vertical edge
            // left as N stacked 1-tile-tall rects pays 2*pad of height N times over
            // — measured, that is what made a box-select's tiling (15 rects,
            // 2.47 M cells) cost more than refining the whole 1.80 M-pixel frame.
            let tx2 = tx
            while (tx2 + 1 < tw && active[ty * tw + tx2 + 1]) tx2 += 1
            // Extend down only while the NEXT row covers the same span exactly;
            // a partial row would either clip the rect or pull in dead tiles.
            let ty2 = ty
            while (ty2 + 1 < th) {
                let full = true
                for (let k = tx; k <= tx2 && full; k += 1) {
                    if (!active[(ty2 + 1) * tw + k]) full = false
                }
                if (!full) break
                ty2 += 1
            }
            for (let r = ty; r <= ty2; r += 1) {
                for (let k = tx; k <= tx2; k += 1) active[r * tw + k] = 0
            }

            const cx0 = bx0 + tx * T
            const cy0 = by0 + ty * T
            const cx1 = Math.min(w, bx0 + (tx2 + 1) * T)
            const cy1 = Math.min(h, by0 + (ty2 + 1) * T)
            area += (Math.min(w, cx1 + pad0) - Math.max(0, cx0 - pad0))
                * (Math.min(h, cy1 + pad0) - Math.max(0, cy0 - pad0))
            rects.push(cx0, cy0, cx1, cy1)
            tx = tx2
        }
    }
    if (!rects.length) return null

    // Tiles win on a ring; ONE rect wins whenever the per-tile pad — paid on all
    // four sides of every tile, and overlapping its neighbours' — costs more than
    // the interior it skips. That is not rare: a box-select's band spans wide rows,
    // and 15 padded 96-tall rects walked 2.47 M cells on a 1.80 M-pixel frame.
    //
    // The choice is made on cost AT THE SUBSAMPLE FACTOR EACH WOULD ACTUALLY USE,
    // because s is what sets accuracy. Same s ⇒ same filter ⇒ the cheaper
    // decomposition is free. A coarser s for the single rect ⇒ keep the tiles;
    // their whole purpose is holding s down, and trading that for speed is the
    // regression this rule exists to prevent.
    const subsample = (cells) => Math.min(Math.max(1, Math.round(scale)),
        Math.max(1, Math.ceil(Math.sqrt(cells / CELL_BUDGET))))
    const sx0 = Math.max(0, bx0 - pad0)
    const sy0 = Math.max(0, by0 - pad0)
    const single = (Math.min(w, bx1 + 1 + pad0) - sx0) * (Math.min(h, by1 + 1 + pad0) - sy0)
    if (single < area && subsample(single) <= subsample(area)) {
        rects.length = 0
        rects.push(bx0, by0, Math.min(w, bx1 + 1), Math.min(h, by1 + 1))
        area = single
    }

    const s = subsample(area)
    const pad = radius * 2 + s * 2

    let ux0 = w, uy0 = h, ux1 = 0, uy1 = 0
    let touched = false
    for (let i = 0; i < rects.length; i += 4) {
        const cx0 = rects[i], cy0 = rects[i + 1], cx1 = rects[i + 2], cy1 = rects[i + 3]
        // Snap the padded origin DOWN to a multiple of s. downsample() bins from
        // the sub-rect's own origin, so without this every rect subsamples on its
        // own phase: tiles disagree with a single rect at the same s, and a 1 px
        // bbox nudge shifts the grid under the whole band. Aligned, every bin
        // falls on a global multiple of s no matter how the band is cut up, which
        // is what makes the decomposition choice above a pure cost question.
        // Snapping down only ever adds context, and at most s-1 columns of it.
        const px0 = Math.max(0, cx0 - pad)
        const py0 = Math.max(0, cy0 - pad)
        const x0 = px0 - (px0 % s)   // s is not always a power of two (3 occurs)
        const y0 = py0 - (py0 % s)
        const x1 = Math.min(w, cx1 + pad)
        const y1 = Math.min(h, cy1 + pad)
        if (x1 - x0 <= 2 || y1 - y0 <= 2) continue

        refineRect(field, rgba, w, x0, y0, x1 - x0, y1 - y0, cx0, cy0, cx1, cy1,
            { radius, eps, scale: s, color, chromaEps })
        touched = true
        if (cx0 < ux0) ux0 = cx0
        if (cy0 < uy0) uy0 = cy0
        if (cx1 > ux1) ux1 = cx1
        if (cy1 > uy1) uy1 = cy1
    }
    lastRefineStats = {
        rects: rects.length / 4, s, pad,
        bandArea: area,
        bboxArea: (bx1 - bx0 + 1) * (by1 - by0 + 1),
        frame: w * h,
        // What the filter actually walked, pad included — the only figure that
        // predicts cost. `bandArea` is the same sum, so a large ratio to
        // bboxArea means the pad, not the band, is what is being paid for.
        cells: area,
    }
    return touched ? [ux0, uy0, ux1, uy1] : null
}

/** Decomposition the last refineField chose. Cost is linear in `cells`. */
export let lastRefineStats = null

/**
 * Refine one padded rect, writing back only its core. The pad exists so the box
 * means see real context; its own values are contaminated by the rect edge and
 * are discarded.
 */
const refineRect = (field, rgba, w, x0, y0, rw, rh, cx0, cy0, cx1, cy1, o) => {
    const n = rw * rh
    const sub = take(Float32Array, n)
    const gy = take(Float32Array, n)
    const cb = o.color ? take(Float32Array, n) : null
    const cr = o.color ? take(Float32Array, n) : null

    const INV255 = 1 / 255
    for (let y = 0; y < rh; y += 1) {
        const srow = (y0 + y) * w + x0
        const drow = y * rw
        sub.set(field.subarray(srow, srow + rw), drow)
        for (let x = 0; x < rw; x += 1) {
            const j = (srow + x) << 2
            const r = rgba[j] * INV255, g = rgba[j | 1] * INV255, b = rgba[j | 2] * INV255
            const Y = KR * r + KG * g + KB * b
            gy[drow + x] = Y
            if (o.color) {
                cb[drow + x] = KCB * (b - Y)
                cr[drow + x] = KCR * (r - Y)
            }
        }
    }

    const ref = o.color
        ? gfColor(sub, gy, cb, cr, rw, rh,
            { radius: o.radius, eps: o.eps, eps2: o.eps * o.chromaEps, eps3: o.eps * o.chromaEps, scale: o.scale })
        : gfLuma(sub, gy, rw, rh, { radius: o.radius, eps: o.eps, scale: o.scale })

    const cw = cx1 - cx0
    for (let y = cy0; y < cy1; y += 1) {
        const s = (y - y0) * rw + (cx0 - x0)
        field.set(ref.subarray(s, s + cw), y * w + cx0)
    }
    // Per-RECT, not per-call: the copy-out above is the last read of anything
    // borrowed for this rect, so every tile reuses the same buffers. Releasing
    // once per refineField would make the pool grow with the tile count.
    release()
}

// ----------------------------------------------------------------- band alpha

/** True when a Uint8ClampedArray can be viewed as u32 (offset and length both
 *  4-aligned). Anything the app builds is, but a caller-supplied subarray need
 *  not be, so the byte path stays as a fallback. */
const u32able = (a) => (a.byteOffset & 3) === 0 && (a.length & 3) === 0

/**
 * Narrow-band alpha (§10 step 3). Inside and outside stay hard; only a strip
 * around the zero crossing gets a soft ramp, so hair and foliage keep a real
 * gradient at negligible cost. `band` is in logit units, not pixels — the field
 * is already edge-aligned by the guided filter, so a value cut is the right
 * cut and costs one pass.
 *
 * The ramp goes in EVERY channel: the app reads coverage from R (>=128) and
 * composites from A, so both have to carry the same soft edge. That makes the
 * 32-bit word byte-symmetric, so it can be stored once per pixel instead of four
 * times and endianness cannot matter.
 */
export const bandAlpha = (field, w, h, band = 1.5, out = null) => {
    const dst = out || new Uint8ClampedArray(w * h * 4)
    const inv = 255 / (2 * band)
    const n = Math.min(field.length, w * h)
    // Branchless clamp: Math.min/max compile to hardware min/max on modern V8
    const clamp = (v) => Math.round(Math.min(255, Math.max(0, (v + band) * inv)))
    if (u32able(dst)) {
        const u32 = new Uint32Array(dst.buffer, dst.byteOffset, dst.length >> 2)
        // 4× unrolled for better JIT pipeline utilization
        const n4 = n & ~3
        for (let i = 0; i < n4; i += 4) {
            u32[i]     = clamp(field[i])     * 0x01010101
            u32[i + 1] = clamp(field[i + 1]) * 0x01010101
            u32[i + 2] = clamp(field[i + 2]) * 0x01010101
            u32[i + 3] = clamp(field[i + 3]) * 0x01010101
        }
        for (let i = n4; i < n; i += 1) u32[i] = clamp(field[i]) * 0x01010101
        return dst
    }
    for (let i = 0, j = 0; i < n; i += 1, j += 4) {
        const a = clamp(field[i])
        dst[j] = dst[j + 1] = dst[j + 2] = dst[j + 3] = a
    }
    return dst
}

/** The same ramp restricted to `rect` — for re-deriving a mask after
 *  refineField rewrote only that band. Outside is left as the caller had it. */
export const bandAlphaRect = (field, w, rect, band, dst) => {
    const [x0, y0, x1, y1] = rect
    const inv = 255 / (2 * band)
    if (u32able(dst)) {
        const u32 = new Uint32Array(dst.buffer, dst.byteOffset, dst.length >> 2)
        for (let y = y0; y < y1; y += 1) {
            const row = y * w
            for (let x = x0; x < x1; x += 1) {
                const v = field[row + x]
                const a = v <= -band ? 0 : (v >= band ? 255 : Math.round((v + band) * inv))
                u32[row + x] = a * 0x01010101
            }
        }
        return dst
    }
    for (let y = y0; y < y1; y += 1) {
        const row = y * w
        for (let x = x0; x < x1; x += 1) {
            const v = field[row + x]
            const a = v <= -band ? 0 : (v >= band ? 255 : Math.round((v + band) * inv))
            const j = (row + x) * 4
            dst[j] = dst[j + 1] = dst[j + 2] = dst[j + 3] = a
        }
    }
    return dst
}
