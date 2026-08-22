/**
 * subject-prior — "where is the subject", answered from the pixels.
 *
 * AI Subject used to probe SAM with a box inset 4 % from the frame edge, which
 * is another way of saying "select the whole photo". Candidate arbitration ranks
 * partly by `inBox`, so a whole-frame box makes that term 1 for every candidate
 * and the ranking loses its only positional signal — a 15 % blob anywhere in the
 * frame then passes the coverage gate and wins. That is how a shallow-DOF macro
 * shot came back with a fabric fold in the top-left corner.
 *
 * No model fixes that, because the mask is on the wrong OBJECT. What is needed
 * is a real answer to "where", computed before SAM is prompted at all. This is
 * that answer: a saliency map from three pixel statistics, and the box / clicks
 * derived from it.
 *
 * Everything runs on a downsample with a 128 px short edge (~25 k cells), so
 * every pass below is sub-millisecond and nothing new is downloaded. It is a
 * PRIOR, not a segmentation: SAM still decides the boundary. Its only job is to
 * point at the right thing.
 *
 * Never throws, and returns null rather than a guess: a flat, evenly-lit scene
 * has no subject to find, and the caller falls back to the old probes.
 */

import { boxMeanInto, ycbcrPlanes } from './mask-refine.js'

/** Short edge of the analysis grid. 128 keeps every pass under a millisecond
 *  and is still finer than SAM's own 256² after the aspect ratio is applied. */
const SHORT_EDGE = 128
/** Box radius for the local-sharpness mean, in analysis cells. */
const FOCUS_RADIUS = 4
/** Second sharpness scale, as a multiple of FOCUS_RADIUS. */
const FOCUS_SPREAD = 3
/** Signal weights. Focus is the strongest cue for photography; the centre prior
 *  is deliberately the weakest — it is a tiebreaker, not the answer. */
const W_FOCUS = 0.45
const W_COLOUR = 0.35
const W_CENTRE = 0.20
/** Floor under each signal before the weighted product, so one weak term
 *  discounts a cell instead of vetoing it. 0.05 vetoed: 0.26x for one floored
 *  term, 0.09x for two — enough to sink a smooth subject the colour of its
 *  own background. */
const FLOOR = 0.18
/** Cells this fraction of the frame from the edge are attenuated: a subject
 *  does not begin at the border, and a background that does is not a subject. */
const BORDER_FRAC = 0.03
const BORDER_PENALTY = 0.35
/** Below this variance the map has no structure to read — a flat scene. */
const MIN_VARIANCE = 0.004
/** A region past this fraction of the frame is the photo, not a subject. */
const MAX_COVER = 0.85
/** Growth stops here and retries at a higher level. Past 60 % of the frame a
 *  box prior excludes almost nothing, so there is no reason to risk it. */
const GROW_COVER = 0.6
/** Non-max suppression radius for the point prompts, as a fraction of the
 *  analysis grid's short edge. */
const NMS_FRAC = 0.1
/** SAM's own logit grid — what `grid256` is sampled onto. */
const SAM_SIDE = 256

/* ─── analysis frame ─────────────────────────────────────────────────────── */

/** The proxy at ~128 px short edge, as RGBA. Null when the canvas is unreadable
 *  (tainted) or degenerate. */
const analysisPixels = (canvas) => {
    const W = canvas?.width | 0
    const H = canvas?.height | 0
    if (!(W > 0) || !(H > 0)) return null
    const k = SHORT_EDGE / Math.max(1, Math.min(W, H))
    const sw = Math.max(8, Math.min(W, Math.round(W * k)))
    const sh = Math.max(8, Math.min(H, Math.round(H * k)))
    try {
        const c = typeof OffscreenCanvas !== 'undefined'
            ? new OffscreenCanvas(sw, sh)
            : Object.assign(document.createElement('canvas'), { width: sw, height: sh })
        const ctx = c.getContext('2d', { willReadFrequently: true })
        ctx.drawImage(canvas, 0, 0, sw, sh)
        return { px: ctx.getImageData(0, 0, sw, sh).data, sw, sh }
    } catch { return null }
}

/* ─── signals ────────────────────────────────────────────────────────────── */

/** 95th percentile via a 256-bin histogram — a sort of 25 k floats costs more
 *  than the precision is worth, and every consumer is a normaliser. */
const p95 = (src, n) => {
    let hi = 0
    for (let i = 0; i < n; i += 1) if (src[i] > hi) hi = src[i]
    if (!(hi > 0)) return 0
    const bins = new Int32Array(256)
    const k = 255 / hi
    for (let i = 0; i < n; i += 1) bins[Math.min(255, (src[i] * k) | 0)] += 1
    const want = Math.ceil(n * 0.95)
    let acc = 0
    for (let b = 0; b < 256; b += 1) {
        acc += bins[b]
        if (acc >= want) return ((b + 1) / 255) * hi
    }
    return hi
}

/** Normalise in place to [0,1] against `ref`, which is a percentile rather than
 *  the max: one specular highlight otherwise flattens the whole map. */
const normalise = (src, n, ref) => {
    const inv = ref > 1e-9 ? 1 / ref : 0
    for (let i = 0; i < n; i += 1) {
        const v = src[i] * inv
        src[i] = v > 1 ? 1 : v
    }
}

/**
 * Local sharpness — the strongest "main subject" cue in photography, and
 * exactly what a shallow depth of field hands over for free. |∇²Y| squared,
 * box-averaged, normalised by its own 95th percentile.
 *
 * Mean of L² rather than a true windowed variance: the Laplacian is zero-mean
 * over any smooth region, so the two agree to within the window's DC term and
 * this costs one box mean instead of two.
 *
 * Two scales, because |∇²Y| measures texture: a tight window calls the interior
 * of any smooth object (painted panel, petal) out of focus. The wide one lets a
 * region inherit its own edges. Each normalised to its own p95 before the max —
 * a wide mean of a sparse edge field is smaller by the window's area ratio and
 * would never win otherwise.
 */
const focusMap = (Y, sw, sh, out) => {
    const n = sw * sh
    const lap = new Float32Array(n)
    for (let y = 1; y < sh - 1; y += 1) {
        const row = y * sw
        for (let x = 1; x < sw - 1; x += 1) {
            const i = row + x
            const l = 4 * Y[i] - Y[i - 1] - Y[i + 1] - Y[i - sw] - Y[i + sw]
            lap[i] = l * l
        }
    }
    boxMeanInto(lap, sw, sh, FOCUS_RADIUS, out)
    normalise(out, n, p95(out, n))
    const wide = new Float32Array(n)
    boxMeanInto(lap, sw, sh, FOCUS_RADIUS * FOCUS_SPREAD, wide)
    normalise(wide, n, p95(wide, n))
    for (let i = 0; i < n; i += 1) if (wide[i] > out[i]) out[i] = wide[i]
    return out
}

/**
 * How far each cell's colour sits from the FRAME BORDER's distribution.
 *
 * The outer ring is the one region a photograph almost always spends on
 * context, so it is the closest thing to a background sample that needs no
 * model. Scored as a Mahalanobis distance against the ring's own 3×3 Y/Cb/Cr
 * covariance, so a busy background is correctly treated as less informative
 * than a clean one instead of lighting up everywhere.
 */
const colourMap = (Y, Cb, Cr, sw, sh, out) => {
    const n = sw * sh
    const ring = Math.max(2, Math.round(Math.min(sw, sh) * 0.08))
    let cnt = 0
    let my = 0; let mb = 0; let mr = 0
    let syy = 0; let sbb = 0; let srr = 0
    let syb = 0; let syr = 0; let sbr = 0
    for (let y = 0; y < sh; y += 1) {
        const inRing = y < ring || y >= sh - ring
        const row = y * sw
        for (let x = 0; x < sw; x += 1) {
            if (!inRing && x >= ring && x < sw - ring) { x = sw - ring - 1; continue }
            const i = row + x
            const a = Y[i]; const b = Cb[i]; const c = Cr[i]
            my += a; mb += b; mr += c
            syy += a * a; sbb += b * b; srr += c * c
            syb += a * b; syr += a * c; sbr += b * c
            cnt += 1
        }
    }
    if (cnt < 8) { out.fill(0); return out }
    const inv = 1 / cnt
    my *= inv; mb *= inv; mr *= inv
    // Regularised: an evenly-toned border is near-singular, and an unregularised
    // inverse would then report astronomic distances for ordinary noise.
    const E = 1e-4
    const a11 = syy * inv - my * my + E
    const a22 = sbb * inv - mb * mb + E
    const a33 = srr * inv - mr * mr + E
    const a12 = syb * inv - my * mb
    const a13 = syr * inv - my * mr
    const a23 = sbr * inv - mb * mr

    const c11 = a22 * a33 - a23 * a23
    const c12 = a13 * a23 - a12 * a33
    const c13 = a12 * a23 - a13 * a22
    let det = a11 * c11 + a12 * c12 + a13 * c13
    if (det > -1e-12 && det < 1e-12) det = det < 0 ? -1e-12 : 1e-12
    const c22 = a11 * a33 - a13 * a13
    const c23 = a13 * a12 - a11 * a23
    const c33 = a11 * a22 - a12 * a12
    const invDet = 1 / det

    for (let i = 0; i < n; i += 1) {
        const dy = Y[i] - my
        const db = Cb[i] - mb
        const dr = Cr[i] - mr
        const xy = (c11 * dy + c12 * db + c13 * dr) * invDet
        const xb = (c12 * dy + c22 * db + c23 * dr) * invDet
        const xr = (c13 * dy + c23 * db + c33 * dr) * invDet
        const d = dy * xy + db * xb + dr * xr
        out[i] = d > 0 ? Math.sqrt(d) : 0
    }
    // Smoothed before normalising: a per-cell distance is noisy, and the map is
    // asked "is this REGION unlike the background", not "is this pixel".
    boxMeanInto(out, sw, sh, 2, out)
    normalise(out, n, p95(out, n))
    return out
}

/* ─── combination ────────────────────────────────────────────────────────── */

/**
 * The three signals, combined as a WEIGHTED PRODUCT rather than a sum, then
 * attenuated at the border.
 *
 * A sum lets one strong term carry a cell on its own, which is precisely the
 * failure being fixed: the centre prior alone would nominate the middle of
 * every photograph. A floored product asks a cell to be plausible on all three
 * before it can win, and the floor keeps a single weak term from vetoing.
 */
const combine = (focus, colour, sw, sh) => {
    const n = sw * sh
    const out = new Float32Array(n)
    const cx = (sw - 1) / 2
    const cy = (sh - 1) / 2
    const sigma = 0.35 * Math.min(sw, sh)
    const inv2s2 = 1 / (2 * sigma * sigma)
    const edgeX = Math.max(1, Math.round(sw * BORDER_FRAC))
    const edgeY = Math.max(1, Math.round(sh * BORDER_FRAC))
    for (let y = 0; y < sh; y += 1) {
        const dy = y - cy
        const row = y * sw
        const yEdge = y < edgeY || y >= sh - edgeY
        for (let x = 0; x < sw; x += 1) {
            const dx = x - cx
            const centre = Math.exp(-(dx * dx + dy * dy) * inv2s2)
            const i = row + x
            const f = focus[i] < FLOOR ? FLOOR : focus[i]
            const c = colour[i] < FLOOR ? FLOOR : colour[i]
            const g = centre < FLOOR ? FLOOR : centre
            let v = (f ** W_FOCUS) * (c ** W_COLOUR) * (g ** W_CENTRE)
            if (yEdge || x < edgeX || x >= sw - edgeX) v *= BORDER_PENALTY
            out[i] = v
        }
    }
    return out
}

/** Mean and variance of the map — the "is there anything here" test. */
const spread = (map, n) => {
    let sum = 0
    for (let i = 0; i < n; i += 1) sum += map[i]
    const mean = sum / n
    let acc = 0
    for (let i = 0; i < n; i += 1) { const d = map[i] - mean; acc += d * d }
    return { mean, variance: acc / n }
}

/** Otsu's threshold over a 256-bin histogram of the map. */
const otsu = (map, n) => {
    let hi = 0
    for (let i = 0; i < n; i += 1) if (map[i] > hi) hi = map[i]
    if (!(hi > 0)) return 0
    const bins = new Int32Array(256)
    const k = 255 / hi
    for (let i = 0; i < n; i += 1) bins[Math.min(255, (map[i] * k) | 0)] += 1
    let total = 0
    for (let b = 0; b < 256; b += 1) total += bins[b] * b
    let wB = 0
    let sumB = 0
    let best = 0
    let bestVar = -1
    for (let b = 0; b < 256; b += 1) {
        wB += bins[b]
        if (!wB) continue
        const wF = n - wB
        if (!wF) break
        sumB += bins[b] * b
        const mB = sumB / wB
        const mF = (total - sumB) / wF
        const between = wB * wF * (mB - mF) * (mB - mF)
        if (between > bestVar) { bestVar = between; best = b }
    }
    return ((best + 1) / 255) * hi
}

/** Otsu again over the cells below `hi` — the split between background and the
 *  merely-unremarkable. A fixed fraction of `hi` cannot do this: how far the
 *  subject's dull parts sit above the background is a property of the scene,
 *  and 0.5x grows straight through a busy one. */
const otsuBelow = (map, n, hi) => {
    const sub = new Float32Array(n)
    let m = 0
    for (let i = 0; i < n; i += 1) if (map[i] < hi) { sub[m] = map[i]; m += 1 }
    return m < 16 ? hi : otsu(sub, m)
}

/* ─── the component holding the subject ──────────────────────────────────── */

/**
 * Components of `map > t`, ranked by how many cells they hold above `seedT`.
 *
 * Two levels because one cannot do both jobs: the level that separates subject
 * from background is set by the most anomalous thing in frame, which is a PART
 * of the subject (a tram's roof, a rose's petals). So the high level seeds and
 * the low one traces. Seed mass, not size, decides — the biggest region above
 * the low level is usually the background it grew out of.
 *
 * Scanline runs plus union-find — the same decomposition `cleanRegions`
 * (mask-select.js) uses, written locally because at 25 k cells it is a dozen
 * lines and importing that module's tuned, stateful scratch for one call would
 * couple two passes that have nothing to do with each other.
 */
const subjectComponent = (map, sw, sh, t, seedT) => {
    const parent = []
    const find = (i) => {
        let r = i
        while (parent[r] !== r) { parent[r] = parent[parent[r]]; r = parent[r] }
        return r
    }
    const union = (a, b) => {
        const ra = find(a); const rb = find(b)
        if (ra === rb) return ra
        if (ra < rb) { parent[rb] = ra; return ra }
        parent[ra] = rb
        return rb
    }
    const size = []
    const seed = []
    const bx0 = []; const by0 = []; const bx1 = []; const by1 = []
    let prev = []
    for (let y = 0; y < sh; y += 1) {
        const row = y * sw
        const cur = []
        let x = 0
        while (x < sw) {
            while (x < sw && !(map[row + x] > t)) x += 1
            if (x >= sw) break
            const s = x
            do { x += 1 } while (x < sw && map[row + x] > t)
            const e = x
            let lab = -1
            for (const r of prev) {
                if (r[1] <= s || r[0] >= e) continue
                const f = find(r[2])
                lab = lab < 0 ? f : union(lab, f)
            }
            if (lab < 0) {
                lab = parent.length
                parent.push(lab); size.push(0); seed.push(0)
                bx0.push(s); bx1.push(e - 1); by0.push(y); by1.push(y)
            }
            if (s < bx0[lab]) bx0[lab] = s
            if (e - 1 > bx1[lab]) bx1[lab] = e - 1
            if (y < by0[lab]) by0[lab] = y
            if (y > by1[lab]) by1[lab] = y
            size[lab] += e - s
            for (let i = s; i < e; i += 1) if (map[row + i] > seedT) seed[lab] += 1
            cur.push([s, e, lab])
        }
        prev = cur
    }
    if (!parent.length) return null
    // Roots always carry the lower id, so one reverse pass folds every child.
    for (let i = parent.length - 1; i >= 0; i -= 1) {
        const r = find(i)
        if (r === i) continue
        size[r] += size[i]
        seed[r] += seed[i]
        if (bx0[i] < bx0[r]) bx0[r] = bx0[i]
        if (by0[i] < by0[r]) by0[r] = by0[i]
        if (bx1[i] > bx1[r]) bx1[r] = bx1[i]
        if (by1[i] > by1[r]) by1[r] = by1[i]
    }
    let best = -1
    for (let i = 0; i < parent.length; i += 1) {
        if (find(i) !== i) continue
        if (best < 0) { best = i; continue }
        if (seed[i] > seed[best] || (seed[i] === seed[best] && size[i] > size[best])) best = i
    }
    if (best < 0 || !size[best]) return null
    return { root: best, find, size: size[best], seed: seed[best], box: [bx0[best], by0[best], bx1[best], by1[best]] }
}

/* ─── prompt extraction ──────────────────────────────────────────────────── */

/** Top-`k` maxima of `map` inside `box`, non-max-suppressed so three prompts
 *  describe three parts of the subject rather than one hot spot three times. */
const peaks = (map, sw, sh, box, k, radius) => {
    const [x0, y0, x1, y1] = box
    const picked = []
    const r2 = radius * radius
    // One pass per pick: the window is a few thousand cells, and a full sort
    // would order 25 k values to read three of them.
    for (let p = 0; p < k; p += 1) {
        let bi = -1
        let bv = -Infinity
        for (let y = y0; y <= y1; y += 1) {
            const row = y * sw
            for (let x = x0; x <= x1; x += 1) {
                const v = map[row + x]
                if (v <= bv) continue
                let clear = true
                for (const q of picked) {
                    const dx = x - q[0]; const dy = y - q[1]
                    if (dx * dx + dy * dy < r2) { clear = false; break }
                }
                if (clear) { bv = v; bi = row + x }
            }
        }
        if (bi < 0) break
        picked.push([bi % sw, (bi / sw) | 0])
    }
    return picked
}

/** The two least subject-like border cells, kept apart so they sample two
 *  different pieces of background rather than one corner twice. */
const borderNegatives = (map, sw, sh) => {
    const cells = []
    const push = (x, y) => cells.push([x, y, map[y * sw + x]])
    for (let x = 0; x < sw; x += 1) { push(x, 0); push(x, sh - 1) }
    for (let y = 1; y < sh - 1; y += 1) { push(0, y); push(sw - 1, y) }
    cells.sort((a, b) => a[2] - b[2])
    const out = []
    const minGap = 0.3 * Math.max(sw, sh)
    for (const c of cells) {
        if (out.some((o) => Math.hypot(o[0] - c[0], o[1] - c[1]) < minGap)) continue
        out.push(c)
        if (out.length === 2) break
    }
    return out.map((c) => [c[0], c[1]])
}

/** The map resampled onto SAM's 256² logit grid, quantised to bytes.
 *  Bytes because this crosses to the SharedWorker on every probe: 64 KB is a
 *  structured clone nobody notices, 256 KB of f32 is four times that for
 *  precision a ranking term cannot use. */
const toSamGrid = (map, sw, sh) => {
    const out = new Uint8Array(SAM_SIDE * SAM_SIDE)
    for (let y = 0; y < SAM_SIDE; y += 1) {
        const sy = Math.min(sh - 1, ((y + 0.5) * sh / SAM_SIDE) | 0)
        const src = sy * sw
        const dst = y * SAM_SIDE
        for (let x = 0; x < SAM_SIDE; x += 1) {
            const sx = Math.min(sw - 1, ((x + 0.5) * sw / SAM_SIDE) | 0)
            const v = map[src + sx]
            out[dst + x] = v <= 0 ? 0 : (v >= 1 ? 255 : (v * 255) | 0)
        }
    }
    return out
}

/* ─── public ─────────────────────────────────────────────────────────────── */

/**
 * The subject prior for one proxy canvas, or null when the scene has no
 * structure to read.
 *
 * `box`, `points` and `negatives` are all in the canvas's OWN pixel space — the
 * same space `select()` takes prompts in — so the caller does no mapping.
 * `grid256` is the saliency map on SAM's logit grid, for the `salienceFit`
 * term in candidate arbitration.
 */
export const computeSubjectPrior = (canvas) => {
    try {
        const frame = analysisPixels(canvas)
        if (!frame) return null
        const { px, sw, sh } = frame
        const n = sw * sh

        const Y = new Float32Array(n)
        const Cb = new Float32Array(n)
        const Cr = new Float32Array(n)
        ycbcrPlanes(px, sw, sh, Y, Cb, Cr)

        const focus = focusMap(Y, sw, sh, new Float32Array(n))
        const colour = colourMap(Y, Cb, Cr, sw, sh, new Float32Array(n))
        const map = combine(focus, colour, sw, sh)

        const { variance } = spread(map, n)
        if (!(variance > MIN_VARIANCE)) return null

        const hiT = otsu(map, n)
        const loT = otsuBelow(map, n, hiT)
        // Lowest level first, backing off as soon as the region stops being an
        // object and starts being the photo — a busy background is reachable
        // from the subject, and on those scenes the grow has to be given up.
        let comp = null
        for (const t of [loT, (loT + hiT) / 2, hiT]) {
            comp = subjectComponent(map, sw, sh, t, hiT)
            if (comp && comp.size <= n * GROW_COVER) break
        }
        // A speck or the whole frame answers the same as no prior at all, and a
        // bad box is worse than none.
        if (!comp || comp.size < n * 0.005 || comp.size > n * MAX_COVER) return null

        const pts = peaks(map, sw, sh, comp.box, 3, NMS_FRAC * Math.min(sw, sh))
        if (!pts.length) return null
        const negs = borderNegatives(map, sw, sh)

        // Analysis cell → canvas pixel, sampling each cell at its centre.
        const kx = canvas.width / sw
        const ky = canvas.height / sh
        const toPx = ([x, y]) => [(x + 0.5) * kx, (y + 0.5) * ky]
        const [cx0, cy0, cx1, cy1] = comp.box
        // One cell of pad on each side: the box is a threshold crossing, not a
        // boundary, and SAM reads a box as "the object is in here".
        const box = [
            Math.max(0, cx0 * kx - kx),
            Math.max(0, cy0 * ky - ky),
            Math.min(canvas.width, (cx1 + 1) * kx + kx),
            Math.min(canvas.height, (cy1 + 1) * ky + ky),
        ]

        return {
            sw,
            sh,
            map,
            box,
            points: pts.map(toPx),
            negatives: negs.map(toPx),
            grid256: toSamGrid(map, sw, sh),
            coverage: comp.size / n,
            variance,
        }
    } catch (err) {
        // A prior is an optimisation. Never let it fail a selection.
        console.warn('[seglab][prior] subject prior unavailable:', err?.message || err)
        return null
    }
}
