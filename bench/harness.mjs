/**
 * Shared bench harness for the mask-refine lane.
 *
 * Reproduces the REAL path on real DSLR pixels:
 *   256² SAM logits → separable bicubic upsample → refineField → bandAlpha
 *
 * The guide is a genuine 45 MP Nikon NEF decoded to the app's proxy size, so the
 * guided filter sees real sensor noise, real chroma and real edges — not a
 * synthetic gradient that would flatter any filter.
 */
import { readFileSync } from 'node:fs'
import path from 'node:path'

export const CORPUS = path.resolve(import.meta.dirname, './corpus')
const MASK_SIDE = 256

export const loadRgba = (name, w, h) => {
    const buf = readFileSync(path.join(CORPUS, name))
    if (buf.length !== w * h * 4) throw new Error(`${name}: ${buf.length} != ${w * h * 4}`)
    return new Uint8ClampedArray(buf.buffer, buf.byteOffset, buf.length)
}

const cubic = (t) => {
    const t2 = t * t, t3 = t2 * t
    return [-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1,
        -1.5 * t3 + 2 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2]
}
const clampIdx = (i) => (i < 0 ? 0 : (i > MASK_SIDE - 1 ? MASK_SIDE - 1 : i))

/** Byte-for-byte port of sam21-adapter's upsampleLogits (+ band bbox). */
export const upsampleLogits = (logits, w, h) => {
    const out = new Float32Array(w * h)
    const sx = MASK_SIDE / w, sy = MASK_SIDE / h
    const xi = new Int32Array(w * 4), xw = new Float32Array(w * 4)
    for (let x = 0; x < w; x += 1) {
        const fx = (x + 0.5) * sx - 0.5
        const x0 = Math.floor(fx)
        const k = cubic(fx - x0)
        for (let t = 0; t < 4; t += 1) { xi[x * 4 + t] = clampIdx(x0 - 1 + t); xw[x * 4 + t] = k[t] }
    }
    const tmp = new Float32Array(MASK_SIDE * w)
    for (let r = 0; r < MASK_SIDE; r += 1) {
        const src = r * MASK_SIDE, dst = r * w
        for (let x = 0; x < w; x += 1) {
            const b = x * 4
            tmp[dst + x] = logits[src + xi[b]] * xw[b] + logits[src + xi[b + 1]] * xw[b + 1]
                + logits[src + xi[b + 2]] * xw[b + 2] + logits[src + xi[b + 3]] * xw[b + 3]
        }
    }
    const BAND = 6
    let minX = w, minY = h, maxX = -1, maxY = -1
    for (let y = 0; y < h; y += 1) {
        const fy = (y + 0.5) * sy - 0.5
        const y0 = Math.floor(fy)
        const k = cubic(fy - y0)
        const r0 = clampIdx(y0 - 1) * w, r1 = clampIdx(y0) * w
        const r2 = clampIdx(y0 + 1) * w, r3 = clampIdx(y0 + 2) * w
        const row = y * w
        for (let x = 0; x < w; x += 1) {
            const v = tmp[r0 + x] * k[0] + tmp[r1 + x] * k[1] + tmp[r2 + x] * k[2] + tmp[r3 + x] * k[3]
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

/**
 * A 256² logit field shaped like real SAM output: a smooth signed field that
 * saturates well inside/outside and crosses zero over a few cells, plus a little
 * high-frequency wobble so the band is not a perfect analytic curve.
 * `cover` is the target area fraction — it drives how much of the frame the
 * refinement band touches, which is the variable perf actually depends on.
 */
export const makeLogits = (cover = 0.25, seed = 7) => {
    const f = new Float32Array(MASK_SIDE * MASK_SIDE)
    // xorshift so runs are reproducible across processes
    let s = seed | 0 || 7
    const rnd = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296 }
    const R = Math.sqrt(cover / Math.PI) * MASK_SIDE
    const cx = MASK_SIDE * 0.5, cy = MASK_SIDE * 0.5
    // a few low-frequency lobes → a non-convex, organic outline
    const lobes = Array.from({ length: 5 }, () => ({
        k: 2 + Math.floor(rnd() * 5), a: rnd() * 0.28, p: rnd() * Math.PI * 2,
    }))
    for (let y = 0; y < MASK_SIDE; y += 1) {
        for (let x = 0; x < MASK_SIDE; x += 1) {
            const dx = x - cx, dy = y - cy
            const th = Math.atan2(dy, dx)
            let rr = R
            for (const L of lobes) rr *= 1 + L.a * Math.sin(L.k * th + L.p)
            const d = Math.hypot(dx, dy) - rr
            // slope ~1.2 logits/cell at the crossing, saturating by ±10
            f[y * MASK_SIDE + x] = Math.max(-10, Math.min(10, -d * 1.2 + (rnd() - 0.5) * 0.35))
        }
    }
    return f
}

/** Median-of-runs timing. Returns {med, min, p90} in ms. */
export const time = (fn, runs = 15, warm = 4) => {
    for (let i = 0; i < warm; i += 1) fn()
    const t = []
    for (let i = 0; i < runs; i += 1) {
        const a = performance.now()
        fn()
        t.push(performance.now() - a)
    }
    t.sort((x, y) => x - y)
    return { med: t[t.length >> 1], min: t[0], p90: t[Math.floor(t.length * 0.9)] }
}

/** Compare two float fields. */
export const diff = (a, b) => {
    let max = 0, sum = 0, n = 0
    for (let i = 0; i < a.length; i += 1) {
        const d = Math.abs(a[i] - b[i])
        if (d > max) max = d
        sum += d; n += 1
    }
    return { max, mean: sum / n }
}

/** Compare two RGBA masks: max channel delta + IoU at the 128 coverage cut. */
export const maskDiff = (a, b) => {
    let max = 0, inter = 0, uni = 0
    for (let i = 0; i < a.length; i += 4) {
        const d = Math.abs(a[i] - b[i])
        if (d > max) max = d
        const pa = a[i] >= 128, pb = b[i] >= 128
        if (pa && pb) inter += 1
        if (pa || pb) uni += 1
    }
    return { max, iou: uni ? inter / uni : 1 }
}

export const PROXIES = [
    { name: 'nef_proxy_1024.rgba', w: 1024, h: 683, label: '1024×683 (45MP NEF proxy)' },
    { name: 'nef_proxy_2048.rgba', w: 2048, h: 1365, label: '2048×1365 (export-scale)' },
]
