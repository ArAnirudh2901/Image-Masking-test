/**
 * In-browser A/B of mask-refine on V8, using the real page's own proxy pixels.
 *
 * The Node numbers are JSC; the app runs on V8, and JIT behaviour differs enough
 * that a speedup has to be confirmed where it ships. Both modules are imported
 * into the live page and run against the actual decoded proxy the app is holding.
 */
import { launch } from './cdp.mjs'
import { copyFileSync, rmSync } from 'node:fs'
import path from 'node:path'

const APP = '/Users/anirudharavalli/Web_Dev/NextJS/Image-Masking-test'
const BASE = process.env.URL || 'http://127.0.0.1:8811/'
const CORPUS = path.resolve(import.meta.dirname, './corpus')
const IMG = process.argv[2] || '2680558334.nef'

// serve.mjs only serves from the project dir, so the pristine copy is staged
// there for the duration of the run and removed after.
const staged = path.join(APP, 'ai', '_orig-mask-refine.js')
copyFileSync(path.resolve(import.meta.dirname, 'orig-mask-refine.js'), staged)

const b = await launch({ headless: true })
try {
    await b.goto(BASE)
    await b.evaluate(`
        for (let i = 0; i < 300; i++) { if (window.__ready) return true; await new Promise(r => setTimeout(r, 100)) }
        throw new Error('not ready')
    `)
    await b.setFiles('input[type=file]', [path.join(CORPUS, IMG)])
    const size = await b.evaluate(`
        for (let i = 0; i < 600; i++) {
            const s = window.__studio.imageSize()
            if (s && s.width) return JSON.stringify(s)
            await new Promise(r => setTimeout(r, 100))
        }
        return null
    `)
    console.log(`image ${IMG} → proxy ${size}`)

    const out = await b.evaluate(`
        const opt  = await import('/ai/mask-refine.js')
        const orig = await import('/ai/_orig-mask-refine.js')
        const { width: W, height: H } = window.__studio.imageSize()

        // The app's own decoded proxy pixels.
        const cvs = document.querySelector('canvas')
        const px = window.__studio.pixels()

        // Same synthetic-but-organic logit field the Node harness uses, so the
        // two benches are comparable.
        const MS = 256
        const makeLogits = (cover, seed = 7) => {
            const f = new Float32Array(MS * MS)
            let s = seed | 0 || 7
            const rnd = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296 }
            const R = Math.sqrt(cover / Math.PI) * MS, cx = MS / 2, cy = MS / 2
            const lobes = Array.from({ length: 5 }, () => ({ k: 2 + Math.floor(rnd() * 5), a: rnd() * 0.28, p: rnd() * Math.PI * 2 }))
            for (let y = 0; y < MS; y++) for (let x = 0; x < MS; x++) {
                const dx = x - cx, dy = y - cy, th = Math.atan2(dy, dx)
                let rr = R
                for (const L of lobes) rr *= 1 + L.a * Math.sin(L.k * th + L.p)
                const d = Math.hypot(dx, dy) - rr
                f[y * MS + x] = Math.max(-10, Math.min(10, -d * 1.2 + (rnd() - 0.5) * 0.35))
            }
            return f
        }
        const cubic = (t) => { const t2 = t*t, t3 = t2*t; return [-0.5*t3+t2-0.5*t, 1.5*t3-2.5*t2+1, -1.5*t3+2*t2+0.5*t, 0.5*t3-0.5*t2] }
        const ci = (i) => i < 0 ? 0 : (i > MS-1 ? MS-1 : i)
        const upsample = (lg, w, h) => {
            const out = new Float32Array(w*h), sx = MS/w, sy = MS/h
            const xi = new Int32Array(w*4), xw = new Float32Array(w*4)
            for (let x=0;x<w;x++){ const fx=(x+0.5)*sx-0.5, x0=Math.floor(fx), k=cubic(fx-x0)
                for(let t=0;t<4;t++){ xi[x*4+t]=ci(x0-1+t); xw[x*4+t]=k[t] } }
            const tmp = new Float32Array(MS*w)
            for (let r=0;r<MS;r++){ const s=r*MS, d=r*w
                for(let x=0;x<w;x++){ const bb=x*4
                    tmp[d+x]=lg[s+xi[bb]]*xw[bb]+lg[s+xi[bb+1]]*xw[bb+1]+lg[s+xi[bb+2]]*xw[bb+2]+lg[s+xi[bb+3]]*xw[bb+3] } }
            let mnX=w,mnY=h,mxX=-1,mxY=-1
            for (let y=0;y<h;y++){ const fy=(y+0.5)*sy-0.5, y0=Math.floor(fy), k=cubic(fy-y0)
                const r0=ci(y0-1)*w,r1=ci(y0)*w,r2=ci(y0+1)*w,r3=ci(y0+2)*w, row=y*w
                for(let x=0;x<w;x++){ const v=tmp[r0+x]*k[0]+tmp[r1+x]*k[1]+tmp[r2+x]*k[2]+tmp[r3+x]*k[3]
                    out[row+x]=v
                    if(v>-6&&v<6){ if(x<mnX)mnX=x; if(x>mxX)mxX=x; if(y<mnY)mnY=y; if(y>mxY)mxY=y } } }
            return { field: out, bbox: mxX<0?null:[mnX,mnY,mxX,mxY] }
        }
        const timeIt = (fn, runs = 11, warm = 4) => {
            for (let i=0;i<warm;i++) fn()
            const t=[]
            for (let i=0;i<runs;i++){ const a=performance.now(); fn(); t.push(performance.now()-a) }
            t.sort((x,y)=>x-y); return t[t.length>>1]
        }
        const iou = (a, bb) => { let i=0,u=0
            for (let k=0;k<a.length;k+=4){ const pa=a[k]>=128, pb=bb[k]>=128; if(pa&&pb)i++; if(pa||pb)u++ }
            return u ? i/u : 1 }

        const rows = []
        for (const cover of [0.001, 0.02, 0.1, 0.35, 0.7]) {
            const { field: f0, bbox } = upsample(makeLogits(cover), W, H)
            const tOrig = timeIt(() => { const f = f0.slice(); orig.refineField(f, px, W, H, bbox, { radius:8, eps:1e-4, scale:4 }) })
            const tOpt  = timeIt(() => { const f = f0.slice(); opt.refineField(f, px, W, H, bbox, { radius:8, eps:1e-4, scale:4 }) })
            const bOrig = timeIt(() => orig.bandAlpha(f0, W, H, 0.9))
            const bOpt  = timeIt(() => opt.bandAlpha(f0, W, H, 0.9))
            // agreement of the thresholded masks
            const fa = f0.slice(); orig.refineField(fa, px, W, H, bbox, { radius:8, eps:1e-4, scale:4 })
            const fb = f0.slice(); opt.refineField(fb, px, W, H, bbox, { radius:8, eps:1e-4, scale:4 })
            const fe = f0.slice(); orig.refineField(fe, px, W, H, bbox, { radius:8, eps:1e-4, scale:1 })
            const mE = orig.bandAlpha(fe, W, H, 0.9)
            rows.push({ cover,
                refineOrig: +tOrig.toFixed(2), refineOpt: +tOpt.toFixed(2),
                bandOrig: +bOrig.toFixed(2), bandOpt: +bOpt.toFixed(2),
                accOrig: +iou(orig.bandAlpha(fa, W, H, 0.9), mE).toFixed(4),
                accOpt:  +iou(opt.bandAlpha(fb, W, H, 0.9), mE).toFixed(4) })
        }
        return JSON.stringify({ W, H, rows })
    `)

    const { W, H, rows } = JSON.parse(out)
    console.log(`\nV8 / real page, proxy ${W}×${H}, guide = the app's own decoded pixels\n`)
    console.log('cover   refine_orig  refine_opt   speedup   band_orig  band_opt  speedup   acc_orig  acc_opt')
    for (const r of rows) {
        console.log([String(r.cover).padEnd(6),
            r.refineOrig.toFixed(2).padStart(11), r.refineOpt.toFixed(2).padStart(12),
            `${(r.refineOrig / r.refineOpt).toFixed(2)}x`.padStart(10),
            r.bandOrig.toFixed(2).padStart(11), r.bandOpt.toFixed(2).padStart(10),
            `${(r.bandOrig / r.bandOpt).toFixed(2)}x`.padStart(9),
            r.accOrig.toFixed(4).padStart(11), r.accOpt.toFixed(4).padStart(9)].join(''))
    }
} finally {
    await b.close()
    rmSync(staged, { force: true })
}
