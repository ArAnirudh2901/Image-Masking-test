/**
 * Whole-workflow timing: open → encode → first mask → subsequent masks → export.
 *
 * Every other bench here looks at one function. This one asks where a user's time
 * actually goes, which is the only way to know whether optimizing a function is
 * worth doing. Stage timings come from the lane's own instrumentation (encodeMs /
 * decodeMs / postMs), so they attribute inside a selection too.
 *
 *   bun run bench/workflow.mjs [image...]
 */
import { launch } from './cdp.mjs'
import path from 'node:path'
import { readdirSync } from 'node:fs'

const BASE = process.env.URL || 'http://127.0.0.1:8811/'
const CORPUS = path.resolve(import.meta.dirname, 'corpus')
const ARGS = process.argv.slice(2)
const IMAGES = ARGS.length ? ARGS : readdirSync(CORPUS)
    .filter((f) => /\.(nef|arw|jpe?g)$/i.test(f) && !/_prev\./.test(f))

const b = await launch({ headless: process.env.HEAD !== '1' })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

try {
    await b.send('Emulation.setDeviceMetricsOverride', { width: 1728, height: 1080, deviceScaleFactor: 2, mobile: false })
    await b.goto(BASE)
    await b.evaluate(`
        for (let i = 0; i < 300; i++) { if (window.__ready) return true; await new Promise(r => setTimeout(r, 100)) }
        throw new Error('not ready')
    `)

    const all = []
    for (const img of IMAGES) {
        const t0 = performance.now()
        await b.setFiles('input[type=file]', [path.join(CORPUS, img)])
        const r = await b.evaluate(`
            const S = new Proxy({}, { get: (_, k) => window.__studio[k] })
            // aiState() is a snapshot taken when __studio was last rebuilt, and
            // React has not committed the busy latch yet when a tool call returns.
            // Polling straight for "not busy" therefore succeeds instantly and
            // measures nothing — wait for busy to APPEAR first, then to clear.
            const settle = async (started) => {
                for (let i = 0; i < 60; i++) {
                    if (S.aiState().busy) break
                    await new Promise(r => setTimeout(r, 10))
                }
                for (let i = 0; i < 3600; i++) {
                    if (!S.aiState().busy) return true
                    await new Promise(r => setTimeout(r, 25))
                }
                return false
            }
            const idle = async () => {
                for (let i = 0; i < 3600; i++) {
                    if (!S.aiState().busy) return true
                    await new Promise(r => setTimeout(r, 25))
                }
                return false
            }
            const out = { }

            // 1 · decode + proxy: how long until the working canvas exists
            const tOpen = performance.now()
            for (let i = 0; i < 1200; i++) {
                const s = S.imageSize()
                if (s && s.width) { out.size = s; break }
                await new Promise(r => setTimeout(r, 25))
            }
            out.openMs = +(performance.now() - tOpen).toFixed(0)
            if (!out.size) return JSON.stringify(out)

            // 2 · the eager encode the image load kicks off. Waiting for it here
            //     means the numbers below are warm-embedding numbers, which is
            //     what every click after the first one actually gets.
            const tWarm = performance.now()
            await idle()
            out.warmMs = +(performance.now() - tWarm).toFixed(0)

            // 3 · AI Subject — three decoder probes over one cached encode
            const tSub = performance.now()
            await S.runSubject()
            await settle()
            out.subjectMs = +(performance.now() - tSub).toFixed(0)
            const eng = S.aiState().engine
            out.device = eng?.device; out.gpuTier = eng?.gpuTier
            out.lastRun = eng?.lastRun || null

            // 4 · a single box-select on the warm embedding — the steady-state click
            const { width: W, height: H } = out.size
            const tBox = performance.now()
            await S.samBox(W * 0.25, H * 0.25, W * 0.75, H * 0.75)
            await settle()
            out.boxMs = +(performance.now() - tBox).toFixed(0)
            out.boxRun = S.aiState().engine?.lastRun || null

            // 5 · click-select, twice: first seeds a layer, second composites
            const tC1 = performance.now(); S.clickSelect(W * 0.5, H * 0.5, 1); await settle()
            out.click1Ms = +(performance.now() - tC1).toFixed(0)
            const tC2 = performance.now(); S.clickSelect(W * 0.4, H * 0.6, 1); await settle()
            out.click2Ms = +(performance.now() - tC2).toFixed(0)

            // 6 · full-resolution export
            const tX = performance.now()
            await S.exportHd()
            await idle()
            out.exportMs = +(performance.now() - tX).toFixed(0)

            out.layers = S.chain().length
            return JSON.stringify(out)
        `)
        const o = JSON.parse(r)
        o.image = img
        o.wallMs = Math.round(performance.now() - t0)
        all.push(o)
        console.log(`${img.padEnd(22)} ${o.size ? `${o.size.width}×${o.size.height}` : 'FAILED'}`)
        // fresh page per image so the next one is not measured warm
        await b.goto(BASE)
        await b.evaluate(`
            for (let i = 0; i < 300; i++) { if (window.__ready) return true; await new Promise(r => setTimeout(r, 100)) }
            throw new Error('not ready')
        `)
        await sleep(300)
    }

    console.log('\nimage                 proxy         open  warmEnc  subject   box  click1  click2  export   dev')
    for (const o of all) {
        if (!o.size) { console.log(`${o.image.padEnd(20)}  FAILED`); continue }
        console.log([
            o.image.padEnd(20),
            `${o.size.width}×${o.size.height}`.padStart(12),
            String(o.openMs).padStart(6),
            String(o.warmMs).padStart(9),
            String(o.subjectMs).padStart(9),
            String(o.boxMs).padStart(6),
            String(o.click1Ms).padStart(8),
            String(o.click2Ms).padStart(8),
            String(o.exportMs).padStart(8),
            ` ${o.device || '?'}`,
        ].join(''))
    }
    const s = all.find((o) => o.boxRun)
    if (s) {
        console.log(`\nsteady-state selection breakdown (${s.image}):`)
        console.log(`  encode ${s.boxRun.encodeMs} ms · decode ${s.boxRun.decodeMs} ms · post ${s.boxRun.postMs} ms · total ${s.boxRun.ms} ms`)
        const ps = s.boxRun.postStages
        if (ps) {
            console.log('  post breakdown: ' + Object.entries(ps)
                .sort((a, c) => c[1] - a[1])
                .filter(([, v]) => typeof v === 'number')
                .map(([k, v]) => `${k.replace(/Ms$/, '')} ${v}`).join(' · '))
            if (ps.refineShape) {
                const r = ps.refineShape
                console.log(`  refine shape: ${r.rects} rect(s), s=${r.s}, pad=${r.pad}, `
                    + `cells ${(r.cells / 1e6).toFixed(2)}M of ${(r.frame / 1e6).toFixed(2)}M frame `
                    + `(bbox ${(r.bboxArea / 1e6).toFixed(2)}M)`)
            }
        }
    }
    const errs = b.consoleLines().filter((l) => /error|Error/.test(l))
    if (errs.length) console.log('\nconsole errors:\n' + errs.slice(0, 8).join('\n'))
} finally {
    await b.close()
}
