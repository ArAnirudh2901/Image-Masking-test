/**
 * Cost of the selection-outline overlay when a GRADE changes.
 *
 * A texture-backed mask (AI / brush) has no vector outline, so in Adjust mode the
 * overlay traces one: threshold the coverage into a silhouette, dilate it around 16
 * offsets, punch the interior back out. That is a getImageData, a per-pixel JS pass
 * and 17 canvas draws at mask resolution, and the overlay effect re-runs on every
 * `chain` change — so dragging a gamma slider on a selected AI mask repeats the
 * whole trace per frame while the texture never changes.
 *
 * This drives that exact path (repeated setGamma, which is what a slider drag emits)
 * and reports the per-update cost, so memoising the trace can be justified or
 * dropped on evidence rather than on how expensive it looks.
 *
 *   bun run bench/overlay.mjs [image]
 */
import { launch } from './cdp.mjs'
import path from 'node:path'

const BASE = process.env.URL || 'http://127.0.0.1:8811/'
const CORPUS = path.resolve(import.meta.dirname, 'corpus')
const IMG = process.argv[2] || '2680558334.nef'

const b = await launch({ headless: process.env.HEAD !== '1' })
try {
    await b.send('Emulation.setDeviceMetricsOverride', { width: 1728, height: 1080, deviceScaleFactor: 2, mobile: false })
    await b.goto(BASE)
    await b.evaluate(`
        for (let i = 0; i < 300; i++) { if (window.__ready) return true; await new Promise(r => setTimeout(r, 100)) }
        throw new Error('not ready')
    `)
    await b.setFiles('input[type=file]', [path.join(CORPUS, IMG)])
    const out = await b.evaluate(`
        const S = new Proxy({}, { get: (_, k) => window.__studio[k] })
        for (let i = 0; i < 1200; i++) {
            const s = S.imageSize(); if (s && s.width) break
            await new Promise(r => setTimeout(r, 25))
        }
        for (let i = 0; i < 900; i++) { if (!S.aiState().busy) break; await new Promise(r => setTimeout(r, 25)) }
        await S.runSubject()
        for (let i = 0; i < 900; i++) { if (!S.aiState().busy) break; await new Promise(r => setTimeout(r, 25)) }
        const c = S.chain().find(e => e.kind === 'semantic')
        if (!c) return null
        const { width: W, height: H } = S.imageSize()

        // Time N grade updates, each flushed to a real paint. Long tasks attribute
        // the synchronous part; the double-rAF wait sets a 33 ms floor on the wall
        // figure, so the long-task total is the number that discriminates.
        const runs = 24
        const measure = async (label) => {
            const tasks = []
            const obs = new PerformanceObserver((l) => { for (const e of l.getEntries()) tasks.push(e.duration) })
            for (let i = 0; i < 4; i++) { S.setGamma(c.id, 1 + i * 1e-3); await new Promise(r => requestAnimationFrame(r)) }
            obs.observe({ entryTypes: ['longtask'] })
            const t0 = performance.now()
            for (let i = 0; i < runs; i++) {
                S.setGamma(c.id, 1.05 + i * 1e-3)
                await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))
            }
            const wall = performance.now() - t0
            obs.disconnect()
            return { label, wallPer: +(wall / runs).toFixed(1),
                blocked: +tasks.reduce((a, x) => a + x, 0).toFixed(0),
                worst: +Math.max(0, ...tasks).toFixed(0), n: tasks.length }
        }

        const rows = []
        S.select(c.id)
        S.setFillMode(c.id, 'fill')
        await new Promise(r => setTimeout(r, 400))
        rows.push(await measure('fill mode (outline skipped)'))
        S.setFillMode(c.id, 'adjust')
        await new Promise(r => setTimeout(r, 400))
        rows.push(await measure('adjust mode (outline traced)'))
        S.select(null)
        await new Promise(r => setTimeout(r, 400))
        rows.push(await measure('nothing selected'))
        return JSON.stringify({ W, H, rows })
    `)
    if (!out) { console.log('no semantic layer produced — cannot measure'); }
    else {
        const { W, H, rows } = JSON.parse(out)
        console.log(`\n${IMG} → proxy ${W}×${H}, 24 gamma updates each\n`)
        console.log('state                          ms/update   blocked ms   worst   longTasks')
        for (const r of rows) {
            console.log([r.label.padEnd(31), String(r.wallPer).padStart(9),
                String(r.blocked).padStart(13), String(r.worst).padStart(8),
                String(r.n).padStart(12)].join(''))
        }
    }
    const errs = b.consoleLines().filter((l) => /error|Error/.test(l))
    if (errs.length) console.log('\nconsole errors:\n' + errs.slice(0, 6).join('\n'))
} finally {
    await b.close()
}
