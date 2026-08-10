/**
 * Deterministic attribution for the interaction path.
 *
 * interact.mjs proves a drag stalls; it does not say which of the three things a
 * pointer event triggers is responsible. This times each one in isolation, with
 * warmup and medians, against the real page state:
 *
 *   render   — the megashader effect (grade the whole proxy frame)
 *   overlay  — the selection-outline effect, per selected mask kind
 *   boundary — drawMaskBoundary on a texture mask (the AI/brush outline trace)
 *   rects    — forced layout reads per pointer event
 *
 *   bun run bench/uiprof.mjs [image]
 */
import { launch } from './cdp.mjs'
import path from 'node:path'

const BASE = process.env.URL || 'http://127.0.0.1:8811/'
const CORPUS = path.resolve(import.meta.dirname, 'corpus')
const IMG = process.argv[2] || 'dslr-cover.jpg'

const b = await launch({ headless: process.env.HEAD !== '1' })
try {
    await b.send('Emulation.setDeviceMetricsOverride', {
        width: 1728, height: 1080, deviceScaleFactor: 2, mobile: false,
    })
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
    await b.evaluate(`
        for (let i = 0; i < 900; i++) { if (!window.__studio.aiState().busy) break; await new Promise(r => setTimeout(r, 100)) }
        return true
    `)

    const out = await b.evaluate(`
        // __studio is rebuilt by an effect whose deps include the chain, so a
        // captured reference goes stale the instant a layer is added — every call
        // has to go through the live global.
        const S = new Proxy({}, { get: (_, k) => window.__studio[k] })
        const { width: W, height: H } = S.imageSize()

        // A layer param nudge is the smallest change that forces BOTH effects to
        // re-run, i.e. exactly what one pointer-move of a handle drag costs. Long
        // tasks attribute the synchronous portion; wall-to-next-frame would just
        // report the 16.7 ms rAF quantum and hide everything.
        const blockedBy = async (fn, runs = 9, warm = 3) => {
            const seen = []
            const obs = new PerformanceObserver((l) => {
                for (const e of l.getEntries()) seen.push(e.duration)
            })
            for (let i = 0; i < warm; i++) { await fn(i); await new Promise(r => requestAnimationFrame(r)) }
            obs.observe({ entryTypes: ['longtask'] })
            const t = []
            for (let i = 0; i < runs; i++) {
                const a = performance.now()
                await fn(i)
                // flush React's effect + the browser's paint
                await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))
                t.push(performance.now() - a)
            }
            obs.disconnect()
            t.sort((x, y) => x - y)
            return { median: +t[t.length >> 1].toFixed(1), longTasks: seen.length,
                     longMax: +Math.max(0, ...seen).toFixed(0) }
        }

        const rows = {}

        // 1 · radial selected: vector outline, no texture trace
        const rid = S.add('radial'); S.select(rid)
        await new Promise(r => setTimeout(r, 300))
        rows.radialParam = await blockedBy((i) => S.update(rid, { feather: 0.5 + i * 1e-4 }))
        rows.radialDrag  = await blockedBy((i) => S.update(rid, { center: { x: W * 0.5 + i, y: H * 0.5 } }))

        // 2 · an AI texture mask selected: adds drawMaskBoundary to every redraw
        await S.runSubject()
        for (let i = 0; i < 900; i++) { if (!S.aiState().busy) break; await new Promise(r => setTimeout(r, 100)) }
        const sem = S.chain().find(e => e.kind === 'semantic')
        let texSize = null
        if (sem) {
            S.select(sem.id)
            await new Promise(r => setTimeout(r, 300))
            rows.semanticParam = await blockedBy((i) => S.update(sem.id, { feather: 0.02 + i * 1e-4 }))
        }

        // 3 · forced layout reads per pointer event. Only counts during an ACTIVE
        //     drag — an idle move over a non-paint tool returns before it reads
        //     anything, so counting idle moves would report a misleading zero.
        const ov = document.querySelectorAll('canvas')[1] || document.querySelector('canvas')
        const orig = Element.prototype.getBoundingClientRect
        const r = orig.call(ov)
        let rects = 0
        Element.prototype.getBoundingClientRect = function () { rects++; return orig.call(this) }
        const ev = (type, x, y, buttons = 1) => ov.dispatchEvent(new PointerEvent(type, {
            bubbles: true, cancelable: true, pointerId: 1, buttons,
            clientX: x, clientY: y, isPrimary: true,
        }))
        // grab the radial centre (a move-drag), then stream moves
        S.select(rid)
        await new Promise(r2 => setTimeout(r2, 200))
        const cxp = r.x + r.width * 0.5, cyp = r.y + r.height * 0.5
        ev('pointerdown', cxp, cyp)
        rects = 0
        const MOVES = 20
        for (let i = 0; i < MOVES; i++) ev('pointermove', cxp + i, cyp)
        const rectsDrag = rects
        ev('pointerup', cxp + MOVES, cyp, 0)
        Element.prototype.getBoundingClientRect = orig

        return JSON.stringify({ W, H, rows, rectsPerMove: rectsDrag / MOVES,
            chain: S.chain().length })
    `)

    const { W, H, rows, rectsPerMove, chain } = JSON.parse(out)
    console.log(`\n${IMG} → proxy ${W}×${H}, ${chain} layer(s)\n`)
    console.log('what                              median ms   longTasks  worst ms')
    const label = {
        radialParam: 'radial: param nudge → paint',
        radialDrag: 'radial: handle move → paint',
        semanticParam: 'AI mask selected: nudge → paint',
    }
    for (const [k, v] of Object.entries(rows)) {
        if (!v) continue
        console.log([(label[k] || k).padEnd(34),
            String(v.median).padStart(9), String(v.longTasks).padStart(11),
            String(v.longMax).padStart(10)].join(''))
    }
    console.log(`\nforced layout reads per pointermove: ${rectsPerMove}`)
    const errs = b.consoleLines().filter((l) => /error|Error/.test(l))
    if (errs.length) console.log('\nconsole errors:\n' + errs.slice(0, 6).join('\n'))
} finally {
    await b.close()
}
