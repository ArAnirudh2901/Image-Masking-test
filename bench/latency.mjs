/**
 * Input → paint latency, per pointer event, during a sustained drag.
 *
 * This is the number that decides whether a control "feels smooth"; throughput
 * benches and long-task counters both miss it. Moves are paced at a real pointer
 * rate (Chrome coalesces to one per frame anyway, so flooding measures nothing),
 * and each one is stamped on arrival and again in the rAF that follows the paint
 * it caused.
 *
 *   bun run bench/latency.mjs [image]
 */
import { launch } from './cdp.mjs'
import path from 'node:path'

const BASE = process.env.URL || 'http://127.0.0.1:8811/'
const CORPUS = path.resolve(import.meta.dirname, 'corpus')
const IMG = process.argv[2] || 'dslr-cover.jpg'
const HZ = Number(process.env.HZ || 120)

const b = await launch({ headless: process.env.HEAD !== '1' })

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

/** Drag paced at HZ, so the page sees the same cadence a trackpad produces. */
const drag = async (x0, y0, x1, y1, steps = 60) => {
    const gap = 1000 / HZ
    await b.send('Input.dispatchMouseEvent', { type: 'mousePressed', x: Math.round(x0), y: Math.round(y0), button: 'left', clickCount: 1, buttons: 1 })
    for (let i = 1; i <= steps; i += 1) {
        const t = i / steps
        const at = performance.now()
        b.send('Input.dispatchMouseEvent', {
            type: 'mouseMoved', button: 'left', buttons: 1,
            x: Math.round(x0 + (x1 - x0) * t), y: Math.round(y0 + (y1 - y0) * t),
        }).catch(() => {})
        const left = gap - (performance.now() - at)
        if (left > 0) await sleep(left)
    }
    await b.send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: Math.round(x1), y: Math.round(y1), button: 'left', buttons: 0 })
    await sleep(300)
}

const pct = (a, p) => {
    if (!a.length) return 0
    const s = [...a].sort((x, y) => x - y)
    return +s[Math.min(s.length - 1, Math.floor(s.length * p))].toFixed(1)
}

try {
    await b.send('Emulation.setDeviceMetricsOverride', { width: 1728, height: 1080, deviceScaleFactor: 2, mobile: false })
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

    // Per-move arrival→paint stamps, plus main-thread busy time per frame. Busy
    // is measured as (frame start - previous frame end): everything the main
    // thread did between two rAF callbacks, which is what steals the budget.
    await b.evaluate(`
        const P = window.__lat = { on: false, moves: [], frames: [], pending: [] }
        window.addEventListener('pointermove', () => {
            if (P.on) P.pending.push(performance.now())
        }, true)
        const tick = (t) => {
            if (P.on) {
                if (P.last !== undefined) P.frames.push(t - P.last)
                P.last = t
                // every move that arrived before this frame was served by it
                for (const at of P.pending) P.moves.push(t - at)
                P.pending.length = 0
            }
            requestAnimationFrame(tick)
        }
        requestAnimationFrame(tick)
        window.__latStart = () => { P.moves.length = 0; P.frames.length = 0; P.pending.length = 0; P.last = undefined; P.on = true }
        window.__latStop  = () => { P.on = false; return { moves: P.moves.slice(), frames: P.frames.slice() } }
        return true
    `)

    const rect = JSON.parse(await b.evaluate(`
        const r = document.querySelector('canvas').getBoundingClientRect()
        return JSON.stringify({ x: r.x, y: r.y, w: r.width, h: r.height })
    `))
    const cx = rect.x + rect.w / 2, cy = rect.y + rect.h / 2
    console.log(`${IMG} → proxy ${size}   stage ${Math.round(rect.w)}×${Math.round(rect.h)}   input ${HZ} Hz\n`)

    const rows = []
    const scenario = async (name, setup, args) => {
        const ok = await b.evaluate(setup)
        if (ok === null) { console.log(`(skipped ${name})`); return }
        await sleep(700)
        await b.evaluate('window.__studio.resetRenderMetrics(); window.__latStart(); return true')
        await drag(...args)
        const { moves, frames } = JSON.parse(await b.evaluate('return JSON.stringify(window.__latStop())'))
        const rm = JSON.parse(await b.evaluate('return JSON.stringify(window.__studio.renderMetrics())'))
        const budget = 1000 / 60
        rows.push({
            name, n: moves.length,
            l50: pct(moves, 0.5), l95: pct(moves, 0.95), lmax: pct(moves, 1),
            f50: pct(frames, 0.5), f95: pct(frames, 0.95), fmax: pct(frames, 1),
            dropped: frames.filter((f) => f > budget * 1.5).length,
            draws: rm.drawCount, frames_n: frames.length,
        })
    }

    await scenario('radial handle drag',
        `const id = window.__studio.add('radial'); window.__studio.select(id); return id`,
        [cx + rect.w * 0.28, cy, cx + rect.w * 0.40, cy - rect.h * 0.12, 60])

    await scenario('brush stroke',
        `document.querySelectorAll('button').forEach(bt => { if (bt.textContent.trim() === 'Brush') bt.click() }); return true`,
        [cx - rect.w * 0.22, cy - rect.h * 0.12, cx + rect.w * 0.22, cy + rect.h * 0.12, 60])

    await scenario('box-select rubber-band',
        `document.querySelectorAll('button').forEach(bt => { if (bt.textContent.trim() === 'AI Box-Select') bt.click() }); return true`,
        [cx - rect.w * 0.2, cy - rect.h * 0.2, cx + rect.w * 0.2, cy + rect.h * 0.2, 60])

    await scenario('brush-refine over AI mask', `
        const S = new Proxy({}, { get: (_, k) => window.__studio[k] })
        await S.runSubject()
        for (let i = 0; i < 900; i++) { if (!S.aiState().busy) break; await new Promise(r => setTimeout(r, 100)) }
        const c = S.chain().find(e => e.kind === 'semantic')
        if (!c) return null
        S.select(c.id); S.refine()
        return c.id
    `, [cx - rect.w * 0.18, cy, cx + rect.w * 0.18, cy + rect.h * 0.10, 60])

    // The boundary-trace path. A texture mask defaults to fillMode 'fill', where
    // the outline is skipped entirely; it only gets traced in Adjust mode, and then
    // once per overlay redraw — so this is the scenario that exercises it, and the
    // only one where memoising the trace can matter.
    await scenario('brush-refine, Adjust mode', `
        const S = new Proxy({}, { get: (_, k) => window.__studio[k] })
        const c = S.chain().find(e => e.kind === 'semantic')
        if (!c) return null
        S.setFillMode(c.id, 'adjust')
        S.select(c.id); S.refine()
        return c.id
    `, [cx - rect.w * 0.18, cy + rect.h * 0.05, cx + rect.w * 0.18, cy + rect.h * 0.15, 60])

    console.log('scenario                     moves   lat50  lat95  latmax   frame50 frame95 framemax  janky  draws/frames')
    for (const r of rows) {
        console.log([r.name.padEnd(29),
            String(r.n).padStart(5),
            String(r.l50).padStart(8), String(r.l95).padStart(7), String(r.lmax).padStart(8),
            String(r.f50).padStart(10), String(r.f95).padStart(8), String(r.fmax).padStart(9),
            String(r.dropped).padStart(7),
            `  ${r.draws}/${r.frames_n}`.padStart(14)].join(''))
    }
    console.log('\nlat* = pointer arrival → the frame that served it (ms). janky = frames over 25 ms.')
    const errs = b.consoleLines().filter((l) => /error|Error/.test(l))
    if (errs.length) console.log('\nconsole errors:\n' + errs.slice(0, 6).join('\n'))
} finally {
    await b.close()
}
