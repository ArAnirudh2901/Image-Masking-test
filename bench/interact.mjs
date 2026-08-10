/**
 * Interaction-latency bench: what a DRAG actually costs.
 *
 * Filter throughput is not what makes an editor feel slow — the per-pointer-event
 * cost is. This drives real CDP mouse events (not synthetic dispatch, so React's
 * own event plumbing and the browser's layout/paint are both in the measurement)
 * across a handle drag, a brush stroke and a brush-refine stroke, and reports the
 * frame-time distribution plus long-task total for each.
 *
 *   bun run bench/interact.mjs [image]
 */
import { launch } from './cdp.mjs'
import path from 'node:path'

const BASE = process.env.URL || 'http://127.0.0.1:8811/'
const CORPUS = path.resolve(import.meta.dirname, 'corpus')
const IMG = process.argv[2] || 'dslr-cover.jpg'

const b = await launch({ headless: process.env.HEAD !== '1' })

/**
 * Drag from a→b in `steps` moves. The moves are fired WITHOUT awaiting each
 * round-trip — awaiting spaces them one per frame, which is not a drag, it is a
 * slideshow, and it hides every coalescing failure this bench exists to find.
 * A real trackpad delivers 120 Hz into a queue; so does this.
 */
const drag = async (x0, y0, x1, y1, steps = 40) => {
    await b.send('Input.dispatchMouseEvent', { type: 'mousePressed', x: x0, y: y0, button: 'left', clickCount: 1, buttons: 1 })
    const moves = []
    for (let i = 1; i <= steps; i += 1) {
        const t = i / steps
        moves.push(b.send('Input.dispatchMouseEvent', {
            type: 'mouseMoved', button: 'left', buttons: 1,
            x: Math.round(x0 + (x1 - x0) * t), y: Math.round(y0 + (y1 - y0) * t),
        }))
    }
    await Promise.all(moves)
    await b.send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: x1, y: y1, button: 'left', buttons: 0 })
}

const stats = (a) => {
    if (!a.length) return null
    const s = [...a].sort((x, y) => x - y)
    return {
        n: s.length,
        p50: +s[s.length >> 1].toFixed(1),
        p95: +s[Math.min(s.length - 1, Math.floor(s.length * 0.95))].toFixed(1),
        max: +s[s.length - 1].toFixed(1),
    }
}

try {
    // A 360px-wide stage measures nothing — the working canvas is proxy-res
    // regardless, but the overlay and hit-test maths scale with the stage.
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
    if (!size) throw new Error('image never loaded')
    // Let the eager encode finish so it is not charged to the drag.
    await b.evaluate(`
        for (let i = 0; i < 600; i++) { if (!window.__studio.aiState().busy) break; await new Promise(r => setTimeout(r, 100)) }
        return true
    `)

    // Frame + long-task recorder, and a counter on the layout-read path so we can
    // show the thrash directly rather than infer it from timings.
    await b.evaluate(`
        window.__prof = { frames: [], tasks: [], rects: 0, moves: 0, on: false }
        const P = window.__prof
        window.addEventListener('pointermove', () => { if (P.on) P.moves++ }, true)
        const rAF = () => {
            let last = performance.now()
            const tick = (t) => { if (P.on) P.frames.push(t - last); last = t; requestAnimationFrame(tick) }
            requestAnimationFrame(tick)
        }
        rAF()
        new PerformanceObserver((l) => { if (P.on) for (const e of l.getEntries()) P.tasks.push(e.duration) })
            .observe({ entryTypes: ['longtask'] })
        const orig = Element.prototype.getBoundingClientRect
        Element.prototype.getBoundingClientRect = function () { if (P.on) P.rects++; return orig.call(this) }
        window.__profStart = () => { P.frames.length = 0; P.tasks.length = 0; P.rects = 0; P.moves = 0; P.on = true }
        window.__profStop = () => { P.on = false; return { frames: P.frames.slice(), tasks: P.tasks.slice(), rects: P.rects, moves: P.moves } }
        return true
    `)

    const rect = JSON.parse(await b.evaluate(`
        const r = document.querySelector('canvas').getBoundingClientRect()
        return JSON.stringify({ x: r.x, y: r.y, w: r.width, h: r.height })
    `))
    const cx = rect.x + rect.w / 2
    const cy = rect.y + rect.h / 2

    const scenarios = []

    /** Run one scenario: set up, profile the drag, tear down. */
    const run = async (name, setup, dragArgs, teardown = 'return true') => {
        await b.evaluate(setup)
        await new Promise((r) => setTimeout(r, 400))
        await b.evaluate('window.__profStart(); return true')
        const t0 = performance.now()
        await drag(...dragArgs)
        // Drain: wait until the page goes two quiet frames without a long task.
        await b.evaluate(`
            const P = window.__prof
            let n = P.tasks.length
            for (let i = 0; i < 200; i++) {
                await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))
                if (P.tasks.length === n) break
                n = P.tasks.length
            }
            return true
        `)
        const wall = performance.now() - t0
        const out = await b.evaluate('return JSON.stringify(window.__profStop())')
        const { frames, tasks, rects, moves } = JSON.parse(out)
        scenarios.push({
            name,
            frame: stats(frames),
            taskMs: +tasks.reduce((a, c) => a + c, 0).toFixed(0),
            taskMax: +Math.max(0, ...tasks).toFixed(0),
            rects, moves, wall: +wall.toFixed(0),
        })
        await b.evaluate(teardown)
    }

    // 1 · radial handle drag — the parametric-control path
    await run('radial handle drag',
        `const id = window.__studio.add('radial'); window.__studio.select(id); window.__radial = id; return id`,
        [cx + rect.w * 0.3, cy, cx + rect.w * 0.42, cy - rect.h * 0.1, 40])

    // 2 · brush stroke — the paint path (new layer, alpha coverage)
    await run('brush stroke',
        `document.querySelectorAll('button').forEach(bt => { if (bt.textContent.trim() === 'Brush') bt.click() }); return true`,
        [cx - rect.w * 0.2, cy - rect.h * 0.1, cx + rect.w * 0.2, cy + rect.h * 0.1, 40])

    // 3 · brush-refine over an AI mask — the path that also redraws the mask
    //     boundary overlay every frame
    const haveAi = await b.evaluate(`
        await window.__studio.runSubject()
        for (let i = 0; i < 900; i++) { if (!window.__studio.aiState().busy) break; await new Promise(r => setTimeout(r, 100)) }
        const c = window.__studio.chain().find(e => e.kind === 'semantic')
        if (!c) return null
        window.__studio.select(c.id); window.__studio.refine()
        return c.id
    `)
    if (haveAi) {
        await run('brush-refine on AI mask', 'return true',
            [cx - rect.w * 0.15, cy, cx + rect.w * 0.15, cy + rect.h * 0.08, 40])
    }

    // 4 · isolated: what one overlay boundary redraw costs on the AI mask
    const overlay = haveAi ? await b.evaluate(`
        const c = window.__studio.chain().find(e => e.kind === 'semantic')
        if (!c) return null
        window.__studio.select(c.id)
        const t = []
        for (let i = 0; i < 12; i++) {
            const a = performance.now()
            window.__studio.update(c.id, { feather: 0.02 + i * 1e-6 })
            await new Promise(r => requestAnimationFrame(r))
            t.push(performance.now() - a)
        }
        t.sort((x, y) => x - y)
        return t[t.length >> 1].toFixed(1)
    `) : null

    console.log(`\nimage ${IMG} → ${size}   stage ${Math.round(rect.w)}×${Math.round(rect.h)}\n`)
    console.log('scenario                    moves  wallms  p50ms  p95ms  maxms  blockedms  worst  rects  rect/move')
    for (const s of scenarios) {
        console.log([
            s.name.padEnd(28),
            String(s.moves).padStart(5),
            String(s.wall).padStart(8),
            String(s.frame?.p50 ?? '-').padStart(7),
            String(s.frame?.p95 ?? '-').padStart(7),
            String(s.frame?.max ?? '-').padStart(7),
            String(s.taskMs).padStart(11),
            String(s.taskMax).padStart(7),
            String(s.rects).padStart(7),
            (s.moves ? (s.rects / s.moves).toFixed(1) : '-').padStart(11),
        ].join(''))
    }
    if (overlay) console.log(`\nselected-mask param update → paint: ${overlay} ms median`)
    const errs = b.consoleLines().filter((l) => /error|Error/.test(l))
    if (errs.length) console.log('\nconsole errors:\n' + errs.slice(0, 6).join('\n'))
} finally {
    await b.close()
}
