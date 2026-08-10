/**
 * CPU-profile a real drag and attribute self-time by function.
 *
 * The rAF/long-task approach has a 33 ms floor (double-rAF) and a 50 ms trigger,
 * so it can prove a drag stalls but cannot say which function did it. V8's
 * sampling profiler can: this drives real CDP mouse input, samples at 100 µs, and
 * sums self-time per (function, file:line). The bundle is unminified, so the
 * names are the real ones.
 *
 *   bun run bench/cpuprof.mjs [image] [scenario]
 *     scenario: radial | brush | refine | all   (default all)
 */
import { launch } from './cdp.mjs'
import path from 'node:path'

const BASE = process.env.URL || 'http://127.0.0.1:8811/'
const CORPUS = path.resolve(import.meta.dirname, 'corpus')
const IMG = process.argv[2] || 'dslr-cover.jpg'
const WANT = process.argv[3] || 'all'

const b = await launch({ headless: process.env.HEAD !== '1' })

const drag = async (x0, y0, x1, y1, steps = 60) => {
    await b.send('Input.dispatchMouseEvent', { type: 'mousePressed', x: Math.round(x0), y: Math.round(y0), button: 'left', clickCount: 1, buttons: 1 })
    const moves = []
    for (let i = 1; i <= steps; i += 1) {
        const t = i / steps
        moves.push(b.send('Input.dispatchMouseEvent', {
            type: 'mouseMoved', button: 'left', buttons: 1,
            x: Math.round(x0 + (x1 - x0) * t), y: Math.round(y0 + (y1 - y0) * t),
        }))
    }
    await Promise.all(moves)
    await b.send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: Math.round(x1), y: Math.round(y1), button: 'left', buttons: 0 })
}

/** Sum self-time per node from a V8 .cpuprofile, hottest first. */
const attribute = (profile) => {
    const { nodes, samples, timeDeltas } = profile
    const byId = new Map(nodes.map((n) => [n.id, n]))
    const self = new Map()
    for (let i = 0; i < samples.length; i += 1) {
        const n = byId.get(samples[i])
        if (!n) continue
        const cf = n.callFrame
        const name = cf.functionName || '(anonymous)'
        const where = cf.url ? `${cf.url.split('/').pop()}:${cf.lineNumber + 1}` : '(native)'
        const key = `${name}  ${where}`
        self.set(key, (self.get(key) || 0) + (timeDeltas[i] || 0) / 1000)
    }
    const total = [...self.values()].reduce((a, c) => a + c, 0)
    return { total, rows: [...self.entries()].sort((a, c) => c[1] - a[1]) }
}

const report = (name, prof) => {
    const { total, rows } = attribute(prof)
    console.log(`\n── ${name} — ${total.toFixed(0)} ms of samples ──`)
    console.log('self ms    %     function')
    for (const [k, ms] of rows.slice(0, 12)) {
        if (ms < total * 0.01) break
        console.log(`${ms.toFixed(1).padStart(7)}  ${((ms / total) * 100).toFixed(1).padStart(5)}%  ${k}`)
    }
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
    const rect = JSON.parse(await b.evaluate(`
        const r = document.querySelector('canvas').getBoundingClientRect()
        return JSON.stringify({ x: r.x, y: r.y, w: r.width, h: r.height })
    `))
    console.log(`${IMG} → proxy ${size}   stage ${Math.round(rect.w)}×${Math.round(rect.h)}`)
    const cx = rect.x + rect.w / 2, cy = rect.y + rect.h / 2

    await b.send('Profiler.enable')
    await b.send('Profiler.setSamplingInterval', { interval: 100 })

    const profileDrag = async (label, setup, args) => {
        await b.evaluate(setup)
        await new Promise((r) => setTimeout(r, 600))
        await b.send('Profiler.start')
        await drag(...args)
        await new Promise((r) => setTimeout(r, 300))
        const { profile } = await b.send('Profiler.stop')
        report(label, profile)
    }

    if (WANT === 'all' || WANT === 'radial') {
        await profileDrag('radial handle drag',
            `const id = window.__studio.add('radial'); window.__studio.select(id); window.__rid = id; return id`,
            [cx + rect.w * 0.28, cy, cx + rect.w * 0.40, cy - rect.h * 0.12, 60])
    }

    if (WANT === 'all' || WANT === 'brush') {
        await profileDrag('brush stroke',
            `document.querySelectorAll('button').forEach(bt => { if (bt.textContent.trim() === 'Brush') bt.click() }); return true`,
            [cx - rect.w * 0.22, cy - rect.h * 0.12, cx + rect.w * 0.22, cy + rect.h * 0.12, 60])
    }

    if (WANT === 'all' || WANT === 'refine') {
        const ok = await b.evaluate(`
            const S = new Proxy({}, { get: (_, k) => window.__studio[k] })
            await S.runSubject()
            for (let i = 0; i < 900; i++) { if (!S.aiState().busy) break; await new Promise(r => setTimeout(r, 100)) }
            const c = S.chain().find(e => e.kind === 'semantic')
            if (!c) return null
            S.select(c.id); S.refine()
            return c.id
        `)
        if (ok) {
            await profileDrag('brush-refine over AI mask', 'return true',
                [cx - rect.w * 0.18, cy, cx + rect.w * 0.18, cy + rect.h * 0.10, 60])
        } else console.log('\n(no semantic layer — refine scenario skipped)')
    }

    const errs = b.consoleLines().filter((l) => /error|Error/.test(l))
    if (errs.length) console.log('\nconsole errors:\n' + errs.slice(0, 6).join('\n'))
} finally {
    await b.close()
}
