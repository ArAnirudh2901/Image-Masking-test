/**
 * CPU-profile a full-resolution export and attribute self-time by function.
 *
 * Export is the one stage a WebGPU compute port could plausibly win (45 MP, no
 * interactivity, one round-trip amortized), but `workflow.mjs` reports it as a
 * single number. This says what that number is made of — and how much of it
 * blocks the main thread, since sam21HdCompose still runs there.
 *
 *   bun run bench/profexport.mjs [image]
 */
import { launch } from './cdp.mjs'
import path from 'node:path'

const BASE = process.env.URL || 'http://127.0.0.1:8811/'
const CORPUS = path.resolve(import.meta.dirname, 'corpus')
const IMG = process.argv[2] || '2680558334.nef'

const b = await launch({ headless: process.env.HEAD !== '1' })

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
        self.set(`${name}  ${where}`, (self.get(`${name}  ${where}`) || 0) + (timeDeltas[i] || 0) / 1000)
    }
    const total = [...self.values()].reduce((a, c) => a + c, 0)
    return { total, rows: [...self.entries()].sort((a, c) => c[1] - a[1]) }
}

try {
    await b.send('Emulation.setDeviceMetricsOverride', { width: 1728, height: 1080, deviceScaleFactor: 2, mobile: false })
    await b.goto(BASE)
    await b.evaluate(`
        for (let i = 0; i < 300; i++) { if (window.__ready) return true; await new Promise(r => setTimeout(r, 100)) }
        throw new Error('not ready')
    `)
    await b.setFiles('input[type=file]', [path.join(CORPUS, IMG)])

    // A selection has to exist before there is anything to export.
    const size = await b.evaluate(`
        const S = new Proxy({}, { get: (_, k) => window.__studio[k] })
        for (let i = 0; i < 1200; i++) { const s = S.imageSize(); if (s && s.width) break; await new Promise(r => setTimeout(r, 25)) }
        for (let i = 0; i < 3600; i++) { if (!S.aiState().busy) break; await new Promise(r => setTimeout(r, 25)) }
        const { width: W, height: H } = S.imageSize()
        await S.samBox(W * 0.25, H * 0.25, W * 0.75, H * 0.75)
        for (let i = 0; i < 60; i++) { if (S.aiState().busy) break; await new Promise(r => setTimeout(r, 10)) }
        for (let i = 0; i < 3600; i++) { if (!S.aiState().busy) break; await new Promise(r => setTimeout(r, 25)) }
        return JSON.stringify(S.imageSize())
    `)
    console.log(`${IMG} → proxy ${size}`)

    await b.send('Profiler.enable')
    await b.send('Profiler.setSamplingInterval', { interval: 100 })
    await b.send('Profiler.start')
    const wall = await b.evaluate(`
        const S = new Proxy({}, { get: (_, k) => window.__studio[k] })
        const tasks = []
        new PerformanceObserver((l) => { for (const e of l.getEntries()) tasks.push(+e.duration.toFixed(1)) })
            .observe({ entryTypes: ['longtask'] })
        const t = performance.now()
        await S.exportHd()
        for (let i = 0; i < 3600; i++) { if (!S.aiState().busy) break; await new Promise(r => setTimeout(r, 25)) }
        return JSON.stringify({ ms: +(performance.now() - t).toFixed(0), longTasks: tasks.sort((a, b) => b - a).slice(0, 6) })
    `)
    const { profile } = await b.send('Profiler.stop')

    const { total, rows } = attribute(profile)
    const w = JSON.parse(wall)
    console.log(`\nexport wall ${w.ms} ms · main-thread long tasks: ${w.longTasks.length ? w.longTasks.join(', ') : 'none'}`)
    console.log(`\n── export self-time — ${total.toFixed(0)} ms of samples ──`)
    console.log('self ms    %     function')
    for (const [k, ms] of rows.slice(0, 16)) {
        if (ms < total * 0.01) break
        console.log(`${ms.toFixed(1).padStart(7)}  ${((ms / total) * 100).toFixed(1).padStart(5)}%  ${k}`)
    }

    const errs = b.consoleLines().filter((l) => /error|Error/.test(l))
    if (errs.length) console.log('\nconsole errors:\n' + errs.slice(0, 6).join('\n'))
} finally {
    await b.close()
}
