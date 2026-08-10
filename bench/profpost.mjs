/**
 * CPU-profile the post-processing pass of a real selection, by function.
 *
 * post is the largest stage of a steady-state click, and its own stage timers say
 * refine owns most of it — but refine is a dozen passes over the frame and the
 * timers cannot say which. This samples V8 at 100 µs across a real box-select and
 * sums self-time per function, so the answer is attributable rather than modelled.
 *
 *   bun run bench/profpost.mjs [image]
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
    const size = JSON.parse(await b.evaluate(`
        const S = new Proxy({}, { get: (_, k) => window.__studio[k] })
        for (let i = 0; i < 1200; i++) {
            const s = S.imageSize()
            if (s && s.width) return JSON.stringify(s)
            await new Promise(r => setTimeout(r, 25))
        }
        return 'null'
    `))
    // warm the embedding so the profile is post-processing, not encoding
    await b.evaluate(`
        const S = new Proxy({}, { get: (_, k) => window.__studio[k] })
        const { width: W, height: H } = S.imageSize()
        await S.samBox(W * 0.25, H * 0.25, W * 0.75, H * 0.75)
        for (let i = 0; i < 900; i++) { if (!S.aiState().busy) break; await new Promise(r => setTimeout(r, 25)) }
        return true
    `)

    await b.evaluate('window.__studio.resetRenderMetrics(); return true')
    await b.send('Profiler.enable')
    await b.send('Profiler.setSamplingInterval', { interval: 100 })
    await b.send('Profiler.start')
    // Several selections so the sample count is meaningful.
    await b.evaluate(`
        const S = new Proxy({}, { get: (_, k) => window.__studio[k] })
        const { width: W, height: H } = S.imageSize()
        for (let k = 0; k < 6; k++) {
            await S.samBox(W * (0.2 + k * 0.01), H * 0.25, W * 0.75, H * 0.75)
            for (let i = 0; i < 900; i++) { if (!S.aiState().busy) break; await new Promise(r => setTimeout(r, 20)) }
        }
        return true
    `)
    const { profile } = await b.send('Profiler.stop')
    const rm = JSON.parse(await b.evaluate('return JSON.stringify(window.__studio.renderMetrics())'))

    const byId = new Map(profile.nodes.map((n) => [n.id, n]))
    const self = new Map()
    for (let i = 0; i < profile.samples.length; i += 1) {
        const n = byId.get(profile.samples[i])
        if (!n) continue
        const cf = n.callFrame
        const key = `${cf.functionName || '(anonymous)'}  ${cf.url ? cf.url.split('/').pop() : ''}:${cf.lineNumber + 1}`
        self.set(key, (self.get(key) || 0) + (profile.timeDeltas[i] || 0) / 1000)
    }
    const rows = [...self.entries()].sort((a, c) => c[1] - a[1])
    const busy = rows.filter(([k]) => !/^\(idle\)|^\(program\)|^\(garbage/.test(k))
        .reduce((a, c) => a + c[1], 0)
    console.log(`\n${IMG} → ${size.width}×${size.height}, 6 selections, ${busy.toFixed(0)} ms of JS\n`)
    console.log('self ms   % of JS   function')
    for (const [k, ms] of rows) {
        if (/^\(idle\)/.test(k)) continue
        if (ms < busy * 0.015) break
        console.log(`${ms.toFixed(1).padStart(7)}  ${((ms / busy) * 100).toFixed(1).padStart(7)}%  ${k}`)
    }
    console.log(`\nrenderer over those 6 selections:`)
    console.log(`  draws ${rm.drawCount} · program cache ${rm.cacheHits} hit / ${rm.cacheMisses} miss`
        + ` · compiles ${rm.compileCount} taking ${rm.totalCompileMs.toFixed(0)} ms`
        + ` · evictions ${rm.evictions} · identity skips ${rm.identityShortCircuits}`)
} finally {
    await b.close()
}
