/**
 * End-to-end: the real app, in a real browser, on real DSLR files.
 *
 * Loads each image through the app's own file input (so the RAW path, proxy
 * policy, capability probe and SAM lane all run for real), then exercises every
 * AI mask tool through window.__studio and reports wall-clock timings.
 */
import { launch } from './cdp.mjs'
import path from 'node:path'

const BASE = process.env.URL || 'http://127.0.0.1:8811/'
const CORPUS = path.resolve(import.meta.dirname, './corpus')
const IMAGES = process.argv.slice(2).length ? process.argv.slice(2) : [
    'dslr-cover.jpg',          // 10 MB JPEG
    '2680558334.nef',          // 45 MP Nikon RAW
    '8250045242.arw',          // Sony RAW
]

const b = await launch({ headless: process.env.HEADED ? false : true })
const fail = []

try {
    await b.goto(BASE)
    await b.evaluate(`
        for (let i = 0; i < 300; i++) {
            if (window.__ready) return true
            await new Promise(r => setTimeout(r, 100))
        }
        throw new Error('window.__ready never set; window.__error=' + window.__error)
    `)
    const engine = await b.evaluate('return JSON.stringify(await window.__studio.engine())')
    console.log('engine:', engine)
    const budget = await b.evaluate('return JSON.stringify(await window.__studio.budget())')
    console.log('budget:', budget)

    for (const img of IMAGES) {
        const file = path.join(CORPUS, img)
        console.log(`\n──────── ${img} ────────`)
        await b.setFiles('input[type=file]', [file])

        const loaded = await b.evaluate(`
            for (let i = 0; i < 600; i++) {
                const s = window.__studio.imageSize()
                if (s && s.width) return JSON.stringify(s)
                await new Promise(r => setTimeout(r, 100))
            }
            return null
        `)
        if (!loaded) { fail.push(`${img}: never decoded`); console.log('  DECODE FAILED'); continue }
        console.log('  decoded proxy:', loaded)

        /**
         * Run one AI op.
         *
         * Two things the obvious harness gets wrong:
         *  - runAi() returns null on the spot while ai.busy is set, and loading an
         *    image kicks off an eager encode, so idle must be awaited or the call
         *    is a silent no-op.
         *  - commitAiMask REPLACES an existing layer of the same kind, so chain
         *    length is not a completion signal. The op is awaited directly (it is
         *    async all the way down); success is then read off the status line,
         *    which runAi sets to the error text on failure.
         */
        const op = async (label, call) => {
            const r = await b.evaluate(`
                for (let i = 0; i < 1200; i++) {
                    if (!window.__studio.aiState().busy) break
                    await new Promise(r => setTimeout(r, 50))
                }
                const t0 = performance.now()
                try { await window.__studio.${call} } catch (e) { return JSON.stringify({ err: String(e) }) }
                for (let i = 0; i < 200; i++) {          // let React flush
                    const s = window.__studio.aiState()
                    if (!s.busy && s.status) break
                    await new Promise(r => setTimeout(r, 25))
                }
                const s = window.__studio.aiState()
                return JSON.stringify({
                    ms: Math.round(performance.now() - t0),
                    status: s.status,
                    lastRun: s.engine && s.engine.lastRun,
                    chain: window.__studio.chain().length,
                })
            `)
            const o = JSON.parse(r)
            // Success statuses all report a result; runAi puts the error text here.
            const good = !o.err && /masked|Selected|coverage|selected|Snapped/i.test(o.status || '')
            if (!good) fail.push(`${img}/${label}: ${o.err || 'status=' + JSON.stringify(o.status)}`)
            const lr = o.lastRun
            console.log(`  ${label.padEnd(12)} ${String(o.ms).padStart(6)} ms  ${good ? 'ok ' : 'BAD'}`
                + (lr ? `  enc=${Math.round(lr.encodeMs)} dec=${Math.round(lr.decodeMs)} post=${Math.round(lr.postMs)}` : '')
                + `  layers=${o.chain}`)
            return o
        }

        await op('subject', 'runSubject()')
        await op('background', 'background()')
        const sz = JSON.parse(loaded)
        await op('clickSelect', `clickSelect(${Math.round(sz.width * 0.5)}, ${Math.round(sz.height * 0.5)})`)
        await op('samBox', `samBox(${Math.round(sz.width * 0.2)}, ${Math.round(sz.height * 0.2)}, ${Math.round(sz.width * 0.8)}, ${Math.round(sz.height * 0.8)})`)

        // the composite must actually produce non-trivial pixels
        const px = await b.evaluate(`
            const d = window.__studio.pixels()
            let nz = 0, sum = 0
            for (let i = 0; i < d.length; i += 4000) { if (d[i] || d[i+1] || d[i+2]) nz++; sum += d[i] }
            return JSON.stringify({ len: d.length, nonzeroSamples: nz, meanR: Math.round(sum / (d.length / 4000)) })
        `)
        console.log('  composite:', px)
        const pxo = JSON.parse(px)
        if (pxo.nonzeroSamples === 0) fail.push(`${img}: composite is blank`)
    }

    const errs = b.consoleLines().filter((l) => /error|Error|exception/i.test(l) && !/favicon/i.test(l))
    if (errs.length) {
        console.log('\nconsole errors:')
        for (const e of errs.slice(0, 15)) console.log('  ' + e)
    }
} finally {
    await b.close()
}

console.log(fail.length ? `\n${fail.length} E2E FAILURE(S):\n` + fail.map((f) => ' - ' + f).join('\n')
    : '\nE2E PASSED ✓')
process.exit(fail.length ? 1 : 0)
