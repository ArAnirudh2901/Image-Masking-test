/**
 * Where AI Subject actually LANDS, per corpus image.
 *
 * e2e.mjs asserts a mask came back; this asserts it came back on the right
 * thing. Reports the mask's bbox, centre of mass and how far that centre sits
 * from the frame's — which is what separates "the subject" from "a fold in the
 * top-left corner" — plus the probe that won and the wall time.
 *
 * Run it on both sides of a change; the columns are meant to be diffed.
 *
 *   PORT=8811 bun run serve.mjs &
 *   bun run bench/subject.mjs [image...]
 */
import { launch } from './cdp.mjs'
import path from 'node:path'
import { readdirSync } from 'node:fs'

const BASE = process.env.URL || 'http://127.0.0.1:8811/'
const CORPUS = path.resolve(import.meta.dirname, 'corpus')
const ARGS = process.argv.slice(2)
const IMAGES = ARGS.length ? ARGS : readdirSync(CORPUS)
    .filter((f) => /\.(nef|arw|jpe?g|png)$/i.test(f) && !/_prev\./.test(f)).sort()

const b = await launch({ headless: process.env.HEADED ? false : true })
const rows = []
try {
    for (const img of IMAGES) {
        // A fresh page per image: the lane caches embeddings and the prior is
        // per-import, so a warm page would measure the wrong thing.
        await b.goto(BASE)
        await b.evaluate(`
            for (let i = 0; i < 300; i++) { if (window.__ready) return true; await new Promise(r => setTimeout(r, 100)) }
            throw new Error('not ready')
        `)
        await b.setFiles('input[type=file]', [path.join(CORPUS, img)])
        const r = await b.evaluate(`
            const S = new Proxy({}, { get: (_, k) => window.__studio[k] })
            for (let i = 0; i < 1200; i++) {
                const s = S.imageSize(); if (s && s.width) break
                await new Promise(r => setTimeout(r, 25))
            }
            // The eager encode fires on import; measure a warm decode, not it.
            for (let i = 0; i < 3600; i++) { if (!S.aiState().busy) break; await new Promise(r => setTimeout(r, 25)) }
            const t0 = performance.now()
            await S.runSubject()
            // busy is not set when the call returns — wait for it to APPEAR.
            for (let i = 0; i < 60; i++) { if (S.aiState().busy) break; await new Promise(r => setTimeout(r, 10)) }
            for (let i = 0; i < 3600; i++) { if (!S.aiState().busy) break; await new Promise(r => setTimeout(r, 25)) }
            const ms = Math.round(performance.now() - t0)
            const st = S.aiState()
            return JSON.stringify({
                ms,
                status: st.status,
                lastRun: st.engine && st.engine.lastRun,
                mask: S.maskStats(),
                candidates: S.candidates ? S.candidates() : null,
                probes: window.__lastSubject || null,
            })
        `)
        const o = JSON.parse(r)
        o.image = img
        rows.push(o)
        const m = o.mask
        console.log(`${img.padEnd(22)} ${String(o.ms).padStart(6)} ms  `
            + (m && m.bbox
                ? `cov=${m.coverage.toFixed(3)} bbox=${m.bbox.join(',')} centroid=${m.centroid.join(',')} off=${m.offCentre}`
                : 'NO MASK')
            + (o.lastRun ? `  enc=${Math.round(o.lastRun.encodeMs)} dec=${Math.round(o.lastRun.decodeMs)} post=${Math.round(o.lastRun.postMs)}` : ''))
        console.log(`  ${o.status}`)
        if (o.probes) {
            for (const pr of o.probes.probes || []) {
                console.log(`    ${String(pr.name).padEnd(14)} ${String(pr.ms).padStart(6)} ms`
                    + (pr.rejected ? `  rejected: ${pr.rejected} (cov ${pr.coverage})`
                        : `  cov=${pr.coverage} sal=${pr.salienceFit} border=${pr.borderFrac} rank=${pr.rank}`))
            }
            console.log(`    ${'post'.padEnd(14)} ${String(o.probes.postMs).padStart(6)} ms  (once, on the winner: ${o.probes.won})`)
        }
    }
} finally { await b.close() }

const ok = rows.filter((r) => r.mask && r.mask.bbox)
console.log(`\n${ok.length}/${rows.length} produced a mask`
    + (ok.length ? ` · median AI Subject ${ok.map((r) => r.ms).sort((a, b) => a - b)[ok.length >> 1]} ms` : ''))
if (process.env.JSON) console.log(JSON.stringify(rows, null, 2))
process.exit(ok.length === rows.length ? 0 : 1)
