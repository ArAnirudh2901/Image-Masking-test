/** Minimal CDP driver: launch Chrome, eval in the page, set files on an input. */
import { spawn } from 'node:child_process'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

export const launch = async ({ port = 9333, headless = true } = {}) => {
    const profile = mkdtempSync(path.join(tmpdir(), 'ms-cdp-'))
    const args = [
        `--remote-debugging-port=${port}`,
        `--user-data-dir=${profile}`,
        '--no-first-run', '--no-default-browser-check',
        // WebGPU in headless needs coaxing; the lane falls back to wasm if it fails
        '--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU',
        '--use-angle=metal',
        '--disable-dev-shm-usage',
        'about:blank',
    ]
    if (headless) args.unshift('--headless=new')
    const proc = spawn(CHROME, args, { stdio: 'ignore', detached: false })

    // wait for the debugger
    let target = null
    for (let i = 0; i < 100; i += 1) {
        try {
            const r = await fetch(`http://127.0.0.1:${port}/json/list`)
            const list = await r.json()
            target = list.find((t) => t.type === 'page')
            if (target) break
        } catch { /* not up yet */ }
        await new Promise((r) => setTimeout(r, 200))
    }
    if (!target) { proc.kill(); throw new Error('chrome did not expose a page target') }

    const ws = new WebSocket(target.webSocketDebuggerUrl)
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej })

    let id = 0
    const pending = new Map()
    const events = []
    ws.onmessage = (m) => {
        const msg = JSON.parse(m.data)
        if (msg.id && pending.has(msg.id)) {
            const { res, rej } = pending.get(msg.id)
            pending.delete(msg.id)
            msg.error ? rej(new Error(JSON.stringify(msg.error))) : res(msg.result)
        } else if (msg.method) events.push(msg)
    }
    const send = (method, params = {}) => new Promise((res, rej) => {
        const n = ++id
        pending.set(n, { res, rej })
        ws.send(JSON.stringify({ id: n, method, params }))
    })

    await send('Page.enable')
    await send('Runtime.enable')
    await send('Log.enable')

    /** Evaluate an async expression, returning its JSON value. */
    const evaluate = async (expr) => {
        const r = await send('Runtime.evaluate', {
            expression: `(async () => { ${expr} })()`,
            awaitPromise: true, returnByValue: true,
        })
        if (r.exceptionDetails) {
            throw new Error('page threw: ' + (r.exceptionDetails.exception?.description
                || r.exceptionDetails.text))
        }
        return r.result.value
    }

    const goto = async (url) => {
        await send('Page.navigate', { url })
        for (let i = 0; i < 200; i += 1) {
            const st = await evaluate('return document.readyState')
            if (st === 'complete') return
            await new Promise((r) => setTimeout(r, 100))
        }
    }

    /** Put real files on an <input type=file>. */
    const setFiles = async (selector, files) => {
        const doc = await send('DOM.getDocument')
        const node = await send('DOM.querySelector', { nodeId: doc.root.nodeId, selector })
        if (!node.nodeId) throw new Error(`no node for ${selector}`)
        await send('DOM.setFileInputFiles', { files, nodeId: node.nodeId })
    }

    const consoleLines = () => events
        .filter((e) => e.method === 'Runtime.consoleAPICalled' || e.method === 'Log.entryAdded')
        .map((e) => {
            if (e.method === 'Log.entryAdded') return `[${e.params.entry.level}] ${e.params.entry.text}`
            return `[${e.params.type}] ` + (e.params.args || [])
                .map((a) => a.value ?? a.description ?? a.type).join(' ')
        })

    const close = async () => { try { ws.close() } catch {} ; try { await fetch(`http://127.0.0.1:${port}/json/close/${target.id}`) } catch {}; proc.kill() }

    return { send, evaluate, goto, setFiles, consoleLines, close }
}
