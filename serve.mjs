/**
 * Tiny static server for the Mask Studio testbed.
 *
 * Mirrors scripts/verify-client-ai.mjs: serves this directory with correct MIME
 * types — crucially `.wasm` → application/wasm so onnxruntime-web (the on-device
 * AI backend) can stream-compile. Avoids relying on `python3 -m http.server`,
 * which doesn't always map .wasm.
 */
import { createServer } from 'node:http'
import { readFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import path from 'node:path'

const DIR = import.meta.dir
const PORT = Number(process.env.PORT || 8810)

const MIME = {
    '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
    '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm',
    '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
    '.webp': 'image/webp', '.svg': 'image/svg+xml', '.map': 'application/json',
}

const server = createServer(async (req, res) => {
    try {
        const url = new URL(req.url, 'http://localhost')
        const rel = url.pathname === '/' ? '/index.html' : decodeURIComponent(url.pathname)
        const file = path.join(DIR, path.normalize(rel))
        if (!file.startsWith(DIR) || !existsSync(file)) {
            res.writeHead(404).end('not found')
            return
        }
        const body = await readFile(file)
        // NB: deliberately NO COOP/COEP — cross-origin isolation would let
        // onnxruntime use threads, but require-corp also blocks the HuggingFace
        // CDN model fetches (no CORP header), breaking on-device AI. Single-
        // threaded WASM is fine. (verify-client-ai.mjs serves the same way.)
        res.writeHead(200, {
            'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
            'Cache-Control': 'no-cache',
        })
        res.end(body)
    } catch (e) {
        res.writeHead(500).end(String(e?.message || e))
    }
})

server.listen(PORT, '127.0.0.1', () => {
    console.log(`Mask Studio → http://127.0.0.1:${PORT}/`)
})
