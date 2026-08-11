/**
 * Static server for Mask Studio.
 *
 * Three things it must get right for the on-device engine to work:
 *
 *  1. `.wasm` → application/wasm, so onnxruntime-web can stream-compile.
 *  2. COOP/COEP, so the page is CROSS-ORIGIN ISOLATED. That unlocks threaded
 *     WASM (SharedArrayBuffer) and — the reason it matters most here —
 *     `performance.measureUserAgentSpecificMemory()`, the only real byte signal
 *     the memory governor has. Every model is vendored under models/ and lib/,
 *     so isolation costs nothing; sw.js re-tags the ORT CDN fallback with CORP
 *     for the case where lib/ort-web is missing.
 *  3. Range requests on the weights, so a browser can resume/stream an 81 MB
 *     encoder instead of restarting the whole download.
 *
 * Build output and index.html are served no-cache (a rebuild must be one
 * refresh away); the immutable, content-versioned model blobs are served with a
 * long max-age so a reload never re-downloads 200 MB.
 */
import { createServer } from 'node:http'
import { createReadStream } from 'node:fs'
import { stat } from 'node:fs/promises'
import path from 'node:path'

const DIR = import.meta.dir
const PORT = Number(process.env.PORT || 8810)

const MIME = {
    '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
    '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm',
    '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
    '.webp': 'image/webp', '.svg': 'image/svg+xml', '.map': 'application/json',
    '.onnx': 'application/octet-stream', '.txt': 'text/plain',
    '.i8': 'application/octet-stream', '.f32': 'application/octet-stream',
}

// Vendored weights and runtime are content-versioned by path; the app's own
// output never is.
const IMMUTABLE = /^\/(models|lib)\//

const server = createServer(async (req, res) => {
    try {
        const url = new URL(req.url, 'http://localhost')
        const rel = url.pathname === '/' ? '/index.html' : decodeURIComponent(url.pathname)
        const file = path.join(DIR, path.normalize(rel))
        // Compare against DIR + separator: a bare startsWith also admits a
        // sibling whose name merely begins with DIR's (…/studio-secrets).
        const root = DIR.endsWith(path.sep) ? DIR : DIR + path.sep
        if (!file.startsWith(root)) { res.writeHead(403).end('forbidden'); return }

        let info
        try { info = await stat(file) } catch { res.writeHead(404).end('not found'); return }
        if (!info.isFile()) { res.writeHead(404).end('not found'); return }

        const headers = {
            'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
            // Cross-origin isolation. Safe because nothing cross-origin is
            // embedded on the vendored path (see the header comment).
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp',
            'Cross-Origin-Resource-Policy': 'same-origin',
            'Accept-Ranges': 'bytes',
            'Cache-Control': IMMUTABLE.test(rel) ? 'public, max-age=31536000, immutable' : 'no-cache',
        }

        // Range: a partial weight fetch must not restart from byte 0.
        // Single ranges only, including the `bytes=-N` suffix form. Multipart is
        // ignored, which is a legal 200 — no client fetches weights that way.
        const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || '')
        if (range && (range[1] || range[2])) {
            const suffix = !range[1]   // `bytes=-N` = the LAST n bytes, not 0..n
            const start = suffix ? Math.max(0, info.size - Number(range[2])) : Number(range[1])
            const end = suffix || !range[2] ? info.size - 1 : Math.min(Number(range[2]), info.size - 1)
            if (start >= info.size || start > end) {
                res.writeHead(416, { 'Content-Range': `bytes */${info.size}` }).end()
                return
            }
            res.writeHead(206, { ...headers, 'Content-Range': `bytes ${start}-${end}/${info.size}`, 'Content-Length': end - start + 1 })
            if (req.method === 'HEAD') { res.end(); return }   // HEAD carries headers, never a body
            createReadStream(file, { start, end }).pipe(res)
            return
        }

        res.writeHead(200, { ...headers, 'Content-Length': info.size })
        if (req.method === 'HEAD') { res.end(); return }
        createReadStream(file).pipe(res)
    } catch (e) {
        res.writeHead(500).end(String(e?.message || e))
    }
})

server.listen(PORT, '127.0.0.1', () => {
    console.log(`Mask Studio → http://127.0.0.1:${PORT}/  (cross-origin isolated)`)
})
