/**
 * Build the test corpus: real DSLR frames decoded to the app's proxy sizes.
 *
 * The guided filter has to be measured against real sensor noise, real chroma and
 * real edges — a synthetic gradient flatters any edge-aware filter. Point SRC at
 * a folder of RAW/JPEG files; RAWs go through `sips` (macOS) for the Nikon path
 * and through their embedded JPEG preview for Sony ARW, which `sips` won't read.
 *
 *   bun run bench/make-corpus.mjs [srcDir]
 */
import { execFileSync } from 'node:child_process'
import { mkdirSync, readdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'node:fs'
import path from 'node:path'

const SRC = process.argv[2] || path.join(process.env.HOME, 'Downloads')
const OUT = path.resolve(import.meta.dirname, 'corpus')
mkdirSync(OUT, { recursive: true })

const RAW = /\.(nef|nrw|cr2|cr3|arw|dng|orf|rw2|raf|pef|srw)$/i
const IMG = /\.(jpe?g|png|tiff?|webp)$/i

/** Largest embedded JPEG in a TIFF-based RAW — the camera's own full preview. */
const embeddedJpeg = (buf) => {
    let best = null
    for (let i = 0; i + 3 < buf.length; i += 1) {
        if (buf[i] === 0xff && buf[i + 1] === 0xd8 && buf[i + 2] === 0xff) {
            const end = buf.indexOf(Buffer.from([0xff, 0xd9]), i + 2)
            if (end > 0 && (!best || end - i > best[1] - best[0])) best = [i, end + 2]
        }
    }
    return best && buf.subarray(best[0], best[1])
}

const files = readdirSync(SRC)
    .filter((f) => RAW.test(f) || IMG.test(f))
    .map((f) => path.join(SRC, f))
    .filter((f) => { try { return statSync(f).size > 1_000_000 } catch { return false } })

console.log(`${files.length} candidate(s) in ${SRC}`)
let copied = 0
for (const f of files) {
    const base = path.basename(f)
    const dst = path.join(OUT, base)
    if (existsSync(dst)) { copied += 1; continue }
    try {
        if (RAW.test(f)) {
            const jpg = embeddedJpeg(readFileSync(f))
            // keep the RAW itself (the app decodes it) and, for ARW, its preview
            writeFileSync(dst, readFileSync(f))
            if (jpg) writeFileSync(path.join(OUT, base.replace(/\.\w+$/, '_prev.jpg')), jpg)
        } else {
            writeFileSync(dst, readFileSync(f))
        }
        copied += 1
    } catch (e) { console.log(`  skip ${base}: ${e.message}`) }
}
console.log(`corpus: ${copied} file(s) → ${OUT}`)

// Raw RGBA proxies for the Node-side filter bench (harness.mjs PROXIES).
const nef = files.find((f) => /\.nef$/i.test(f))
if (nef) {
    const png = path.join(OUT, 'nef.png')
    if (!existsSync(png)) execFileSync('sips', ['-s', 'format', 'png', nef, '--out', png])
    for (const [w, h, name] of [[1024, 683, 'nef_proxy_1024.rgba'], [2048, 1365, 'nef_proxy_2048.rgba']]) {
        const out = path.join(OUT, name)
        if (existsSync(out)) continue
        execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', png,
            '-vf', `scale=${w}:${h}`, '-pix_fmt', 'rgba', '-f', 'rawvideo', out, '-y'])
        console.log(`proxy ${name}`)
    }
} else {
    console.log('no .nef found — the Node filter bench needs one (see harness.mjs PROXIES)')
}
