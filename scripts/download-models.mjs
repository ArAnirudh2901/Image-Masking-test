#!/usr/bin/env node
/**
 * Vendors every runtime asset SEGLAB needs, so the app serves fully offline:
 *   lib/ort-web/…              — onnxruntime-web (WebGPU ESM bundle + wasm loader)
 *   models/sam21/…             — SAM 2.1 small, fp16 (core)
 *   models/yoloe/…, clip-text/ — open-vocab text search (--detector)
 *   models/manifest.json       — presence signal read at runtime
 *
 * No hub publishes the exact ONNX artifacts this app runs, so the weights come
 * from this repo's own `weights-v1` release rather than being rebuilt: ONNX
 * export is not byte-reproducible across torch/onnx versions, and the encoder
 * is fp16-sensitive enough that a re-export is a different model in practice.
 * Each bundle is SHA-256 pinned. The export scripts remain the source of truth
 * for producing a NEW release, not for reproducing this one.
 *
 * lib/ and models/ are gitignored (~25 MB runtime, ~197 MB weights).
 * Idempotent: complete files are skipped.
 *
 * Usage: bun run models          (core)
 *        bun run models:all      (core + detector)
 *
 *   (core)       click/box/lasso selection + export run with no network.
 *   --detector   also fetches the open-vocabulary text-search artifacts.
 */

import { execFile } from 'node:child_process'
import { createHash } from 'node:crypto'
import { createWriteStream } from 'node:fs'
import { mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

// The ONE runtime. There is no second ORT build and no transformers.js: the
// SlimSAM lane that needed them is gone, and with it ~35 MB of wasm and a
// version-pin check that existed only to keep those two in step.
//
// MUST be >= 1.24.3. The 1.22 WebGPU fp16 kernels compute SAM 2.1's encoder
// WRONG — silently, with a confident-looking mask covering 99.6% of the frame
// (spikes/sam21/FINDINGS.md §1.2). fp16 is what keeps resident memory under
// 2 GB, so downgrading this re-breaks the memory contract as well as quality.
// spikes/sam21/ortver.html is the gate: re-run it on any change here.
const ORT_WEB_VERSION = '1.27.0'

const ORT_WEB_FILES = [
    'ort.webgpu.bundle.min.mjs',
    // 1.23+ renamed the WebGPU-capable wasm from .jsep to .asyncify; the bundle
    // dynamically imports it, so the old names produce "no available backend".
    'ort-wasm-simd-threaded.asyncify.mjs',
    'ort-wasm-simd-threaded.asyncify.wasm',
]

// The mask lane. Core, not optional — without these the app cannot segment at
// all, so a missing one is a hard failure rather than a note.
const CORE_ASSETS = [
    ['models/sam21/encoder.fp16.onnx', 'python3 scripts/export-sam21.py'],
    ['models/sam21/decoder.fp16.onnx', 'python3 scripts/export-sam21.py'],
    ['models/sam21/model.json', 'python3 scripts/export-sam21.py'],
]

// Open-vocabulary text search. Built locally: no hub publishes a YOLOE
// text-prompt ONNX, and the text tower has to be the MobileCLIP2-B one
// YOLOE-26L was trained against or RepRTA receives out-of-distribution vectors.
const DETECTOR_ASSETS = [
    ['models/yoloe/yoloe-26l-text.fp16.onnx', 'python3 scripts/export-yoloe-text.py'],
    ['models/clip-text/mclip2-text.q4.onnx', 'python3 scripts/export-clip-text.py'],
    ['models/clip-text/mclip2-embed.i8', 'python3 scripts/export-clip-text.py'],
    ['models/clip-text/mclip2-embed.scale.f32', 'python3 scripts/export-clip-text.py'],
    ['models/clip-text/merges.txt', 'python3 scripts/export-clip-text.py'],
]

// Prebuilt weights, published from a machine that had the export toolchain.
// Bumping a bundle means a new tag + new digest — never overwrite an asset in
// place, or pinned checkouts silently change models.
const WEIGHTS_TAG = 'weights-v1'
const releaseUrl = (name) =>
    `https://github.com/ArAnirudh2901/Image-Masking-test/releases/download/${WEIGHTS_TAG}/${name}`

const WEIGHT_BUNDLES = {
    core: {
        asset: 'weights-core.tar.gz',
        sha256: 'fce3b92db7e1eb915291d92da2d104a03952ff7bcde779e584cb085e552e1f4b',
    },
    detector: {
        asset: 'weights-detector.tar.gz',
        sha256: '58bedb872662f244eacb67f88208937c8e0a3bf1adebd837764fa3925aa52115',
    },
}

const withDetector = process.argv.includes('--detector')

const jobs = ORT_WEB_FILES.map((f) => ({
    url: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_WEB_VERSION}/dist/${f}`,
    dest: `lib/ort-web/${f}`,
}))

const log = (msg) => console.log(`[download-models] ${msg}`)

// ORT-Web forwards only a fixed key list from a WebGPU EP options object to the
// native EP, and the buffer cache modes are not on it — so the one setting that
// keeps this app under its RAM ceiling is unreachable through public API.
// (js/ort-loader.js webgpuEP: Bucket 1128 MB of GPU process vs 290 MB with
// lazyRelease, identical logits.) Forward an `epConfig` bag instead.
//
// A one-line, anchored patch on a DERIVED artifact. It throws if the anchor is
// gone, so an ORT bump fails here rather than silently costing ~840 MB;
// verify.mjs gates the vendored copy carrying it.
const ORT_EP_ANCHOR = 'S.validationMode&&ot(l,"validationMode",S.validationMode,s)'
const ORT_EP_PATCH = ',S.epConfig&&Object.entries(S.epConfig).forEach(([Ck,Cv])=>ot(l,Ck,String(Cv),s))'
const patchOrtBundle = (buf) => {
    const src = buf.toString('utf8')
    if (src.includes('S.epConfig')) return buf
    if (!src.includes(ORT_EP_ANCHOR)) {
        throw new Error(`ORT ${ORT_WEB_VERSION}: epConfig anchor missing — re-derive the patch in download-models.mjs`)
    }
    return Buffer.from(src.replace(ORT_EP_ANCHOR, ORT_EP_ANCHOR + ORT_EP_PATCH), 'utf8')
}
const mb = (n) => `${(n / 1024 / 1024).toFixed(2)} MB`

// Idempotency is manifest-based: CDN content-length reports the compressed
// size when content-encoding is active, so it cannot be compared to disk.
const priorBytes = await readFile(path.join(ROOT, 'models', 'manifest.json'), 'utf8')
    .then((s) => new Map(JSON.parse(s).files.map((f) => [f.path, f.bytes])))
    .catch(() => new Map())

const download = async ({ url, dest, optional }) => {
    const target = path.join(ROOT, dest)
    const local = await stat(target).catch(() => null)
    if (local && priorBytes.get(dest) === local.size) {
        log(`have ${dest} (${mb(local.size)})`)
        return { path: dest, bytes: local.size }
    }
    const res = await fetch(url)
    if (!res.ok) {
        if (optional) { log(`skip (optional, HTTP ${res.status}): ${dest}`); return null }
        throw new Error(`GET failed for ${url} (HTTP ${res.status})`)
    }
    let buf = Buffer.from(await res.arrayBuffer())
    const remoteBytes = Number(res.headers.get('content-length')) || null
    const encoded = !!res.headers.get('content-encoding')
    if (!encoded && remoteBytes && buf.length !== remoteBytes) {
        throw new Error(`size mismatch for ${dest}: got ${buf.length}, expected ${remoteBytes}`)
    }
    if (buf.length === 0) throw new Error(`empty download for ${dest}`)
    if (dest.endsWith('ort.webgpu.bundle.min.mjs')) buf = patchOrtBundle(buf)
    await mkdir(path.dirname(target), { recursive: true })
    await writeFile(target, buf)
    log(`got  ${dest} (${mb(buf.length)})`)
    return { path: dest, bytes: buf.length }
}

/** Fetch + extract a weight bundle if any of its assets are absent. Streams to
 *  a temp file: the detector bundle is 113 MB and buffering it costs more RAM
 *  than the app is allowed at runtime. Digest is checked before extraction, so
 *  a truncated or swapped asset never lands in models/. */
const ensureBundle = async ({ asset, sha256 }, assets) => {
    const present = await Promise.all(
        assets.map(([rel]) => stat(path.join(ROOT, rel)).then((s) => !!s.size).catch(() => false)),
    )
    if (present.every(Boolean)) return true

    // Before the download, not after: discovering a missing tar at extraction
    // time throws away 108 MB of transfer. bsdtar ships with macOS, Windows 10+
    // and every mainstream Linux, so this is a rare path, not a common one.
    if (!(await execFileAsync('tar', ['--version']).then(() => true).catch(() => false))) {
        log(`FAILED ${asset}: no \`tar\` on PATH — install it, then re-run`)
        return false
    }

    const url = releaseUrl(asset)
    const tmp = path.join(tmpdir(), `${asset}.${process.pid}.part`)
    log(`fetching ${asset} …`)
    try {
        const res = await fetch(url)
        if (!res.ok) throw new Error(`GET failed for ${url} (HTTP ${res.status})`)
        const hash = createHash('sha256')
        let bytes = 0
        await pipeline(
            Readable.fromWeb(res.body),
            async function* (source) {
                for await (const chunk of source) { hash.update(chunk); bytes += chunk.length; yield chunk }
            },
            createWriteStream(tmp),
        )
        const got = hash.digest('hex')
        if (got !== sha256) throw new Error(`digest mismatch for ${asset}\n  expected ${sha256}\n  got      ${got}`)
        // Bundles expand to models/…, so ROOT is the extraction point.
        await execFileAsync('tar', ['-xzf', tmp, '-C', ROOT])
        log(`got  ${asset} (${mb(bytes)}) — extracted`)
        return true
    } catch (err) {
        log(`FAILED ${asset}: ${err.message}`)
        return false
    } finally {
        await rm(tmp, { force: true })
    }
}

/** Confirm each asset landed, and say exactly how to produce a missing one
 *  instead of failing with a bare path. */
const verifyBuilt = async (assets, files) => {
    const missing = []
    for (const [rel, how] of assets) {
        const local = await stat(path.join(ROOT, rel)).catch(() => null)
        if (local?.size) { log(`have ${rel} (${mb(local.size)})`); files.push({ path: rel, bytes: local.size }) }
        else missing.push([rel, how])
    }
    for (const [rel, how] of missing) log(`MISSING ${rel} — re-run, or rebuild it with: ${how}`)
    return missing
}

const files = []
for (const job of jobs) {
    const entry = await download(job)
    if (entry) files.push(entry)
}

// Also on the cache-hit path: a checked-out tree already has the bundle at the
// manifest's size, so nothing would re-download it and the patch would be lost.
{
    const entry = files.find((f) => f.path.endsWith('ort.webgpu.bundle.min.mjs'))
    const target = path.join(ROOT, entry.path)
    const patched = patchOrtBundle(await readFile(target))
    if (patched.length !== entry.bytes) {
        await writeFile(target, patched)
        entry.bytes = patched.length
        log(`patched ${entry.path} (epConfig passthrough)`)
    }
}

await ensureBundle(WEIGHT_BUNDLES.core, CORE_ASSETS)
const missingCore = await verifyBuilt(CORE_ASSETS, files)

let detectorReady = false
if (withDetector) {
    await ensureBundle(WEIGHT_BUNDLES.detector, DETECTOR_ASSETS)
    detectorReady = (await verifyBuilt(DETECTOR_ASSETS, files)).length === 0
}

const manifest = {
    onnxruntimeWeb: ORT_WEB_VERSION,
    lane: 'sam2.1-small',
    precision: 'fp16',
    detector: detectorReady ? 'yoloe-26l-text + mobileclip2-b' : null,
    generatedAt: new Date().toISOString(),
    files,
}
await mkdir(path.join(ROOT, 'models'), { recursive: true })
await writeFile(path.join(ROOT, 'models', 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`)
const total = files.reduce((sum, f) => sum + f.bytes, 0)
log(`done — ${files.length} files, ${mb(total)} total; wrote models/manifest.json`)
if (!withDetector) log('open-vocab text search not registered — run `bun run models:all` to fetch it')
// Loud, and last, so it is the thing left on screen: the app cannot segment
// without these, and a silent manifest would let that surface as a runtime bug.
if (missingCore.length) {
    throw new Error(`mask lane incomplete — ${missingCore.length} core asset(s) missing; see MISSING lines above`)
}
