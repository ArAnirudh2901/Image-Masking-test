#!/usr/bin/env bun
/**
 * One command to make a fresh clone runnable: `bun run setup`.
 *
 * The repo alone does not build. app.jsx imports the megashader engine and the
 * real editor UI from phosmith over the `@/` alias, and build.mjs resolves that
 * to a SIBLING checkout — so without phosmith on disk the bundler fails with
 * eight "Cannot find module './src/lib/…'" lines that never mention phosmith.
 * React must also resolve from phosmith/node_modules (build.mjs dedupes every
 * react import there), so phosmith needs its own `bun install`, not just a
 * clone.
 *
 * Steps, all idempotent:
 *   1. phosmith checkout at the pinned commit   (cloned only if absent)
 *   2. bun install here and in phosmith
 *   3. bun run models[:all]                     (ORT + weights, SHA-256 pinned)
 *   4. public/wasm presence check               (tracked, so this catches LFS/
 *                                                partial-clone accidents)
 *   5. bun run build                            — the real proof it works
 *
 * Usage: bun run setup           core weights (~111 MB)
 *
 *   PHOSMITH_DIR=/path/to/phosmith   use a checkout that is not ../phosmith
 *   --skip-models                    wiring only, no download
 */

import { spawn } from 'node:child_process'
import { readFile, stat } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

const log = (msg) => console.log(`[setup] ${msg}`)
const die = (msg) => { console.error(`[setup] FAILED — ${msg}`); process.exit(1) }

/** Inherit stdio: bun install and the model download both report progress, and
 *  swallowing it makes a 227 MB step look like a hang. */
const run = (cmd, args, opts = {}) =>
    new Promise((resolve) => {
        const child = spawn(cmd, args, { stdio: 'inherit', cwd: ROOT, ...opts })
        child.on('error', () => resolve(-1))
        child.on('close', (code) => resolve(code ?? -1))
    })

const capture = (cmd, args, cwd = ROOT) =>
    new Promise((resolve) => {
        const child = spawn(cmd, args, { cwd, stdio: ['ignore', 'pipe', 'ignore'] })
        let out = ''
        child.on('error', () => resolve(null))
        child.stdout.on('data', (d) => { out += d })
        child.on('close', (code) => resolve(code === 0 ? out.trim() : null))
    })

const exists = (p) => stat(p).then(() => true).catch(() => false)

const args = process.argv.slice(2)
const skipModels = args.includes('--skip-models')

const pkg = JSON.parse(await readFile(path.join(ROOT, 'package.json'), 'utf8'))
const PIN = pkg.phosmith ?? {}
if (!PIN.repo || !PIN.commit) die('package.json has no `phosmith` {repo, commit} pin')

// ── 0. toolchain ─────────────────────────────────────────────────────────────
// Bun is a given (this file runs under it); git is not, and its absence shows
// up as a confusing clone failure three steps later.
if (!(await capture('git', ['--version']))) die('git not found on PATH — needed to fetch phosmith')

// ── 1. phosmith ──────────────────────────────────────────────────────────────
const PHOSMITH = path.resolve(process.env.PHOSMITH_DIR || path.join(ROOT, '..', 'phosmith'))

if (await exists(path.join(PHOSMITH, 'src'))) {
    log(`phosmith found at ${PHOSMITH}`)
    const head = await capture('git', ['rev-parse', 'HEAD'], PHOSMITH)
    if (head && head !== PIN.commit) {
        // Never rewrite a checkout the user may have work in — say what differs
        // and let them decide.
        log(`NOTE  phosmith is at ${head.slice(0, 8)}, pinned is ${PIN.commit.slice(0, 8)}`)
        log(`      to match this repo exactly:  git -C ${PHOSMITH} checkout ${PIN.commit}`)
    }
} else {
    if (process.env.PHOSMITH_DIR) die(`PHOSMITH_DIR=${PHOSMITH} has no src/ — wrong path?`)
    log(`cloning phosmith → ${PHOSMITH}`)
    // Full clone, not --depth 1: the pinned commit is often not the branch tip.
    if ((await run('git', ['clone', PIN.repo, PHOSMITH])) !== 0) die('git clone of phosmith failed')
    // Detached on purpose — this checkout is a build input pinned to a commit,
    // not a branch to develop on.
    if ((await run('git', ['-C', PHOSMITH, 'checkout', '--detach', PIN.commit])) !== 0) {
        die(`phosmith has no commit ${PIN.commit} — the pin in package.json is stale`)
    }
    log(`phosmith pinned at ${PIN.commit.slice(0, 8)} (detached HEAD, by design)`)
}

// ── 2. dependencies ──────────────────────────────────────────────────────────
log('bun install (mask-studio)')
if ((await run('bun', ['install'])) !== 0) die('bun install failed')

// phosmith's node_modules is not optional: build.mjs pins react/react-dom/
// scheduler there so exactly ONE React copy is bundled. Two copies crash at
// runtime with "Cannot read properties of null (reading 'useState')".
log('bun install (phosmith — build.mjs resolves React from here)')
if ((await run('bun', ['install'], { cwd: PHOSMITH })) !== 0) die('bun install in phosmith failed')

// ── 3. runtime + weights ─────────────────────────────────────────────────────
if (skipModels) {
    log('skipping model download (--skip-models)')
} else {
    const modelArgs = ['scripts/download-models.mjs']
    if ((await run('bun', modelArgs)) !== 0) die('model download failed — re-run `bun run models`')
}

// ── 4. compiled wasm ─────────────────────────────────────────────────────────
// These are committed, so a miss means a broken clone, not a missing step.
const WASM = ['cv-refine.js', 'cv-refine.wasm', 'raw-develop.js', 'raw-develop.wasm']
const missingWasm = []
for (const f of WASM) {
    const p = path.join(ROOT, 'public', 'wasm', f)
    const s = await stat(p).catch(() => null)
    if (!s?.size) missingWasm.push(f)
}
if (missingWasm.length) {
    die(`public/wasm is incomplete (${missingWasm.join(', ')}) — re-clone; these are committed files`)
}

// ── 5. prove it builds ───────────────────────────────────────────────────────
if ((await run('bun', ['run', 'build.mjs'])) !== 0) die('build failed')

console.log(`
[setup] ready.

  bun run dev        build + serve on http://127.0.0.1:8810

Needs WebGPU with shader-f16 — Chrome/Edge 121+, or Safari 18+ on Apple
Silicon. The mask lane is fp16-only; there is no WASM fallback.
Drop any photo into the page, or add a test.png here to have one auto-load.`)
