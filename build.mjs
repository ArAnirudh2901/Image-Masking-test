/**
 * Standalone bundler for the Mask Studio testbed.
 *
 * The phosmith UI components we reuse (ProRulerSlider, LabeledSlider,
 * KindParamEditor, MaskChainCard) import via the project's `@/*` alias.
 * Bun's CLI bundler only reads tsconfig.json paths, and this project uses
 * jsconfig.json (adding a root tsconfig.json would flip Next into TS mode),
 * so we resolve `@/` → `<repo>/src/` with a tiny onResolve plugin instead.
 */
import path from 'node:path'
import { readdirSync, rmSync } from 'node:fs'

const REPO = path.resolve(import.meta.dir, '../phosmith') // phosmith root
const SRC = path.join(REPO, 'src')

// Clean previous output (entry + split chunks) so stale chunks don't linger.
for (const f of readdirSync(import.meta.dir)) {
    if (f === 'app.js' || /^chunk-.*\.js$/.test(f) || f.endsWith('.js.map')) {
        rmSync(path.join(import.meta.dir, f), { force: true })
    }
}

const aliasPlugin = {
    name: 'phosmith-@-alias',
    setup(build) {
        build.onResolve({ filter: /^@\// }, (args) => {
            const mapped = './' + path.join('src', args.path.slice(2))
            return { path: Bun.resolveSync(mapped, REPO) }
        })
        build.onResolve({ filter: /^@hooks\// }, (args) => {
            const mapped = './' + path.join('hooks', args.path.slice('@hooks/'.length))
            return { path: Bun.resolveSync(mapped, REPO) }
        })
        // Dedupe React to a SINGLE copy. The entry (app.jsx) resolves `react`
        // from Image-Masking-test/node_modules while the @/-aliased phosmith
        // components resolve it from phosmith/node_modules, so Bun bundles TWO
        // React instances — the second's exports are null at runtime, crashing
        // with "Cannot read properties of null (reading 'useState')" in
        // LayerGradeEditor. Pin all React imports to the phosmith copy so there
        // is exactly one. (Only react* — NOT @huggingface/transformers, whose
        // node/browser build variant must keep resolving from the entry.)
        build.onResolve(
            { filter: /^(react|react-dom|scheduler)(\/.*)?$/ },
            (args) => ({ path: Bun.resolveSync(args.path, REPO) }),
        )
    },
}

const result = await Bun.build({
    entrypoints: [path.join(import.meta.dir, 'app.jsx')],
    outdir: import.meta.dir,
    // `splitting` lets the heavy on-device-AI module (client-ai.js +
    // transformers.js) load as a lazy chunk via dynamic import(), so the core
    // studio (React + megashader + MaskChainCard) is small and loads instantly.
    splitting: true,
    naming: { entry: 'app.js', chunk: 'chunk-[hash].js', asset: '[name]-[hash].[ext]' },
    format: 'esm',
    target: 'browser',
    minify: false,
    sourcemap: 'none',
    plugins: [aliasPlugin],
    define: { 'process.env.NODE_ENV': '"production"' },
})

if (!result.success) {
    console.error('BUILD FAILED')
    for (const log of result.logs) console.error(log)
    process.exit(1)
}
let total = 0
for (const o of result.outputs) total += o.size
console.log(`OK → ${result.outputs.length} file(s), ${(total / 1024).toFixed(1)} KB total`)
for (const o of result.outputs) {
    console.log(`   ${path.basename(o.path)}  ${(o.size / 1024).toFixed(1)} KB`)
}
