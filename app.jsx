/**
 * Mask Studio — standalone, no-auth testbed for the phosmith megashader masking
 * engine. Loads an image, lets you add MULTIPLE masks of every kind (radial /
 * linear / pen·lasso / brush / luminance / color / AI), draw each region on the
 * canvas, and colour-grade ONLY that region with the REAL phosmith mask card
 * (ProRulerSlider + MaskChainCard + the per-layer adjustment sliders).
 *
 * Two production engines spliced together, both verbatim:
 *   · phosmith  — the megashader mask/grade engine + the real editor UI.
 *   · seglab    — the on-device segmentation stack under `ai/`: SAM 2.1 fp16 on
 *                 WebGPU, bounded decode workers, wasm mask refinement, a
 *                 one-at-a-time heavy-job queue and a live memory governor.
 *
 * Everything runs in the browser. There is no upload endpoint, no Python
 * service and no CDN model fetch on the vendored path — see `ai/studio-bridge.js`
 * for the whole surface this file is allowed to touch.
 *
 * Copyright (C) 2026 Anirudh Aravalli
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * See LICENSE for the full text; third-party terms, including the
 * research-only Apple model license, are in NOTICE.
 */
import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { createRoot } from 'react-dom/client'
import { AnimatePresence } from 'framer-motion'
import { renderMegashader, getRenderMetrics, resetRenderMetrics } from '@/lib/megashader/megashader-renderer'
import {
    radialLayer, linearLayer, pathLayer, lassoLayer, brushLayer,
    luminanceLayer, colorLayer, semanticLayer, smartBrushLayer,
    sanitiseLayer, setMaskTexture, getMaskTexture, getMaskTextureVersion,
} from '@/lib/megashader/mask-types'
import { growMaskCanvas } from '@/lib/mask-grow'
import { rasterisePath, smoothToBezier } from '@/lib/megashader/path-raster'
import { buildPackedLutFromCurves } from '@/lib/curve-lut'
import { computeImageHistogram } from '@/lib/image-histogram'
import {
    MaskChainCard, getKindMeta, ToolEmptyState,
} from '@/app/(main)/editor/[projectId]/_components/tools/_pixel-tool-ui.jsx'
import { LayerGradeEditor } from '@/app/(main)/editor/[projectId]/_components/tools/_layer-grade-editor.jsx'
import { rgbToHsb } from '@/lib/color-utils'

const ACCENT = '#53d8ff'
const uid = () => Math.random().toString(36).slice(2, 9)

// Where the corresponding source lives, for the AGPL-3.0 §13 offer in the top
// bar. Point this at YOUR fork if you deploy a modified build — §13 asks for the
// source of the running version, not of upstream.
const SOURCE_URL = 'https://github.com/ArAnirudh2901/Image-Masking-test'

/**
 * The seglab engine, loaded through a RUNTIME import so Bun leaves it alone.
 * Its lane resolves workers and weights with `new URL(…, import.meta.url)`,
 * which only holds while `ai/*.js` is served from its real path — bundling it
 * into app.js would rewrite those URLs and break every worker and model fetch.
 */
let bridgePromise = null
const engine = () => (bridgePromise ??= import(
    /* @vite-ignore */ new URL('ai/studio-bridge.js', document.baseURI).href
))

// True when a layer carries any grade the engine renders (gamma ≠ 1, a tone
// curve, or a non-zero colour wheel). Used to skip the base pre-grade pass when
// the global grade is identity, and to gate the LayerGradeEditor reset buttons.
const wheelOn = (w) => Array.isArray(w) && w.some((v) => Math.abs(v) > 1e-4)
const hasGrade = (l) => !!l && (
    (typeof l.gamma === 'number' && l.gamma !== 1)
    || !!l.curveLutKey
    || wheelOn(l.wheelShadows) || wheelOn(l.wheelMidtones) || wheelOn(l.wheelHighlights)
)

// One unified mask palette — geometric, parametric, freehand AND AI masks are
// all just mask "types" here (no separation). Pen and Brush fold their two
// engine kinds into one tool via a toggle (no redundant buttons). The `ai` field
// routes the click to an on-device model instead of a canvas interaction.
const MASK_TOOLS = [
    { id: 'radial', kind: 'radial', label: 'Radial', hint: 'ellipse / circle gradient' },
    { id: 'linear', kind: 'linear', label: 'Linear', hint: 'linear gradient' },
    { id: 'brush', kind: 'brush', label: 'Brush', hint: 'paint a region (edge-aware optional)' },
    { id: 'pen', kind: 'path', label: 'Pen / Lasso', hint: 'click points · close' },
    { id: 'luminance', kind: 'luminance', label: 'Luminance', hint: 'tonal range' },
    { id: 'color', kind: 'color', label: 'Color', hint: 'colour range · click to sample' },
    { id: 'subject', kind: 'semantic', label: 'AI Subject', hint: 'auto-mask the main subject (SAM 2.1, on-device)', ai: 'subject' },
    { id: 'background', kind: 'semantic', label: 'AI Sky / Bg', hint: 'select the ENTIRE sky / background (mask the subject, then invert)', ai: 'background' },
    { id: 'clickselect', kind: 'semantic', label: 'AI Click-Select', hint: 'click any object · click again to add · Alt+click to remove', ai: 'clickselect' },
    { id: 'sam', kind: 'semantic', label: 'AI Box-Select', hint: 'drag a box around an object', ai: 'sam' },
    { id: 'ailasso', kind: 'semantic', label: 'AI Lasso', hint: 'draw a rough loop — it snaps to the object and can never bleed outside', ai: 'ailasso' },
]

// Banner titles for the transient on-canvas modes (tool ids are terse).
const TOOL_TITLE = {
    sam: 'box-select', clickselect: 'click-select', ailasso: 'ai lasso',
    refine: 'brush-refine', pen: 'pen', brush: 'brush', color: 'color',
}

// ── geometry helpers for on-canvas mask handles (all in image-space px) ──────
const HANDLE_PX = 9 // on-screen handle hit/half-size, converted to image px per use
const rotv = (dx, dy, a) => ({ x: dx * Math.cos(a) - dy * Math.sin(a), y: dx * Math.sin(a) + dy * Math.cos(a) })
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y)
// Radial handles: 4 cardinal resize points (on the rotated ellipse), a centre
// move point, and a rotation handle just beyond north.
const radialHandles = (l) => {
    const c = l.center, rx = l.radius.x, ry = l.radius.y, a = l.rotation || 0
    const add = (v) => ({ x: c.x + v.x, y: c.y + v.y })
    return {
        center: c,
        e: add(rotv(rx, 0, a)), w: add(rotv(-rx, 0, a)),
        n: add(rotv(0, -ry, a)), s: add(rotv(0, ry, a)),
        rot: add(rotv(0, -(ry + Math.max(rx, ry) * 0.22 + 2), a)),
    }
}
const linearHandles = (l) => ({ p1: l.p1, p2: l.p2, mid: { x: (l.p1.x + l.p2.x) / 2, y: (l.p1.y + l.p2.y) / 2 } })
const nearestHandle = (handles, p, thr, order) => order.find((k) => handles[k] && dist(handles[k], p) <= thr) || null
const insideEllipse = (l, p) => {
    const a = l.rotation || 0, dx = p.x - l.center.x, dy = p.y - l.center.y
    const lx = dx * Math.cos(a) + dy * Math.sin(a), ly = -dx * Math.sin(a) + dy * Math.cos(a)
    return (lx * lx) / (l.radius.x * l.radius.x) + (ly * ly) / (l.radius.y * l.radius.y) <= 1
}
const distToSeg = (a, b, p) => {
    const vx = b.x - a.x, vy = b.y - a.y, wx = p.x - a.x, wy = p.y - a.y
    const c1 = vx * wx + vy * wy
    if (c1 <= 0) return dist(p, a)
    const c2 = vx * vx + vy * vy
    if (c2 <= c1) return dist(p, b)
    const t = c1 / c2
    return dist(p, { x: a.x + t * vx, y: a.y + t * vy })
}

// Trace a thin accent boundary around a texture mask's coverage (instead of a
// heavy fill) so the SELECTED region is identifiable while the image colours
// underneath stay visible. Thresholds the coverage (luma for opaque masks, or
// alpha for the painted-alpha brush) into a silhouette, dilates it around a ring
// of offsets, then punches the interior back out — leaving just the edge.
// Tracing the boundary costs two full-size canvases, a getImageData, a per-pixel
// JS pass and 16 dilation draws — far too much to repeat per frame while a mask is
// merely SELECTED. Memoised on the texture's content version (so a brush stroke
// still invalidates it) plus the geometry that changes its appearance.
// Insertion-ordered LRU: one slot forced a full re-trace every time selection
// alternated between two masks. Capped at 3 — each entry is a full-size RGBA
// canvas (~16 MB at 2048²), so this trades bounded memory for the re-trace.
const BOUNDARY_CACHE_MAX = 3
const boundaryCache = new Map()

const traceBoundary = (tex, key, W, H, ipx) => {
    const tw = tex.width, th = tex.height
    const id = `${key}@${getMaskTextureVersion(key)}:${tw}x${th}:${W}x${H}:${ipx.toFixed(3)}`
    const hit = boundaryCache.get(id)
    if (hit) { boundaryCache.delete(id); boundaryCache.set(id, hit); return hit } // touch → most-recent

    const r = Math.max(1.2, 2.2 * ipx * (tw / (W || tw)))

    // 1. Threshold mask → binary silhouette via compositing (no pixel loop)
    const sil = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(tw, th) : (() => { const c = document.createElement('canvas'); c.width = tw; c.height = th; return c })()
    const sc = sil.getContext('2d')
    sc.drawImage(tex, 0, 0)
    sc.globalCompositeOperation = 'source-in'
    sc.fillStyle = '#53d8ff'
    sc.fillRect(0, 0, tw, th)

    // 2. Dilate via offset draws (adaptive step count instead of fixed 16)
    const out = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(tw, th) : (() => { const c = document.createElement('canvas'); c.width = tw; c.height = th; return c })()
    const oc = out.getContext('2d')
    const steps = Math.max(8, Math.ceil(Math.PI * r))
    for (let i = 0; i < steps; i += 1) {
        const a = (i / steps) * Math.PI * 2
        oc.drawImage(sil, Math.cos(a) * r, Math.sin(a) * r)
    }
    // 3. Punch out interior
    oc.globalCompositeOperation = 'destination-out'
    oc.drawImage(sil, 0, 0)

    boundaryCache.set(id, out)
    if (boundaryCache.size > BOUNDARY_CACHE_MAX) boundaryCache.delete(boundaryCache.keys().next().value)
    return out
}

const drawMaskBoundary = (g, tex, key, W, H, ipx) => {
    if (!tex || !tex.width) return
    g.drawImage(traceBoundary(tex, key, W, H, ipx), 0, 0, W, H)
}

// Human-readable byte counter for the model-download progress line.
const mb = (n) => `${(n / (1024 * 1024)).toFixed(0)} MB`

// Radial / linear / path geometry is image-space px, so exporting at a larger
// resolution has to scale it. Texture-backed masks need nothing — the shader
// samples them in normalised UV, so a ≤proxy mask upsamples with its own soft
// coverage intact.
const scaleLayer = (l, k) => {
    if (!l || k === 1) return l
    const pt = (p) => (p ? { x: p.x * k, y: p.y * k } : p)
    const out = { ...l }
    if (out.center) out.center = pt(out.center)
    if (out.radius) out.radius = { x: out.radius.x * k, y: out.radius.y * k }
    if (out.p1) out.p1 = pt(out.p1)
    if (out.p2) out.p2 = pt(out.p2)
    if (Array.isArray(out.points)) out.points = out.points.map(pt)
    return out
}

function App() {
    const srcRef = useRef(null)          // working-res source canvas
    const dispRef = useRef(null)         // visible result canvas
    const overlayRef = useRef(null)      // image-res overlay for guides + pointer capture
    const brushRef = useRef(null)        // { canvas, ctx, key, baseKey, layerId, mode } for the active brush layer
    const refineRef = useRef(null)       // same shape — painting INTO the selected mask (brush-refine)
    const dragRef = useRef(null)         // transient drag state for radial/linear/brush
    const chainRef = useRef([])           // always-current mirror of `chain` for callbacks
    const baseLayerRef = useRef(null)      // always-current mirror of `baseLayer`
    const lastSnapVersionsRef = useRef(new Map())  // U8: texture versions at last snapshot
    const rafRef = useRef(0)
    const cursorRef = useRef(null)       // floating brush-size ring (brush / refine)
    const stageRef = useRef(null)        // zoom/pan clip box (gestures listens here)
    const frameRef = useRef(null)        // the only element gestures transforms
    const zoomRef = useRef(null)         // zoom readout, and the way back to 1x
    const gestureHintRef = useRef(null)  // binding hint for the current input device

    const [imageSize, setImageSize] = useState(null)
    const [chain, setChain] = useState([])         // [{ layer, op }]
    chainRef.current = chain  // sync on every render — callbacks read this instead of closing over state
    const [baseLayer, setBaseLayer] = useState(null) // full-frame global/base grade (id 'base')
    baseLayerRef.current = baseLayer
    const [histogram, setHistogram] = useState(null) // source histogram for the curve graphs
    const [compare, setCompare] = useState(false)  // hold-to-compare: show the ungraded source
    const [selectedId, setSelectedId] = useState(null)
    const [tool, setTool] = useState(null)
    const [penSmooth, setPenSmooth] = useState(true)   // pen (bézier) vs lasso (straight)
    const [edgeAware, setEdgeAware] = useState(false)  // brush vs smartBrush (reserved)
    const [brushSize, setBrushSize] = useState(60)
    const [refineErase, setRefineErase] = useState(false) // brush-refine: remove coverage instead of add
    const [draft, setDraft] = useState([])         // in-progress pen/lasso points (image px)
    const [samBox, setSamBox] = useState(null)     // in-progress box-select rubber-band {x0,y0,x1,y1}
    const [clickPoints, setClickPoints] = useState([])   // [{x,y,label}] for click-select (label: 1=pos, 0=neg)
    // Ref mirror: appends must read the latest list even when two clicks land
    // before React re-renders. Always write through putClickPoints, never the
    // raw setter, or the ref goes stale and the next append drops a point.
    const clickPointsRef = useRef([])
    const putClickPoints = useCallback((next) => { clickPointsRef.current = next; setClickPoints(next) }, [])
    const [clickMaskId, setClickMaskId] = useState(null)  // layer ID of the live click-select mask being refined
    const [overlayMode, setOverlayMode] = useState(false)
    const [globalInvert, setGlobalInvert] = useState(false)
    const [preview, setPreview] = useState(false)  // hide all guides/handles → clean result
    const [tick, setTick] = useState(0)
    const [hasImage, setHasImage] = useState(false)
    const [status, setStatus] = useState('')
    // On-device segmentation engine (ai/studio-bridge.js → seglab SAM 2.1 lane).
    const [ai, setAi] = useState({ busy: false, status: '', engine: null, progress: null, pressure: 0 })
    // The latch that actually gates concurrent runs. `ai.busy` drives the UI but
    // lags a render behind, so two clicks inside one frame both read false and
    // launch. Guards must test this ref; `ai.busy` is for display only.
    const aiBusyRef = useRef(false)
    const [aiLasso, setAiLasso] = useState(null)   // in-progress AI-lasso stroke (image px)

    const bump = () => setTick((t) => t + 1)
    const W = imageSize?.width || 0
    const H = imageSize?.height || 0
    const aiSet = useCallback((patch) => setAi((s) => ({ ...s, ...patch })), [])
    // Texture-backed layer count, in a ref so the governor's measure callback
    // can read it without re-subscribing on every chain change.
    const textureLayers = useRef(0)

    /* ── image import (seglab asset-store: blob-only custody + bounded proxy) ─ */
    // One canvas for the life of the page: importOriginal resizes it in place,
    // so the megashader source, the mask textures and the model's interaction
    // frame are the same buffer and no mask ever needs a resample.
    const proxyCanvas = useRef(null)
    if (!proxyCanvas.current && typeof document !== 'undefined') {
        proxyCanvas.current = document.createElement('canvas')
    }

    const loadImage = useCallback(async (source, label) => {
        const canvas = proxyCanvas.current
        try {
            setStatus('Reading image…')
            const mod = await engine()
            const transform = await mod.importImage(source, {
                proxyCanvas: canvas,
                onStage: (msg) => setStatus(msg),
            })
            if (!transform) return  // a newer import owns the canvas now
            const w = canvas.width
            const h = canvas.height

            srcRef.current = canvas
            brushRef.current = null
            refineRef.current = null
            dragRef.current = null        // Bug #3: prevent dangling pointer capture
            boundaryCache.clear()         // Bug #5: release stale boundary canvases
            // Base/global grade carrier: a full-WHITE texture so one semantic
            // layer covers EVERY pixel (white = full coverage — no smoothstep
            // boundary, unlike a luminance(0..1) mask which under-covers pure
            // black/white). Pass 1 of the render bakes its grade into the frame.
            const baseKey = 'base-tex'
            const wc = document.createElement('canvas'); wc.width = w; wc.height = h
            const wx = wc.getContext('2d'); wx.fillStyle = '#fff'; wx.fillRect(0, 0, w, h)
            setMaskTexture(baseKey, wc)
            setBaseLayer(sanitiseLayer({ ...semanticLayer({ maskTextureKey: baseKey, feather: 0, label: 'Base' }), id: 'base', fillMode: 'adjust' }))
            setHistogram(computeImageHistogram({ getElement: () => canvas }))
            setImageSize({ width: w, height: h })
            setChain([]); setSelectedId(null); setTool(null); setDraft([])
            putClickPoints([]); setClickMaskId(null); setSamBox(null); setAiLasso(null)
            setCycleTarget(null)   // a new photo parks nothing
            setHasImage(true)
            const native = `${transform.originalW}×${transform.originalH}`
            setStatus(`${label || 'image'} · ${native} → ${w}×${h} interaction frame${transform.sourceWasRaw ? ' (RAW preview)' : ''}`)
            bump()
        } catch (err) {
            console.error('[studio] import failed', err)
            setStatus('Import failed: ' + (err?.message || err))
        }
    }, [])

    // Auto-load the bundled sample so the page is usable (and testable) instantly.
    useEffect(() => {
        fetch('test.png')
            .then((r) => (r.ok ? r.blob() : Promise.reject(new Error('no sample'))))
            .then((blob) => loadImage(blob, 'sample'))
            .catch(() => setStatus('Load an image to begin'))
    }, [loadImage])

    const onFile = (e) => {
        const f = e.target.files?.[0]
        if (f) loadImage(f, f.name)
        e.target.value = ''  // re-selecting the same file must re-import
    }

    // Drop / paste an image (or a camera RAW) anywhere on the page.
    useEffect(() => {
        const onDrop = (e) => {
            e.preventDefault()
            const f = e.dataTransfer?.files?.[0]
            if (f) loadImage(f, f.name)
        }
        const onPaste = (e) => {
            const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'))
            const f = item?.getAsFile()
            if (f) loadImage(f, f.name || 'pasted image')
        }
        const stop = (e) => e.preventDefault()
        window.addEventListener('dragover', stop)
        window.addEventListener('drop', onDrop)
        window.addEventListener('paste', onPaste)
        return () => {
            window.removeEventListener('dragover', stop)
            window.removeEventListener('drop', onDrop)
            window.removeEventListener('paste', onPaste)
        }
    }, [loadImage])

    /* ── chain mutations ───────────────────────────────────────────────── */
    const commit = useCallback((layer, { select = true } = {}) => {
        const clean = sanitiseLayer(layer)
        setChain((prev) => [...prev, { layer: clean, op: prev.length === 0 ? 'replace' : 'add' }])
        if (select) setSelectedId(clean.id)
        return clean.id
    }, [])

    const updateLayer = useCallback((id, patch) => {
        if (id === 'base') { setBaseLayer((b) => (b ? { ...b, ...patch } : b)); return }
        setChain((prev) => prev.map((e) => (e.layer.id === id ? { ...e, layer: { ...e.layer, ...patch } } : e)))
    }, [])
    const setOp = useCallback((id, op) => {
        setChain((prev) => prev.map((e) => (e.layer.id === id ? { ...e, op } : e)))
    }, [])
    const setFillMode = useCallback((id, mode) => updateLayer(id, { fillMode: mode }), [updateLayer])
    const removeLayer = useCallback((id) => {
        setChain((prev) => prev.filter((e) => e.layer.id !== id))
        setSelectedId((s) => (s === id ? null : s))
    }, [])
    const moveLayer = useCallback((id, dir) => {
        setChain((prev) => {
            const i = prev.findIndex((e) => e.layer.id === id)
            if (i < 0) return prev
            const j = dir === 'up' ? i - 1 : i + 1
            if (j < 0 || j >= prev.length) return prev
            const next = prev.slice();[next[i], next[j]] = [next[j], next[i]]
            return next
        })
    }, [])

    // Register a texture-backed mask: stores the PRISTINE canvas under a base
    // key (kept untouched so boundary grow/shrink is reversible & non-cumulative)
    // and a live key the renderer samples. Returns both keys for the layer.
    const registerTexture = (canvas) => {
        const baseTextureKey = 'base-' + uid()
        const maskTextureKey = 'mask-' + uid()
        // Distinct canvases: nothing mutates this one in place today, but sharing
        // the object is a footgun — any future in-place edit would silently
        // rewrite the pristine base the boundary slider restores from.
        const base = document.createElement('canvas')
        base.width = canvas.width; base.height = canvas.height
        base.getContext('2d').drawImage(canvas, 0, 0)
        setMaskTexture(baseTextureKey, base)
        setMaskTexture(maskTextureKey, canvas)
        return { maskTextureKey, baseTextureKey, growPx: 0 }
    }

    // Readjust a texture mask's edge: grow (+) / shrink (−) its boundary by an
    // absolute px amount from the pristine base (0 restores it). Drives the
    // MaskChainCard "Boundary" slider for AI Subject / lasso / brush masks.
    const onExpandBoundary = useCallback((id, px) => {
        // Bug #1: read from chainRef (always current) instead of closed-over chain
        const layer = chainRef.current.find((e) => e.layer.id === id)?.layer
        if (!layer) return
        const base = getMaskTexture(layer.baseTextureKey || layer.maskTextureKey)
        if (!base || typeof base.getContext !== 'function') return // need a real canvas
        const grown = px === 0 ? base : growMaskCanvas(base, px)
        setMaskTexture(layer.maskTextureKey, grown)
        updateLayer(id, { growPx: px })
        bump()
    }, [updateLayer])  // no longer depends on `chain`

    /* ── add-mask handlers ─────────────────────────────────────────────── */
    const defaultFill = (kind) => {
        const m = getKindMeta(kind)
        const hex = (m.color || ACCENT).replace('#', '')
        const n = parseInt(hex.length === 3 ? hex.replace(/(.)/g, '$1$1') : hex, 16)
        return { r: ((n >> 16) & 255) / 255, g: ((n >> 8) & 255) / 255, b: (n & 255) / 255 }
    }
    const newFillProps = (kind) => ({ fillMode: 'fill', fillColor: defaultFill(kind), fillStrength: 0.55 })

    const addMask = useCallback((toolDef) => {
        if (!imageSize) return
        const { kind } = toolDef
        const imgSz = imageSize
        if (kind === 'radial') {
            const l = { ...radialLayer({ center: { x: W * 0.5, y: H * 0.5 }, radius: { x: W * 0.26, y: H * 0.26 }, feather: 0.5, imageSize: imgSz }), ...newFillProps(kind) }
            commit(l); setTool(null)
            setStatus('Radial added — drag the ring handles to resize, inside to move, top handle to rotate')
        } else if (kind === 'linear') {
            const l = { ...linearLayer({ p1: { x: W * 0.5, y: H * 0.2 }, p2: { x: W * 0.5, y: H * 0.7 }, feather: 0.5, imageSize: imgSz }), ...newFillProps(kind) }
            commit(l); setTool(null)
            setStatus('Linear added — drag the endpoints to aim the gradient, the line to move it')
        } else if (kind === 'path') {
            setDraft([]); setTool('pen')
            setStatus('Pen: click points on the image, then Close (or double-click)')
        } else if (kind === 'brush') {
            // Create an EMPTY brush texture + layer; strokes paint into it. The
            // coverage channel differs by kind: the plain brush samples ALPHA, so
            // its canvas must stay TRANSPARENT (an opaque base would read as a
            // full-frame selection and tint the whole image); smartBrush samples
            // RED, so it needs an OPAQUE black base to carry a soft R falloff.
            const key = 'brush-' + uid()
            const baseKey = 'brushbase-' + uid()
            const bc = document.createElement('canvas'); bc.width = W; bc.height = H
            const bx = bc.getContext('2d', { willReadFrequently: true })
            if (edgeAware) { bx.fillStyle = '#000'; bx.fillRect(0, 0, W, H) }
            // Base needs its OWN canvas: strokes mutate bc in place, so sharing the
            // object would let paint bleed into the pristine copy and make
            // boundary grow/shrink cumulative instead of reversible.
            const bbase = document.createElement('canvas'); bbase.width = W; bbase.height = H
            bbase.getContext('2d').drawImage(bc, 0, 0)
            setMaskTexture(key, bc); setMaskTexture(baseKey, bbase)
            // Edge-aware → smartBrush kind (bilateral edge snap in the shader);
            // else a plain brush. smartBrush samples brushTextureKey — we ALSO set
            // maskTextureKey=key so the boundary-grow + paint-sync use one key.
            const layerObj = edgeAware
                ? { ...smartBrushLayer({ brushTextureKey: key }), maskTextureKey: key }
                : { ...brushLayer({ maskTextureKey: key }) }
            const lid = commit({ ...layerObj, baseTextureKey: baseKey, growPx: 0, ...newFillProps(edgeAware ? 'smartBrush' : 'brush') })
            brushRef.current = { canvas: bc, ctx: bx, key, baseKey, layerId: lid, mode: edgeAware ? 'red' : 'alpha' }
            setTool('brush')
            setStatus(edgeAware ? 'Smart Brush (edge-aware): paint — it snaps to edges' : 'Brush: drag to paint · Alt-drag to erase')
        } else if (kind === 'luminance') {
            const l = { ...luminanceLayer({ min: 0.55, max: 1, softness: 0.15 }), ...newFillProps(kind) }
            commit(l); setTool(null)
        } else if (kind === 'color') {
            const l = { ...colorLayer({ target: { h: 0, s: 0.7, b: 0.7 }, tolerance: 0.18, softness: 0.12 }), ...newFillProps(kind) }
            commit(l); setTool('color')
            setStatus('Color: click the image to sample the target colour')
        }
    }, [imageSize, W, H, commit, edgeAware])

    /* ── pointer → image-space ─────────────────────────────────────────── */
    const toImage = (e) => {
        const r = overlayRef.current.getBoundingClientRect()
        return {
            x: ((e.clientX - r.left) / r.width) * W,
            y: ((e.clientY - r.top) / r.height) * H,
        }
    }

    // image-px per on-screen px (the overlay is rendered at image-res but
    // displayed scaled-to-fit), so hit-thresholds + handle sizes feel constant.
    const imgPerScreen = () => {
        const r = overlayRef.current?.getBoundingClientRect()
        return r && r.width ? W / r.width : 1
    }

    // Floating brush-size ring that tracks the cursor while a paint tool is
    // active (standalone Brush or brush-refine). Sized in screen px from the
    // image-space radius; positioned imperatively so moving it costs no re-render.
    const updateCursor = (e) => {
        const el = cursorRef.current
        if (!el) return
        if (tool !== 'brush' && tool !== 'refine') { el.style.display = 'none'; return }
        const stage = el.parentElement.getBoundingClientRect()
        const diam = (brushSize * 2) / imgPerScreen()
        el.style.display = 'block'
        el.style.width = `${diam}px`; el.style.height = `${diam}px`
        el.style.left = `${e.clientX - stage.left}px`; el.style.top = `${e.clientY - stage.top}px`
        el.classList.toggle('brush-cursor--erase', tool === 'refine' && (refineErase || e.altKey))
    }

    const onPointerDown = (e) => {
        if (!imageSize) return
        updateCursor(e)
        const p = toImage(e)
        // 1 · active drawing tools take precedence
        if (tool === 'pen') {
            if (draft.length >= 3 && dist(p, draft[0]) < Math.max(W, H) * 0.02) { closePen(); return }
            setDraft((d) => [...d, p]); return
        }
        if (tool === 'brush' || tool === 'refine') {
            const target = tool === 'refine' ? refineRef.current : brushRef.current
            if (!target) return
            const erase = (tool === 'refine' && refineErase) || e.altKey
            e.preventDefault(); overlayRef.current.setPointerCapture?.(e.pointerId)
            dragRef.current = { kind: 'paint', last: p, erase, target }
            stampInto(target.ctx, p, erase, target.mode); scheduleBrushSync(target); return
        }
        if (tool === 'color') { sampleColor(p); setTool(null); setStatus(''); return }
        if (tool === 'clickselect') {
            e.preventDefault()
            // Drop clicks while a selection is in flight rather than dropping a
            // marker the run behind it will never honour.
            if (aiBusyRef.current) return
            const label = e.altKey ? 0 : 1  // Alt+click = negative point
            const newPts = [...clickPointsRef.current, { x: p.x, y: p.y, label }]
            putClickPoints(newPts)
            onClickSelect(newPts)
            return
        }
        if (tool === 'sam') {
            e.preventDefault(); overlayRef.current.setPointerCapture?.(e.pointerId)
            dragRef.current = { kind: 'sambox', start: p }
            setSamBox({ x0: p.x, y0: p.y, x1: p.x, y1: p.y }); return
        }
        if (tool === 'ailasso') {
            e.preventDefault(); overlayRef.current.setPointerCapture?.(e.pointerId)
            // Sampled, not accumulated per event: a dense stroke costs the same
            // prompt (bbox + centroid) and a much cheaper clamp raster.
            dragRef.current = { kind: 'ailasso', pts: [p] }
            setAiLasso([p]); return
        }
        // 2 · otherwise: direct-manipulate the SELECTED spatial mask via handles
        const sel = chain.find((c) => c.layer.id === selectedId)?.layer
        if (!sel) return
        const thr = HANDLE_PX * imgPerScreen() * 1.4
        let grabbed = null
        if (sel.kind === 'radial') {
            const h = radialHandles(sel)
            const hit = nearestHandle(h, p, thr, ['rot', 'e', 'w', 'n', 's'])
            if (hit === 'rot') grabbed = { kind: 'radial-rot', center: { ...sel.center } }
            else if (hit) grabbed = { kind: 'radial-resize', axis: hit, center: { ...sel.center }, rotation: sel.rotation || 0, radius: { ...sel.radius } }
            else if (insideEllipse(sel, p)) grabbed = { kind: 'radial-move', offset: { x: sel.center.x - p.x, y: sel.center.y - p.y } }
        } else if (sel.kind === 'linear') {
            const h = linearHandles(sel)
            const hit = nearestHandle(h, p, thr, ['p1', 'p2', 'mid'])
            if (hit === 'p1') grabbed = { kind: 'linear-p1' }
            else if (hit === 'p2') grabbed = { kind: 'linear-p2' }
            else if (hit === 'mid' || distToSeg(sel.p1, sel.p2, p) <= thr * 1.6) {
                grabbed = { kind: 'linear-move', o1: { x: sel.p1.x - p.x, y: sel.p1.y - p.y }, o2: { x: sel.p2.x - p.x, y: sel.p2.y - p.y } }
            }
        }
        if (grabbed) { e.preventDefault(); overlayRef.current.setPointerCapture?.(e.pointerId); dragRef.current = grabbed }
    }

    const onPointerMove = (e) => {
        updateCursor(e)
        const d = dragRef.current
        if (!d) return
        const p = toImage(e)
        const id = selectedId
        if (d.kind === 'paint') { stampLineInto(d.target.ctx, d.last, p, d.erase, d.target.mode); d.last = p; scheduleBrushSync(d.target); return }
        if (d.kind === 'sambox') { setSamBox({ x0: d.start.x, y0: d.start.y, x1: p.x, y1: p.y }); return }
        if (d.kind === 'ailasso') {
            const last = d.pts[d.pts.length - 1]
            if (dist(last, p) < Math.max(W, H) * 0.004) return
            d.pts.push(p)
            setAiLasso(d.pts.slice())
            return
        }
        if (d.kind === 'radial-move') {
            updateLayer(id, { center: { x: p.x + d.offset.x, y: p.y + d.offset.y } })
        } else if (d.kind === 'radial-resize') {
            const a = d.rotation, c = d.center
            const dx = p.x - c.x, dy = p.y - c.y
            const lx = dx * Math.cos(a) + dy * Math.sin(a)   // project onto the rotated axes
            const ly = -dx * Math.sin(a) + dy * Math.cos(a)
            const radius = (d.axis === 'e' || d.axis === 'w')
                ? { x: Math.max(6, Math.abs(lx)), y: d.radius.y }
                : { x: d.radius.x, y: Math.max(6, Math.abs(ly)) }
            updateLayer(id, { radius })
        } else if (d.kind === 'radial-rot') {
            updateLayer(id, { rotation: Math.atan2(p.y - d.center.y, p.x - d.center.x) + Math.PI / 2 })
        } else if (d.kind === 'linear-p1') {
            updateLayer(id, { p1: p })
        } else if (d.kind === 'linear-p2') {
            updateLayer(id, { p2: p })
        } else if (d.kind === 'linear-move') {
            updateLayer(id, { p1: { x: p.x + d.o1.x, y: p.y + d.o1.y }, p2: { x: p.x + d.o2.x, y: p.y + d.o2.y } })
        }
    }

    const onPointerUp = () => {
        const d = dragRef.current
        dragRef.current = null
        if (d?.kind === 'sambox') {
            const x0 = Math.max(0, Math.min(d.start.x, samBox?.x1 ?? d.start.x))
            const y0 = Math.max(0, Math.min(d.start.y, samBox?.y1 ?? d.start.y))
            const x1 = Math.min(W, Math.max(d.start.x, samBox?.x1 ?? d.start.x))
            const y1 = Math.min(H, Math.max(d.start.y, samBox?.y1 ?? d.start.y))
            setSamBox(null)
            if (x1 - x0 >= 6 && y1 - y0 >= 6) onSamBox([x0, y0, x1, y1])
            else setStatus('Box too small — drag a larger rectangle around the object')
            return
        }
        if (d?.kind === 'ailasso') {
            const pts = d.pts
            setAiLasso(null)
            if (pts.length >= 3) onAiLasso(pts)
            else setStatus('Loop too short — draw around the object')
            return
        }
        if (d?.kind === 'paint' && d.target) {
            const t = d.target
            setMaskTexture(t.key, t.canvas)
            // snapshot the painted canvas as the boundary base so grow/shrink
            // readjusts from the latest stroke (resets any prior boundary edit)
            if (t.baseKey) {
                // Bug #7: use actual canvas dimensions, not closure W/H
                const cw = t.canvas.width, ch = t.canvas.height
                const snap = document.createElement('canvas'); snap.width = cw; snap.height = ch
                snap.getContext('2d').drawImage(t.canvas, 0, 0)
                setMaskTexture(t.baseKey, snap)
            }
            if (t.layerId) updateLayer(t.layerId, { growPx: 0 })
            bump()
        }
    }

    const closePen = () => {
        if (draft.length >= 3) {
            const pts = draft
            const anchors = penSmooth ? smoothToBezier(pts, { closed: true }) : pts
            const canvas = rasterisePath(anchors, W, H, { closed: true })
            const keys = registerTexture(canvas)
            const kind = penSmooth ? 'path' : 'lasso'
            const factory = penSmooth ? pathLayer : lassoLayer
            // keep the raw points on the layer so the selection outline can be
            // traced on the canvas (the rasterised texture has no path info).
            const l = { ...factory({ maskTextureKey: keys.maskTextureKey, feather: penSmooth ? 0.04 : 0.05 }), baseTextureKey: keys.baseTextureKey, growPx: 0, ...newFillProps(kind), points: pts }
            commit(l)
        }
        setDraft([]); setTool(null); setStatus('')
    }

    /* ── brush painting (shared by the standalone Brush + brush-refine) ──── */
    // Stamp one soft dab into a target ctx. `mode` picks the coverage channel:
    //  • 'alpha' (plain brush): coverage IS the alpha — add paints white alpha,
    //    erase punches alpha out (destination-out), since painting black over
    //    alpha would only raise it.
    //  • 'red' (semantic / smart-brush, opaque A=255): coverage is the red/luma —
    //    add paints white, erase paints black, both source-over.
    const stampInto = (ctx, p, erase, mode) => {
        if (!ctx) return
        const r = brushSize
        if (mode === 'alpha') {
            ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over'
            const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r)
            g.addColorStop(0, 'rgba(255,255,255,0.85)'); g.addColorStop(1, 'rgba(255,255,255,0)')
            ctx.fillStyle = g
            ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill()
            ctx.globalCompositeOperation = 'source-over'
        } else {
            const c = erase ? '0,0,0' : '255,255,255'
            const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r)
            g.addColorStop(0, `rgba(${c},0.85)`); g.addColorStop(1, `rgba(${c},0)`)
            ctx.fillStyle = g
            ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill()
        }
    }
    const stampLineInto = (ctx, a, b, erase, mode) => {
        const d = Math.hypot(b.x - a.x, b.y - a.y)
        const step = Math.max(1, brushSize * 0.25)
        const n = Math.max(1, Math.ceil(d / step))
        for (let i = 1; i <= n; i++) stampInto(ctx, { x: a.x + (b.x - a.x) * (i / n), y: a.y + (b.y - a.y) * (i / n) }, erase, mode)
    }
    // Push the painted canvas back to its mask texture (rAF-coalesced) + redraw.
    const scheduleBrushSync = (target) => {
        if (rafRef.current) return
        rafRef.current = requestAnimationFrame(() => {
            rafRef.current = 0
            if (target) setMaskTexture(target.key, target.canvas)
            bump()
        })
    }

    // Brush-refine: paint directly on the SELECTED texture mask (AI Subject /
    // brush) to add (drag) or remove (Alt-drag or the Erase toggle) coverage.
    // The texture is normalised to a W×H working canvas so the image-space brush
    // maths is shared with the standalone Brush regardless of the mask's native
    // resolution, while preserving the kind's coverage channel ('alpha' for the
    // plain brush, opaque 'red' for semantic / smart-brush).
    const startRefine = useCallback(() => {
        const sel = chain.find((e) => e.layer.id === selectedId)?.layer
        if (!sel) return
        const key = sel.maskTextureKey || sel.brushTextureKey
        if (!key) return
        const mode = sel.kind === 'brush' ? 'alpha' : 'red'
        const tex = getMaskTexture(key)
        const cv = document.createElement('canvas'); cv.width = W; cv.height = H
        const cx = cv.getContext('2d', { willReadFrequently: true })
        if (mode === 'red') { cx.fillStyle = '#000'; cx.fillRect(0, 0, W, H) }
        if (tex && tex.width) cx.drawImage(tex, 0, 0, W, H)
        setMaskTexture(key, cv)
        refineRef.current = { canvas: cv, ctx: cx, key, baseKey: sel.baseTextureKey, layerId: sel.id, mode }
        setRefineErase(false)
        setTool('refine')
        setStatus('Brush-refine: drag to add to the mask · Alt-drag (or Erase) to remove')
    }, [chain, selectedId, W, H])

    // Tone curves — build the packed 256×1 LUT from per-channel points (reusing
    // phosmith's buildLut math via curve-lut), register it as a texture, and
    // point the layer at it. Identity curves clear the LUT (engine skips it).
    const applyCurve = useCallback((id, curves) => {
        const { packed, identity } = buildPackedLutFromCurves(curves || {})
        if (identity) { updateLayer(id, { curveLutKey: undefined, curves: undefined }); bump(); return }
        const key = 'curve-' + id
        setMaskTexture(key, new ImageData(new Uint8ClampedArray(packed), 256, 1))
        updateLayer(id, { curveLutKey: key, curves })
        bump()
    }, [updateLayer])

    /* ── undo / redo — FULL history incl. mask textures ─────────────────── */
    // A snapshot captures the chain + base grade AND a clone of every mask
    // texture they reference (brush strokes, AI masks, boundary edits, curve
    // LUTs), so ANY edit is reversible — not just parametric ones. Changes are
    // debounced into history so a drag collapses to a single undo step.
    // SAM's other granularity candidates for the most recent AI mask:
    // { layerId, count, index }. Null whenever nothing on screen owns them.
    const [cycleTarget, setCycleTarget] = useState(null)
    const cycleTargetRef = useRef(null)
    cycleTargetRef.current = cycleTarget

    const historyRef = useRef({ past: [], present: null, future: [] })
    const histTimerRef = useRef(0)
    const restoringRef = useRef(false)
    const TEX_KEYS = ['maskTextureKey', 'baseTextureKey', 'brushTextureKey', 'depthMapKey', 'curveLutKey']
    const cloneTex = (tex) => {
        if (!tex) return null
        if (typeof tex.getContext === 'function') {
            const cv = document.createElement('canvas'); cv.width = tex.width; cv.height = tex.height
            cv.getContext('2d').drawImage(tex, 0, 0); return cv
        }
        if (typeof ImageData !== 'undefined' && tex instanceof ImageData) return new ImageData(new Uint8ClampedArray(tex.data), tex.width, tex.height)
        return null
    }
    const snapTextures = (chainArr, base) => {
        const keys = new Set()
        const collect = (l) => l && TEX_KEYS.forEach((k) => l[k] && keys.add(l[k]))
        chainArr.forEach((e) => collect(e.layer)); collect(base)
        const map = new Map()
        keys.forEach((k) => { const c = cloneTex(getMaskTexture(k)); if (c) map.set(k, c) })
        return map
    }
    // Immutable-update patterns (updateLayer/applyCurve always replace nested
    // objects) make a shallow layer copy safe to retain in a snapshot.
    // Bug #2: reads from refs so the debounced timer always gets current state.
    // U8: structural sharing — only clone textures whose version changed.
    const snapshot = () => {
        const curChain = chainRef.current
        const curBase = baseLayerRef.current
        const keys = new Set()
        const collect = (l) => l && TEX_KEYS.forEach((k) => l[k] && keys.add(l[k]))
        curChain.forEach((e) => collect(e.layer)); collect(curBase)

        const prevVersions = lastSnapVersionsRef.current
        const prevTextures = historyRef.current.present?.textures
        const nextVersions = new Map()
        const textures = new Map()
        keys.forEach((k) => {
            const ver = getMaskTextureVersion(k)
            nextVersions.set(k, ver)
            if (prevVersions.get(k) === ver && prevTextures?.has(k)) {
                textures.set(k, prevTextures.get(k))  // reuse — unchanged since last snap
            } else {
                const c = cloneTex(getMaskTexture(k))
                if (c) textures.set(k, c)
            }
        })
        lastSnapVersionsRef.current = nextVersions
        return {
            chain: curChain.map((e) => ({ op: e.op, layer: { ...e.layer } })),
            base: curBase ? { ...curBase } : null,
            textures,
        }
    }
    const restore = (snap) => {
        snap.textures.forEach((tex, key) => { const c = cloneTex(tex); if (c) setMaskTexture(key, c) })
        restoringRef.current = true
        setChain(snap.chain.map((e) => ({ op: e.op, layer: { ...e.layer } })))
        setBaseLayer(snap.base ? { ...snap.base } : null)
        bump()
    }
    const undo = useCallback(() => {
        const h = historyRef.current
        if (!h.past.length) return
        const prev = h.past.pop()
        if (h.present) h.future.push(h.present)
        h.present = prev
        restore(prev)
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])
    const redo = useCallback(() => {
        const h = historyRef.current
        if (!h.future.length) return
        const next = h.future.pop()
        if (h.present) h.past.push(h.present)
        h.present = next
        restore(next)
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    /* ── on-device AI masks (seglab SAM 2.1 lane via ai/studio-bridge.js) ─── */
    // Every model runs in this browser on WebGPU. Nothing is uploaded, there is
    // no service to probe and no fallback engine to pick between: one lane, so
    // the app can always say which model produced a mask.

    // Boot the engine once: probe the device, size the budget, start the memory
    // governor, then warm the lane speculatively behind the first import.
    useEffect(() => {
        let alive = true
        let offEvent = null
        ;(async () => {
            try {
                const mod = await engine()
                offEvent = mod.onEngineEvent((event) => {
                    if (!alive) return
                    if (event.type === 'progress') {
                        const d = event.detail || {}
                        if (!d.total) return
                        aiSet({ progress: { file: d.file || d.lane || 'model', loaded: d.loaded || 0, total: d.total } })
                        if ((d.loaded || 0) >= d.total) setTimeout(() => alive && aiSet({ progress: null }), 400)
                    } else if (event.type === 'pressure') {
                        aiSet({ pressure: event.level, status: event.level >= 3
                            ? 'Memory pressure — running in safe mode.'
                            : 'Memory pressure — heavy caches released.' })
                    } else if (event.type === 'waiting') {
                        aiSet({ status: 'Waiting for the segmentation lane…' })
                    } else {
                        aiSet({ engine: mod.status() })
                    }
                })
                await mod.boot({
                    // The governor's ledger is the only memory signal that exists
                    // on WebKit, so every buffer React owns has to be in it —
                    // including the mask textures, which are the term that grows
                    // with use (each texture-backed layer holds a live canvas
                    // plus its pristine boundary base, both at frame size).
                    measure: () => {
                        const px = (c) => (c ? c.width * c.height * 4 : 0)
                        const frame = px(srcRef.current)
                        return {
                            pixelBytes: frame + px(overlayRef.current) + px(dispRef.current)
                                + textureLayers.current * frame * 2,
                        }
                    },
                })
                if (!alive) return
                aiSet({ engine: mod.status() })
                await mod.warm({ withEncoder: false })
                if (alive) aiSet({ engine: mod.status() })
            } catch (err) {
                console.error('[studio] engine boot failed', err)
                if (alive) aiSet({ status: 'Engine unavailable: ' + (err?.message || err) })
            }
        })()
        return () => { alive = false; offEvent?.() }
    }, [aiSet])

    // Register a lane mask as a new semantic layer. `res` is what the bridge
    // returns; its canvas is already the opaque R=G=B=coverage texture the
    // semantic shader samples, at exactly the working canvas's dimensions.
    const commitAiMask = useCallback((res, { label, invert = false, feather = 0.02 } = {}) => {
        const keys = registerTexture(res.canvas)
        const id = commit({
            ...semanticLayer({ maskTextureKey: keys.maskTextureKey, feather, label }),
            baseTextureKey: keys.baseTextureKey,
            growPx: 0,
            inverted: invert,
            // Which of the lane's parked score fields this mask came from. The
            // adapter holds exactly ONE, and imageKey cannot tell two masks of
            // the same photo apart, so an export has to prove the field it is
            // about to matte still belongs to this layer.
            decodeId: res.decodeId || 0,
            ...newFillProps('semantic'),
        })
        // SAM computed three granularity candidates for this prompt and they are
        // already in memory. Arm the cycling control on the layer that owns them
        // — a later selection parks its own set and takes the arming with it.
        setCycleTarget(res.candidates && res.candidates.count > 1
            ? { layerId: id, count: res.candidates.count, index: res.candidates.index }
            : null)
        return id
    }, [commit])


    // One place where every AI tool's async lifecycle lives: busy latch, stale
    // guard, engine-status refresh and error surface. Tools describe the run;
    // they never re-implement it.
    const runAi = useCallback(async (startStatus, work) => {
        if (aiBusyRef.current || !srcRef.current) return null
        aiBusyRef.current = true
        aiSet({ busy: true, status: startStatus })
        try {
            const mod = await engine()
            const res = await work(mod, srcRef.current)
            if (!res || res.stale) { aiSet({ busy: false, status: '' }); return null }
            if (!res.usable) { aiSet({ busy: false, status: res.reason || 'Nothing selectable there — try another prompt' }); return null }
            aiSet({ busy: false, engine: mod.status(), status: '' })
            return res
        } catch (err) {
            console.error('[studio] ai run failed', err)
            aiSet({ busy: false, status: (err?.message || String(err)) })
            return null
        } finally {
            aiBusyRef.current = false
        }
    }, [aiSet])

    /**
     * Step to SAM's next / previous candidate for the selected AI mask.
     *
     * Arbitration picks the best default, but a first click on a genuinely
     * ambiguous subject has no single right answer — "the petal" and "the bloom"
     * are both correct for one point on a rose — and the subject prior can pick
     * the right OBJECT at the wrong scope. The other two planes are already in
     * memory, so this is a repaint, not a decode.
     */
    const cycleMask = useCallback(async (delta) => {
        const target = cycleTargetRef.current
        if (!target || aiBusyRef.current) return
        const entry = chainRef.current.find((e) => e.layer.id === target.layerId)
        if (!entry) { setCycleTarget(null); return }
        const res = await runAi('Trying SAM\u2019s next candidate\u2026', (mod) => mod.cycleMask(delta))
        if (!res) return
        const l = entry.layer
        if (l.maskTextureKey) setMaskTexture(l.maskTextureKey, res.canvas)
        // Re-base the boundary grow/shrink origin, same as a click-select refine.
        if (l.baseTextureKey) {
            const snap = document.createElement('canvas'); snap.width = W; snap.height = H
            snap.getContext('2d').drawImage(res.canvas, 0, 0)
            setMaskTexture(l.baseTextureKey, snap)
        }
        // The layer change plus bump() are what push the undo entry: the history
        // effect snapshots textures whose version moved, and setMaskTexture
        // moved both of these.
        updateLayer(l.id, { growPx: 0, decodeId: res.decodeId || 0 })
        const c = res.candidates
        setCycleTarget(c && c.count > 1 ? { layerId: l.id, count: c.count, index: c.index } : null)
        bump()
        aiSet({ status: `Candidate ${(c?.index ?? 0) + 1} of ${c?.count ?? 1} — ${(res.coverage * 100).toFixed(0)}% coverage · [ and ] cycle` })
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [runAi, updateLayer, aiSet, W, H])

    // AI Subject / AI Sky·Background — the same mask, inverted for the second.
    // See studio-bridge.selectSubject: prompts come from a pixel-statistics
    // prior, every probe is decode-only, and the post pipeline runs once.
    const runSubject = useCallback(async ({ invert = false } = {}) => {
        const noun = invert ? 'Background' : 'Subject'
        const res = await runAi(`AI ${noun}: segmenting…`, (mod, canvas) => mod.selectSubject(canvas))
        if (!res) return
        commitAiMask(res, { label: invert ? 'AI Sky / Background' : 'AI Subject', invert })
        // Per-probe audit for bench/subject.mjs — which prompts ran, what each
        // cost, and why the winner won. Debug surface only; nothing reads it.
        if (typeof window !== 'undefined' && res.probes) {
            window.__lastSubject = { probes: res.probes, postMs: res.postMs, won: res.probe }
        }
        aiSet({ status: `${noun} masked — SAM 2.1 (${res.probe}, ${(res.coverage * 100).toFixed(0)}% coverage, ${res.ms} ms)`
            + (invert ? ' → inverted = the entire sky / background' : '')
            + ((res.candidates?.count || 0) > 1 ? ' · wrong scope? press [ or ]' : '') })
    }, [runAi, commitAiMask, aiSet])

    // AI Box-Select — drag a box, SAM returns the object inside it.
    const onSamBox = useCallback(async (box) => {
        const res = await runAi('Box-Select: segmenting…', (mod, canvas) => mod.select(canvas, { box }))
        if (!res) return
        commitAiMask(res, { label: 'AI Box-Select' })
        aiSet({ status: `Selected in ${res.ms} ms — drag another box, or press Done` })
    }, [runAi, commitAiMask, aiSet])

    // AI Lasso — a rough loop is a PROMPT, not a cut: its bbox and centroid go
    // to SAM, which snaps to the real object boundary, and the result is then
    // clamped to the loop ∪ margin so it can never bleed onto a neighbour.
    const onAiLasso = useCallback(async (poly) => {
        const res = await runAi('Lasso: snapping to the object…',
            (mod, canvas) => mod.select(canvas, { lasso: poly.map((p) => [p.x, p.y]) }))
        if (!res) return
        commitAiMask(res, { label: 'AI Lasso' })
        aiSet({ status: `Snapped to the object inside the loop (${res.ms} ms) — draw another, or press Done` })
    }, [runAi, commitAiMask, aiSet])

    /**
     * AI Click-Select / predictive refine. Each click segments the object under
     * the cursor and composites it INTO the active mask — a plain click unions
     * it in, Alt+click subtracts it. SAM is always prompted POSITIVELY at the
     * point (to find what is there); the click's intent is applied afterwards by
     * sam-core's op model, which keeps the refined boundary's soft falloff on an
     * add and dilates the subtract channel so removing an object leaves no
     * one-pixel residue ring. With no mask active the first click seeds a fresh
     * layer instead.
     */
    const onClickSelect = useCallback(async (pts) => {
        const last = pts[pts.length - 1]
        if (!last) return
        const add = last.label !== 0
        const res = await runAi(add ? 'Click-Select: finding the region to add…' : 'Click-Select: finding the region to remove…',
            (mod, canvas) => mod.select(canvas, { clicks: [[last.x, last.y, 1]] }))
        if (!res) return
        const mod = await engine()
        const sel = clickMaskId ? chain.find((e) => e.layer.id === clickMaskId) : null
        if (sel) {
            const key = sel.layer.maskTextureKey
            const merged = mod.composeMask(key ? getMaskTexture(key) : null, res.canvas, add ? 'add' : 'sub', W, H)
            if (key) setMaskTexture(key, merged)
            // Re-base the boundary grow/shrink origin to the refined edge.
            if (sel.layer.baseTextureKey) {
                const snap = document.createElement('canvas'); snap.width = W; snap.height = H
                snap.getContext('2d').drawImage(merged, 0, 0)
                setMaskTexture(sel.layer.baseTextureKey, snap)
            }
            updateLayer(clickMaskId, { growPx: 0 })
            bump()
            aiSet({ status: `${add ? 'Added' : 'Removed'} region (${res.ms} ms) — click to ADD · Alt+click to REMOVE · Done to finish` })
        } else {
            setClickMaskId(commitAiMask(res, { label: 'AI Click-Select' }))
            aiSet({ status: `Selected (${res.ms} ms) — click to ADD another region · Alt+click to REMOVE · Done to finish` })
        }
    }, [runAi, commitAiMask, aiSet, clickMaskId, chain, updateLayer, W, H])

    // Engine self-check: what the device actually resolved to, and a real
    // end-to-end selection on the current frame rather than a synthetic scene.
    const testDevice = useCallback(async () => {
        aiSet({ busy: true, status: 'Checking the device…' })
        try {
            const mod = await engine()
            const { capability } = await mod.boot()
            await mod.warm({ withEncoder: true })
            const st = mod.status()
            const canvas = srcRef.current
            let probeMs = null
            let ok = false
            if (canvas) {
                const t0 = Date.now()
                const r = await mod.select(canvas, { clicks: [[canvas.width / 2, canvas.height / 2, 1]] })
                probeMs = Date.now() - t0
                ok = !!r && !r.stale && r.usable
            }
            aiSet({
                busy: false,
                engine: st,
                status: ok
                    ? `On-device AI works — ${st.lane} on ${String(st.device || 'webgpu').toUpperCase()}, selection in ${probeMs} ms`
                    : 'Engine is up but the probe selection returned nothing usable',
                report: {
                    caps: [
                        ['WebGPU', !!capability?.webgpu],
                        ['shader-f16', !!capability?.f16],
                        [String(capability?.gpuTier || 'gpu'), capability?.gpuTier === 'accelerated'],
                        ['Selection', ok],
                    ],
                },
            })
        } catch (err) { aiSet({ busy: false, status: 'Device check failed: ' + (err?.message || err) }) }
    }, [aiSet])

    /* ── HD export ─────────────────────────────────────────────────────────
     * The graded frame at export resolution. The original is re-decoded from
     * the bytes asset-store kept (never a resident full-res RGBA), the mask
     * textures upscale cleanly because they carry continuous coverage rather
     * than a bitmask, and the megashader renders the same chain at the larger
     * size. Bounded by the budget, so the peak is predictable. */
    const exportHd = useCallback(async () => {
        if (!srcRef.current || aiBusyRef.current) return
        aiBusyRef.current = true   // shares runAi's latch: an export blocks AI runs and vice versa
        aiSet({ busy: true, status: 'Re-decoding the original for export…' })
        let owned = null
        try {
            const mod = await engine()
            const out = await mod.exportSource()
            if (!out) throw new Error('no original held — re-import the image')
            owned = out.owned ? out.source : null
            const hd = document.createElement('canvas')
            hd.width = out.width
            hd.height = out.height
            hd.getContext('2d').drawImage(out.source, 0, 0, hd.width, hd.height)

            // Bug #4: guard against W===0 (imageSize null during rapid swap)
            if (!W || !H) { aiSet({ busy: false, stage: '' }); return }
            const k = hd.width / W
            const scaled = chain.map((e) => ({ op: e.op, layer: scaleLayer(e.layer, k) }))
            let working = hd
            if (hasGrade(baseLayer)) {
                const baked = renderMegashader(hd, { chain: [{ layer: sanitiseLayer(scaleLayer(baseLayer, k)), op: 'replace' }] }, {})
                if (baked) working = baked
            }
            const result = renderMegashader(working, { chain: scaled.map((e) => ({ layer: sanitiseLayer(e.layer), op: e.op })) }, { globalInvert })
            const src = result || working
            // PNG encode off-thread: OffscreenCanvas.convertToBlob runs the
            // encoder on a compositor thread, removing ~45 ms of main-thread
            // blocking. The drawImage is cheap (canvas→canvas, same process).
            // Falls back to the synchronous toBlob for browsers without
            // OffscreenCanvas (Safari < 16.4).
            let blob
            if (typeof OffscreenCanvas !== 'undefined') {
                const osc = new OffscreenCanvas(src.width, src.height)
                osc.getContext('2d').drawImage(src, 0, 0)
                blob = await osc.convertToBlob({ type: 'image/png' })
            } else {
                blob = await new Promise((res) => src.toBlob(res, 'image/png'))
            }
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `mask-studio-${hd.width}x${hd.height}.png`
            a.click()
            setTimeout(() => URL.revokeObjectURL(url), 10_000)
            aiSet({ busy: false, status: `Exported ${hd.width}×${hd.height}${out.bounded ? ' (bounded by the memory budget)' : ''}` })
        } catch (err) {
            console.error('[studio] export failed', err)
            aiSet({ busy: false, status: 'Export failed: ' + (err?.message || err) })
        } finally {
            aiBusyRef.current = false
            try { owned?.close?.() } catch { /* already gone */ }
        }
    }, [aiSet, chain, baseLayer, globalInvert, W])

    // Unified palette dispatch: AI tools run the on-device lane; everything else
    // opens its canvas interaction — no separation between AI and regular masks.
    const onToolClick = useCallback((t) => {
        if (t.ai === 'subject') return runSubject()
        if (t.ai === 'background') return runSubject({ invert: true })
        if (t.ai === 'sam') { setTool('sam'); setStatus('Drag a box around an object to select it'); return }
        if (t.ai === 'ailasso') { setAiLasso(null); setTool('ailasso'); setStatus('Draw a rough loop around the object — release to snap'); return }
        if (t.ai === 'clickselect') {
            putClickPoints([])
            // If a semantic mask (AI Subject / Box-Select / a prior Click-Select)
            // is selected, REFINE it: clicks add/remove regions on that mask. With
            // nothing selected, the first click seeds a fresh selection instead.
            const sel = chain.find((e) => e.layer.id === selectedId)?.layer
            const refine = !!sel && sel.kind === 'semantic' && !!sel.maskTextureKey
            setClickMaskId(refine ? sel.id : null)
            setTool('clickselect')
            setStatus(refine
                ? 'Refining the selected mask · click to ADD a region · Alt+click to REMOVE · Done to finish'
                : 'Click on an object to select it · Alt+click to exclude regions · click more to refine')
            return
        }
        return addMask(t)
    }, [runSubject, addMask, chain, selectedId])

    const sampleColor = (p) => {
        const src = srcRef.current
        if (!src || !selectedId) return
        const x = Math.max(0, Math.min(W - 1, Math.round(p.x)))
        const y = Math.max(0, Math.min(H - 1, Math.round(p.y)))
        const d = src.getContext('2d', { willReadFrequently: true }).getImageData(x, y, 1, 1).data
        const hsb = rgbToHsb(d[0], d[1], d[2])
        updateLayer(selectedId, { target: { h: hsb.h, s: hsb.s, b: hsb.b } })
    }

    // Commit settled state into history (debounced so a drag = one step). The
    // `restoringRef` guard prevents an undo/redo from itself pushing a new entry.
    useEffect(() => {
        if (restoringRef.current) { restoringRef.current = false; historyRef.current.present = snapshot(); return }
        clearTimeout(histTimerRef.current)
        histTimerRef.current = setTimeout(() => {
            const snap = snapshot()
            const h = historyRef.current
            if (h.present) { h.past.push(h.present); if (h.past.length > 60) h.past.shift() }
            h.present = snap
            h.future = []
        }, 350)
        return () => clearTimeout(histTimerRef.current)
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [chain, baseLayer, tick])

    // ⌘/Ctrl+Z undo · ⌘/Ctrl+Shift+Z (or Ctrl+Y) redo · [ / ] cycle candidates.
    // Through a ref so the binding is installed once: cycleMask closes over the
    // chain, and re-registering a window listener per keystroke-worth of state
    // is how a repaint turns into a leak.
    const cycleMaskRef = useRef(cycleMask)
    cycleMaskRef.current = cycleMask
    useEffect(() => {
        const onKey = (e) => {
            if (!(e.metaKey || e.ctrlKey)) {
                if (e.key !== '[' && e.key !== ']') return
                const t = e.target
                if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return
                e.preventDefault()
                cycleMaskRef.current(e.key === '[' ? -1 : 1)
                return
            }
            const k = e.key.toLowerCase()
            if (k === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo() }
            else if (k === 'y') { e.preventDefault(); redo() }
        }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
    }, [undo, redo])

    // Zoom and pan. The bindings belong to the DEVICE, not the tool — the same
    // two-finger movement is a scroll on a trackpad and a pinch on glass — so
    // they live in their own module. Loaded through a runtime import like the
    // rest of ai/, which keeps that directory out of the bundle.
    const hasImageRef = useRef(false)
    hasImageRef.current = hasImage
    useEffect(() => {
        if (!hasImage) return
        let g = null
        let dead = false
        import(/* @vite-ignore */ new URL('ai/gestures.js', document.baseURI).href)
            .then(({ createGestures }) => {
                if (dead || !stageRef.current || !frameRef.current) return
                g = createGestures({
                    stage: stageRef.current,
                    frame: frameRef.current,
                    surface: overlayRef.current,
                    hint: gestureHintRef.current,
                    readout: zoomRef.current,
                    active: () => hasImageRef.current,
                    // A second finger is a view gesture, never a stroke: drop
                    // whatever the first one started, before it commits a mask
                    // nobody asked for.
                    onGestureStart: () => {
                        dragRef.current = null
                        setDraft([])
                        setSamBox(null)
                        setAiLasso(null)
                    },
                })
            })
            .catch((err) => console.warn('[studio] gestures unavailable', err))
        return () => { dead = true; g?.destroy() }
    }, [hasImage])

    /* ── live render ───────────────────────────────────────────────────── */
    // Deliberately NOT rAF-coalesced. It looks like it should be — every pass
    // renders the whole proxy on the GPU and reads it back — but the renderer's own
    // drawCount says the waste is not there: measured over paced 120 Hz drags, a
    // brush stroke draws 17 times in 52 frames, a rubber-band once in 43, and a
    // gradeless handle drag not at all (the stack short-circuits). React's batching
    // and scheduleBrushSync's rAF already hold it to at most one draw per frame.
    // Adding another rAF only pushed the brush a frame later — 5 frames over 25 ms
    // in a stroke that previously had none.
    useEffect(() => {
        const src = srcRef.current
        const disp = dispRef.current
        if (!src || !disp) return
        let result
        try {
            if (compare) {
                // Before/after — the ungraded source (no base, no chain).
                result = src
            } else {
                // Pass 1: bake the global/base grade into a full-frame canvas so
                // local mask grades STACK on top of it. The engine grades the
                // ORIGINAL source per-layer (glsl c_i = layerColor(srcRgb)), so a
                // base applied as just another chain layer would NOT stack — it
                // must be pre-baked and fed in as the source for pass 2.
                let working = src
                if (hasGrade(baseLayer)) {
                    const baked = renderMegashader(
                        src,
                        { chain: [{ layer: sanitiseLayer(baseLayer), op: 'replace' }] },
                        {},
                    )
                    if (baked) working = baked
                }
                // Pass 2: the mask chain over the (optionally) base-graded image.
                result = renderMegashader(
                    working,
                    { chain: chain.map((e) => ({ layer: sanitiseLayer(e.layer), op: e.op })) },
                    { maskOverlay: overlayMode, globalInvert, overlayColor: { r: 0.33, g: 0.85, b: 1 } },
                )
            }
        } catch (err) {
            console.error('[studio] render failed', err)
            window.__error = String(err)
            return
        }
        if (!result) return
        if (disp.width !== result.width) disp.width = result.width
        if (disp.height !== result.height) disp.height = result.height
        const ctx = disp.getContext('2d')
        ctx.clearRect(0, 0, disp.width, disp.height)
        ctx.drawImage(result, 0, 0)
    }, [chain, baseLayer, compare, overlayMode, globalInvert, tick, imageSize])

    /* ── selection outline + handles ───────────────────────────────────── */
    useEffect(() => {
        const ov = overlayRef.current
        if (!ov || !imageSize) return
        if (ov.width !== W) ov.width = W
        if (ov.height !== H) ov.height = H
        const g = ov.getContext('2d')
        g.clearRect(0, 0, W, H)
        if (preview) return  // clean view — no outlines/handles
        const ipx = imgPerScreen()
        const HR = HANDLE_PX * ipx            // handle radius in image px
        const LW = 1.7 * ipx                  // outline width in image px
        const handle = (x, y, c = ACCENT) => {
            g.setLineDash([])
            g.beginPath(); g.arc(x, y, HR, 0, Math.PI * 2)
            g.fillStyle = c; g.fill()
            g.lineWidth = 1.5 * ipx; g.strokeStyle = 'rgba(8,10,14,0.95)'; g.stroke()
        }
        const sel = chain.find((e) => e.layer.id === selectedId)?.layer

        if (sel?.kind === 'radial' && sel.center && sel.radius) {
            g.save(); g.translate(sel.center.x, sel.center.y); g.rotate(sel.rotation || 0)
            g.fillStyle = 'rgba(83,216,255,0.10)'
            g.beginPath(); g.ellipse(0, 0, sel.radius.x, sel.radius.y, 0, 0, Math.PI * 2); g.fill()
            g.lineWidth = LW; g.strokeStyle = ACCENT; g.setLineDash([]); g.stroke()
            g.restore()
            const h = radialHandles(sel)
            g.strokeStyle = ACCENT; g.lineWidth = LW; g.setLineDash([])
            g.beginPath(); g.moveTo(h.n.x, h.n.y); g.lineTo(h.rot.x, h.rot.y); g.stroke()  // rotation stem
            handle(h.e.x, h.e.y); handle(h.w.x, h.w.y); handle(h.n.x, h.n.y); handle(h.s.x, h.s.y)
            handle(h.rot.x, h.rot.y, '#9bf95b')                                            // rotate (green)
            g.strokeStyle = ACCENT; g.lineWidth = LW                                       // centre cross = move
            g.beginPath()
            g.moveTo(sel.center.x - HR, sel.center.y); g.lineTo(sel.center.x + HR, sel.center.y)
            g.moveTo(sel.center.x, sel.center.y - HR); g.lineTo(sel.center.x, sel.center.y + HR); g.stroke()
        } else if (sel?.kind === 'linear' && sel.p1 && sel.p2) {
            const dx = sel.p2.x - sel.p1.x, dy = sel.p2.y - sel.p1.y
            const len = Math.hypot(dx, dy) || 1
            const px = -dy / len, py = dx / len
            const guide = Math.min(W, H) * 0.5
            const mid = { x: (sel.p1.x + sel.p2.x) / 2, y: (sel.p1.y + sel.p2.y) / 2 }
            g.strokeStyle = ACCENT; g.lineWidth = LW
            for (const [pt, dash] of [[sel.p1, [3 * ipx, 4 * ipx]], [mid, []], [sel.p2, [3 * ipx, 4 * ipx]]]) {
                g.setLineDash(dash)
                g.beginPath(); g.moveTo(pt.x - px * guide, pt.y - py * guide); g.lineTo(pt.x + px * guide, pt.y + py * guide); g.stroke()
            }
            g.setLineDash([]); g.beginPath(); g.moveTo(sel.p1.x, sel.p1.y); g.lineTo(sel.p2.x, sel.p2.y); g.stroke()
            handle(sel.p1.x, sel.p1.y); handle(sel.p2.x, sel.p2.y)
        } else if ((sel?.kind === 'path' || sel?.kind === 'lasso') && Array.isArray(sel.points) && sel.points.length > 1) {
            g.fillStyle = 'rgba(83,216,255,0.10)'; g.strokeStyle = ACCENT; g.lineWidth = LW; g.setLineDash([])
            g.beginPath(); g.moveTo(sel.points[0].x, sel.points[0].y)
            for (let i = 1; i < sel.points.length; i++) g.lineTo(sel.points[i].x, sel.points[i].y)
            g.closePath(); g.fill(); g.stroke()
        } else if (sel && (!tool || tool === 'refine') && sel.fillMode !== 'fill' && ['semantic', 'brush', 'smartBrush'].includes(sel.kind)) {
            // Texture-backed (AI Subject / Click-Select / brush) masks have no
            // vector outline. Rather than tint the whole coverage (which hides the
            // image colours you're grading), trace just a thin boundary around the
            // region — so it stays identifiable in Adjust / Erase (and while
            // brush-refining) while the pixels underneath stay fully visible. FILL
            // mode already paints the region in its fill colour, so skip it there.
            const key = sel.maskTextureKey || sel.brushTextureKey
            drawMaskBoundary(g, key && getMaskTexture(key), key, W, H, ipx)
        }

        // active pen draft
        if (tool === 'pen' && draft.length) {
            g.strokeStyle = ACCENT; g.lineWidth = LW; g.setLineDash([6 * ipx, 4 * ipx])
            g.beginPath(); g.moveTo(draft[0].x, draft[0].y)
            for (let i = 1; i < draft.length; i++) g.lineTo(draft[i].x, draft[i].y)
            g.stroke()
            draft.forEach((p, i) => handle(p.x, p.y, i === 0 ? '#fff' : ACCENT))
        }

        // active AI-lasso stroke — drawn closed, because that is the shape the
        // clamp actually uses (the stroke is auto-closed before it is rasterised).
        if (aiLasso && aiLasso.length > 1) {
            g.strokeStyle = ACCENT; g.lineWidth = LW * 1.4; g.setLineDash([7 * ipx, 5 * ipx])
            g.lineJoin = 'round'; g.lineCap = 'round'
            g.beginPath(); g.moveTo(aiLasso[0].x, aiLasso[0].y)
            for (let i = 1; i < aiLasso.length; i++) g.lineTo(aiLasso[i].x, aiLasso[i].y)
            g.closePath(); g.stroke()
            g.setLineDash([])
            g.fillStyle = 'rgba(83,216,255,0.08)'; g.fill()
        }

        // active box-select rubber-band
        if (samBox) {
            const x = Math.min(samBox.x0, samBox.x1), y = Math.min(samBox.y0, samBox.y1)
            const w = Math.abs(samBox.x1 - samBox.x0), h = Math.abs(samBox.y1 - samBox.y0)
            g.fillStyle = 'rgba(83,216,255,0.10)'; g.fillRect(x, y, w, h)
            g.strokeStyle = ACCENT; g.lineWidth = LW; g.setLineDash([6 * ipx, 4 * ipx])
            g.strokeRect(x, y, w, h); g.setLineDash([])
        }

        // click-select point markers
        if (tool === 'clickselect' && clickPoints.length > 0) {
            clickPoints.forEach(p => {
                const r = HR * 1.2
                g.beginPath(); g.arc(p.x, p.y, r, 0, Math.PI * 2)
                g.fillStyle = p.label === 1 ? 'rgba(83,255,120,0.9)' : 'rgba(255,83,83,0.9)'
                g.fill()
                g.lineWidth = 1.5 * ipx; g.strokeStyle = 'rgba(8,10,14,0.95)'; g.stroke()
                // cross for negative points
                if (p.label === 0) {
                    g.strokeStyle = '#fff'; g.lineWidth = 1.5 * ipx
                    g.beginPath()
                    g.moveTo(p.x - r * 0.5, p.y - r * 0.5); g.lineTo(p.x + r * 0.5, p.y + r * 0.5)
                    g.moveTo(p.x + r * 0.5, p.y - r * 0.5); g.lineTo(p.x - r * 0.5, p.y + r * 0.5)
                    g.stroke()
                }
            })
        }
    }, [chain, selectedId, tool, draft, samBox, aiLasso, clickPoints, imageSize, W, H, tick, preview])

    /* ── imperative hooks for the dev-browser test ─────────────────────── */
    useEffect(() => {
        window.__studio = {
            ready: true,
            imageSize: () => imageSize,
            chain: () => chain.map((e) => ({ id: e.layer.id, kind: e.layer.kind, op: e.op, fillMode: e.layer.fillMode })),
            // add a fully-specified layer (used by tests) — merges opts into the factory
            add: (kind, opts = {}) => {
                const f = { radial: radialLayer, linear: linearLayer, luminance: luminanceLayer, color: colorLayer }[kind]
                if (!f) return null
                const base = kind === 'radial'
                    ? radialLayer({ center: { x: W * 0.5, y: H * 0.5 }, radius: { x: W * 0.3, y: H * 0.3 }, feather: 0.5, imageSize })
                    : kind === 'linear'
                        ? linearLayer({ p1: { x: W * 0.5, y: 0 }, p2: { x: W * 0.5, y: H * 0.6 }, feather: 0.5, imageSize })
                        : f({})
                return commit({ ...base, fillMode: 'adjust', ...opts })
            },
            update: (id, patch) => updateLayer(id, patch),
            setFillMode: (id, mode) => setFillMode(id, mode),
            applyCurve: (id, curves) => applyCurve(id, curves),
            setGamma: (id, v) => updateLayer(id, { gamma: v }),
            setWheel: (id, which, off) => updateLayer(id, { [which]: off }),
            base: () => baseLayer,
            undo: () => undo(),
            redo: () => redo(),
            compare: (on) => setCompare(!!on),
            select: (id) => setSelectedId(id),
            expandBoundary: (id, px) => onExpandBoundary(id, px),
            refine: () => startRefine(),
            cycle: (delta = 1) => cycleMask(delta),
            candidates: () => cycleTarget,
            runSubject: () => runSubject(),
            background: () => runSubject({ invert: true }),
            samBox: (x0, y0, x1, y1) => onSamBox([x0, y0, x1, y1]),
            aiLasso: (pts) => onAiLasso(pts),
            clickSelect: (x, y, label = 1) => { const pts = [...clickPointsRef.current, { x, y, label }]; putClickPoints(pts); onClickSelect(pts) },
            testDevice: () => testDevice(),
            exportHd: () => exportHd(),
            engine: async () => (await engine()).status(),
            budget: async () => (await engine()).engineBudget(),
            shed: async (level) => (await engine()).shed(level),
            aiState: () => ai,
            layer: (id) => {
                const l = chain.find((e) => e.layer.id === id)?.layer
                return l ? { kind: l.kind, center: l.center, radius: l.radius, rotation: l.rotation, p1: l.p1, p2: l.p2 } : null
            },
            pixels: () => {
                const d = dispRef.current
                return d.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, d.width, d.height).data
            },
            // Renderer counters — program-cache hit rate, compile time, draw
            // count. The only way to tell a shader recompile from a slow draw.
            // Where a mask actually LANDED, for the subject-accuracy bench. A
            // status line reports coverage; only the bbox and centroid can say
            // whether the mask is on the subject or on a corner of the frame.
            maskStats: (id) => {
                const l = (id ? chainRef.current.find((e) => e.layer.id === id)
                    : chainRef.current[chainRef.current.length - 1])?.layer
                const tex = l && getMaskTexture(l.maskTextureKey)
                if (!tex || !tex.width) return null
                const c = document.createElement('canvas'); c.width = W; c.height = H
                c.getContext('2d').drawImage(tex, 0, 0, W, H)
                const d = c.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, W, H).data
                let n = 0; let sx = 0; let sy = 0
                let x0 = W; let y0 = H; let x1 = -1; let y1 = -1
                for (let y = 0; y < H; y += 1) {
                    for (let x = 0; x < W; x += 1) {
                        if (d[(y * W + x) * 4] < 128) continue
                        n += 1; sx += x; sy += y
                        if (x < x0) x0 = x
                        if (x > x1) x1 = x
                        if (y < y0) y0 = y
                        if (y > y1) y1 = y
                    }
                }
                if (!n) return { id: l.id, coverage: 0 }
                return {
                    id: l.id,
                    w: W,
                    h: H,
                    coverage: +(n / (W * H)).toFixed(4),
                    bbox: [x0, y0, x1, y1],
                    centroid: [Math.round(sx / n), Math.round(sy / n)],
                    // Fraction of the frame's diagonal between the mask's centre
                    // of mass and the frame's — the single number that separated
                    // "the sphere" from "the top-left fold".
                    offCentre: +(Math.hypot(sx / n - W / 2, sy / n - H / 2) / Math.hypot(W / 2, H / 2)).toFixed(3),
                }
            },
            renderMetrics: () => getRenderMetrics(),
            resetRenderMetrics: () => resetRenderMetrics(),
        }
        window.__ready = true
    }, [imageSize, chain, baseLayer, W, H, commit, updateLayer, setFillMode, applyCurve, onExpandBoundary, ai, runSubject, testDevice, exportHd, onSamBox, onAiLasso, onClickSelect, clickPoints, undo, redo, startRefine, cycleMask, cycleTarget])

    // Engine chip: what the lane actually resolved to, never what it intends to
    // use. Stays "starting…" until a session exists.
    const engineChip = useMemo(() => {
        const e = ai.engine
        if (!e) return { text: 'engine…', ok: false, title: 'probing the device' }
        if (!e.ready) return { text: 'starting…', ok: false, title: `${e.lane} · ${e.gpuTier} GPU` }
        const dev = String(e.device || 'webgpu').toUpperCase()
        const ms = e.lastRun?.ms
        return {
            text: `${dev} ✓${e.pressure ? ` · P${e.pressure}` : ''}`,
            ok: true,
            title: `${e.lane} · ${e.mode || 'worker'} · proxy ≤${e.proxyMax}px${ms ? ` · last selection ${ms} ms` : ''}`,
        }
    }, [ai.engine])

    textureLayers.current = chain.reduce(
        (n, e) => n + (e.layer.maskTextureKey || e.layer.brushTextureKey ? 1 : 0), 0,
    )

    const selected = chain.find((e) => e.layer.id === selectedId) || null
    // Brush-refine targets texture-backed masks whose coverage is paintable
    // (AI Subject / Click-Select / brush / smart-brush). Parametric & vector
    // masks (radial / linear / pen / lasso) are edited via their handles instead.
    const canRefine = !!selected && ['semantic', 'brush', 'smartBrush'].includes(selected.layer.kind)
        && !!(selected.layer.maskTextureKey || selected.layer.brushTextureKey)
    // The lane parks ONE candidate set, so cycling is offered only on the layer
    // that owns it — anything else would repaint a mask with another one's planes.
    const canCycle = !!cycleTarget && selectedId === cycleTarget.layerId

    /* ── render ────────────────────────────────────────────────────────── */
    return (
        <div className="studio">
            <div className="studio-top">
                <h1>Mask <b>Studio</b></h1>
                <label className="mask-btn" style={{ cursor: 'pointer' }}>
                    Load image
                    <input type="file" accept="image/*,.nef,.nrw,.cr2,.cr3,.arw,.dng,.orf,.rw2,.raf,.pef,.srw" onChange={onFile} style={{ display: 'none' }} />
                </label>
                <button className="mask-btn" disabled={!hasImage || ai.busy} onClick={exportHd}
                    title="Re-decode the original at export resolution and render the full mask chain onto it">
                    ⤓ Export HD
                </button>
                <div className="spacer" />
                <button className="mask-btn" onClick={undo} title="Undo (⌘/Ctrl+Z)">↶ Undo</button>
                <button className="mask-btn" onClick={redo} title="Redo (⌘/Ctrl+Shift+Z)">↷ Redo</button>
                <label className="mask-toggle"><input type="checkbox" checked={preview} onChange={(e) => setPreview(e.target.checked)} /> Preview (hide guides)</label>
                <label className="mask-toggle"><input type="checkbox" checked={overlayMode} onChange={(e) => setOverlayMode(e.target.checked)} /> Show mask overlay</label>
                <label className="mask-toggle"><input type="checkbox" checked={globalInvert} onChange={(e) => setGlobalInvert(e.target.checked)} /> Invert all</label>
                <span className="hint">{status}</span>
                {/* Written by ai/gestures.js: it names the bindings the current input
                    device actually has, and rewrites them when the device changes. */}
                <span className="hint gesture-hint" ref={gestureHintRef} />
                {/* AGPL-3.0 §13: whoever interacts with this over a network has to be
                    offered the corresponding source, so the offer lives in the UI. */}
                <a className="src-link" href={SOURCE_URL} target="_blank" rel="noreferrer"
                    title="Mask Studio is free software under the GNU AGPL v3 — get the source">Source</a>
            </div>

            <div className="studio-main">
                <div className="stage-wrap">
                    {hasImage ? (
                        <div className="stage" ref={stageRef}>
                            {/* Only .frame is zoomed and panned. The brush ring, the tool
                                banner and the zoom pill are its SIBLINGS, so they keep
                                their real size at every zoom level, and every
                                pointer-to-image conversion still works untouched: it goes
                                through getBoundingClientRect, which already reports the
                                transformed box. */}
                            <div className="frame" ref={frameRef}>
                                <canvas ref={dispRef} className="display" />
                                <canvas
                                    ref={overlayRef}
                                    className={`overlay ${tool ? 'drawing' : ''}`}
                                    onPointerDown={onPointerDown}
                                    onPointerMove={onPointerMove}
                                    onPointerUp={onPointerUp}
                                    onPointerLeave={() => { if (cursorRef.current) cursorRef.current.style.display = 'none' }}
                                    onDoubleClick={() => tool === 'pen' && closePen()}
                                    style={{ pointerEvents: preview ? 'none' : 'auto', cursor: tool === 'brush' || tool === 'refine' ? 'none' : tool ? 'crosshair' : 'default', touchAction: 'none' }}
                                />
                            </div>
                            {/* No children and a constant `hidden` in JSX: React never
                                diffs either, so gestures owns the label and the flag. */}
                            <button className="zoom-reset" ref={zoomRef} type="button" hidden title="Reset zoom (0)" />
                            {(tool === 'brush' || tool === 'refine') && <div ref={cursorRef} className="brush-cursor" />}
                            {tool && (
                                <div className="tool-banner">
                                    <span><b>{TOOL_TITLE[tool] || tool}</b> · {ai.busy ? (ai.status || 'working…') : (status || 'draw on the image')}</span>
                                    {tool === 'pen' && <button className="mask-btn" onClick={closePen}>Close path</button>}
                                    {tool === 'refine' && (
                                        <div className="seg" role="group" aria-label="brush-refine mode">
                                            <button className={`seg-btn ${!refineErase ? 'seg-btn--active' : ''}`} onClick={() => setRefineErase(false)}>Add</button>
                                            <button className={`seg-btn ${refineErase ? 'seg-btn--active' : ''}`} onClick={() => setRefineErase(true)}>Erase</button>
                                        </div>
                                    )}
                                    {(tool === 'brush' || tool === 'refine') && (
                                        <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                                            size
                                            <input type="range" className="mask-range" min={8} max={200} value={brushSize} onChange={(e) => setBrushSize(Number(e.target.value))} style={{ width: 90 }} />
                                        </label>
                                    )}
                                    {tool === 'clickselect' && (
                                        <>
                                            <span className="hint" style={{ marginLeft: 8 }}>
                                                {clickPoints.filter(p => p.label === 1).length} pos · {clickPoints.filter(p => p.label === 0).length} neg
                                            </span>
                                            <button className="mask-btn" disabled={!clickPoints.length}
                                                onClick={() => { putClickPoints([]); setStatus('Click to ADD a region · Alt+click to REMOVE · Done to finish') }}
                                                title="Clear the click markers (use ⌘/Ctrl+Z to undo individual add/remove edits)">
                                                Clear points
                                            </button>
                                        </>
                                    )}
                                    <button className="mask-btn" onClick={() => { setTool(null); setDraft([]); setSamBox(null); setAiLasso(null); putClickPoints([]); setClickMaskId(null); refineRef.current = null; setStatus('') }}>Done</button>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="stage-placeholder">
                            <b>Load an image</b> to begin.<br />A sample auto-loads if present.
                        </div>
                    )}
                </div>

                <div className="panel">
                    <div className="panel-scroll">
                        <div className="section">
                            <div className="section-head">
                                <label className="panel-label">Base · Global grade</label>
                                <button className="link-btn" disabled={!baseLayer}
                                    onPointerDown={() => setCompare(true)}
                                    onPointerUp={() => setCompare(false)}
                                    onPointerLeave={() => setCompare(false)}
                                    title="Hold to compare against the ungraded original">
                                    {compare ? 'before' : 'hold: before/after'}
                                </button>
                            </div>
                            {baseLayer ? (
                                <LayerGradeEditor
                                    layer={baseLayer}
                                    onUpdate={(patch) => updateLayer('base', patch)}
                                    onApplyCurve={applyCurve}
                                    histogram={histogram}
                                    dominantColor={ACCENT}
                                />
                            ) : (
                                <span className="hint">Load an image to grade the whole frame.</span>
                            )}
                        </div>

                        <div className="section">
                            <div className="section-head">
                                <label className="panel-label">Masks</label>
                                <span className={`ai-badge ${engineChip.ok ? 'ai-badge--ok' : ''}`} title={engineChip.title}>
                                    {engineChip.text}
                                </span>
                            </div>
                            <div className="add-grid">
                                {MASK_TOOLS.map((t) => {
                                    const m = getKindMeta(t.kind)
                                    const active = !t.ai && tool === t.id
                                    return (
                                        <button key={t.id}
                                            className={`mask-btn ${active ? 'mask-btn--active' : ''}`}
                                            disabled={!hasImage || (!!t.ai && ai.busy)}
                                            onClick={() => onToolClick(t)} title={t.hint}>
                                            <span className="dot" style={{ background: m.color }} />
                                            {t.label}
                                        </button>
                                    )
                                })}
                            </div>
                            <div className="global-row">
                                <label className="mask-toggle"><input type="checkbox" checked={penSmooth} onChange={(e) => setPenSmooth(e.target.checked)} /> Pen: smooth</label>
                                <label className="mask-toggle"><input type="checkbox" checked={edgeAware} onChange={(e) => setEdgeAware(e.target.checked)} /> Brush: edge-aware</label>
                            </div>
                            {ai.progress && (
                                <div className="ai-progress" title={ai.progress.file}>
                                    <div className="ai-progress-bar" style={{ width: `${Math.min(100, (ai.progress.loaded / ai.progress.total) * 100).toFixed(1)}%` }} />
                                    <span>{ai.progress.file} · {mb(ai.progress.loaded)} / {mb(ai.progress.total)}</span>
                                </div>
                            )}
                            {ai.status && <div className="ai-status">{ai.busy ? '⏳ ' : ''}{ai.status}</div>}
                            {ai.pressure > 0 && (
                                <div className="ai-status">⚠︎ Memory pressure level {ai.pressure} — the governor released heavy caches; selections still work, exports are capped.</div>
                            )}
                            {ai.report?.caps && (
                                <div className="ai-report">
                                    {ai.report.caps.map(([label, ok], i) => (
                                        <span key={i} className={`ai-cap ${ok ? 'ok' : 'no'}`}>{label} {ok ? '✓' : '✗'}</span>
                                    ))}
                                </div>
                            )}
                            <div className="studio-help">
                                <b>1.</b> Pick a mask — it appears on the image. <b>2.</b> Drag the <span style={{ color: ACCENT }}>○ handles</span> (resize / move / rotate), or use a layer's <b>Boundary</b> to grow/shrink AI &amp; painted masks. <b>3.</b> Switch the card to <b>Adjust</b> to grade only that region — AI / brush masks then show just a <span style={{ color: ACCENT }}>boundary</span> so the colours stay visible. <b>4.</b> <b>Brush-refine</b> an AI / brush mask to paint its coverage (Alt-drag to erase).
                            </div>
                            <div className="ai-foot">
                                <button className="link-btn" disabled={ai.busy} onClick={testDevice}>{ai.busy ? 'checking…' : 'Test device AI'}</button>
                                <span className="ai-note">
                                    SAM 2.1 (fp16) runs in this browser on WebGPU. No upload endpoint, no server, no cloud — drop a JPEG, PNG, HEIF or a camera RAW and every pixel stays on this device.
                                </span>
                            </div>
                        </div>

                        <div className="section">
                            <div className="section-head">
                                <label className="panel-label">Mask layers</label>
                                <span className="hint">{chain.length}</span>
                            </div>
                            {chain.length > 0 && (
                                <button
                                    className={`mask-btn ${tool === 'refine' ? 'mask-btn--active' : ''}`}
                                    disabled={!canRefine}
                                    onClick={startRefine}
                                    title={canRefine ? 'Paint on the selected mask to add / erase coverage (Alt-drag to erase)' : 'Select an AI Subject or brush mask to refine it'}
                                >
                                    ✎ {tool === 'refine' ? 'Refining selected mask…' : 'Brush-refine selected mask'}
                                </button>
                            )}
                            {canCycle && (
                                <button
                                    className="mask-btn"
                                    disabled={ai.busy}
                                    onClick={() => cycleMask(1)}
                                    title="SAM returns three nested answers (part / object / whole). Step through them — no re-decode. Keyboard: [ and ]"
                                >
                                    ⇄ Next mask scope ({cycleTarget.index + 1}/{cycleTarget.count}) · [ ]
                                </button>
                            )}
                            {chain.length === 0 ? (
                                <ToolEmptyState icon={SquareIcon} title="No masks yet" subtitle="Add a mask above, draw its region, then switch it to Adjust to grade only that area." />
                            ) : (
                                <div className="chain-list">
                                    <AnimatePresence initial={false}>
                                        {chain.map((entry, i) => (
                                            <MaskChainCard
                                                key={entry.layer.id}
                                                entry={entry}
                                                index={i + 1}
                                                total={chain.length}
                                                isFirst={i === 0}
                                                selected={entry.layer.id === selectedId}
                                                onSelect={setSelectedId}
                                                onUpdate={(patch) => updateLayer(entry.layer.id, patch)}
                                                onRemove={removeLayer}
                                                onMove={moveLayer}
                                                onSetOp={setOp}
                                                onSetFillMode={setFillMode}
                                                onExpandBoundary={onExpandBoundary}
                                                onApplyCurve={applyCurve}
                                                histogram={histogram}
                                                dominantColor={ACCENT}
                                                imageSize={imageSize}
                                            />
                                        ))}
                                    </AnimatePresence>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

// minimal inline icon for the empty state (avoid pulling a lucide import here)
const SquareIcon = (props) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" {...props}>
        <rect x="3" y="3" width="18" height="18" rx="3" />
        <path d="M3 14l5-5 4 4 4-4 5 5" />
    </svg>
)

createRoot(document.getElementById('root')).render(<App />)
