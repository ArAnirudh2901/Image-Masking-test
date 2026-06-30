/**
 * Mask Studio — standalone, no-auth testbed for the phosmith megashader masking
 * engine. Loads an image, lets you add MULTIPLE masks of every client-side kind
 * (radial / linear / pen·lasso / brush / luminance / color), draw each region on
 * the canvas, and colour-grade ONLY that region with the REAL phosmith mask card
 * (ProRulerSlider + MaskChainCard + the per-layer adjustment sliders).
 *
 * Reuses the production modules verbatim — this is the now-fixed engine + the
 * real editor UI in isolation, for later integration into the app.
 *
 * Phase A: core masks + the 13-field per-mask grade + multi-mask stacking.
 * (On-device AI + tone-curves/colour-wheels engine extension land next.)
 */
import React, { useState, useRef, useEffect, useCallback } from 'react'
import { createRoot } from 'react-dom/client'
import { AnimatePresence } from 'framer-motion'
import { renderMegashader } from '@/lib/megashader/megashader-renderer'
import {
    radialLayer, linearLayer, pathLayer, lassoLayer, brushLayer,
    luminanceLayer, colorLayer, semanticLayer, depthLayer, smartBrushLayer,
    sanitiseLayer, setMaskTexture, getMaskTexture,
} from '@/lib/megashader/mask-types'
import { growMaskCanvas } from '@/lib/mask-grow'
import { cleanSubjectMatte } from '@/lib/subject-mask-cleanup'
import { rasterisePath, smoothToBezier } from '@/lib/megashader/path-raster'
import { buildPackedLutFromCurves } from '@/lib/curve-lut'
import { computeImageHistogram } from '@/lib/image-histogram'
import {
    MaskChainCard, getKindMeta, ToolEmptyState,
} from '@/app/(main)/editor/[projectId]/_components/tools/_pixel-tool-ui.jsx'
import { LayerGradeEditor } from '@/app/(main)/editor/[projectId]/_components/tools/_layer-grade-editor.jsx'
import { rgbToHsb } from '@/lib/color-utils'

const MAX_DIM = 1400
const ACCENT = '#53d8ff'
const uid = () => Math.random().toString(36).slice(2, 9)

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
    { id: 'subject', kind: 'semantic', label: 'AI Subject', hint: 'auto-mask the subject (on-device)', ai: 'subject' },
    { id: 'background', kind: 'semantic', label: 'AI Sky / Bg', hint: 'select the ENTIRE sky / background (detect the subject, then invert)', ai: 'background' },
    { id: 'sam', kind: 'semantic', label: 'AI Box-Select', hint: 'drag a box around an object (SAM 3.1)', ai: 'sam' },
    { id: 'clickselect', kind: 'semantic', label: 'AI Click-Select', hint: 'click any object for a precise mask (multi-point)', ai: 'clickselect' },
    { id: 'depth', kind: 'depth', label: 'AI Depth', hint: 'near / far selection (on-device)', ai: 'depth' },
]

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
const drawMaskBoundary = (g, tex, W, H, ipx) => {
    if (!tex || !tex.width) return
    const tw = tex.width, th = tex.height
    const sil = document.createElement('canvas'); sil.width = tw; sil.height = th
    const sc = sil.getContext('2d', { willReadFrequently: true })
    sc.drawImage(tex, 0, 0)
    const id = sc.getImageData(0, 0, tw, th); const d = id.data
    for (let i = 0; i < d.length; i += 4) {
        const cov = d[i + 3] >= 250 ? d[i] : d[i + 3] // luma (opaque) or painted alpha
        const on = cov >= 128 ? 255 : 0
        d[i] = 0x53; d[i + 1] = 0xd8; d[i + 2] = 0xff; d[i + 3] = on
    }
    sc.putImageData(id, 0, 0)
    const out = document.createElement('canvas'); out.width = tw; out.height = th
    const oc = out.getContext('2d')
    const r = Math.max(1.2, 2.2 * ipx * (tw / (W || tw))) // ≈ constant on-screen width
    for (let a = 0; a < Math.PI * 2 - 1e-3; a += Math.PI / 8) oc.drawImage(sil, Math.cos(a) * r, Math.sin(a) * r)
    oc.globalCompositeOperation = 'destination-out'
    oc.drawImage(sil, 0, 0)
    g.drawImage(out, 0, 0, W, H)
}

// SAM output → white-on-black mask canvas. post_process_masks gives an array of
// Tensors ([nMasks,H,W] at the original image size); pick the highest-IoU mask.
const samMaskToCanvas = (m, iou) => {
    const dims = m.dims
    const w = dims[dims.length - 1], h = dims[dims.length - 2]
    const nMasks = dims.length >= 3 ? dims[dims.length - 3] : 1
    const scores = iou && iou.data ? Array.from(iou.data) : []
    let bi = 0, best = -Infinity
    for (let i = 0; i < nMasks; i += 1) { const s = scores[i] ?? 0; if (s > best) { best = s; bi = i } }
    const plane = w * h, off = bi * plane
    const cv = document.createElement('canvas'); cv.width = w; cv.height = h
    const ctx = cv.getContext('2d'); const img = ctx.createImageData(w, h)
    const d = m.data
    for (let p = 0; p < plane; p += 1) { const v = d[off + p] ? 255 : 0; const q = p * 4; img.data[q] = v; img.data[q + 1] = v; img.data[q + 2] = v; img.data[q + 3] = 255 }
    ctx.putImageData(img, 0, 0)
    return cv
}

// ── optional bridge to the local Python mask service (SAM 3.1) ───────────────
// The testbed is on-device by default; when the mask service (services/segment)
// is reachable it is PREFERRED for subject/concept masks (server-side SAM 3.1,
// far better on stylized art) and for click-select. Every call fails soft, so
// the on-device engines remain the fallback. The service must allow this origin
// via CORS_ORIGINS (see services/segment/.env).
const SERVICE_URL = () => (typeof window !== 'undefined' && window.__MASK_SERVICE_URL) || 'http://127.0.0.1:8001'
const SUBJECT_CONCEPT = 'main subject' // default prompt the AI Subject button sends

const canvasToBlob = (canvas) =>
    new Promise((res, rej) => canvas.toBlob((b) => (b ? res(b) : rej(new Error('toBlob failed'))), 'image/png'))

// SAM resizes inputs to ~1024px internally, so uploading the full working
// canvas (≤MAX_DIM) just wastes PNG-encode + network + server-side decode. Send
// the service a ≤SAM_INPUT copy instead; the returned mask is scaled back to the
// requested display size by pngToMaskCanvas, so there's no visible quality loss.
// Returns the scaled canvas and the scale factor (needed to scale box prompts).
const SAM_INPUT = 1024
const downscaleForSam = (canvas, maxDim = SAM_INPUT) => {
    const scale = Math.min(1, maxDim / Math.max(canvas.width, canvas.height))
    if (scale === 1) return { canvas, scale: 1 }
    const c = document.createElement('canvas')
    c.width = Math.round(canvas.width * scale)
    c.height = Math.round(canvas.height * scale)
    c.getContext('2d').drawImage(canvas, 0, 0, c.width, c.height)
    return { canvas: c, scale }
}

// Service masks come back as a greyscale PNG (white = subject) or, for /segment,
// an RGBA cutout carrying the mask in alpha. Decode either into the opaque
// R=G=B=coverage / A=255 canvas the semantic shader samples.
const pngToMaskCanvas = (src, w, h) =>
    new Promise((resolve, reject) => {
        const img = new Image()
        img.onload = () => {
            const cv = document.createElement('canvas')
            cv.width = w || img.naturalWidth
            cv.height = h || img.naturalHeight
            const ctx = cv.getContext('2d', { willReadFrequently: true })
            ctx.drawImage(img, 0, 0, cv.width, cv.height)
            const id = ctx.getImageData(0, 0, cv.width, cv.height)
            const d = id.data
            for (let i = 0; i < d.length; i += 4) {
                const v = d[i + 3] < 250 ? d[i + 3] : d[i]
                d[i] = v; d[i + 1] = v; d[i + 2] = v; d[i + 3] = 255
            }
            ctx.putImageData(id, 0, 0)
            resolve(cv)
        }
        img.onerror = () => reject(new Error('mask decode failed'))
        img.src = src
    })

// GET /health — is the service up, and is SAM 3.1 actually loaded (vs fallback)?
const checkService = async (timeoutMs = 1500) => {
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), timeoutMs)
    try {
        const r = await fetch(SERVICE_URL() + '/health', { signal: ctrl.signal })
        if (!r.ok) return { available: false }
        const j = await r.json()
        return {
            available: true,
            sam3: !!j.sam3_available,
            sam3Loaded: !!j.sam3_loaded,
            subjectEngine: j.subject_engine || (j.sam3_available ? 'sam3' : 'saliency'),
            model: j.sam3_model || j.model || '',
        }
    } catch { return { available: false } }
    finally { clearTimeout(timer) }
}

// Tight [x0,y0,x1,y1] bounding box of a mask canvas's coverage (white = selected,
// opaque → coverage in the red channel), in the canvas's own pixel space. Returns
// null when the mask is essentially empty.
const bboxOfMaskCanvas = (canvas, thresh = 24) => {
    const w = canvas.width, h = canvas.height
    if (!w || !h) return null
    const d = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h).data
    let x0 = w, y0 = h, x1 = -1, y1 = -1
    for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
            if (d[(y * w + x) * 4] > thresh) {
                if (x < x0) x0 = x
                if (x > x1) x1 = x
                if (y < y0) y0 = y
                if (y > y1) y1 = y
            }
        }
    }
    return x1 < 0 ? null : [x0, y0, x1, y1]
}

// POST /segment/instances — SAM 3.1 concept segmentation; returns the UNION of
// all matching instances as a white-on-black mask canvas (+ count/mode/model).
const serviceSubjectMask = async (srcCanvas, { concept = SUBJECT_CONCEPT, width, height } = {}) => {
    const { canvas: small } = downscaleForSam(srcCanvas)
    const blob = await canvasToBlob(small)
    const form = new FormData()
    form.append('image', blob, 'image.png')
    form.append('prompt', concept)
    // Ask the service for its fast subject path: skip the always-failing text
    // grounding for the abstract "main subject" concept and box-seed SAM 3.1 from
    // the saliency matte in ONE image encode. Older services ignore this field
    // and return saliency, in which case runSubject box-seeds client-side.
    form.append('subject_box', 'true')
    const r = await fetch(SERVICE_URL() + '/segment/instances', { method: 'POST', body: form })
    if (!r.ok) throw new Error('service ' + r.status)
    const j = await r.json()
    if (!j.union_png) throw new Error(j.count === 0 ? 'no subject found' : 'no mask')
    const canvas = await pngToMaskCanvas('data:image/png;base64,' + j.union_png, width || j.width, height || j.height)
    return { canvas, count: j.count || 0, mode: j.mode || 'sam3', model: j.model || '' }
}

// POST /segment/box — SAM 3.1 box-prompted object select (greyscale PNG mask).
const serviceSamBox = async (srcCanvas, box, { width, height } = {}) => {
    // Box is in working-canvas space; scale it to match the downscaled upload.
    const { canvas: small, scale } = downscaleForSam(srcCanvas)
    const blob = await canvasToBlob(small)
    const form = new FormData()
    form.append('image', blob, 'image.png')
    form.append('box', JSON.stringify(box.map((v) => Math.round(v * scale))))
    const r = await fetch(SERVICE_URL() + '/segment/box', { method: 'POST', body: form })
    if (!r.ok) throw new Error('service ' + r.status)
    const blobOut = await r.blob()
    const url = URL.createObjectURL(blobOut)
    try { return await pngToMaskCanvas(url, width, height) }
    finally { URL.revokeObjectURL(url) }
}

// POST /sam2/click — SAM point-prompted click-select (service).
const serviceSamClick = async (srcCanvas, points, labels, { width, height } = {}) => {
    const { canvas: small, scale } = downscaleForSam(srcCanvas)
    const blob = await canvasToBlob(small)
    const form = new FormData()
    form.append('image', blob, 'image.png')
    form.append('points', JSON.stringify(points.map(([x, y]) => [Math.round(x * scale), Math.round(y * scale)])))
    form.append('labels', JSON.stringify(labels))
    const r = await fetch(SERVICE_URL() + '/sam2/click', { method: 'POST', body: form })
    if (!r.ok) throw new Error('service ' + r.status)
    const blobOut = await r.blob()
    const url = URL.createObjectURL(blobOut)
    try { return await pngToMaskCanvas(url, width, height) }
    finally { URL.revokeObjectURL(url) }
}

// POST /ground/text — SAM 3.1 open-vocabulary TEXT grounding (binds the phrase
// to a region). Returns results[0].maskPng (base64 greyscale, white = selected).
const serviceGroundText = async (srcCanvas, phrase, { width, height } = {}) => {
    const { canvas: small } = downscaleForSam(srcCanvas)
    const blob = await canvasToBlob(small)
    const form = new FormData()
    form.append('image', blob, 'image.png')
    form.append('phrases', JSON.stringify([phrase]))
    const r = await fetch(SERVICE_URL() + '/ground/text', { method: 'POST', body: form })
    if (!r.ok) throw new Error('service ' + r.status)
    const j = await r.json()
    const res = Array.isArray(j.results) ? j.results[0] : null
    if (!res || !res.found || !res.maskPng) throw new Error(`no region matched “${phrase}”`)
    const canvas = await pngToMaskCanvas('data:image/png;base64,' + res.maskPng, width || j.width, height || j.height)
    return { canvas, engine: j.engine || 'clipseg', score: res.score, coverage: res.coverage }
}

function App() {
    const srcRef = useRef(null)          // working-res source canvas
    const dispRef = useRef(null)         // visible result canvas
    const overlayRef = useRef(null)      // image-res overlay for guides + pointer capture
    const brushRef = useRef(null)        // { canvas, ctx, key, baseKey, layerId, mode } for the active brush layer
    const refineRef = useRef(null)       // same shape — painting INTO the selected mask (brush-refine)
    const dragRef = useRef(null)         // transient drag state for radial/linear/brush
    const rafRef = useRef(0)
    const cursorRef = useRef(null)       // floating brush-size ring (brush / refine)

    const [imageSize, setImageSize] = useState(null)
    const [chain, setChain] = useState([])         // [{ layer, op }]
    const [baseLayer, setBaseLayer] = useState(null) // full-frame global/base grade (id 'base')
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
    const [clickMaskId, setClickMaskId] = useState(null)  // layer ID of the live click-select mask being refined
    const [overlayMode, setOverlayMode] = useState(false)
    const [globalInvert, setGlobalInvert] = useState(false)
    const [preview, setPreview] = useState(false)  // hide all guides/handles → clean result
    const [tick, setTick] = useState(0)
    const [hasImage, setHasImage] = useState(false)
    const [status, setStatus] = useState('')
    // on-device AI (client-ai.js, loaded lazily so transformers.js stays a chunk)
    const [ai, setAi] = useState({ busy: false, status: '', device: null, report: null, service: { status: 'unchecked' }, sensitivity: 0.5, fillHoles: true, lastFragmented: false })
    const [aiPhrase, setAiPhrase] = useState('')
    const aiModRef = useRef(null)
    const samRef = useRef(null)     // cached on-device SAM fallback { model, processor, image, src }

    const bump = () => setTick((t) => t + 1)
    const W = imageSize?.width || 0
    const H = imageSize?.height || 0

    /* ── image loading ─────────────────────────────────────────────────── */
    const loadImage = useCallback((url, label) => {
        // Downscale the decoded source into the ≤MAX_DIM working canvas.
        const buildWorkingCanvas = (bitmapOrImg, naturalW, naturalH) => {
            const scale = Math.min(1, MAX_DIM / Math.max(naturalW, naturalH))
            const w = Math.round(naturalW * scale)
            const h = Math.round(naturalH * scale)
            const c = document.createElement('canvas')
            c.width = w; c.height = h
            c.getContext('2d', { willReadFrequently: true }).drawImage(bitmapOrImg, 0, 0, w, h)
            if (bitmapOrImg.close) bitmapOrImg.close()
            return { c, w, h }
        }

        const onReady = ({ c, w, h }) => {
            srcRef.current = c
            brushRef.current = null
            // Base/global grade carrier: a full-WHITE texture so one semantic
            // layer covers EVERY pixel (white = full coverage — no smoothstep
            // boundary, unlike a luminance(0..1) mask which under-covers pure
            // black/white). Pass 1 of the render bakes its grade into the frame.
            const baseKey = 'base-tex'
            const wc = document.createElement('canvas'); wc.width = w; wc.height = h
            const wx = wc.getContext('2d'); wx.fillStyle = '#fff'; wx.fillRect(0, 0, w, h)
            setMaskTexture(baseKey, wc)
            setBaseLayer(sanitiseLayer({ ...semanticLayer({ maskTextureKey: baseKey, feather: 0, label: 'Base' }), id: 'base', fillMode: 'adjust' }))
            setHistogram(computeImageHistogram({ getElement: () => c }))
            setImageSize({ width: w, height: h })
            setChain([]); setSelectedId(null); setTool(null); setDraft([])
            setHasImage(true)
            setStatus(`${label || 'image'} · ${w}×${h}`)
            bump()
        }

        // Classic decode path — fallback for environments without
        // createImageBitmap and for any fetch failure.
        const loadViaImage = () => {
            const image = new Image()
            image.crossOrigin = 'anonymous'
            image.onload = () => onReady(buildWorkingCanvas(image, image.naturalWidth, image.naturalHeight))
            image.onerror = () => setStatus('image failed to load')
            image.src = url
        }

        if (typeof createImageBitmap === 'function') {
            // fetch → blob → createImageBitmap decodes off the main thread, so a
            // 4K/8K original doesn't block the UI during load.
            fetch(url)
                .then((r) => { if (!r.ok) throw new Error('fetch ' + r.status); return r.blob() })
                .then((blob) => createImageBitmap(blob))
                .then((bmp) => onReady(buildWorkingCanvas(bmp, bmp.width, bmp.height)))
                .catch(loadViaImage)
        } else {
            loadViaImage()
        }
    }, [])

    // Auto-load the bundled sample so the page is usable (and testable) instantly.
    useEffect(() => { loadImage('test.png?' + Date.now(), 'sample') }, [loadImage])

    const onFile = (e) => {
        const f = e.target.files?.[0]
        if (f) loadImage(URL.createObjectURL(f), f.name)
    }

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
        setMaskTexture(baseTextureKey, canvas)
        setMaskTexture(maskTextureKey, canvas)
        return { maskTextureKey, baseTextureKey, growPx: 0 }
    }

    // Readjust a texture mask's edge: grow (+) / shrink (−) its boundary by an
    // absolute px amount from the pristine base (0 restores it). Drives the
    // MaskChainCard "Boundary" slider for AI Subject / lasso / brush masks.
    const onExpandBoundary = useCallback((id, px) => {
        const layer = chain.find((e) => e.layer.id === id)?.layer
        if (!layer) return
        const base = getMaskTexture(layer.baseTextureKey || layer.maskTextureKey)
        if (!base || typeof base.getContext !== 'function') return // need a real canvas
        const grown = px === 0 ? base : growMaskCanvas(base, px)
        setMaskTexture(layer.maskTextureKey, grown)
        updateLayer(id, { growPx: px })
        bump()
    }, [chain, updateLayer])

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
            setMaskTexture(key, bc); setMaskTexture(baseKey, bc)
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
    }, [imageSize, W, H, commit])

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
            const label = e.altKey ? 0 : 1  // Alt+click = negative point
            const newPts = [...clickPoints, { x: p.x, y: p.y, label }]
            setClickPoints(newPts)
            onClickSelect(newPts)
            return
        }
        if (tool === 'sam') {
            e.preventDefault(); overlayRef.current.setPointerCapture?.(e.pointerId)
            dragRef.current = { kind: 'sambox', start: p }
            setSamBox({ x0: p.x, y0: p.y, x1: p.x, y1: p.y }); return
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
        if (d?.kind === 'paint' && d.target) {
            const t = d.target
            setMaskTexture(t.key, t.canvas)
            // snapshot the painted canvas as the boundary base so grow/shrink
            // readjusts from the latest stroke (resets any prior boundary edit)
            if (t.baseKey) {
                const snap = document.createElement('canvas'); snap.width = W; snap.height = H
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
    const snapshot = () => ({
        chain: chain.map((e) => ({ op: e.op, layer: { ...e.layer } })),
        base: baseLayer ? { ...baseLayer } : null,
        textures: snapTextures(chain, baseLayer),
    })
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

    /* ── on-device AI masks (reuse phosmith's in-browser engines) ───────── */
    // All inference is in-browser (WebGPU→WASM); nothing is sent to a server.
    const loadAi = async () => {
        if (!aiModRef.current) {
            // CRITICAL: point onnxruntime-web at the LOCAL, version-matched
            // runtime files. Bundlers (bun/esbuild) break onnxruntime's runtime
            // fetch of ort-wasm-simd-threaded.jsep.{mjs,wasm} — the JSEP glue that
            // defines `webgpuInit` — so WebGPU init throws "webgpuInit is not a
            // function" and WASM reports "no available backend found". Serving the
            // exact dist files from /ort/ and setting wasmPaths makes BOTH backends
            // load deterministically. Single-threaded (no cross-origin isolation
            // here → no SharedArrayBuffer for multi-threaded WASM).
            const tf = await import('@huggingface/transformers')
            tf.env.backends.onnx.wasm.wasmPaths = new URL('ort/', document.baseURI).href
            tf.env.backends.onnx.wasm.numThreads = 1
            aiModRef.current = await import('@/lib/client-ai')
        }
        return aiModRef.current
    }
    const aiSet = (patch) => setAi((s) => ({ ...s, ...patch }))

    // Probe the local mask service once on load so the AI tools can PREFER
    // SAM 3.1 when it's up and fall back on-device when it isn't. Fails soft.
    useEffect(() => {
        let alive = true
        ;(async () => {
            aiSet({ service: { status: 'checking' } })
            const s = await checkService()
            if (alive) aiSet({ service: { status: s.available ? 'available' : 'offline', sam3: s.sam3, sam3Loaded: s.sam3Loaded, model: s.model, subjectEngine: s.subjectEngine } })
        })()
        return () => { alive = false }
    }, [])

    // Return the live mask-service info, RE-PROBING when we don't already know
    // it's up — so SAM 3.1 drives subject / concept / click-select even when the
    // Python service is started AFTER this page loaded (no reload needed). The
    // server SAM 3.1 path is preferred for almost everything; the on-device
    // engines are the fallback. A localhost probe rejects instantly when the
    // service is down, so this adds no real latency to the fallback path.
    const ensureService = useCallback(async () => {
        if (ai.service?.status === 'available') return ai.service
        const s = await checkService()
        const info = s.available
            ? { status: 'available', sam3: s.sam3, sam3Loaded: s.sam3Loaded, model: s.model, subjectEngine: s.subjectEngine }
            : { status: 'offline' }
        aiSet({ service: info })
        return info
    }, [ai.service])

    // AI Subject — prefer the SAM 3.1 mask service (concept segmentation of the
    // whole subject) when it's up; otherwise on-device RMBG-1.4 + matte cleanup
    // (binarize / close / fill-holes / keep-significant + luminance assist),
    // which rescues the holes RMBG leaves on stylized / backlit silhouettes.
    const runSubject = useCallback(async (conceptArg, { invert = false, label: labelArg } = {}) => {
        if (!srcRef.current) return
        const concept = (typeof conceptArg === 'string' && conceptArg.trim()) || SUBJECT_CONCEPT
        // invert=true → mask the subject, then flip it to select the ENTIRE
        // background/sky. Text grounding ("sky") only binds CLIPSeg to the bright
        // sky blob (~40% here); detect-subject-and-invert covers everything that
        // ISN'T the subject — the only reliable "whole sky/background" selection.
        const label = labelArg || (invert ? 'AI Sky / Background' : 'AI Subject')
        const noun = invert ? 'Background' : 'Subject'
        const svc = await ensureService()
        if (svc.status === 'available') {
            aiSet({ busy: true, status: `${noun}: segmenting “${concept}” via mask service…` })
            try {
                let { canvas, mode, model } = await serviceSubjectMask(srcRef.current, { concept, width: W, height: H })
                // SAM 3.1's open-vocab detector only grounds CONCRETE nouns, not
                // the abstract concept "main subject" — so /segment/instances
                // returns the saliency matte (mode != 'sam3') for the generic
                // subject prompt. Seed SAM 3.1 with that matte's bounding box (a
                // box prompt — the strongest single-object prompt) to upgrade it
                // to a crisp SAM 3.1 mask of the same subject.
                if (mode !== 'sam3') {
                    const box = bboxOfMaskCanvas(canvas)
                    if (box && (box[2] - box[0]) >= 8 && (box[3] - box[1]) >= 8) {
                        aiSet({ busy: true, status: `${noun}: refining with SAM 3.1 (box-seeded from saliency)…` })
                        try {
                            const samCanvas = await serviceSamBox(srcRef.current, box, { width: W, height: H })
                            if (samCanvas && bboxOfMaskCanvas(samCanvas)) { canvas = samCanvas; mode = 'sam3'; model = model || 'sam3.1' }
                        } catch { /* SAM 3.1 box failed — keep the saliency matte */ }
                    }
                }
                const keys = registerTexture(canvas)
                commit({ ...semanticLayer({ maskTextureKey: keys.maskTextureKey, feather: 0.02, label }), baseTextureKey: keys.baseTextureKey, growPx: 0, inverted: invert, ...newFillProps('semantic') })
                const sam3 = mode === 'sam3'
                aiSet({ busy: false, device: sam3 ? 'sam3.1' : 'service', lastFragmented: false, status: sam3 ? `${noun} masked — SAM 3.1 (${model || 'sam3.1'})${invert ? ' → inverted = entire sky/background' : ''}` : `${noun} masked — service fallback (${mode})${invert ? ' → inverted = entire sky/background' : '; install SAM 3.1 for best results'}` })
                return
            } catch (e) {
                aiSet({ status: `Service subject failed (${e?.message || e}) — falling back on-device…` })
            }
        }
        aiSet({ busy: true, status: `AI ${noun}: loading background-removal model (first run ~44 MB)…` })
        try {
            const mod = await loadAi()
            const raw = await mod.clientSubjectMask(srcRef.current, { width: W, height: H })
            const { canvas, diagnostics } = cleanSubjectMatte(raw, {
                threshold: Math.max(0.05, Math.min(0.95, 1 - (ai.sensitivity ?? 0.5))),
                fillHoles: ai.fillHoles !== false,
                luminanceAssist: true,
                sourceCanvas: srcRef.current,
            })
            const keys = registerTexture(canvas)
            commit({ ...semanticLayer({ maskTextureKey: keys.maskTextureKey, feather: 0.02, label }), baseTextureKey: keys.baseTextureKey, growPx: 0, inverted: invert, ...newFillProps('semantic') })
            const st = mod.getClientAIState()
            const tip = diagnostics.fragmented ? ' — matte looked fragmented; AI Box-Select usually nails stylized subjects' : ''
            aiSet({ busy: false, device: st.device, lastFragmented: diagnostics.fragmented, status: `${noun} masked on ${st.device || 'device'} (cleaned)${invert ? ' → inverted = entire sky/background' : ''}${tip}` })
        } catch (e) { aiSet({ busy: false, status: `${noun} failed: ` + (e?.message || e) }) }
    }, [W, H, commit, ensureService, ai.sensitivity, ai.fillHoles])

    const runDepth = useCallback(async () => {
        if (!srcRef.current) return
        aiSet({ busy: true, status: 'Depth: loading Depth-Anything (first run downloads ~50 MB)…' })
        try {
            const mod = await loadAi()
            const canvas = await mod.clientDepthMap(srcRef.current, { width: W, height: H })
            const key = 'ai-depth-' + uid()
            setMaskTexture(key, canvas)
            // default to the NEAR half of the depth range (white = near)
            commit({ ...depthLayer({ depthMapKey: key, min: 0.5, max: 1, softness: 0.15, label: 'AI Depth (near)' }), ...newFillProps('depth') })
            const st = mod.getClientAIState()
            aiSet({ busy: false, status: `Depth map ready on ${st.device || 'device'}`, device: st.device })
        } catch (e) { aiSet({ busy: false, status: 'Depth failed: ' + (e?.message || e) }) }
    }, [W, H, commit])

    const runText = useCallback(async (phrase) => {
        if (!srcRef.current || !phrase) return
        // "sky"/"horizon" are real visual concepts SAM 3.1 grounds directly and
        // completely (≈80% at the 0.25 confidence default), so they go through
        // /ground/text below. Abstract "background"/"backdrop" aren't SAM
        // concepts — segment the subject and invert (its COMPLEMENT) instead.
        if (/\b(background|backdrop|backg(?:round)?|scenery|behind)\b/i.test(phrase.trim())) {
            return runSubject(phrase, { invert: true, label: `AI: ${phrase}` })
        }
        // Use the service's TEXT-grounding endpoint (/ground/text → SAM 3 when
        // loaded, else CLIPSeg). NOT /segment/instances: that's concept seg that
        // falls back to a saliency SUBJECT mask when SAM 3.1 is absent, so every
        // phrase returns the same salient object regardless of the words (the
        // "search returns the subject for everything" bug).
        const svc = await ensureService()
        if (svc.status === 'available') {
            aiSet({ busy: true, status: `Grounding “${phrase}” via mask service…` })
            try {
                const { canvas, engine, score } = await serviceGroundText(srcRef.current, phrase, { width: W, height: H })
                const keys = registerTexture(canvas)
                commit({ ...semanticLayer({ maskTextureKey: keys.maskTextureKey, feather: 0.03, label: `AI: ${phrase}` }), baseTextureKey: keys.baseTextureKey, growPx: 0, ...newFillProps('semantic') })
                aiSet({ busy: false, device: engine === 'sam3' ? 'sam3.1' : 'clipseg', status: `“${phrase}” matched (${engine}${typeof score === 'number' ? `, score ${score.toFixed(2)}` : ''})` })
                return
            } catch (e) {
                aiSet({ status: `Service grounding failed (${e?.message || e}) — trying on-device CLIPSeg…` })
            }
        }
        aiSet({ busy: true, status: `Text: grounding “${phrase}” (first run downloads CLIPSeg ≈600 MB)…` })
        try {
            const mod = await loadAi()
            const { canvas, score } = await mod.clientGroundPhrase(srcRef.current, phrase, { width: W, height: H })
            if (!canvas) throw new Error('no region matched')
            const keys = registerTexture(canvas)
            commit({ ...semanticLayer({ maskTextureKey: keys.maskTextureKey, feather: 0.03, label: `AI: ${phrase}` }), baseTextureKey: keys.baseTextureKey, growPx: 0, ...newFillProps('semantic') })
            const st = mod.getClientAIState()
            aiSet({ busy: false, status: `“${phrase}” matched (score ${score?.toFixed?.(2) ?? '?'}) on ${st.device || 'device'}`, device: st.device })
        } catch (e) { aiSet({ busy: false, status: 'Text grounding failed: ' + (e?.message || e) }) }
    }, [W, H, commit, ensureService, runSubject])

    // Lightweight device check: an INSTANT WebGPU/WASM probe, then confirm
    // end-to-end with ONLY the smallest model (RMBG subject, ~44 MB) on a tiny
    // synthetic scene — far faster than the 3-model self-test (~250 MB) the user
    // found slow.
    const testDevice = useCallback(async () => {
        aiSet({ busy: true, status: 'Checking device…', report: null })
        try {
            let device = 'wasm'
            try { if (navigator.gpu && (await navigator.gpu.requestAdapter())) device = 'webgpu' } catch { /* no webgpu */ }
            aiSet({ device, status: `Device supports ${device.toUpperCase()} — confirming with the subject model (one-time ~44 MB download)…` })
            const mod = await loadAi()
            const probe = document.createElement('canvas'); probe.width = 256; probe.height = 256
            const pc = probe.getContext('2d')
            pc.fillStyle = '#23304a'; pc.fillRect(0, 0, 256, 256)
            pc.fillStyle = '#d8b552'; pc.beginPath(); pc.arc(128, 152, 60, 0, Math.PI * 2); pc.fill(); pc.fillRect(110, 72, 36, 90)
            const t0 = Date.now()
            const out = await mod.clientSubjectMask(probe, { width: 256, height: 256 })
            const ms = ((Date.now() - t0) / 1000).toFixed(1)
            const st = mod.getClientAIState()
            const ok = !!(out && out.width)
            aiSet({
                busy: false, device: st.device || device,
                status: ok ? `On-device AI works — subject model ran in ${ms}s on ${st.device || device}` : 'Model loaded but returned no mask',
                report: { caps: [['Subject', ok], [String(st.device || device).toUpperCase(), true]] },
            })
        } catch (e) { aiSet({ busy: false, status: 'Device AI check failed: ' + (e?.message || e) }) }
    }, [])

    // On-device SlimSAM — offline fallback when the SAM 3.1 service is down.
    // Loads once; image encodes per call (SlimSAM is tiny).
    const ensureSam = async () => {
        const tf = await import('@huggingface/transformers')
        tf.env.backends.onnx.wasm.wasmPaths = new URL('ort/', document.baseURI).href
        tf.env.backends.onnx.wasm.numThreads = 1
        if (!samRef.current || samRef.current.src !== srcRef.current) {
            let device = 'wasm'
            try { if (navigator.gpu && (await navigator.gpu.requestAdapter())) device = 'webgpu' } catch { /* no webgpu */ }
            const model = samRef.current?.model || await tf.SamModel.from_pretrained('Xenova/slimsam-77-uniform', { device })
            const processor = samRef.current?.processor || await tf.AutoProcessor.from_pretrained('Xenova/slimsam-77-uniform')
            const blob = await new Promise((res, rej) => srcRef.current.toBlob((b) => (b ? res(b) : rej(new Error('toBlob failed'))), 'image/png'))
            const image = await tf.RawImage.fromBlob(blob)
            // Encode the image ONCE (the slow ViT pass) and cache the embeddings —
            // every subsequent click reuses them, so multi-point refine is fast.
            let embeddings = null
            try { embeddings = await model.get_image_embeddings(await processor(image)) } catch { /* fall back to per-call encode */ }
            samRef.current = { model, processor, image, embeddings, src: srcRef.current, device }
        }
        return samRef.current
    }

    const runSamBox = async (box) => {
        const { model, processor, image, embeddings } = await ensureSam()
        const input_boxes = [[[Math.round(box[0]), Math.round(box[1]), Math.round(box[2]), Math.round(box[3])]]]
        const inputs = await processor(image, { input_boxes })
        let outputs
        if (embeddings) {
            try { outputs = await model({ ...embeddings, input_boxes: inputs.input_boxes }) } catch { outputs = null }
        }
        if (!outputs) outputs = await model(inputs) // fallback: full encode
        const masks = await processor.post_process_masks(outputs.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes)
        return samMaskToCanvas(masks[0], outputs.iou_scores)
    }

    // On-device SlimSAM point-prompt — click-select fallback.
    const runSamClick = async (points, labels) => {
        const { model, processor, image, embeddings } = await ensureSam()
        const input_points = [[points.map(([x, y]) => [Math.round(x), Math.round(y)])]]
        const input_labels = [[labels]]
        const inputs = await processor(image, { input_points, input_labels })
        let outputs
        if (embeddings) {
            try { outputs = await model({ ...embeddings, input_points: inputs.input_points, input_labels: inputs.input_labels }) } catch { outputs = null }
        }
        if (!outputs) outputs = await model(inputs)
        const masks = await processor.post_process_masks(outputs.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes)
        return samMaskToCanvas(masks[0], outputs.iou_scores)
    }

    // Box-select: drag a box around an object → SAM 3.1 (service) returns its
    // mask. Falls back to on-device SlimSAM when the service is unavailable.
    const onSamBox = useCallback(async (box) => {
        if (ai.busy || !srcRef.current) return
        aiSet({ busy: true, status: 'Box-Select: segmenting…' })
        try {
            let canvas = null
            const svc = await ensureService()
            if (svc.status === 'available') {
                try { canvas = await serviceSamBox(srcRef.current, box, { width: W, height: H }) } catch { canvas = null }
            }
            let onDevice = false
            if (!canvas) { canvas = await runSamBox(box); onDevice = true }
            const keys = registerTexture(canvas)
            commit({ ...semanticLayer({ maskTextureKey: keys.maskTextureKey, feather: 0.02, label: 'AI Box-Select' }), baseTextureKey: keys.baseTextureKey, growPx: 0, ...newFillProps('semantic') })
            const st = onDevice ? (samRef.current?.device || 'device') : 'SAM 3.1'
            aiSet({ busy: false, status: `Selected on ${st} — drag another box to select again, or press Done`, device: st })
        } catch (e) { aiSet({ busy: false, status: 'Box-Select failed: ' + (e?.message || e) }) }
    }, [ai.busy, W, H, commit, ensureService])

    // Click-select / predictive refine: each click runs SAM at that point to get
    // the object under the cursor, then composites it INTO the active mask — a
    // normal click UNIONS the region in (add), Alt+click SUBTRACTS it (remove).
    // SAM is always prompted POSITIVELY at the point (to find what's there); the
    // click's intent (add vs remove) is decided when compositing. This ALTERS the
    // current selection (e.g. an AI Subject mask) instead of replacing it. With no
    // mask active, the first click seeds a fresh AI Click-Select layer.
    const onClickSelect = useCallback(async (pts) => {
        if (ai.busy || !srcRef.current || !pts.length) return
        const last = pts[pts.length - 1]
        const add = last.label !== 0   // Alt+click (label 0) = remove the region
        aiSet({ busy: true, status: add ? 'Click-Select: finding region to add…' : 'Click-Select: finding region to remove…' })
        try {
            // Prompt SAM positively at the click to segment the object there.
            let blob = null
            const svc = await ensureService()
            if (svc.status === 'available') {
                try { blob = await serviceSamClick(srcRef.current, [[last.x, last.y]], [1], { width: W, height: H }) } catch { blob = null }
            }
            let onDevice = false
            if (!blob) { blob = await runSamClick([[last.x, last.y]], [1]); onDevice = true }

            const sel = clickMaskId ? chain.find(e => e.layer.id === clickMaskId) : null
            if (sel) {
                // Composite the SAM blob into the active mask's coverage (white =
                // selected, opaque). Union via 'lighten'; subtract by multiplying
                // in the blob's inverse (white-blob → black → zeroes coverage).
                const key = sel.layer.maskTextureKey
                const acc = document.createElement('canvas'); acc.width = W; acc.height = H
                const ax = acc.getContext('2d')
                ax.fillStyle = '#000'; ax.fillRect(0, 0, W, H)
                const cur = key && getMaskTexture(key)
                if (cur && cur.width) ax.drawImage(cur, 0, 0, W, H)
                if (add) {
                    ax.globalCompositeOperation = 'lighten'
                    ax.drawImage(blob, 0, 0, W, H)
                } else {
                    const inv = document.createElement('canvas'); inv.width = W; inv.height = H
                    const ix = inv.getContext('2d')
                    ix.drawImage(blob, 0, 0, W, H)
                    ix.globalCompositeOperation = 'difference'
                    ix.fillStyle = '#fff'; ix.fillRect(0, 0, W, H)  // 255 − blob = inverse
                    ax.globalCompositeOperation = 'multiply'
                    ax.drawImage(inv, 0, 0)
                }
                ax.globalCompositeOperation = 'source-over'
                if (key) setMaskTexture(key, acc)
                // Re-base the boundary grow/shrink origin to the refined edge.
                const baseKey = sel.layer.baseTextureKey
                if (baseKey) {
                    const snap = document.createElement('canvas'); snap.width = W; snap.height = H
                    snap.getContext('2d').drawImage(acc, 0, 0)
                    setMaskTexture(baseKey, snap)
                }
                updateLayer(clickMaskId, { growPx: 0 })
                bump()
            } else {
                // No active mask — first click seeds a fresh AI Click-Select layer.
                const keys = registerTexture(blob)
                const lid = commit({ ...semanticLayer({ maskTextureKey: keys.maskTextureKey, feather: 0.02, label: 'AI Click-Select' }), baseTextureKey: keys.baseTextureKey, growPx: 0, ...newFillProps('semantic') })
                setClickMaskId(lid)
            }

            const st = onDevice ? (samRef.current?.device || 'device') : 'SAM 3.1'
            aiSet({ busy: false, status: `${sel ? (add ? 'Added' : 'Removed') + ' region' : 'Selected'} on ${st} — click to ADD · Alt+click to REMOVE · Done to finish`, device: st })
        } catch (e) { aiSet({ busy: false, status: 'Click-Select failed: ' + (e?.message || e) }) }
    }, [ai.busy, W, H, commit, ensureService, clickMaskId, chain, updateLayer])

    // Unified palette dispatch: AI tools run an on-device model; everything else
    // opens its canvas interaction — no separation between AI and regular masks.
    const onToolClick = useCallback((t) => {
        if (t.ai === 'subject') return runSubject()
        if (t.ai === 'background') return runSubject(undefined, { invert: true })
        if (t.ai === 'depth') return runDepth()
        if (t.ai === 'sam') { setTool('sam'); setStatus('Drag a box around an object to select it (SAM 3.1)'); return }
        if (t.ai === 'clickselect') {
            setClickPoints([])
            // If a semantic mask (AI Subject / Box-Select / a prior Click-Select)
            // is selected, REFINE it: clicks add/remove regions on that mask. With
            // nothing selected, the first click seeds a fresh selection instead.
            const sel = chain.find(e => e.layer.id === selectedId)?.layer
            const refine = !!sel && sel.kind === 'semantic' && !!sel.maskTextureKey
            setClickMaskId(refine ? sel.id : null)
            setTool('clickselect')
            setStatus(refine
                ? 'Refining the selected mask · click to ADD a region · Alt+click to REMOVE · Done to finish'
                : 'Click on an object to select it · Alt+click to exclude regions · click more to refine')
            return
        }
        return addMask(t)
    }, [runSubject, runDepth, addMask, chain, selectedId])

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

    // ⌘/Ctrl+Z undo · ⌘/Ctrl+Shift+Z (or Ctrl+Y) redo.
    useEffect(() => {
        const onKey = (e) => {
            if (!(e.metaKey || e.ctrlKey)) return
            const k = e.key.toLowerCase()
            if (k === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo() }
            else if (k === 'y') { e.preventDefault(); redo() }
        }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
    }, [undo, redo])

    /* ── live render ───────────────────────────────────────────────────── */
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
            drawMaskBoundary(g, key && getMaskTexture(key), W, H, ipx)
        }

        // active pen draft
        if (tool === 'pen' && draft.length) {
            g.strokeStyle = ACCENT; g.lineWidth = LW; g.setLineDash([6 * ipx, 4 * ipx])
            g.beginPath(); g.moveTo(draft[0].x, draft[0].y)
            for (let i = 1; i < draft.length; i++) g.lineTo(draft[i].x, draft[i].y)
            g.stroke()
            draft.forEach((p, i) => handle(p.x, p.y, i === 0 ? '#fff' : ACCENT))
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
    }, [chain, selectedId, tool, draft, samBox, clickPoints, imageSize, W, H, tick, preview])

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
            runSubject: () => runSubject(),
            background: () => runSubject(undefined, { invert: true }),
            runDepth: () => runDepth(),
            samBox: (x0, y0, x1, y1) => onSamBox([x0, y0, x1, y1]),
            clickSelect: (x, y, label = 1) => { const pts = [...clickPoints, { x, y, label }]; setClickPoints(pts); onClickSelect(pts) },
            testDevice: () => testDevice(),
            serviceStatus: () => ai.service,
            checkService: async () => {
                aiSet({ service: { status: 'checking' } })
                const s = await checkService()
                aiSet({ service: { status: s.available ? 'available' : 'offline', sam3: s.sam3, sam3Loaded: s.sam3Loaded, model: s.model, subjectEngine: s.subjectEngine } })
                return s
            },
            runConcept: (concept) => runSubject(concept),
            setSensitivity: (v) => aiSet({ sensitivity: Number(v) }),
            setFillHoles: (v) => aiSet({ fillHoles: !!v }),
            aiState: () => ai,
            layer: (id) => {
                const l = chain.find((e) => e.layer.id === id)?.layer
                return l ? { kind: l.kind, center: l.center, radius: l.radius, rotation: l.rotation, p1: l.p1, p2: l.p2 } : null
            },
            pixels: () => {
                const d = dispRef.current
                return d.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, d.width, d.height).data
            },
        }
        window.__ready = true
    }, [imageSize, chain, baseLayer, W, H, commit, updateLayer, setFillMode, applyCurve, onExpandBoundary, ai, runSubject, runDepth, testDevice, onSamBox, onClickSelect, clickPoints, undo, redo, startRefine])

    const selected = chain.find((e) => e.layer.id === selectedId) || null
    // Brush-refine targets texture-backed masks whose coverage is paintable
    // (AI Subject / Click-Select / brush / smart-brush). Parametric & vector
    // masks (radial / linear / pen / lasso) are edited via their handles instead.
    const canRefine = !!selected && ['semantic', 'brush', 'smartBrush'].includes(selected.layer.kind)
        && !!(selected.layer.maskTextureKey || selected.layer.brushTextureKey)

    /* ── render ────────────────────────────────────────────────────────── */
    return (
        <div className="studio">
            <div className="studio-top">
                <h1>Mask <b>Studio</b></h1>
                <label className="mask-btn" style={{ cursor: 'pointer' }}>
                    Load image
                    <input type="file" accept="image/*" onChange={onFile} style={{ display: 'none' }} />
                </label>
                <div className="spacer" />
                <button className="mask-btn" onClick={undo} title="Undo (⌘/Ctrl+Z)">↶ Undo</button>
                <button className="mask-btn" onClick={redo} title="Redo (⌘/Ctrl+Shift+Z)">↷ Redo</button>
                <label className="mask-toggle"><input type="checkbox" checked={preview} onChange={(e) => setPreview(e.target.checked)} /> Preview (hide guides)</label>
                <label className="mask-toggle"><input type="checkbox" checked={overlayMode} onChange={(e) => setOverlayMode(e.target.checked)} /> Show mask overlay</label>
                <label className="mask-toggle"><input type="checkbox" checked={globalInvert} onChange={(e) => setGlobalInvert(e.target.checked)} /> Invert all</label>
                <span className="hint">{status}</span>
            </div>

            <div className="studio-main">
                <div className="stage-wrap">
                    {hasImage ? (
                        <div className="stage">
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
                            {(tool === 'brush' || tool === 'refine') && <div ref={cursorRef} className="brush-cursor" />}
                            {tool && (
                                <div className="tool-banner">
                                    <span><b>{tool === 'sam' ? 'box-select' : tool === 'clickselect' ? 'click-select' : tool === 'refine' ? 'brush-refine' : tool}</b> · {status || 'draw on the image'}</span>
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
                                                onClick={() => { setClickPoints([]); setStatus('Click to ADD a region · Alt+click to REMOVE · Done to finish') }}
                                                title="Clear the click markers (use ⌘/Ctrl+Z to undo individual add/remove edits)">
                                                Clear points
                                            </button>
                                        </>
                                    )}
                                    <button className="mask-btn" onClick={() => { setTool(null); setDraft([]); setSamBox(null); setClickPoints([]); setClickMaskId(null); refineRef.current = null; setStatus('') }}>Done</button>
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
                                <span className="ai-badge" title="mask service / AI backend">
                                    {ai.service?.status === 'available'
                                        ? (ai.service.sam3 ? 'SAM 3.1 ✓' : 'service ✓')
                                        : ai.service?.status === 'checking' ? 'service…'
                                            : (ai.device ? String(ai.device).toUpperCase() : 'on-device')}
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
                            <div className="ai-row">
                                <input className="ai-input" placeholder="AI: describe a region… e.g. the sky" value={aiPhrase}
                                    disabled={ai.busy}
                                    onChange={(e) => setAiPhrase(e.target.value)}
                                    onKeyDown={(e) => { if (e.key === 'Enter') runText(aiPhrase) }} />
                                <button className="mask-btn" disabled={!hasImage || ai.busy || !aiPhrase} onClick={() => runText(aiPhrase)}>Find</button>
                            </div>
                            <div className="global-row">
                                <label className="mask-toggle"><input type="checkbox" checked={penSmooth} onChange={(e) => setPenSmooth(e.target.checked)} /> Pen: smooth</label>
                                <label className="mask-toggle"><input type="checkbox" checked={edgeAware} onChange={(e) => setEdgeAware(e.target.checked)} /> Brush: edge-aware</label>
                            </div>
                            <div className="global-row" title="On-device AI Subject cleanup — used when the SAM 3.1 service is offline">
                                <label className="mask-toggle" style={{ gap: 6 }}>
                                    Sensitivity
                                    <input type="range" className="mask-range" min={0} max={1} step={0.05}
                                        value={ai.sensitivity ?? 0.5} disabled={ai.busy}
                                        onChange={(e) => aiSet({ sensitivity: Number(e.target.value) })} style={{ width: 80 }} />
                                </label>
                                <label className="mask-toggle"><input type="checkbox" checked={ai.fillHoles !== false} disabled={ai.busy} onChange={(e) => aiSet({ fillHoles: e.target.checked })} /> Fill holes</label>
                            </div>
                            {ai.status && <div className="ai-status">{ai.busy ? '⏳ ' : ''}{ai.status}</div>}
                            {ai.lastFragmented && !ai.busy && (
                                <div className="ai-status">💡 The auto subject matte looked fragmented — <b>AI Box-Select</b> (drag a box around the object) usually nails stylized / backlit subjects.</div>
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
                                <span className="ai-note">AI runs in-browser (WebGPU→WASM); models cache after first use; nothing leaves your device.</span>
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
