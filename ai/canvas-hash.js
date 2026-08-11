/**
 * canvas-hash — shared FNV-1a content hash over a 16×16 downsample.
 *
 * Both sam-client.js (contentKey) and asset-store.js (hashCanvas) implemented
 * the same hash independently, each allocating a throwaway 16×16 canvas per
 * call. This module deduplicates them behind a single reusable canvas, cutting
 * GC pressure to zero and the hash loop from 1024 → 256 iterations (u32 path).
 */

const _canvas = typeof document !== 'undefined' ? document.createElement('canvas') : null
if (_canvas) { _canvas.width = 16; _canvas.height = 16 }
const _ctx = _canvas?.getContext('2d', { willReadFrequently: true }) || null

/**
 * FNV-1a hash of a canvas's visual content, downsampled to 16×16.
 * Returns a hex string. ~1 ms — negligible next to any model call.
 */
export const canvasHash = (src) => {
    if (!_ctx) return '0'
    _ctx.drawImage(src, 0, 0, 16, 16)
    const px = _ctx.getImageData(0, 0, 16, 16).data
    // 4-bytes-at-a-time: 256 iterations instead of 1024
    const u32 = new Uint32Array(px.buffer, px.byteOffset, px.byteLength >> 2)
    let h = 0x811c9dc5
    for (let i = 0; i < u32.length; i++) {
        h ^= u32[i]
        h = Math.imul(h, 0x01000193) >>> 0
    }
    return h.toString(16)
}

/**
 * Content key for the embedding cache: dims + hash.
 * `doc:` namespaces whole-document embeddings; crop re-encodes live under
 * `crop:${hash}:${rect}` and must never collide with these.
 */
export const contentKey = (canvas) =>
    `doc:${canvas.width}x${canvas.height}:${canvasHash(canvas)}`

/**
 * Bare hash for asset-store's assetKey (no `doc:` prefix).
 */
export const hashCanvas = (canvas) =>
    `${canvas.width}x${canvas.height}:${canvasHash(canvas)}`
