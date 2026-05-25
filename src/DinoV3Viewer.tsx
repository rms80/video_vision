// Dense-patch-feature viewer for DINOv3 outputs produced by
// scripts/run_dinov3.py. Renders the current frame and overlays a
// per-patch cosine-similarity heatmap.
//
// Interaction model:
//   - Hover (empty feature set): live preview against the patch under
//     the cursor.
//   - Click: capture that patch as a positive feature, clearing the set;
//     scrubbing frames re-applies it against each new frame.
//   - Ctrl/Cmd+click: append the patch as a positive feature.
//   - Alt/Option+click: append the patch as a negative feature.
//   - Reset (bumped via the resetVersion prop): clears the set.
// Scoring:
//   For each patch, score = max_p cos(patch, q_p) - max_n cos(patch, q_n),
//   where p/n range over captured positive/negative features. An empty
//   side contributes 0, so positives-only reduces to the original max-
//   cosine behavior and negatives-only shows the anti-pattern (regions
//   least like the negatives).
//
// Storage: features live at /analysis/<stem>/_scene/dinov3/{meta.json,
// NNNNNN.npz}. Each .npz holds { patches: (grid_h, grid_w, embed_dim)
// fp16 }. We L2-normalize features once on load so cosine sim is a dot
// product on the hot path; captured features inherit that normalization.

import { createSignal, createEffect, onCleanup, untrack, Show } from "solid-js";
import { parseNpz } from "./npz";

export interface DinoV3Meta {
  model: string;
  patch_size: number;
  input_width: number;
  input_height: number;
  grid_width: number;
  grid_height: number;
  embed_dim: number;
  subsample_every: number;
  frame_indices: number[];
  source_width: number;
  source_height: number;
}

interface CachedFrame {
  /** L2-normalized features, shape (gridH * gridW, embedDim), row-major. */
  feats: Float32Array;
}

export interface DinoV3ViewerProps {
  videoName: string | null;
  currentFrame: number;
  /** True iff the dinov3 tab is currently selected; viewer is mounted
   *  unconditionally so the cached features survive tab switches. */
  visible: boolean;
  /** Heatmap-over-image opacity, 0..1. 0 = image only, 1 = heatmap only.
   *  Owned by App.tsx so the slider can live in the tab toolbar. */
  heatmapOpacity: number;
  /** Bumped by App.tsx whenever a new DinoV3 run finishes for the active
   *  video, so the viewer can drop its in-memory feature cache and
   *  cache-bust the meta + .npz fetches against the browser HTTP cache. */
  dataVersion: number;
  /** Bumped by App.tsx (via the Reset button) to discard any captured
   *  feature set and clear the heatmap. */
  resetVersion: number;
  /** Visualization style. "heatmap" = viridis on normalized cosine sim.
   *  "contour" = binary mask of patches at/above `threshold`. */
  mode: "heatmap" | "contour";
  /** Cosine-sim cutoff used in contour mode (0..1). Ignored in heatmap mode. */
  threshold: number;
  /** Called whenever the loaded DinoV3 meta changes (incl. to/from null).
   *  App.tsx uses this for cross-cutting UI like frame-slider snapping
   *  to subsampled frames and a Patch Grid info readout. */
  onMeta?: (meta: DinoV3Meta | null) => void;
}

/** A patch feature plucked out of some frame, along with where it came
 *  from so we can draw a marker when the user scrubs back to that frame.
 *  `kind` controls how it contributes to the score: positives raise it,
 *  negatives subtract via the margin formulation. */
interface CapturedFeature {
  /** L2-normalized embedding, length embed_dim. */
  feat: Float32Array;
  kind: "positive" | "negative";
  fromFrame: number;
  fromPx: number;
  fromPy: number;
}

const VIRIDIS_RGB: [number, number, number][] = [];
{
  const ctrl = [
    [0.267004, 0.004874, 0.329415], [0.282327, 0.140926, 0.457517],
    [0.253935, 0.265254, 0.529983], [0.206756, 0.371758, 0.553117],
    [0.163625, 0.471133, 0.558148], [0.127568, 0.566949, 0.550556],
    [0.134692, 0.658636, 0.517649], [0.266941, 0.748751, 0.440573],
    [0.477504, 0.821444, 0.318195], [0.741388, 0.873449, 0.149561],
    [0.993248, 0.906157, 0.143936],
  ];
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (ctrl.length - 1);
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, ctrl.length - 1);
    const f = t - lo;
    VIRIDIS_RGB.push([
      Math.round((ctrl[lo][0] * (1 - f) + ctrl[hi][0] * f) * 255),
      Math.round((ctrl[lo][1] * (1 - f) + ctrl[hi][1] * f) * 255),
      Math.round((ctrl[lo][2] * (1 - f) + ctrl[hi][2] * f) * 255),
    ]);
  }
}

function videoStem(name: string): string {
  return name.replace(/\.[^.]+$/, "");
}

function nearestIdx(indices: number[], target: number): number {
  // Indices are produced by ::subsample, so they're sorted ascending.
  let best = indices[0];
  let bestDist = Math.abs(target - best);
  for (let i = 1; i < indices.length; i++) {
    const d = Math.abs(target - indices[i]);
    if (d < bestDist) { best = indices[i]; bestDist = d; }
  }
  return best;
}

export default function DinoV3Viewer(props: DinoV3ViewerProps) {
  const [meta, setMeta] = createSignal<DinoV3Meta | null>(null);
  const [activeFrame, setActiveFrame] = createSignal<number | null>(null);
  const [featureSet, setFeatureSet] = createSignal<CapturedFeature[]>([]);
  const [hover, setHover] = createSignal<{ px: number; py: number } | null>(null);

  // LRU-ish cache (cap by count); patches are ~3 MB float32 each at ViT-L/16,
  // so 20 frames ≈ 60 MB — plenty for hover scrubbing.
  const cache = new Map<number, CachedFrame>();
  const CACHE_CAP = 20;

  let imgEl: HTMLImageElement | undefined;
  let heatmapCanvas: HTMLCanvasElement | undefined;
  let containerEl: HTMLDivElement | undefined;

  // Fetch meta whenever the video changes OR App bumps dataVersion (e.g.
  // after a fresh run). Clears the in-memory feature cache too, since the
  // grid shape / embed_dim may have changed; ?v= cache-busts the HTTP
  // response so the browser doesn't hand us yesterday's meta.json.
  createEffect(() => {
    const v = props.videoName;
    const version = props.dataVersion;
    setMeta(null);
    cache.clear();
    setFeatureSet([]);
    setHover(null);
    setActiveFrame(null);
    clearHeatmap();
    if (!v) return;
    const url = `/analysis/${encodeURIComponent(videoStem(v))}/_scene/dinov3/meta.json?v=${version}`;
    // 404 is the normal "no outputs yet" state — App.tsx hides the DinoV3
    // tab in that case, so we silently leave meta null. Real failures
    // (parse errors, non-404 HTTP) are logged for debugging.
    fetch(url, { cache: "no-store" })
      .then(async (r) => {
        if (!r.ok) {
          if (r.status !== 404) console.warn(`[dinov3] meta.json HTTP ${r.status}`);
          return;
        }
        try { setMeta(await r.json()); }
        catch (e) { console.warn("[dinov3] meta.json parse error:", e); }
      })
      .catch((e) => console.warn("[dinov3] meta.json fetch error:", e));
  });

  // Track the nearest available frame as the user scrubs. Don't fetch — the
  // .npz fetch is kicked off lazily on hover, on click, and (when a feature
  // set is captured) on every frame change, so the heatmap stays alive while
  // scrubbing.
  createEffect(() => {
    const m = meta();
    if (!m) return;
    const idx = nearestIdx(m.frame_indices, props.currentFrame);
    setActiveFrame(idx);
  });

  async function ensureFrame(idx: number): Promise<CachedFrame | null> {
    if (cache.has(idx)) return cache.get(idx)!;
    const v = props.videoName;
    const m = meta();
    if (!v || !m) return null;
    const padded = String(idx).padStart(6, "0");
    // Cache-bust both layers: the in-memory cache is keyed by frame idx
    // (cleared whenever dataVersion bumps), and the query string makes the
    // browser HTTP cache treat a re-run's .npz as a different URL.
    const url = `/analysis/${encodeURIComponent(videoStem(v))}/_scene/dinov3/${padded}.npz?v=${props.dataVersion}`;
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) return null;
      const buf = await r.arrayBuffer();
      const arrays = await parseNpz(buf);
      const a = arrays["patches"];
      if (!a) return null;
      // parseNpz already converts fp16 → Float32Array.
      const f32 = a.data as Float32Array;
      const n = m.grid_height * m.grid_width;
      const d = m.embed_dim;
      // L2-normalize row-wise so future cosine sims are dot products.
      for (let i = 0; i < n; i++) {
        let sumsq = 0;
        const base = i * d;
        for (let k = 0; k < d; k++) sumsq += f32[base + k] * f32[base + k];
        const inv = sumsq > 0 ? 1 / Math.sqrt(sumsq) : 0;
        if (inv !== 1) {
          for (let k = 0; k < d; k++) f32[base + k] *= inv;
        }
      }
      const entry: CachedFrame = { feats: f32 };
      if (cache.size >= CACHE_CAP) {
        // Drop the oldest entry — Map preserves insertion order.
        const first = cache.keys().next().value;
        if (first !== undefined) cache.delete(first);
      }
      cache.set(idx, entry);
      return entry;
    } catch {
      return null;
    }
  }

  /** Render an overlay for the given frame, using the margin formulation
   *  score[i] = max_p cos(patch_i, q_p) - max_n cos(patch_i, q_n). Each
   *  side falls back to 0 when empty, so positives-only matches the
   *  original max-cosine behavior. Visualization is either a viridis
   *  heatmap (props.mode === "heatmap") or a binary mask thresholded at
   *  props.threshold (props.mode === "contour"). Canvas-level opacity is
   *  controlled by the heatmap slider in App.tsx. */
  function renderHeatmapAgainst(frameIdx: number, positives: Float32Array[], negatives: Float32Array[]) {
    const m = meta();
    const canvas = heatmapCanvas;
    if (!m || !canvas || (positives.length === 0 && negatives.length === 0)) return;
    const entry = cache.get(frameIdx);
    if (!entry) return;
    const gw = m.grid_width;
    const gh = m.grid_height;
    const d = m.embed_dim;
    const n = gw * gh;
    const feats = entry.feats;

    // Cosine sim = dot product (both sides are L2-normalized). For each
    // patch, take the max over positives and the max over negatives, then
    // subtract: score = max_pos - max_neg.
    const sims = new Float32Array(n);
    let smin = Infinity, smax = -Infinity;
    for (let i = 0; i < n; i++) {
      const base = i * d;
      let maxPos = 0;
      if (positives.length > 0) {
        maxPos = -Infinity;
        for (let j = 0; j < positives.length; j++) {
          const q = positives[j];
          let s = 0;
          for (let k = 0; k < d; k++) s += feats[base + k] * q[k];
          if (s > maxPos) maxPos = s;
        }
      }
      let maxNeg = 0;
      if (negatives.length > 0) {
        maxNeg = -Infinity;
        for (let j = 0; j < negatives.length; j++) {
          const q = negatives[j];
          let s = 0;
          for (let k = 0; k < d; k++) s += feats[base + k] * q[k];
          if (s > maxNeg) maxNeg = s;
        }
      }
      const score = maxPos - maxNeg;
      sims[i] = score;
      if (score < smin) smin = score;
      if (score > smax) smax = score;
    }
    if (props.mode === "contour") {
      // Bilinearly upsample the patch-grid sim field to a finer grid, then
      // threshold there — gives a true isocontour of the sim field rather
      // than relying on the browser's alpha interpolation between patch
      // cells. The CSS scale (canvas → image size) then renders the
      // upsampled mask with normal smoothing, so edges look sharp.
      const S = 8;
      const fw = gw * S;
      const fh = gh * S;
      canvas.width = fw;
      canvas.height = fh;
      const ctx = canvas.getContext("2d")!;
      const img = ctx.createImageData(fw, fh);
      const thr = props.threshold;
      const xDen = fw > 1 ? fw - 1 : 1;
      const yDen = fh > 1 ? fh - 1 : 1;
      for (let fy = 0; fy < fh; fy++) {
        const yf = (fy / yDen) * (gh - 1);
        const y0 = Math.floor(yf);
        const y1 = Math.min(y0 + 1, gh - 1);
        const wy = yf - y0;
        const row0 = y0 * gw;
        const row1 = y1 * gw;
        for (let fx = 0; fx < fw; fx++) {
          const xf = (fx / xDen) * (gw - 1);
          const x0 = Math.floor(xf);
          const x1 = Math.min(x0 + 1, gw - 1);
          const wx = xf - x0;
          const s00 = sims[row0 + x0];
          const s10 = sims[row0 + x1];
          const s01 = sims[row1 + x0];
          const s11 = sims[row1 + x1];
          const sx0 = s00 + (s10 - s00) * wx;
          const sx1 = s01 + (s11 - s01) * wx;
          const s = sx0 + (sx1 - sx0) * wy;
          const idx = (fy * fw + fx) * 4;
          img.data[idx + 0] = 233;
          img.data[idx + 1] = 69;
          img.data[idx + 2] = 96;
          img.data[idx + 3] = s >= thr ? 255 : 0;
        }
      }
      ctx.putImageData(img, 0, 0);
    } else {
      // Stretch contrast to [smin, smax]; for a single query the source patch
      // pegs smax at 1, while smin reflects the worst-matching patch.
      canvas.width = gw;
      canvas.height = gh;
      const ctx = canvas.getContext("2d")!;
      const img = ctx.createImageData(gw, gh);
      const range = smax - smin || 1;
      for (let i = 0; i < n; i++) {
        const t = Math.max(0, Math.min(255, Math.round(((sims[i] - smin) / range) * 255)));
        const [r, g, b] = VIRIDIS_RGB[t];
        img.data[i * 4 + 0] = r;
        img.data[i * 4 + 1] = g;
        img.data[i * 4 + 2] = b;
        img.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    }
  }

  /** Copy a single patch's (already L2-normalized) feature vector out of
   *  the cache so it survives even after the source frame is evicted. */
  function extractFeature(frameIdx: number, px: number, py: number): Float32Array | null {
    const m = meta();
    const entry = cache.get(frameIdx);
    if (!m || !entry) return null;
    const d = m.embed_dim;
    const off = (py * m.grid_width + px) * d;
    const out = new Float32Array(d);
    for (let k = 0; k < d; k++) out[k] = entry.feats[off + k];
    return out;
  }

  function clearHeatmap() {
    const canvas = heatmapCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
  }

  /** Pointer position → (grid_x, grid_y) on the active frame's patch grid. */
  function pointerToGrid(e: PointerEvent): { px: number; py: number } | null {
    const m = meta();
    const img = imgEl;
    if (!m || !img) return null;
    const rect = img.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (x < 0 || y < 0 || x >= rect.width || y >= rect.height) return null;
    const px = Math.floor((x / rect.width) * m.grid_width);
    const py = Math.floor((y / rect.height) * m.grid_height);
    return {
      px: Math.max(0, Math.min(m.grid_width - 1, px)),
      py: Math.max(0, Math.min(m.grid_height - 1, py)),
    };
  }

  async function onPointerMove(e: PointerEvent) {
    // While the user has a captured feature set, the heatmap is locked to
    // it — don't let hover overwrite the visualization.
    if (featureSet().length > 0) return;
    const f = activeFrame();
    if (f == null) return;
    const g = pointerToGrid(e);
    if (!g) return;
    setHover(g);
    if (!cache.has(f)) {
      await ensureFrame(f);
      // After the await, bail if the user has scrubbed, captured a feature,
      // or moved away.
      if (featureSet().length > 0 || activeFrame() !== f || hover() !== g) return;
    }
    const q = extractFeature(f, g.px, g.py);
    if (q) renderHeatmapAgainst(f, [q], []);
  }

  function onPointerLeave() {
    // Leaving the image preserves the captured-set heatmap if one exists.
    if (featureSet().length > 0) return;
    setHover(null);
    clearHeatmap();
  }

  async function onPointerDown(e: PointerEvent) {
    if (e.button !== 0) return;
    const f = activeFrame();
    if (f == null) return;
    const g = pointerToGrid(e);
    if (!g) return;
    await ensureFrame(f);
    // Could have been reset while the .npz fetch was in flight.
    if (activeFrame() !== f) return;
    const feat = extractFeature(f, g.px, g.py);
    if (!feat) return;
    // Modifier semantics:
    //   Ctrl/Cmd   → append as positive.
    //   (none)     → replace the set with a single positive.
    // The negative-feature path below is intentionally disabled for now
    // (the margin scoring/marker plumbing it feeds is still in place).
    let kind: "positive" | "negative" = "positive";
    let append = false;
    // if (e.altKey) {
    //   kind = "negative";
    //   append = true;
    // } else
    if (e.ctrlKey || e.metaKey) {
      kind = "positive";
      append = true;
    }
    const captured: CapturedFeature = { feat, kind, fromFrame: f, fromPx: g.px, fromPy: g.py };
    if (append) {
      setFeatureSet([...featureSet(), captured]);
    } else {
      setFeatureSet([captured]);
    }
    const fs = featureSet();
    renderHeatmapAgainst(
      f,
      fs.filter((c) => c.kind === "positive").map((c) => c.feat),
      fs.filter((c) => c.kind === "negative").map((c) => c.feat),
    );
  }

  // Re-render whenever the active frame, feature set, viz mode, or contour
  // threshold changes — so scrubbing applies the captured features against
  // each new frame, adding to the set updates the overlay immediately, and
  // mode/threshold tweaks refresh the visualization. Mode and threshold are
  // tracked here (not inside renderHeatmapAgainst) because the `await`
  // below ends the reactive context.
  createEffect(async () => {
    const f = activeFrame();
    const fs = featureSet();
    void props.mode;
    void props.threshold;
    if (f == null || fs.length === 0) return;
    await ensureFrame(f);
    // Bail if the set was cleared (e.g. Reset) while fetching.
    const fsNow = featureSet();
    if (fsNow.length === 0 || activeFrame() !== f) return;
    renderHeatmapAgainst(
      f,
      fsNow.filter((c) => c.kind === "positive").map((c) => c.feat),
      fsNow.filter((c) => c.kind === "negative").map((c) => c.feat),
    );
  });

  // Hover preview lives outside Solid's reactive graph (it's driven by a
  // pointermove event handler), so a mode/threshold change while hovering
  // wouldn't otherwise repaint. This effect closes that gap; it deliberately
  // does *not* track hover/featureSet/activeFrame to avoid duplicating the
  // effect above.
  createEffect(() => {
    void props.mode;
    void props.threshold;
    const fs = untrack(featureSet);
    if (fs.length > 0) return; // captured-set repaint is handled above
    const f = untrack(activeFrame);
    const h = untrack(hover);
    if (f == null || !h || !cache.has(f)) return;
    const q = extractFeature(f, h.px, h.py);
    if (q) renderHeatmapAgainst(f, [q], []);
  });

  // Broadcast meta changes to the parent so it can snap the global frame
  // slider to subsampled indices and display patch-grid dims.
  createEffect(() => {
    props.onMeta?.(meta());
  });

  // Reset button (App-owned counter): drop the captured set and clear the
  // overlay. Tracked separately from videoName/dataVersion so it can be
  // triggered independently.
  let lastResetVersion = props.resetVersion;
  createEffect(() => {
    const v = props.resetVersion;
    if (v === lastResetVersion) return;
    lastResetVersion = v;
    setFeatureSet([]);
    clearHeatmap();
  });

  onCleanup(() => cache.clear());

  const frameImgUrl = () => {
    const v = props.videoName;
    const f = activeFrame();
    if (!v || f == null) return null;
    return `/analysis/${encodeURIComponent(videoStem(v))}/_scene/frames/${String(f).padStart(6, "0")}.jpg`;
  };

  /** Pixel positions of source-patch crosshairs, relative to the img.
   *  When the feature set is empty, a single hover marker. Otherwise, one
   *  pinned marker per captured feature whose source frame is the active
   *  frame (other captures belong to other frames and are invisible here).
   *  `kind` (positive/negative) colors the marker. */
  type Marker = { left: number; top: number; pinned: boolean; kind: "positive" | "negative" };
  const sourceMarkers = (): Marker[] => {
    const m = meta();
    if (!m || !imgEl) return [];
    const rect = imgEl.getBoundingClientRect();
    if (rect.width === 0) return [];
    const cellW = rect.width / m.grid_width;
    const cellH = rect.height / m.grid_height;
    const fs = featureSet();
    if (fs.length === 0) {
      const h = hover();
      if (!h) return [];
      return [{
        left: h.px * cellW + cellW / 2,
        top: h.py * cellH + cellH / 2,
        pinned: false,
        kind: "positive",
      }];
    }
    const af = activeFrame();
    return fs
      .filter((c) => c.fromFrame === af)
      .map((c) => ({
        left: c.fromPx * cellW + cellW / 2,
        top: c.fromPy * cellH + cellH / 2,
        pinned: true,
        kind: c.kind,
      }));
  };

  return (
    <>
      {/* Main view — same shape as the source-tab container so the parent
          viewport's flex centering applies directly. Kept mounted across
          tab switches (display:none when hidden) so the feature cache and
          any pinned patch survive. The DinoV3 tab is hidden by App.tsx
          when no meta is on disk, so we never render an empty state. */}
      <Show when={meta()}>
        <div
          ref={containerEl!}
          style={{
            position: "relative",
            display: props.visible ? "inline-block" : "none",
            "max-width": "100%",
            height: "100%",
          }}
        >
          <Show when={frameImgUrl()}>
            <img
              ref={imgEl!}
              src={frameImgUrl()!}
              draggable={false}
              onPointerMove={onPointerMove}
              onPointerLeave={onPointerLeave}
              onPointerDown={onPointerDown}
              onContextMenu={(e) => e.preventDefault()}
              style={{
                "max-width": "100%",
                "max-height": "100%",
                display: "block",
                cursor: "crosshair",
                "user-select": "none",
              }}
            />
            <canvas
              ref={heatmapCanvas!}
              style={{
                position: "absolute",
                inset: "0",
                width: "100%",
                height: "100%",
                "pointer-events": "none",
                "image-rendering": "auto",
                opacity: props.heatmapOpacity,
              }}
            />
            {sourceMarkers().map((mk) => (
              <div
                style={{
                  position: "absolute",
                  left: `${mk.left}px`,
                  top: `${mk.top}px`,
                  width: "10px",
                  height: "10px",
                  "border-radius": "50%",
                  transform: "translate(-50%, -50%)",
                  background: !mk.pinned
                    ? "rgba(255,255,255,0.8)"
                    : mk.kind === "negative" ? "#38bdf8" : "#e94560",
                  border: "2px solid #000",
                  "pointer-events": "none",
                  "box-shadow": "0 0 4px rgba(0,0,0,0.7)",
                }}
              />
            ))}
          </Show>
        </div>
      </Show>
    </>
  );
}
