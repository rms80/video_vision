import { createSignal, createEffect, createMemo, For, Show, onMount, onCleanup, Switch, Match } from "solid-js";
import { parseNpz } from "./npz";
import type { CamerasJson } from "./depthMesh";
import ThreeDepthViewer, { type BoxerResult } from "./ThreeDepthViewer";
import DinoV3Viewer, { type DinoV3Meta } from "./DinoV3Viewer";
import {
  SCENE_PLUGINS,
  SCENE_PLUGINS_BY_ID,
  DEFAULT_SCENE_PLUGIN_ID,
  getScenePluginOrDefault,
} from "./scenePlugins";
import {
  BOX_SOLVER_PLUGINS,
  BOX_SOLVER_PLUGINS_BY_ID,
  DEFAULT_BOX_SOLVER_ID,
} from "./boxSolverPlugins";

/** Show the active scene-analysis plugin name overlaid on the right viewport. */
const SHOW_SCENE_NAME_OVERLAY = true;

/** Gates the depth-tab click-drag color-rescale UI (the drawable line + its
 *  control panel). Disabled for now — the interaction was removed, but the
 *  underlying dynamic-rescale support (depthRange → colormap) is kept intact
 *  so it can be driven again later. Flip to re-enable the canvas wiring and
 *  the panel together. */
const DEPTH_RANGE_UI_ENABLED = false;

/** Compact human-readable duration: `0.4s`, `12s`, `1m 23s`, `1h 4m`. */
function formatElapsed(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "0s";
  if (seconds < 1) return `${seconds.toFixed(1)}s`;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) {
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}m ${s}s`;
  }
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

/** Compact depth-value readout: a few significant digits, scientific only for
 *  very small/large magnitudes. Returns "—" for missing/non-finite values. */
function formatDepth(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.01 || a >= 10000)) return v.toExponential(2);
  return v.toFixed(a < 1 ? 4 : a < 100 ? 2 : 1);
}

/** Viridis colormap (256 entries) — [r,g,b] each 0–255 */
const VIRIDIS: [number, number, number][] = [];
{
  // Generate from the standard viridis control points
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
    VIRIDIS.push([
      Math.round((ctrl[lo][0] * (1 - f) + ctrl[hi][0] * f) * 255),
      Math.round((ctrl[lo][1] * (1 - f) + ctrl[hi][1] * f) * 255),
      Math.round((ctrl[lo][2] * (1 - f) + ctrl[hi][2] * f) * 255),
    ]);
  }
}

export default function App() {
  const [videoSrc, setVideoSrc] = createSignal<string | null>(null);
  const [videoName, setVideoName] = createSignal<string | null>(null);
  const [videos, setVideos] = createSignal<string[]>([]);
  const [playing, setPlaying] = createSignal(false);
  const [currentTime, setCurrentTime] = createSignal(0);
  const [duration, setDuration] = createSignal(0);
  const [currentFrame, setCurrentFrame] = createSignal(0);
  const [totalFrames, setTotalFrames] = createSignal(0);
  const [videoSize, setVideoSize] = createSignal<{ w: number; h: number } | null>(null);
  const fps = () => 30;
  const [dragOver, setDragOver] = createSignal(false);
  const [status, setStatusRaw] = createSignal("Drop a video file to begin");
  // Accumulated, timestamped status log. Every setStatus() call appends
  // here (consecutive duplicates are skipped), and the "Log" button opens
  // a popup that shows the whole history with a copy-to-clipboard action.
  const [statusLog, setStatusLog] = createSignal<string[]>([]);
  const [statusLogOpen, setStatusLogOpen] = createSignal(false);
  const [statusLogCopied, setStatusLogCopied] = createSignal(false);
  const setStatus = (msg: string) => {
    setStatusRaw(msg);
    setStatusLog((prev) => {
      const ts = new Date().toLocaleTimeString();
      const line = `[${ts}] ${msg}`;
      // Skip exact-duplicate messages (ignoring timestamp) to keep
      // polling progress updates from spamming the log.
      if (prev.length > 0) {
        const last = prev[prev.length - 1];
        const lastMsg = last.replace(/^\[[^\]]*\]\s*/, "");
        if (lastMsg === msg) return prev;
      }
      return [...prev, line];
    });
  };
  const clearStatusLog = () => {
    setStatusLog([]);
  };
  const copyStatusLog = async () => {
    const text = statusLog().join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setStatusLogCopied(true);
      window.setTimeout(() => setStatusLogCopied(false), 1500);
    } catch {
      // Fallback for environments without clipboard permission.
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); } catch {}
      document.body.removeChild(ta);
      setStatusLogCopied(true);
      window.setTimeout(() => setStatusLogCopied(false), 1500);
    }
  };
  const [detectLabel, setDetectLabel] = createSignal("chair");
  const [settingSeed, setSettingSeed] = createSignal(false);
  const [seedPoint, setSeedPoint] = createSignal<{ x: number; y: number } | null>(null);
  const [detecting, setDetecting] = createSignal(false);
  const [analyses, setAnalyses] = createSignal<string[]>([]);
  const [currentAnalysis, setCurrentAnalysis] = createSignal<string | null>(null);
  const [tracking, setTracking] = createSignal(false);
  const [trackData, setTrackData] = createSignal<{
    imageWidth: number;
    imageHeight: number;
    frames: { frame: number; bbox: [number, number, number, number] | null }[];
  } | null>(null);
  const [sceneStatus, setSceneStatus] = createSignal<{
    artifacts: Record<string, boolean>;
    job: {
      pluginId?: string;
      stage: string;
      running: boolean;
      error: string | null;
      cancelled?: boolean;
      startedAt?: number;
      finishedAt?: number;
      progress?: string | null;
    } | null;
  } | null>(null);
  // Which plugin (if any) we have locally initiated a prepare for. The
  // backend is authoritative via /api/scene/status.job.running, but we
  // optimistically set this to flip UI state immediately on click.
  const [preparingPluginId, setPreparingPluginId] = createSignal<string | null>(null);
  const savedSceneSource = localStorage.getItem("segviewer:sceneSource");
  const [sceneSource, setSceneSource] = createSignal<string>(
    savedSceneSource && SCENE_PLUGINS_BY_ID[savedSceneSource] ? savedSceneSource : DEFAULT_SCENE_PLUGIN_ID,
  );
  // null = not yet fetched; once loaded, only plugins/solvers with id in
  // these sets are shown in their respective dropdowns. SAM2/SAM3 weights
  // gate the Detect/Track buttons.
  const [availablePluginIds, setAvailablePluginIds] = createSignal<Set<string> | null>(null);
  const [availableBoxSolverIds, setAvailableBoxSolverIds] = createSignal<Set<string> | null>(null);
  const [sam2Available, setSam2Available] = createSignal<boolean>(true);
  const [sam3Available, setSam3Available] = createSignal<boolean>(true);
  const [dinov3Available, setDinov3Available] = createSignal<boolean>(true);
  // dinov3 meta + job state from /api/dinov3/status. `meta` is non-null iff
  // a prior run wrote meta.json for the current video. `job` reflects the
  // most recent prepare invocation (running, errored, cancelled, etc.).
  const [dinov3Status, setDinov3Status] = createSignal<{
    meta: {
      subsample_every?: number;
      scaling?: number;
      grid_width?: number;
      grid_height?: number;
      input_width?: number;
      input_height?: number;
      model?: string;
    } | null;
    job: {
      running: boolean;
      error: string | null;
      cancelled?: boolean;
      startedAt?: number;
      finishedAt?: number;
      progress?: string | null;
      subsample?: number;
      scaling?: number;
    } | null;
  } | null>(null);
  // Per-video user input fields for the next dinov3 run. Persisted to
  // localStorage so picks survive reloads when there's no saved meta yet;
  // overridden by meta.json values whenever a video with stored results is
  // loaded.
  const [dinov3Subsample, setDinov3Subsample] = createSignal<string>(
    localStorage.getItem("segviewer:dinov3Subsample") ?? "2",
  );
  createEffect(() => localStorage.setItem("segviewer:dinov3Subsample", dinov3Subsample()));
  const DINOV3_SCALING_OPTIONS = ["1", "0.75", "0.5", "0.25"] as const;
  const [dinov3Scaling, setDinov3Scaling] = createSignal<string>(
    (() => {
      const stored = localStorage.getItem("segviewer:dinov3Scaling");
      return stored && (DINOV3_SCALING_OPTIONS as readonly string[]).includes(stored) ? stored : "0.5";
    })(),
  );
  createEffect(() => localStorage.setItem("segviewer:dinov3Scaling", dinov3Scaling()));
  const availablePlugins = () => {
    const set = availablePluginIds();
    return set ? SCENE_PLUGINS.filter((p) => set.has(p.id)) : SCENE_PLUGINS;
  };
  const availableBoxSolvers = () => {
    const set = availableBoxSolverIds();
    return set ? BOX_SOLVER_PLUGINS.filter((p) => set.has(p.id)) : BOX_SOLVER_PLUGINS;
  };
  const SAM_INSTALL_HINT = "Run `python setup/plugin_sam.py` from the project root to download the weights, then refresh.";
  const sam2DisabledReason = () => sam2Available() ? null
    : `SAM2 weights not installed (expected models/weights/sam2.1_l.pt). ${SAM_INSTALL_HINT}`;
  const sam3DisabledReason = () => sam3Available() ? null
    : `SAM3 weights not installed (expected models/weights/sam3.pt). ${SAM_INSTALL_HINT}`;
  let scenePollTimer: number | undefined;
  type ViewTab = "source" | "depth" | "3d" | "3d-scene" | "3d-object" | "dinov3";
  const storedTab = localStorage.getItem("segviewer:viewTab") as ViewTab | null;
  const [viewTab, setViewTab] = createSignal<ViewTab>(
    storedTab && ["source", "depth", "3d", "3d-scene", "3d-object", "dinov3"].includes(storedTab) ? storedTab : "source"
  );
  const [showSourceMask, setShowSourceMask] = createSignal(
    localStorage.getItem("segviewer:showSourceMask") !== "false"
  );
  const [showSourceBbox, setShowSourceBbox] = createSignal(
    localStorage.getItem("segviewer:showSourceBbox") !== "false"
  );
  createEffect(() => localStorage.setItem("segviewer:showSourceMask", showSourceMask() ? "true" : "false"));
  createEffect(() => localStorage.setItem("segviewer:showSourceBbox", showSourceBbox() ? "true" : "false"));
  const [cameras, setCameras] = createSignal<CamerasJson | null>(null);
  const [depthFrames, setDepthFrames] = createSignal<number[]>([]);
  const [depthStem, setDepthStem] = createSignal<string>("");
  const [depthCanvas, setDepthCanvas] = createSignal<HTMLCanvasElement | null>(null);
  const [depthLoading, setDepthLoading] = createSignal(false);
  // Depth color-map override set by click-dragging a line on the depth map:
  // the colormap is rescaled so depth at the two endpoints spans the full
  // viridis ramp. null = auto per-frame min/max. Cleared by the Reset button
  // and whenever the depth source changes (refreshDepthFrames).
  const [depthRange, setDepthRange] = createSignal<{ min: number; max: number } | null>(null);
  // The drawn line, in depth-map pixel coordinates, plus the depth sampled at
  // each endpoint (d1=start, d2=end). Kept for the on-canvas overlay and the
  // control-panel readout. null when no line is active.
  const [depthDrag, setDepthDrag] = createSignal<
    { x1: number; y1: number; x2: number; y2: number; d1: number | null; d2: number | null } | null
  >(null);
  const [settingFloor, setSettingFloor] = createSignal(false);
  const [floorPoints, setFloorPoints] = createSignal<{ x: number; y: number; frame: number }[]>([]);
  const [aligning, setAligning] = createSignal(false);
  // Active 3D-box solver: which method produced the result currently shown,
  // and which method will run when the user clicks "Compute Boxes".
  const savedBoxSolverId = localStorage.getItem("segviewer:boxSolver");
  const [boxSolverId, setBoxSolverId] = createSignal<string>(
    savedBoxSolverId && BOX_SOLVER_PLUGINS_BY_ID[savedBoxSolverId]
      ? savedBoxSolverId
      : DEFAULT_BOX_SOLVER_ID,
  );
  createEffect(() => localStorage.setItem("segviewer:boxSolver", boxSolverId()));

  const [boxResult, setBoxResult] = createSignal<BoxerResult | null>(null);
  const [boxRunning, setBoxRunning] = createSignal(false);
  // Per-solver option toggles, keyed by solver id then option key. Initialized
  // from each plugin's defaults; persisted to localStorage so re-runs keep
  // the user's last choice.
  const savedBoxSolverOptionsRaw = localStorage.getItem("segviewer:boxSolverOptions");
  const initialBoxSolverOptions: Record<string, Record<string, boolean>> = (() => {
    const base: Record<string, Record<string, boolean>> = {};
    for (const p of BOX_SOLVER_PLUGINS) {
      base[p.id] = {};
      for (const opt of p.options) base[p.id][opt.key] = opt.defaultValue;
    }
    if (savedBoxSolverOptionsRaw) {
      try {
        const saved = JSON.parse(savedBoxSolverOptionsRaw) as Record<string, Record<string, boolean>>;
        for (const p of BOX_SOLVER_PLUGINS) {
          for (const opt of p.options) {
            const v = saved?.[p.id]?.[opt.key];
            if (typeof v === "boolean") base[p.id][opt.key] = v;
          }
        }
      } catch {}
    }
    return base;
  })();
  const [boxSolverOptions, setBoxSolverOptions] =
    createSignal<Record<string, Record<string, boolean>>>(initialBoxSolverOptions);
  createEffect(() => localStorage.setItem("segviewer:boxSolverOptions", JSON.stringify(boxSolverOptions())));
  // Per-object world-space point cloud (depth points filtered by per-frame
  // tracking masks, fused across frames). Stored at
  // <analysis>/object_pointmap/<source>.npz; both `running` and `ready`
  // are scoped to (currentAnalysis, sceneSource).
  const [objectPointmapRunning, setObjectPointmapRunning] = createSignal(false);
  const [objectPointmapReady, setObjectPointmapReady] = createSignal(false);
  // Mask erosion radius (depth-map pixels) applied before unprojection — peels
  // off silhouette boundary pixels where interpolated depth produces fly-aways
  // behind the object. Persisted across sessions.
  const [objectErode, setObjectErode] = createSignal<string>(
    localStorage.getItem("segviewer:objectErode") ?? "2",
  );
  createEffect(() => localStorage.setItem("segviewer:objectErode", objectErode()));
  // Scene-pointmap fetch status (driven by ThreeDepthViewer.onScenePointmapStatus).
  // `progress` is null until the manifest reports total bytes; `pointCount`
  // is null until the first chunk lands; chunk counters are null until the
  // manifest itself has been fetched.
  const [scenePmLoading, setScenePmLoading] = createSignal(false);
  const [scenePmProgress, setScenePmProgress] = createSignal<number | null>(null);
  const [scenePmPoints, setScenePmPoints] = createSignal<number | null>(null);
  const [scenePmChunksLoaded, setScenePmChunksLoaded] = createSignal<number | null>(null);
  const [scenePmTotalChunks, setScenePmTotalChunks] = createSignal<number | null>(null);
  // Object-pointmap fetch status — same contract as scenePm*, scoped to the
  // 3D (Object) tab.
  const [objectPmLoading, setObjectPmLoading] = createSignal(false);
  const [objectPmProgress, setObjectPmProgress] = createSignal<number | null>(null);
  const [objectPmPoints, setObjectPmPoints] = createSignal<number | null>(null);
  const [objectPmChunksLoaded, setObjectPmChunksLoaded] = createSignal<number | null>(null);
  const [objectPmTotalChunks, setObjectPmTotalChunks] = createSignal<number | null>(null);
  const [pointmapView, setPointmapView] = createSignal(false);
  const [dataVersion, setDataVersion] = createSignal(0);
  const [showCameraPath, setShowCameraPath] = createSignal(true);
  const [gpuStatus, setGpuStatus] = createSignal<{ used: number; total: number; util: number | null } | null>(null);
  const storedMeshSub = Number(localStorage.getItem("segviewer:meshSubsample"));
  const [meshSubsample, setMeshSubsample] = createSignal<number>(
    [1, 2, 4].includes(storedMeshSub) ? storedMeshSub : 4,
  );
  createEffect(() => localStorage.setItem("segviewer:meshSubsample", String(meshSubsample())));
  // Bumped after a successful DinoV3 run so DinoV3Viewer drops its
  // in-memory feature cache and cache-busts its meta + .npz HTTP fetches.
  // Kept separate from `dataVersion` so a dinov3 re-run doesn't make
  // ThreeDepthViewer refetch its depth maps for no reason.
  const [dinov3DataVersion, setDinov3DataVersion] = createSignal<number>(0);
  // Bumped by the Reset button in the dinov3 toolbar to discard the
  // captured-feature set inside DinoV3Viewer.
  const [dinov3ResetVersion, setDinov3ResetVersion] = createSignal<number>(0);
  // DinoV3 heatmap opacity (0..1); slider lives in the dinov3 tab toolbar.
  // 0 = image only, 1 = heatmap only (image fully hidden).
  // NB: read the raw string first — a missing key returns null, and
  // Number(null) === 0 would pass the range check and clobber the 0.75
  // default (then the effect below persists that 0). Key is :v2 to discard
  // any 0 written by the earlier buggy version.
  const storedDinoOpacityRaw = localStorage.getItem("segviewer:dinov3Opacity:v2");
  const storedDinoOpacity = storedDinoOpacityRaw === null ? NaN : Number(storedDinoOpacityRaw);
  const [dinov3Opacity, setDinov3Opacity] = createSignal<number>(
    Number.isFinite(storedDinoOpacity) && storedDinoOpacity >= 0 && storedDinoOpacity <= 1
      ? storedDinoOpacity
      : 0.75,
  );
  createEffect(() => localStorage.setItem("segviewer:dinov3Opacity:v2", String(dinov3Opacity())));
  // DinoV3 visualization mode: "heatmap" (viridis on normalized cosine sim)
  // or "contour" (binary mask of patches above the threshold). Persisted.
  const storedDinoMode = localStorage.getItem("segviewer:dinov3Mode");
  const [dinov3Mode, setDinov3Mode] = createSignal<"heatmap" | "contour">(
    storedDinoMode === "contour" ? "contour" : "heatmap",
  );
  createEffect(() => localStorage.setItem("segviewer:dinov3Mode", dinov3Mode()));
  // Cosine-sim threshold for contour mode (0..1). Patches at or above this
  // are filled in the overlay color; below are transparent. (The viewer's
  // scoring also subtracts a max-negative term, but negatives are currently
  // unreachable from the UI, so the score stays in [0, 1] in practice.)
  // Same null-coercion guard as dinov3Opacity above: a missing key returns
  // null, and Number(null) === 0 would pass the range check and clobber the
  // default. Key is :v2 to discard any 0 the earlier buggy version persisted.
  const storedDinoThrRaw = localStorage.getItem("segviewer:dinov3Threshold:v2");
  const storedDinoThr = storedDinoThrRaw === null ? NaN : Number(storedDinoThrRaw);
  const [dinov3Threshold, setDinov3Threshold] = createSignal<number>(
    Number.isFinite(storedDinoThr) && storedDinoThr >= 0 && storedDinoThr <= 1
      ? storedDinoThr
      : 0.5,
  );
  createEffect(() => localStorage.setItem("segviewer:dinov3Threshold:v2", String(dinov3Threshold())));
  // Mirror of DinoV3Viewer's loaded meta, pushed up via onMeta. Drives the
  // dinov3-tab Patch Grid readout and snaps the timeline slider to the
  // subsampled frames that actually have features on disk.
  const [dinov3Meta, setDinov3Meta] = createSignal<DinoV3Meta | null>(null);
  let threeViewerActions: { snapCamera: () => void; fitAll: () => void } | null = null;
  const [savedWorldUp, setSavedWorldUp] = createSignal<{ x: number; y: number; frame: number }[]>([]);
  const [worldUpId, setWorldUpId] = createSignal<string>("");
  const isAligned = () => {
    const cam = cameras();
    const wuId = worldUpId();
    return !!(cam?.worldup_id && wuId && cam.worldup_id === wuId);
  };
  // Cache: frame number → { data, width, height }
  const depthCache = new Map<number, { data: Float32Array; width: number; height: number }>();
  const [detection, setDetection] = createSignal<{
    bbox: [number, number, number, number];
    maskDataUrl: string;
    imageWidth: number;
    imageHeight: number;
    confidence: number;
    label: string;
  } | null>(null);

  let videoEl!: HTMLVideoElement;
  let videoContainerEl!: HTMLDivElement;
  let animFrameId: number | null = null;

  // Fetch existing uploads on mount, then restore last session
  onMount(async () => {
    try {
      const r = await fetch("/api/availability");
      if (r.ok) {
        const data = await r.json();
        const sceneIds = new Set(
          Object.entries(data.scenePlugins ?? {})
            .filter(([, v]) => v)
            .map(([k]) => k),
        );
        setAvailablePluginIds(sceneIds);
        if (!sceneIds.has(sceneSource())) {
          const first = SCENE_PLUGINS.find((p) => sceneIds.has(p.id));
          if (first) setSceneSource(first.id);
        }
        const solverIds = new Set(
          Object.entries(data.boxSolvers ?? {})
            .filter(([, v]) => v)
            .map(([k]) => k),
        );
        setAvailableBoxSolverIds(solverIds);
        if (!solverIds.has(boxSolverId())) {
          const first = BOX_SOLVER_PLUGINS.find((p) => solverIds.has(p.id));
          if (first) setBoxSolverId(first.id);
        }
        setSam2Available(Boolean(data.sam2));
        setSam3Available(Boolean(data.sam3));
        setDinov3Available(Boolean(data.dinov3));
      }
    } catch {}
    await refreshVideoList();
    const savedVideo = localStorage.getItem("segviewer:video");
    const savedAnalysis = localStorage.getItem("segviewer:analysis");
    if (savedVideo && videos().includes(savedVideo)) {
      loadVideo(savedVideo);
      if (savedAnalysis) {
        // loadVideo clears the stored analysis — restore it and load
        await refreshAnalyses(savedVideo);
        if (analyses().includes(savedAnalysis)) {
          loadAnalysis(savedAnalysis);
        }
      }
    }
  });

  // Global keyboard handler: floor-click mode + arrow-key frame nav
  function handleGlobalKey(e: KeyboardEvent) {
    if (settingFloor()) {
      if (e.key === "Escape") {
        setSettingFloor(false);
        setFloorPoints([]);
        setStatus("World-up point selection cancelled");
      } else if (e.key === "Enter") {
        setSettingFloor(false);
        const pts = floorPoints();
        if (pts.length < 3) {
          setStatus(`Need at least 3 world-up points (have ${pts.length})`);
        } else {
          const v = videoName();
          if (v) saveWorldUp(v, pts);
          setStatus(`${pts.length} world-up points saved — click "Align Scene"`);
        }
      }
      return;
    }
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      e.preventDefault();
      navigateFrames(e.key === "ArrowRight" ? 1 : -1);
    }
  }
  onMount(() => { window.addEventListener("keydown", handleGlobalKey); });
  onCleanup(() => {
    window.removeEventListener("keydown", handleGlobalKey);
    if (animFrameId !== null) cancelAnimationFrame(animFrameId);
  });

  // Poll GPU memory every 2s (nvidia-smi via /api/gpu-status)
  let gpuTimer: number | undefined;
  onMount(() => {
    const tick = async () => {
      try {
        const res = await fetch("/api/gpu-status");
        if (res.ok) setGpuStatus(await res.json());
        else setGpuStatus(null);
      } catch { setGpuStatus(null); }
    };
    tick();
    gpuTimer = window.setInterval(tick, 2000);
  });
  onCleanup(() => { if (gpuTimer !== undefined) clearInterval(gpuTimer); });
  onCleanup(() => { if (dinov3PollTimer !== undefined) clearInterval(dinov3PollTimer); });

  // Persist UI state to localStorage
  createEffect(() => localStorage.setItem("segviewer:sceneSource", sceneSource()));
  createEffect(() => localStorage.setItem("segviewer:viewTab", viewTab()));

  // Per-plugin subsample input value, keyed by plugin id. Initialized from
  // each plugin's subsampleDefault (which mirrors the runner script's own
  // default), then overwritten by cameras.json `subsample_every` whenever
  // the active source's analysis is loaded.
  const [pluginSubsamples, setPluginSubsamples] = createSignal<Record<string, string>>(
    Object.fromEntries(
      SCENE_PLUGINS
        .filter((p) => p.subsampleDefault !== undefined)
        .map((p) => [p.id, String(p.subsampleDefault)]),
    ),
  );
  createEffect(() => {
    const id = sceneSource();
    const plugin = SCENE_PLUGINS_BY_ID[id];
    if (!plugin || plugin.subsampleDefault === undefined) return;
    const cam = cameras();
    const n = cam?.subsample_every;
    if (typeof n === "number" && n > 0) {
      setPluginSubsamples((prev) => ({ ...prev, [id]: String(n) }));
    }
  });

  // Per-plugin target-frame count, keyed by plugin id. Sent to the backend
  // as --num-frames; the runner picks exactly that many evenly-spaced
  // frames. After a run, sync the input from num_registered so switching
  // to an existing analysis shows what was computed; reset to the plugin's
  // targetFramesDefault when no analysis exists for the video.
  const [pluginTargetFrames, setPluginTargetFrames] = createSignal<Record<string, string>>(
    Object.fromEntries(
      SCENE_PLUGINS
        .filter((p) => p.targetFramesDefault !== undefined)
        .map((p) => [p.id, String(p.targetFramesDefault)]),
    ),
  );
  // Per-plugin output-resolution multiplier, keyed by plugin id. Initialized
  // from each plugin's upscaleDefault; overwritten by cameras.json `upscale`
  // when the active source's analysis is loaded. Stored as a stringified
  // number so the <select>'s value/option comparison stays exact.
  const [pluginUpscales, setPluginUpscales] = createSignal<Record<string, string>>(
    Object.fromEntries(
      SCENE_PLUGINS
        .filter((p) => p.upscaleDefault !== undefined)
        .map((p) => [p.id, String(p.upscaleDefault)]),
    ),
  );
  createEffect(() => {
    const id = sceneSource();
    const plugin = SCENE_PLUGINS_BY_ID[id];
    if (!plugin || plugin.upscaleDefault === undefined) return;
    const cam = cameras() as { upscale?: number } | null;
    const u = cam?.upscale;
    if (typeof u === "number" && u > 0) {
      setPluginUpscales((prev) => ({ ...prev, [id]: String(u) }));
    }
  });
  createEffect(() => {
    const id = sceneSource();
    const plugin = SCENE_PLUGINS_BY_ID[id];
    if (!plugin || plugin.targetFramesDefault === undefined) return;
    const cam = cameras();
    const next = (cam && typeof cam.num_registered === "number" && cam.num_registered > 0)
      ? String(cam.num_registered)
      : String(plugin.targetFramesDefault);
    setPluginTargetFrames((prev) => ({ ...prev, [id]: next }));
  });

  // For plugins with requiresCameraSource (e.g. InfiniDepth): the list of
  // ready upstream plugins fetched from /api/scene/camera-sources, plus the
  // user's per-plugin choice. Refreshed whenever the active video, the
  // active plugin, or the artifact map changes (so a new upstream finishing
  // adds itself to the list automatically).
  const [cameraSourceOptions, setCameraSourceOptions] = createSignal<
    { id: string; label: string }[]
  >([]);
  // The user's *explicit* dropdown pick per plugin. Only written from the
  // dropdown's onChange — nothing else touches it, so clicking Run can't
  // clobber it. Keyed by `${video}:${pluginId}` so a pick on video A doesn't
  // leak into video B.
  const [pluginCameraSources, setPluginCameraSources] = createSignal<Record<string, string>>({});
  // What's recorded in self's cameras.json (per video+plugin), as reported by
  // /api/scene/camera-sources. This is what was actually used by the last run.
  const [cameraSourceOnDisk, setCameraSourceOnDisk] = createSignal<Record<string, string | null>>({});
  createEffect(() => {
    const v = videoName();
    const id = sceneSource();
    const plugin = SCENE_PLUGINS_BY_ID[id];
    // Re-evaluate whenever the artifact map shifts (something upstream finishes).
    sceneStatus()?.artifacts;
    if (!v || !plugin?.requiresCameraSource) {
      setCameraSourceOptions([]);
      return;
    }
    fetch(`/api/scene/camera-sources?video=${encodeURIComponent(v)}&self=${id}`)
      .then((r) => r.json())
      .then((data) => {
        const opts: { id: string; label: string }[] = data?.sources ?? [];
        setCameraSourceOptions(opts);
        const cs: string | null = data?.currentSource ?? null;
        setCameraSourceOnDisk((prev) => ({ ...prev, [`${v}:${id}`]: cs }));
      })
      .catch(() => setCameraSourceOptions([]));
  });
  // Effective camera-source selection: prefer the user's explicit pick (if it
  // still exists), else what's recorded on disk (the last run's source), else
  // the first ready option. The dropdown displays this, and runScenePlugin
  // submits it. Clicking Run never writes to `pluginCameraSources`, so the
  // value can't change just by clicking Run.
  const effectiveCameraSource = createMemo(() => {
    const v = videoName();
    const id = sceneSource();
    if (!v) return "";
    const opts = cameraSourceOptions();
    const userPick = pluginCameraSources()[`${v}:${id}`];
    if (userPick && opts.some((o) => o.id === userPick)) return userPick;
    const onDisk = cameraSourceOnDisk()[`${v}:${id}`];
    if (onDisk && opts.some((o) => o.id === onDisk)) return onDisk;
    return opts[0]?.id ?? "";
  });
  // Fall back from "3D (Scene)" tab when switching to a plugin that doesn't support it
  createEffect(() => {
    if (viewTab() === "3d-scene" && !SCENE_PLUGINS_BY_ID[sceneSource()]?.features?.scenePointmap) {
      setViewTab("3d");
    }
  });
  // Fall back from "3D (Object)" tab when no object cloud is available for the
  // current (analysis, source) combo.
  createEffect(() => {
    if (viewTab() === "3d-object" && !objectPointmapReady()) setViewTab("3d");
  });
  // Fall back from the "DinoV3" tab when the active video has no features
  // on disk (e.g. just loaded a different video, or outputs were wiped).
  createEffect(() => {
    if (viewTab() === "dinov3" && dinov3Status()?.meta == null) setViewTab("source");
  });
  // Refetch object-cloud state when the active scene source or analysis
  // changes — each (analysis, source) pair has its own .npz file.
  createEffect(() => {
    const v = videoName();
    const a = currentAnalysis();
    const s = sceneSource();
    if (v && a) refreshObjectPointmap(v, a, s);
    else { setObjectPointmapReady(false); setObjectPointmapRunning(false); }
  });
  // Save frame position, but skip saving during initial load (before video is ready)
  let videoReady = false;
  createEffect(() => {
    const f = currentFrame();
    if (videoReady) localStorage.setItem("segviewer:frame", String(f));
  });

  // Track playback time
  function startTimeTracking() {
    function tick() {
      if (videoEl) {
        setCurrentTime(videoEl.currentTime);
        setCurrentFrame(Math.floor(videoEl.currentTime * fps()));
      }
      animFrameId = requestAnimationFrame(tick);
    }
    tick();
  }

  function stopTimeTracking() {
    if (animFrameId !== null) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
  }

  async function refreshVideoList() {
    const res = await fetch("/api/videos");
    const data = await res.json();
    setVideos(data.videos);
  }

  async function uploadVideo(file: File) {
    setStatus(`Uploading ${file.name}: re-encoding and extracting frames...`);
    const res = await fetch("/api/upload", {
      method: "POST",
      headers: {
        "x-filename": encodeURIComponent(file.name),
      },
      body: file,
    });
    const data = await res.json();
    if (data.ok) {
      if (data.skipped) {
        setStatus(`Already uploaded: ${data.filename}`);
      } else {
        setStatus(`Ready: ${data.filename}`);
      }
      await refreshVideoList();
      loadVideo(data.filename);
    } else {
      setStatus(`Upload failed: ${data.error}`);
    }
  }

  async function refreshAnalyses(video: string) {
    try {
      const res = await fetch(`/api/analyses?video=${encodeURIComponent(video)}`);
      const data = await res.json();
      setAnalyses(data.analyses ?? []);
    } catch { setAnalyses([]); }
  }

  async function loadAnalysis(name: string) {
    const video = videoName();
    if (!video) return;
    try {
      const res = await fetch(`/api/analysis-result?video=${encodeURIComponent(video)}&name=${encodeURIComponent(name)}`);
      const data = await res.json();
      if (data.error) { setStatus(`Load failed: ${data.error}`); return; }
      setDetection({
        bbox: data.bbox,
        maskDataUrl: `data:image/png;base64,${data.mask_png_base64}`,
        imageWidth: data.image_width,
        imageHeight: data.image_height,
        confidence: data.confidence,
        label: data.label,
      });
      if (data.seed_x != null && data.seed_y != null) {
        setSeedPoint({ x: data.seed_x, y: data.seed_y });
      }
      if (data.label) {
        setDetectLabel(data.label);
      }
      setCurrentAnalysis(name);
      localStorage.setItem("segviewer:analysis", name);
      setTrackData(null);
      setBoxResult(null);
      // Try to load an existing track result for this analysis
      try {
        const tr = await fetch(`/api/track-result?video=${encodeURIComponent(video)}&name=${encodeURIComponent(name)}`);
        if (tr.ok) {
          const td = await tr.json();
          setTrackData({ imageWidth: td.image_width, imageHeight: td.image_height, frames: td.frames });
        }
      } catch {}
      // Try to load existing 3D-box result for the currently selected solver.
      // Switching solver later refetches via the solver-change effect.
      await refreshBoxResult(video, name, boxSolverId());
      await refreshObjectPointmap(video, name, sceneSource());
      setStatus(`Loaded analysis: ${name}`);
    } catch (err: any) { setStatus(`Load error: ${err.message}`); }
  }

  async function trackThroughVideo() {
    const video = videoName();
    const analysis = currentAnalysis();
    if (!video || !analysis) {
      setStatus("Run or load a detection first before tracking");
      return;
    }
    setTracking(true);
    setBoxResult(null);
    setStatus(`Tracking through video with SAM2 (this can take a while)...`);
    try {
      const res = await fetch("/api/track", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video, analysis }),
      });
      const data = await res.json();
      if (data.cancelled) { setStatus("Tracking cancelled"); return; }
      if (data.error) { setStatus(`Tracking failed: ${data.error}`); return; }
      setTrackData({ imageWidth: data.image_width, imageHeight: data.image_height, frames: data.frames });
      setStatus(`Tracked ${data.frame_count} frames with ${data.model}`);
    } catch (err: any) {
      setStatus(`Tracking error: ${err.message}`);
    } finally {
      setTracking(false);
    }
  }

  async function cancelTrack() {
    const video = videoName();
    const analysis = currentAnalysis();
    if (!video || !analysis) return;
    // Flip the UI flag immediately — the server's DELETE waits for the
    // tracker to actually exit (taskkill /T then close), which can take
    // a beat on Windows when CUDA is mid-kernel.
    setTracking(false);
    setStatus("Cancelling tracking…");
    try {
      const r = await fetch(
        `/api/track?video=${encodeURIComponent(video)}&analysis=${encodeURIComponent(analysis)}`,
        { method: "DELETE" },
      );
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        setStatus(`Cancel failed: ${data.error ?? r.statusText}`);
      } else if (data.exited === false) {
        setStatus("Cancel sent but tracker did not exit cleanly — check log");
      } else {
        setStatus("Tracking cancelled");
      }
    } catch (e: any) {
      setStatus(`Cancel error: ${e.message}`);
    }
  }

  // Fetch /api/box-solver-result for (video, analysis, solverId), drop into
  // boxResult on success, clear on 404. Used both at analysis-load time and
  // when the user switches solver in the dropdown.
  async function refreshBoxResult(video: string, analysis: string, solverId: string) {
    try {
      const r = await fetch(
        `/api/box-solver-result?video=${encodeURIComponent(video)}` +
        `&name=${encodeURIComponent(analysis)}&solverId=${encodeURIComponent(solverId)}`,
      );
      if (r.ok) {
        setBoxResult(await r.json());
        return;
      }
    } catch {}
    setBoxResult(null);
  }

  // Poll/timeout handles for the in-flight solver run, captured so
  // cancelBoxSolver() can clear them when the user aborts. (number on
  // browser, NodeJS.Timeout in tests — just store as any.)
  let boxPollInterval: ReturnType<typeof setInterval> | null = null;
  let boxPollTimeout: ReturnType<typeof setTimeout> | null = null;
  // Identifies the currently-running solver run so the poll loop can
  // detect a cancel-and-restart and bail out cleanly.
  let boxRunToken = 0;

  function clearBoxPolling() {
    if (boxPollInterval !== null) { clearInterval(boxPollInterval); boxPollInterval = null; }
    if (boxPollTimeout !== null) { clearTimeout(boxPollTimeout); boxPollTimeout = null; }
  }

  async function cancelBoxSolver() {
    const video = videoName();
    const analysis = currentAnalysis();
    const solverId = boxSolverId();
    const solver = BOX_SOLVER_PLUGINS_BY_ID[solverId];
    if (!video || !analysis || !solver) return;
    boxRunToken++;          // invalidate the in-flight poll loop
    clearBoxPolling();
    setBoxRunning(false);
    setStatus(`Cancelling ${solver.label}…`);
    try {
      const r = await fetch(
        `/api/box-solver?video=${encodeURIComponent(video)}` +
        `&analysis=${encodeURIComponent(analysis)}&solverId=${encodeURIComponent(solverId)}`,
        { method: "DELETE" },
      );
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        setStatus(`${solver.label} cancel failed: ${data.error ?? r.statusText}`);
      } else if (data.exited === false) {
        setStatus(`${solver.label} cancel sent but process did not exit — check log`);
      } else {
        setStatus(`${solver.label} cancelled`);
      }
    } catch (e: any) {
      setStatus(`${solver.label} cancel error: ${e.message}`);
    }
  }

  async function runBoxSolver() {
    const video = videoName();
    const analysis = currentAnalysis();
    const solverId = boxSolverId();
    const solver = BOX_SOLVER_PLUGINS_BY_ID[solverId];
    if (!video || !analysis || !trackData() || !solver) {
      setStatus("Run tracking first before computing 3D boxes");
      return;
    }
    if (solver.requiresDepth && depthFrames().length === 0) {
      setStatus(`${solver.label} needs depth maps — run a scene plugin first`);
      return;
    }
    const myToken = ++boxRunToken;
    clearBoxPolling();
    setBoxRunning(true);
    setBoxResult(null);
    setStatus(`Running ${solver.label} 3D bounding box lifting...`);
    try {
      const det = detection();
      const res = await fetch("/api/box-solver", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video, analysis, solverId,
          label: det?.label ?? "object",
          source: sceneSource(),
          options: boxSolverOptions()[solverId] ?? {},
        }),
      });
      const data = await res.json();
      if (data.error) {
        setStatus(`${solver.label} failed: ${data.error}`);
        setBoxRunning(false);
        return;
      }
      setStatus(`${solver.label} running — waiting for result...`);
      boxPollInterval = setInterval(async () => {
        // Bail out if the user switched solvers, cancelled, or kicked off
        // a new run; the owning run's token won't match.
        if (boxRunToken !== myToken || boxSolverId() !== solverId) {
          clearBoxPolling();
          return;
        }
        try {
          const r = await fetch(
            `/api/box-solver-result?video=${encodeURIComponent(video)}` +
            `&name=${encodeURIComponent(analysis)}&solverId=${encodeURIComponent(solverId)}`,
          );
          if (r.ok && boxRunToken === myToken) {
            clearBoxPolling();
            const result = await r.json();
            setBoxResult(result);
            setBoxRunning(false);
            setStatus(`${solver.label}: ${result.num_frames_with_boxes} frames with 3D boxes`);
          }
        } catch {}
      }, 2000);
      boxPollTimeout = setTimeout(() => {
        if (boxRunToken !== myToken) return;
        clearBoxPolling();
        if (boxRunning()) {
          setBoxRunning(false);
          setStatus(`${solver.label} timed out`);
        }
      }, 600000);
    } catch (err: any) {
      setStatus(`${solver.label} error: ${err.message}`);
      setBoxRunning(false);
    }
  }

  function objectPointmapUrl(video: string, analysis: string, source: string): string {
    // Returns the chunk-manifest URL. The viewer fetches this first, then
    // streams each <source>_NNN.npz chunk listed inside it.
    const stem = video.replace(/\.[^.]+$/, "");
    return `/analysis/${encodeURIComponent(stem)}/${encodeURIComponent(analysis)}/object_pointmap/${encodeURIComponent(source)}_chunks.json`;
  }

  async function refreshObjectPointmap(video: string, analysis: string, source: string) {
    try {
      const r = await fetch(
        `/api/object-pointmap-status?video=${encodeURIComponent(video)}` +
        `&analysis=${encodeURIComponent(analysis)}&source=${encodeURIComponent(source)}`,
      );
      if (!r.ok) { setObjectPointmapReady(false); setObjectPointmapRunning(false); return; }
      const s = await r.json();
      setObjectPointmapReady(!!s.ready);
      setObjectPointmapRunning(!!s.job?.running);
    } catch {
      setObjectPointmapReady(false);
      setObjectPointmapRunning(false);
    }
  }

  async function runObjectPointmap() {
    const video = videoName();
    const analysis = currentAnalysis();
    const source = sceneSource();
    if (!video || !analysis || !trackData()) {
      setStatus("Run tracking first before building the object point cloud");
      return;
    }
    if (depthFrames().length === 0) {
      setStatus("No depth maps available — run a scene plugin first");
      return;
    }
    setObjectPointmapRunning(true);
    setObjectPointmapReady(false);
    setStatus("Building object point cloud (per-frame depth ∩ mask, fused)...");
    try {
      const erodeN = Math.max(0, Math.round(Number(objectErode())) || 0);
      const res = await fetch("/api/object-pointmap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video, analysis, source, options: { erode: erodeN } }),
      });
      const data = await res.json();
      if (data.error) {
        setStatus(`Object cloud failed: ${data.error}`);
        setObjectPointmapRunning(false);
        return;
      }
      const poll = setInterval(async () => {
        // Bail out if the user changed source mid-run; the new source's
        // refresh will pick up its own state.
        if (sceneSource() !== source || currentAnalysis() !== analysis) {
          clearInterval(poll);
          return;
        }
        try {
          const r = await fetch(
            `/api/object-pointmap-status?video=${encodeURIComponent(video)}` +
            `&analysis=${encodeURIComponent(analysis)}&source=${encodeURIComponent(source)}`,
          );
          if (!r.ok) return;
          const s = await r.json();
          if (s.job && !s.job.running) {
            clearInterval(poll);
            setObjectPointmapRunning(false);
            if (s.job.error) {
              setStatus(`Object cloud failed: ${s.job.error}`);
              setObjectPointmapReady(false);
            } else {
              setObjectPointmapReady(!!s.ready);
              const elapsed = formatElapsed((s.job.finishedAt - s.job.startedAt) / 1000);
              setStatus(`Object cloud built in ${elapsed}`);
              setDataVersion((v) => v + 1);  // force viewer to refetch
            }
          } else if (s.job?.running && s.job?.progress) {
            setStatus(`Object cloud: ${s.job.progress}`);
          }
        } catch {}
      }, 2000);
      setTimeout(() => {
        clearInterval(poll);
        if (objectPointmapRunning()) {
          setObjectPointmapRunning(false);
          setStatus("Object cloud timed out");
        }
      }, 600000);
    } catch (err: any) {
      setStatus(`Object cloud error: ${err.message}`);
      setObjectPointmapRunning(false);
    }
  }

  async function deleteVideo(filename: string) {
    if (!filename) return;
    if (!window.confirm(`Delete "${filename}" and all its analyses? This cannot be undone.`)) return;
    try {
      const res = await fetch(`/api/videos?name=${encodeURIComponent(filename)}`, { method: "DELETE" });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        setStatus(`Delete failed: ${data.error ?? res.statusText}`);
        return;
      }
      // If the deleted video was loaded, clear video-related state.
      if (videoName() === filename) {
        videoReady = false;
        setVideoSrc(null);
        setVideoName(null);
        setPlaying(false);
        setCurrentTime(0);
        setCurrentFrame(0);
        setSeedPoint(null);
        setDetection(null);
        setCurrentAnalysis(null);
        setTrackData(null);
        setBoxResult(null);
        setObjectPointmapRunning(false);
        setObjectPointmapReady(false);
        setFloorPoints([]);
        setSettingFloor(false);
        setAnalyses([]);
        localStorage.removeItem("segviewer:video");
        localStorage.removeItem("segviewer:analysis");
      }
      await refreshVideoList();
      setStatus(`Deleted ${filename}`);
    } catch (err: any) {
      setStatus(`Delete error: ${err.message ?? err}`);
    }
  }

  async function deleteAnalysis(name: string) {
    const video = videoName();
    if (!video || !name) return;
    if (!window.confirm(`Delete analysis "${name}"? This cannot be undone.`)) return;
    try {
      const res = await fetch(
        `/api/analyses?video=${encodeURIComponent(video)}&name=${encodeURIComponent(name)}`,
        { method: "DELETE" },
      );
      const data = await res.json();
      if (!res.ok || !data.ok) {
        setStatus(`Delete failed: ${data.error ?? res.statusText}`);
        return;
      }
      if (currentAnalysis() === name) {
        setCurrentAnalysis(null);
        setDetection(null);
        setSeedPoint(null);
        setTrackData(null);
        setBoxResult(null);
        setObjectPointmapRunning(false);
        setObjectPointmapReady(false);
        localStorage.removeItem("segviewer:analysis");
      }
      await refreshAnalyses(video);
      setStatus(`Deleted analysis ${name}`);
    } catch (err: any) {
      setStatus(`Delete error: ${err.message ?? err}`);
    }
  }

  function loadVideo(filename: string) {
    videoReady = false;
    setVideoSrc(`/uploads/${filename}`);
    setVideoName(filename);
    setPlaying(false);
    setCurrentTime(0);
    setCurrentFrame(0);
    setSeedPoint(null);
    setDetection(null);
    setCurrentAnalysis(null);
    setTrackData(null);
    setBoxResult(null);
    setObjectPointmapRunning(false);
    setObjectPointmapReady(false);
    setFloorPoints([]);
    setSettingFloor(false);
    setStatus(`Loaded: ${filename}`);
    localStorage.setItem("segviewer:video", filename);
    localStorage.removeItem("segviewer:analysis");
    refreshAnalyses(filename);
    refreshSceneStatus(filename);
    refreshDepthFrames(filename);
    loadWorldUp(filename);
  }

  async function loadWorldUp(video: string) {
    try {
      const res = await fetch(`/api/scene/worldup?video=${encodeURIComponent(video)}`);
      if (res.ok) {
        const data = await res.json();
        setSavedWorldUp(data.points ?? []);
        setWorldUpId(data.id ?? "");
      }
    } catch { setSavedWorldUp([]); setWorldUpId(""); }
  }

  async function saveWorldUp(video: string, points: { x: number; y: number; frame: number }[]) {
    try {
      await fetch("/api/scene/worldup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video, points }),
      });
      setSavedWorldUp(points);
      // Reload to get the new ID
      await loadWorldUp(video);
    } catch {}
  }

  async function refreshDepthFrames(video: string) {
    depthCache.clear();
    setCameras(null);
    resetDepthRange();
    const source = sceneSource();
    try {
      const res = await fetch(`/api/depth-frames?video=${encodeURIComponent(video)}&source=${source}`);
      const data = await res.json();
      setDepthFrames(data.frames ?? []);
      setDepthStem(data.stem ?? "");
      // Also fetch cameras.json for the 3D view
      if (data.stem) {
        const camDir = getScenePluginOrDefault(source).camerasDir;
        try {
          const camRes = await fetch(`/analysis/${data.stem}/_scene/${camDir}/cameras.json`);
          if (camRes.ok) setCameras(await camRes.json());
        } catch {}
      }
    } catch { setDepthFrames([]); }
  }

  /** Find the nearest depth frame index for the current video frame */
  function nearestDepthFrame(): number | null {
    const frames = depthFrames();
    if (!frames.length) return null;
    const cur = currentFrame();
    let best = frames[0];
    let bestDist = Math.abs(cur - best);
    for (const f of frames) {
      const d = Math.abs(cur - f);
      if (d < bestDist) { best = f; bestDist = d; }
    }
    return best;
  }

  /** Fetch and render depth for the current frame onto the depth canvas */
  async function renderDepthFrame() {
    const canvas = depthCanvas();
    const frameIdx = nearestDepthFrame();
    if (!canvas || frameIdx == null) return;
    const stem = depthStem();
    if (!stem) return;

    let cached = depthCache.get(frameIdx);
    if (!cached) {
      setDepthLoading(true);
      try {
        const padded = String(frameIdx).padStart(6, "0");
        const depthDir = getScenePluginOrDefault(sceneSource()).depthDir;
        const url = `/analysis/${stem}/_scene/${depthDir}/${padded}.npz`;
        const resp = await fetch(url);
        if (!resp.ok) { setDepthLoading(false); return; }
        const buf = await resp.arrayBuffer();
        const arrays = await parseNpz(buf);
        const depthArr = arrays["depth"];
        if (!depthArr) { setDepthLoading(false); return; }
        const [h, w] = depthArr.shape;
        cached = { data: new Float32Array(depthArr.data), width: w, height: h };
        depthCache.set(frameIdx, cached);
      } catch { setDepthLoading(false); return; }
      setDepthLoading(false);
    }

    const { data, width, height } = cached;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d")!;
    const imgData = ctx.createImageData(width, height);
    // Color range: an explicit override from a dragged line (depthRange),
    // otherwise auto-normalize to this frame's finite min/max.
    let min: number, max: number;
    const override = depthRange();
    if (override) {
      min = override.min;
      max = override.max;
    } else {
      min = Infinity;
      max = -Infinity;
      for (let i = 0; i < data.length; i++) {
        const v = data[i];
        if (Number.isFinite(v)) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
      if (!Number.isFinite(min)) { min = 0; max = 1; }
    }
    const range = max - min || 1;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      const t = Number.isFinite(v) ? Math.max(0, Math.min(255, Math.round(((v - min) / range) * 255))) : 0;
      const [r, g, b] = VIRIDIS[t];
      imgData.data[i * 4] = r;
      imgData.data[i * 4 + 1] = g;
      imgData.data[i * 4 + 2] = b;
      imgData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);

    // Overlay the rescale line + endpoint markers, if one is active. Sized in
    // canvas pixels but scaled by the on-screen display factor so the line
    // and dots stay a constant apparent thickness regardless of zoom.
    const drag = depthDrag();
    if (drag) {
      const rect = canvas.getBoundingClientRect();
      const scale = rect.width > 0 ? canvas.width / rect.width : 1;
      ctx.lineWidth = 1.5 * scale;
      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.shadowColor = "rgba(0,0,0,0.85)";
      ctx.shadowBlur = 2 * scale;
      ctx.beginPath();
      ctx.moveTo(drag.x1 + 0.5, drag.y1 + 0.5);
      ctx.lineTo(drag.x2 + 0.5, drag.y2 + 0.5);
      ctx.stroke();
      ctx.shadowBlur = 0;
      const rad = 3.5 * scale;
      // start = green, end = red, matching the control-panel readout.
      for (const [mx, my, fill] of [
        [drag.x1, drag.y1, "#3ad29f"],
        [drag.x2, drag.y2, "#e94560"],
      ] as const) {
        ctx.beginPath();
        ctx.arc(mx + 0.5, my + 0.5, rad, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.lineWidth = scale;
        ctx.strokeStyle = "rgba(0,0,0,0.85)";
        ctx.stroke();
      }
    }
  }

  // ── Depth color-range drag interaction ──────────────────────────────────
  // Pixel the most recent pointerdown landed on (depth-map coordinates).
  let depthDragStart: { px: number; py: number } | null = null;
  let depthDragging = false;

  /** Map a pointer event to depth-map pixel coordinates. The canvas is shown
   *  scaled to fit, so divide by the displayed rect rather than intrinsic size. */
  function depthEventToPixel(e: PointerEvent): { px: number; py: number } | null {
    const canvas = depthCanvas();
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    const px = Math.round(((e.clientX - rect.left) / rect.width) * canvas.width);
    const py = Math.round(((e.clientY - rect.top) / rect.height) * canvas.height);
    return {
      px: Math.max(0, Math.min(canvas.width - 1, px)),
      py: Math.max(0, Math.min(canvas.height - 1, py)),
    };
  }

  /** Sample the current frame's depth at a pixel, falling back to the nearest
   *  finite neighbor (depth maps carry NaN holes at sky / invalid regions). */
  function sampleDepthAt(px: number, py: number): number | null {
    const frameIdx = nearestDepthFrame();
    if (frameIdx == null) return null;
    const cached = depthCache.get(frameIdx);
    if (!cached) return null;
    const { data, width, height } = cached;
    const cx = Math.max(0, Math.min(width - 1, px));
    const cy = Math.max(0, Math.min(height - 1, py));
    const v = data[cy * width + cx];
    if (Number.isFinite(v)) return v;
    for (let r = 1; r <= 5; r++) {
      for (let yy = cy - r; yy <= cy + r; yy++) {
        for (let xx = cx - r; xx <= cx + r; xx++) {
          if (xx < 0 || yy < 0 || xx >= width || yy >= height) continue;
          const vv = data[yy * width + xx];
          if (Number.isFinite(vv)) return vv;
        }
      }
    }
    return null;
  }

  /** Recompute the drawn line + color range from the stored start pixel to a
   *  new end pixel. Called live on every drag move and on release. */
  function updateDepthLine(end: { px: number; py: number }) {
    const start = depthDragStart;
    if (!start) return;
    const d1 = sampleDepthAt(start.px, start.py);
    const d2 = sampleDepthAt(end.px, end.py);
    setDepthDrag({ x1: start.px, y1: start.py, x2: end.px, y2: end.py, d1, d2 });
    if (d1 == null || d2 == null) return;
    const lo = Math.min(d1, d2);
    const hi = Math.max(d1, d2);
    if (hi - lo > 1e-9) setDepthRange({ min: lo, max: hi });
  }

  function onDepthPointerDown(e: PointerEvent) {
    const p = depthEventToPixel(e);
    if (!p) return;
    e.preventDefault();
    depthDragging = true;
    depthDragStart = p;
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    const d = sampleDepthAt(p.px, p.py);
    setDepthDrag({ x1: p.px, y1: p.py, x2: p.px, y2: p.py, d1: d, d2: d });
  }
  function onDepthPointerMove(e: PointerEvent) {
    if (!depthDragging) return;
    const p = depthEventToPixel(e);
    if (p) updateDepthLine(p);
  }
  function onDepthPointerUp(e: PointerEvent) {
    if (!depthDragging) return;
    depthDragging = false;
    const p = depthEventToPixel(e);
    if (p) updateDepthLine(p);
    (e.currentTarget as Element).releasePointerCapture?.(e.pointerId);
  }
  /** Discard the line + range override, restoring auto per-frame scaling. */
  function resetDepthRange() {
    depthDragStart = null;
    depthDragging = false;
    setDepthDrag(null);
    setDepthRange(null);
  }

  // Re-render depth when frame changes, tab switches, canvas mounts, or the
  // color range / drag line changes (the latter drives the live overlay).
  createEffect(() => {
    const _frame = currentFrame();
    const _tab = viewTab();
    const _canvas = depthCanvas();
    const _range = depthRange();
    const _drag = depthDrag();
    if (_tab === "depth" && _canvas) renderDepthFrame();
  });

  // Poll handle for dinov3 status updates while a job is running.
  let dinov3PollTimer: number | undefined;

  async function refreshDinov3Status(video: string) {
    try {
      const res = await fetch(`/api/dinov3/status?video=${encodeURIComponent(video)}`);
      if (!res.ok) { setDinov3Status(null); return; }
      const data = await res.json();
      setDinov3Status(data);
      // Sync the input fields from meta.json so the dropdowns reflect what
      // was actually computed. Only writes when the meta value is valid;
      // doesn't clobber a user-edited input that has since diverged.
      const meta = data?.meta;
      if (meta) {
        if (typeof meta.subsample_every === "number" && meta.subsample_every > 0) {
          setDinov3Subsample(String(meta.subsample_every));
        }
        if (typeof meta.scaling === "number") {
          // Snap to the nearest preset so the dropdown shows the correct
          // option even if the value was hand-edited slightly off.
          const closest = (DINOV3_SCALING_OPTIONS as readonly string[]).reduce((best, o) =>
            Math.abs(Number(o) - meta.scaling!) < Math.abs(Number(best) - meta.scaling!) ? o : best,
          );
          setDinov3Scaling(closest);
        }
      }
      // Poll while running; clear once the backend reports done.
      const running = Boolean(data?.job?.running);
      if (running) {
        if (dinov3PollTimer === undefined) {
          dinov3PollTimer = window.setInterval(() => {
            const v = videoName();
            if (v) refreshDinov3Status(v);
          }, 2000);
        }
        const detail = data?.job?.progress?.trim() || "Detecting features";
        setStatus(`DinoV3: ${detail}`);
      } else if (dinov3PollTimer !== undefined) {
        clearInterval(dinov3PollTimer);
        dinov3PollTimer = undefined;
        if (data?.job) {
          if (data.job.error) {
            setStatus(`DinoV3 failed: ${data.job.error}`);
          } else if (data.job.cancelled) {
            // The cancel handler already set a status.
          } else {
            const elapsed = data.job.startedAt && data.job.finishedAt
              ? formatElapsed((data.job.finishedAt - data.job.startedAt) / 1000)
              : null;
            setStatus(elapsed ? `DinoV3 features computed in ${elapsed}` : "DinoV3 features computed");
            // Force the viewer to drop its cached features + re-fetch
            // meta.json so a new grid shape / scaling takes effect without
            // a page reload.
            setDinov3DataVersion((v) => v + 1);
          }
        }
      }
    } catch {
      setDinov3Status(null);
    }
  }

  // Refetch dinov3 status whenever the active video changes.
  createEffect(() => {
    const v = videoName();
    if (v) refreshDinov3Status(v);
    else setDinov3Status(null);
  });

  /** True iff the meta.json on disk was computed with the inputs the user
   *  currently has selected. Drives the "green ready" button state. */
  const dinov3ReadyMatches = () => {
    const meta = dinov3Status()?.meta;
    if (!meta) return false;
    const wantSub = Math.max(1, Math.round(Number(dinov3Subsample())));
    const wantDs = Number(dinov3Scaling());
    if (meta.subsample_every !== wantSub) return false;
    if (typeof meta.scaling !== "number" || Math.abs(meta.scaling - wantDs) > 1e-6) return false;
    return true;
  };

  async function runDinov3() {
    const v = videoName();
    if (!v) return;
    const subN = Math.max(1, Math.round(Number(dinov3Subsample())));
    const ds = Number(dinov3Scaling());
    if (!Number.isFinite(ds) || ds <= 0) {
      setStatus("DinoV3: pick a valid scaling");
      return;
    }
    // Optimistically reflect "running" so the button flips without waiting
    // for the next poll cycle.
    setDinov3Status({
      meta: dinov3Status()?.meta ?? null,
      job: { running: true, error: null, startedAt: Date.now(), progress: null, subsample: subN, scaling: ds },
    });
    setStatus("Starting DinoV3 features...");
    try {
      const res = await fetch("/api/dinov3/prepare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video: v, subsample: subN, scaling: ds }),
      });
      const data = await res.json();
      if (data.error) {
        setStatus(`DinoV3 failed: ${data.error}`);
        await refreshDinov3Status(v);
        return;
      }
      refreshDinov3Status(v);
    } catch (err: any) {
      setStatus(`DinoV3 error: ${err.message ?? err}`);
      await refreshDinov3Status(v);
    }
  }

  async function cancelDinov3() {
    const v = videoName();
    if (!v) return;
    setStatus("Cancelling DinoV3…");
    // Optimistically flip — DELETE waits for the child process to exit.
    const cur = dinov3Status();
    if (cur?.job) {
      setDinov3Status({ ...cur, job: { ...cur.job, running: false, cancelled: true } });
    }
    try {
      const r = await fetch(`/api/dinov3/prepare?video=${encodeURIComponent(v)}`, { method: "DELETE" });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        setStatus(`Cancel failed: ${data.error ?? r.statusText}`);
      } else if (data.exited === false) {
        setStatus("Cancel sent but DinoV3 did not exit — check log");
      } else {
        setStatus("DinoV3 cancelled");
      }
    } catch (e: any) {
      setStatus(`Cancel error: ${e.message}`);
    } finally {
      await refreshDinov3Status(v);
    }
  }

  async function refreshSceneStatus(video: string) {
    try {
      const res = await fetch(`/api/scene/status?video=${encodeURIComponent(video)}`);
      const data = await res.json();
      setSceneStatus(data);
      const wasRunning = preparingPluginId() !== null;
      const isRunning = Boolean(data.job?.running);
      setPreparingPluginId(isRunning ? (data.job?.pluginId ?? null) : null);
      if (isRunning) {
        if (data.job?.stage) {
          const label = data.job?.pluginId
            ? (SCENE_PLUGINS_BY_ID[data.job.pluginId]?.label ?? data.job.pluginId)
            : "Scene";
          // Show the live progress line from the script when we have one,
          // otherwise fall back to the pipeline stage name.
          const detail = data.job?.progress?.trim() || data.job.stage;
          setStatus(`${label}: ${detail}`);
        }
        if (scenePollTimer === undefined) {
          scenePollTimer = window.setInterval(() => {
            const v = videoName();
            if (v) refreshSceneStatus(v);
          }, 2000);
        }
      } else if (!isRunning && scenePollTimer !== undefined) {
        clearInterval(scenePollTimer);
        scenePollTimer = undefined;
        if (wasRunning) {
          if (data.job?.error) {
            setStatus(`Scene prep failed: ${data.job.error}`);
          } else if (data.job?.cancelled) {
            // cancelScene() already set a status; don't overwrite with "complete".
          } else {
            const label = data.job?.pluginId
              ? (SCENE_PLUGINS_BY_ID[data.job.pluginId]?.label ?? data.job.pluginId)
              : "Scene";
            const elapsed = data.job?.startedAt && data.job?.finishedAt
              ? formatElapsed((data.job.finishedAt - data.job.startedAt) / 1000)
              : null;
            setStatus(elapsed
              ? `${label} complete in ${elapsed}`
              : `${label} complete.`);
            const v = videoName();
            if (v) await refreshDepthFrames(v);
            setDataVersion((v) => v + 1);
          }
        }
      }
    } catch { setSceneStatus(null); }
  }

  async function alignScene() {
    const video = videoName();
    const pts = floorPoints().length >= 3 ? floorPoints() : savedWorldUp();
    if (!video || pts.length < 3) {
      setStatus("Need at least 3 world-up points");
      return;
    }
    setAligning(true);
    setSettingFloor(false);
    setBoxResult(null);
    setStatus("Aligning scene to floor plane...");
    try {
      const res = await fetch("/api/scene/align", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video, points: pts, source: sceneSource() }),
      });
      const data = await res.json();
      if (data.error) {
        setStatus(`Align failed: ${data.error}`);
      } else {
        setStatus("Scene aligned — reloading cameras");
        await refreshDepthFrames(video);
        setDataVersion((v) => v + 1);
        setFloorPoints([]);
      }
    } catch (err: any) {
      setStatus(`Align error: ${err.message}`);
    } finally {
      setAligning(false);
    }
  }

  async function runScenePlugin(pluginId: string) {
    const v = videoName();
    if (!v) return;
    const plugin = SCENE_PLUGINS_BY_ID[pluginId];
    if (!plugin) return;
    setPreparingPluginId(pluginId);
    setBoxResult(null);
    setStatus(`Starting ${plugin.label}...`);
    try {
      const options: Record<string, unknown> = {};
      if (plugin.subsampleDefault !== undefined) {
        const n = Math.max(1, Math.round(Number(pluginSubsamples()[pluginId])));
        if (Number.isFinite(n)) options.subsample = n;
      }
      if (plugin.targetFramesDefault !== undefined) {
        const n = Math.max(1, Math.round(Number(pluginTargetFrames()[pluginId])));
        if (Number.isFinite(n)) options.numFrames = n;
      }
      if (plugin.upscaleDefault !== undefined) {
        const u = Number(pluginUpscales()[pluginId]);
        if (Number.isFinite(u) && u > 0) options.upscale = u;
      }
      if (plugin.requiresCameraSource) {
        const src = effectiveCameraSource();
        if (!src) {
          setStatus(`${plugin.label}: pick a camera source first`);
          setPreparingPluginId(null);
          return;
        }
        options.cameraSource = src;
      }
      const res = await fetch("/api/scene/prepare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video: v, pluginId, options }),
      });
      const data = await res.json();
      if (data.error) {
        setStatus(`${plugin.label} failed: ${data.error}`);
        setPreparingPluginId(null);
        return;
      }
      // refreshSceneStatus starts a 2s poll that clears preparingPluginId
      // when the backend reports job.running === false, then fetches depth
      // frames + cameras for the newly produced artifacts.
      const prevSource = sceneSource();
      refreshSceneStatus(v);
      // Auto-switch the viewer to this plugin once its artifacts appear.
      const watch = setInterval(async () => {
        await refreshSceneStatus(v);
        const ready = sceneStatus()?.artifacts?.[pluginId];
        if (ready) {
          clearInterval(watch);
          if (prevSource !== pluginId) setSceneSource(pluginId);
          await refreshDepthFrames(v);
          setDataVersion((x) => x + 1);
        }
        if (preparingPluginId() === null) clearInterval(watch);
      }, 3000);
      setTimeout(() => clearInterval(watch), 600000);
    } catch (err: any) {
      setStatus(`${plugin.label} error: ${err.message}`);
      setPreparingPluginId(null);
    }
  }

  async function cancelScene() {
    const v = videoName();
    if (!v) return;
    // Flip the UI flag immediately — the server's DELETE waits for the
    // active python step to actually exit, which can take a beat on
    // Windows when CUDA is mid-kernel.
    setPreparingPluginId(null);
    setStatus("Cancelling scene prep…");
    try {
      const r = await fetch(`/api/scene/prepare?video=${encodeURIComponent(v)}`, { method: "DELETE" });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        setStatus(`Cancel failed: ${data.error ?? r.statusText}`);
      } else if (data.exited === false) {
        setStatus("Cancel sent but process did not exit — check log");
      } else {
        setStatus("Scene prep cancelled");
      }
    } catch (e: any) {
      setStatus(`Cancel error: ${e.message}`);
    }
  }

  async function detectObject() {
    const seed = seedPoint();
    const name = videoName();
    if (!seed || !name) return;
    const label = detectLabel().trim() || "object";
    setDetecting(true);
    setStatus(`Detecting "${label}" at (${seed.x}, ${seed.y})...`);
    try {
      const res = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video: name, x: seed.x, y: seed.y, label }),
      });
      const data = await res.json();
      if (data.cancelled) { setStatus("Detection cancelled"); return; }
      if (data.error) {
        setStatus(`Detection failed: ${data.error}`);
        return;
      }
      setDetection({
        bbox: data.bbox,
        maskDataUrl: `data:image/png;base64,${data.mask_png_base64}`,
        imageWidth: data.image_width,
        imageHeight: data.image_height,
        confidence: data.confidence,
        label: data.label,
      });
      if (data.analysis) setCurrentAnalysis(data.analysis);
      setTrackData(null);
      setStatus(`Detected ${data.label} (conf=${data.confidence.toFixed(2)}) bbox=[${data.bbox.join(", ")}]`);
      refreshAnalyses(name);
    } catch (err: any) {
      setStatus(`Detection error: ${err.message}`);
    } finally {
      setDetecting(false);
    }
  }

  async function cancelDetect() {
    const name = videoName();
    if (!name) return;
    setDetecting(false);
    setStatus("Cancelling detection…");
    try {
      const r = await fetch(`/api/detect?video=${encodeURIComponent(name)}`, { method: "DELETE" });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        setStatus(`Cancel failed: ${data.error ?? r.statusText}`);
      } else if (data.exited === false) {
        setStatus("Cancel sent but detector did not exit cleanly — check log");
      } else {
        setStatus("Detection cancelled");
      }
    } catch (e: any) {
      setStatus(`Cancel error: ${e.message}`);
    }
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer?.files[0];
    if (file && file.type.startsWith("video/")) {
      uploadVideo(file);
    } else {
      setStatus("Please drop a video file");
    }
  }

  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    setDragOver(true);
  }

  function handleDragLeave() {
    setDragOver(false);
  }

  function togglePlay() {
    if (!videoEl) return;
    if (videoEl.paused) {
      videoEl.play();
      setPlaying(true);
      startTimeTracking();
    } else {
      videoEl.pause();
      setPlaying(false);
      stopTimeTracking();
    }
  }

  let seekTimer: number | undefined;
  let pendingSeek: number | undefined;
  function seek(time: number) {
    if (!videoEl) return;
    setCurrentTime(time);
    setCurrentFrame(Math.floor(time * fps()));
    pendingSeek = time;
    if (seekTimer === undefined) {
      videoEl.currentTime = time;
      seekTimer = window.setTimeout(() => {
        seekTimer = undefined;
        if (pendingSeek !== undefined && pendingSeek !== videoEl!.currentTime) {
          videoEl!.currentTime = pendingSeek;
        }
        pendingSeek = undefined;
      }, 50);
    }
  }

  function stepFrame(delta: number) {
    if (!videoEl) return;
    videoEl.pause();
    setPlaying(false);
    stopTimeTracking();
    const newTime = Math.max(0, Math.min(videoEl.currentTime + delta / fps(), duration()));
    videoEl.currentTime = newTime;
    setCurrentTime(newTime);
    setCurrentFrame(Math.floor(newTime * fps()));
  }

  // Arrow-key navigation. On the source tab steps one frame at a time;
  // on data-bearing tabs jumps to the next/previous frame that actually
  // has data for that view.
  function navigateFrames(dir: 1 | -1) {
    if (!videoEl) return;
    const tab = viewTab();
    if (tab === "source") {
      stepFrame(dir);
      return;
    }
    let keyframes: number[];
    if (tab === "depth") {
      keyframes = depthFrames();
    } else {
      const cam = cameras();
      keyframes = cam ? cam.frames.filter((f) => f.registered).map((f) => f.idx) : [];
    }
    if (keyframes.length === 0) return;
    const sorted = [...keyframes].sort((a, b) => a - b);
    const cur = currentFrame();
    let target: number | undefined;
    if (dir === 1) {
      target = sorted.find((f) => f > cur);
    } else {
      for (const f of sorted) {
        if (f >= cur) break;
        target = f;
      }
    }
    if (target === undefined) return;
    videoEl.pause();
    setPlaying(false);
    stopTimeTracking();
    const newTime = Math.max(0, Math.min(target / fps(), duration()));
    videoEl.currentTime = newTime;
    setCurrentTime(newTime);
    setCurrentFrame(target);
  }

  function handleVideoLoaded() {
    if (!videoEl) return;
    videoEl.muted = true;
    setDuration(videoEl.duration);
    setTotalFrames(Math.floor(videoEl.duration * fps()));
    // Restore saved frame position
    const savedFrame = localStorage.getItem("segviewer:frame");
    if (savedFrame) {
      const frame = parseInt(savedFrame, 10);
      if (Number.isFinite(frame) && frame > 0) {
        const t = frame / fps();
        if (t < videoEl.duration) {
          videoEl.currentTime = t;
          setCurrentTime(t);
          setCurrentFrame(frame);
        }
      }
    }
    videoReady = true;
    setVideoSize({ w: videoEl.videoWidth, h: videoEl.videoHeight });
    setStatus(`Ready: ${videoName()} | ${videoEl.videoWidth}x${videoEl.videoHeight} | ${videoEl.duration.toFixed(1)}s`);
  }

  function handleVideoClick(e: MouseEvent) {
    if (!videoEl) return;
    if (!settingSeed() && !settingFloor()) return;
    // Get click position relative to the video's rendered area
    const rect = videoEl.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    // Convert to video pixel coordinates
    const scaleX = videoEl.videoWidth / rect.width;
    const scaleY = videoEl.videoHeight / rect.height;
    const vx = Math.round(clickX * scaleX);
    const vy = Math.round(clickY * scaleY);

    if (settingSeed()) {
      setSeedPoint({ x: vx, y: vy });
      setSettingSeed(false);
      setStatus(`Seed point set: (${vx}, ${vy})`);
    } else if (settingFloor()) {
      const frame = currentFrame();
      setFloorPoints((prev) => [...prev, { x: vx, y: vy, frame }]);
      setStatus(`World-up point ${floorPoints().length + 1}: (${vx}, ${vy}) frame ${frame} — click more, Enter to finish, Esc to cancel`);
    }
  }

  /** Convert video-pixel coords to CSS position relative to the video element */
  function seedOverlayPos() {
    const s = seedPoint();
    if (!s || !videoEl) return null;
    const rect = videoEl.getBoundingClientRect();
    const scaleX = rect.width / videoEl.videoWidth;
    const scaleY = rect.height / videoEl.videoHeight;
    return { left: s.x * scaleX, top: s.y * scaleY };
  }

  /** Convert a bbox in video pixel coords to CSS rect relative to the video element */
  function bboxOverlayRect() {
    const d = detection();
    if (!d || !videoEl) return null;
    const rect = videoEl.getBoundingClientRect();
    const scaleX = rect.width / videoEl.videoWidth;
    const scaleY = rect.height / videoEl.videoHeight;
    const [x1, y1, x2, y2] = d.bbox;
    return {
      left: x1 * scaleX,
      top: y1 * scaleY,
      width: (x2 - x1) * scaleX,
      height: (y2 - y1) * scaleY,
    };
  }

  /** URL of the tracked mask PNG for the current frame, if available */
  function trackMaskUrl(): string | null {
    const t = trackData();
    const v = videoName();
    const run = currentAnalysis();
    if (!t || !v || !run) return null;
    const f = t.frames[currentFrame()];
    if (!f || !f.bbox) return null;
    const stem = v.replace(/\.[^.]+$/, "");
    const frameStr = String(currentFrame()).padStart(6, "0");
    return `/analysis/${encodeURIComponent(stem)}/${encodeURIComponent(run)}/masks/${frameStr}.png`;
  }

  /** CSS rect for the tracked bbox at the current frame, if any */
  function trackBboxRect() {
    const t = trackData();
    if (!t || !videoEl) return null;
    const f = t.frames[currentFrame()];
    if (!f || !f.bbox) return null;
    const rect = videoEl.getBoundingClientRect();
    const scaleX = rect.width / videoEl.videoWidth;
    const scaleY = rect.height / videoEl.videoHeight;
    const [x1, y1, x2, y2] = f.bbox;
    return {
      left: x1 * scaleX,
      top: y1 * scaleY,
      width: (x2 - x1) * scaleX,
      height: (y2 - y1) * scaleY,
    };
  }

  function formatTime(t: number): string {
    const m = Math.floor(t / 60);
    const s = (t % 60).toFixed(2);
    return `${m}:${s.padStart(5, "0")}`;
  }

  // ── Styles ──
  const sidebarStyle = {
    width: "280px",
    "min-width": "280px",
    background: "#16213e",
    "border-right": "1px solid #0f3460",
    display: "flex",
    "flex-direction": "column",
    overflow: "hidden",
  } as const;

  const headerStyle = {
    display: "block",
    "border-bottom": "1px solid #0f3460",
  } as const;

  const logoStyle = {
    display: "block",
    width: "100%",
    height: "auto",
  } as const;

  const btnStyle = (active = true) => ({
    padding: "6px 14px",
    background: active ? "#0f3460" : "#333",
    color: active ? "#e0e0e0" : "#666",
    border: "none",
    "border-radius": "3px",
    cursor: active ? "pointer" : "not-allowed",
    "font-size": "12px",
    "font-family": "inherit",
  });

  const accentBtnStyle = (active = true, done = false) => ({
    ...btnStyle(active),
    background: done ? "#2ecc71" : active ? "#e94560" : "#555",
    color: done ? "#000" : "#fff",
    "font-weight": "600",
  });

  /** Style for "Running (click to cancel)" buttons — clickable, orange. */
  const cancellableRunningStyle = () => ({
    ...btnStyle(true),
    background: "#e67e22",
    color: "#fff",
    "font-weight": "600",
    cursor: "pointer",
  });

  const deleteIconBtnStyle = (active = true) => ({
    padding: "0 10px",
    background: "transparent",
    color: active ? "#e94560" : "#444",
    border: `1px solid ${active ? "#0f3460" : "#222"}`,
    "border-radius": "3px",
    cursor: active ? "pointer" : "not-allowed",
    "font-size": "16px",
    "line-height": "1",
    "font-family": "inherit",
  });

  return (
    <div style={{ display: "flex", width: "100%", height: "100%" }}>
      {/* ── Sidebar (full window height) ── */}
      <div style={sidebarStyle}>
          <div style={headerStyle}>
            <img src="/logo.png" alt="VideoVision" style={logoStyle} />
          </div>

          <div style={{ "overflow-y": "auto", flex: "1", padding: "0" }}>
            {/* Video selector */}
            <div style={{ padding: "12px 16px", "border-bottom": "1px solid #0f3460" }}>
              <div style={{ "font-size": "11px", "text-transform": "uppercase", "letter-spacing": "0.5px", color: "#888", "margin-bottom": "8px" }}>
                Video
              </div>
              <div style={{ display: "flex", gap: "4px" }}>
                <select
                  value={videoName() ?? ""}
                  onChange={(e) => {
                    const v = e.currentTarget.value;
                    if (v) loadVideo(v);
                  }}
                  title="Select an uploaded video. Drop a new file anywhere on the page to upload — uploads are re-encoded for smooth scrubbing (half-second keyframes) and all frames are extracted up front, so 3D scene plugins can run without first running COLMAP."
                  style={{
                    flex: "1",
                    "min-width": "0",
                    padding: "6px 8px",
                    background: "#0a0e1a",
                    border: "1px solid #0f3460",
                    color: "#e0e0e0",
                    "border-radius": "3px",
                    "font-size": "13px",
                    "font-family": "inherit",
                    cursor: "pointer",
                  }}
                >
                  <option value="" disabled>
                    {videos().length ? "Select a video..." : "No videos uploaded"}
                  </option>
                  <For each={videos()}>
                    {(name) => <option value={name}>{name}</option>}
                  </For>
                </select>
                <button
                  title="Delete this video and all its analyses"
                  onClick={() => { const v = videoName(); if (v) deleteVideo(v); }}
                  disabled={!videoName()}
                  style={deleteIconBtnStyle(!!videoName())}
                >
                  ×
                </button>
              </div>
              <Show when={videoSrc()}>
                <div style={{ display: "flex", gap: "4px", "margin-top": "6px" }}>
                  <select
                    value={currentAnalysis() ?? ""}
                    onChange={(e) => {
                      const v = e.currentTarget.value;
                      if (v) {
                        loadAnalysis(v);
                      } else {
                        setCurrentAnalysis(null);
                        setDetection(null);
                        setSeedPoint(null);
                        setTrackData(null);
                        localStorage.removeItem("segviewer:analysis");
                        setStatus("New analysis");
                      }
                    }}
                    title="An analysis is a per-object detect+track run, named '<label>_<N>' (e.g. chair_1). Selecting one loads its frame-0 detection, mask, tracking results, and any 3D placement / mesh outputs. Choose '(create new)' to start fresh — a new analysis folder is created the next time you Detect."
                    style={{
                      flex: "1",
                      "min-width": "0",
                      padding: "6px 8px",
                      background: "#0a0e1a",
                      border: "1px solid #0f3460",
                      color: "#e0e0e0",
                      "border-radius": "3px",
                      "font-size": "13px",
                      "font-family": "inherit",
                      cursor: "pointer",
                    }}
                  >
                    <option value="">(create new)</option>
                    <For each={analyses()}>
                      {(name) => <option value={name}>{name}</option>}
                    </For>
                  </select>
                  <button
                    title="Delete this analysis"
                    onClick={() => { const a = currentAnalysis(); if (a) deleteAnalysis(a); }}
                    disabled={!currentAnalysis()}
                    style={deleteIconBtnStyle(!!currentAnalysis())}
                  >
                    ×
                  </button>
                </div>
              </Show>
            </div>

            {/* Annotation */}
            <div style={{ padding: "12px 16px" }}>
              <div style={{ "font-size": "11px", "text-transform": "uppercase", "letter-spacing": "0.5px", color: "#888", "margin-bottom": "8px" }}>
                Annotation
              </div>
              <div style={{ display: "flex", gap: "4px" }}>
                <button
                  style={{
                    ...accentBtnStyle(depthFrames().length > 0 && !settingFloor()),
                    flex: "1",
                    background: settingFloor() ? "#3498db"
                      : savedWorldUp().length >= 3 ? "#2ecc71"
                      : depthFrames().length > 0 ? "#e94560" : "#555",
                  }}
                  onClick={() => {
                    setFloorPoints([]);
                    setSettingFloor(true);
                    setStatus("Click on horizontal surfaces (floor, table, etc). Enter to finish, Esc to cancel.");
                  }}
                  disabled={depthFrames().length === 0 || settingFloor()}
                  title="Click 3+ points on flat horizontal surfaces (floor, table) to define the up direction. You can scrub to different frames and click across multiple frames. Press Enter or click Done to finish."
                >
                  Set World-Up Points
                </button>
                <button
                  style={{
                    ...accentBtnStyle(true),
                    padding: "6px 10px",
                    background: settingFloor() ? "#3498db" : "#555",
                  }}
                  onClick={() => {
                    if (settingFloor()) {
                      setSettingFloor(false);
                      const pts = floorPoints();
                      if (pts.length < 3) {
                        setStatus(`Need at least 3 world-up points (have ${pts.length})`);
                      } else {
                        const v = videoName();
                        if (v) saveWorldUp(v, pts);
                        setStatus(`${pts.length} world-up points saved — click "Align Scene"`);
                      }
                    } else {
                      setFloorPoints([]);
                      setSavedWorldUp([]);
                      setWorldUpId("");
                      const v = videoName();
                      if (v) saveWorldUp(v, []);
                      setStatus("World-up points cleared");
                    }
                  }}
                  disabled={!settingFloor() && savedWorldUp().length === 0 && floorPoints().length === 0}
                  title={settingFloor()
                    ? "Finish picking world-up points (same as pressing Enter). Needs at least 3 points to save."
                    : "Discard the current and any saved world-up points for this video so you can pick them again."}
                >
                  {settingFloor() ? "Done" : "Clear"}
                </button>
              </div>
              <Show when={(floorPoints().length > 0 || savedWorldUp().length > 0) && !settingFloor()}>
                <div style={{ "font-size": "11px", color: "#888", "margin-top": "4px" }}>
                  {floorPoints().length > 0
                    ? `${floorPoints().length} world-up points set`
                    : `${savedWorldUp().length} saved world-up points`}
                </div>
              </Show>
            </div>

            {/* 3D Scene Analysis */}
            <div style={{ padding: "12px 16px" }}>
              <div style={{ "font-size": "11px", "text-transform": "uppercase", "letter-spacing": "0.5px", color: "#888", "margin-bottom": "8px" }}>
                Per-Frame Depth & Cameras
              </div>
              <select
                value={availablePlugins().length === 0 ? "" : sceneSource()}
                disabled={availablePlugins().length === 0}
                onChange={(e) => {
                  setSceneSource(e.currentTarget.value);
                  setBoxResult(null);
                  const v = videoName();
                  if (v) refreshDepthFrames(v);
                }}
                title="Pick which scene-reconstruction method produces camera poses + per-frame depth. COLMAP is classical SfM (slow, robust, geometry-only). The neural plugins (CUT3R, VGGT, Pi3, MapAnything, WorldMirror, DA3) infer poses + depth in one feed-forward pass — usually faster and don't depend on COLMAP. Pi3 / MapAnything / WorldMirror also produce a global scene pointmap for the 3D (Scene) tab. WildDet3D additionally runs 3D object detection. Only plugins whose setup script has been run appear here."
                style={{
                  width: "100%",
                  padding: "6px 8px",
                  "margin-bottom": "6px",
                  background: "#0a0e1a",
                  border: "1px solid #0f3460",
                  color: "#e0e0e0",
                  "border-radius": "3px",
                  "font-size": "12px",
                  "font-family": "inherit",
                  cursor: availablePlugins().length === 0 ? "not-allowed" : "pointer",
                }}
              >
                <Show
                  when={availablePlugins().length > 0}
                  fallback={<option value="">(none available)</option>}
                >
                  <For each={availablePlugins()}>
                    {(p) => <option value={p.id}>{p.label}</option>}
                  </For>
                </Show>
              </select>
              <Show when={SCENE_PLUGINS_BY_ID[sceneSource()]?.requiresCameraSource}>
                <div
                  style={{ display: "flex", "align-items": "center", gap: "6px", "margin-bottom": "6px" }}
                  title="Which scene plugin's cameras.json supplies the per-frame poses and intrinsics fed into this plugin. Only plugins with an existing analysis for this video show up here — run one of them first."
                >
                  <label style={{ "font-size": "11px", color: "#aaa" }}>Source</label>
                  <select
                    value={effectiveCameraSource()}
                    disabled={cameraSourceOptions().length === 0}
                    onChange={(e) => {
                      const id = sceneSource();
                      const v = videoName();
                      const val = e.currentTarget.value;
                      if (!v) return;
                      setPluginCameraSources((prev) => ({ ...prev, [`${v}:${id}`]: val }));
                    }}
                    style={{
                      flex: "1",
                      padding: "4px 6px",
                      background: "#0a0e1a",
                      border: "1px solid #0f3460",
                      color: "#e0e0e0",
                      "border-radius": "3px",
                      "font-size": "12px",
                      "font-family": "inherit",
                      cursor: cameraSourceOptions().length === 0 ? "not-allowed" : "pointer",
                    }}
                  >
                    <Show
                      when={cameraSourceOptions().length > 0}
                      fallback={<option value="">(run another plugin first)</option>}
                    >
                      <For each={cameraSourceOptions()}>
                        {(o) => <option value={o.id}>{o.label}</option>}
                      </For>
                    </Show>
                  </select>
                </div>
              </Show>
              <Show when={SCENE_PLUGINS_BY_ID[sceneSource()]?.subsampleDefault !== undefined}>
                <div
                  style={{ display: "flex", "align-items": "center", gap: "6px", "margin-bottom": "6px" }}
                  title="Use every Nth extracted frame as input to this plugin. Higher N = faster + less VRAM but coarser camera trajectory and depth coverage. Each plugin's default reflects what the underlying script uses when no override is passed."
                >
                  <label style={{ "font-size": "11px", color: "#aaa" }}>Subsample every</label>
                  <input
                    type="number"
                    min="1"
                    step="1"
                    value={pluginSubsamples()[sceneSource()] ?? ""}
                    onInput={(e) => {
                      const id = sceneSource();
                      const v = e.currentTarget.value;
                      setPluginSubsamples((prev) => ({ ...prev, [id]: v }));
                    }}
                    style={{
                      width: "60px",
                      padding: "4px 6px",
                      background: "#0a0e1a",
                      border: "1px solid #0f3460",
                      color: "#e0e0e0",
                      "border-radius": "3px",
                      "font-size": "12px",
                      "font-family": "inherit",
                    }}
                  />
                  <span style={{ "font-size": "11px", color: "#888" }}>frames</span>
                </div>
              </Show>
              <Show when={SCENE_PLUGINS_BY_ID[sceneSource()]?.upscaleOptions}>
                <div
                  style={{ display: "flex", "align-items": "center", gap: "6px", "margin-bottom": "6px" }}
                  title="Multiplier on the encoder's input resolution for the decoded depth. 1x = native; >1 super-resolves the depth via the implicit field (sharper edges, more VRAM, slower)."
                >
                  <label style={{ "font-size": "11px", color: "#aaa" }}>Upscale</label>
                  <select
                    value={pluginUpscales()[sceneSource()] ?? ""}
                    onChange={(e) => {
                      const id = sceneSource();
                      const v = e.currentTarget.value;
                      setPluginUpscales((prev) => ({ ...prev, [id]: v }));
                    }}
                    style={{
                      padding: "4px 6px",
                      background: "#0a0e1a",
                      border: "1px solid #0f3460",
                      color: "#e0e0e0",
                      "border-radius": "3px",
                      "font-size": "12px",
                      "font-family": "inherit",
                      cursor: "pointer",
                    }}
                  >
                    <For each={SCENE_PLUGINS_BY_ID[sceneSource()]?.upscaleOptions ?? []}>
                      {(u) => <option value={String(u)}>{u}x</option>}
                    </For>
                  </select>
                </div>
              </Show>
              <Show when={SCENE_PLUGINS_BY_ID[sceneSource()]?.targetFramesDefault !== undefined}>
                <div
                  style={{ display: "flex", "align-items": "center", gap: "6px", "margin-bottom": "6px" }}
                  title="Process exactly N frames, evenly spaced across the video. Higher = better coverage / longer trajectory, but more VRAM."
                >
                  <label style={{ "font-size": "11px", color: "#aaa" }}>Target frames</label>
                  <input
                    type="number"
                    min="1"
                    step="1"
                    value={pluginTargetFrames()[sceneSource()] ?? ""}
                    onInput={(e) => {
                      const id = sceneSource();
                      const v = e.currentTarget.value;
                      setPluginTargetFrames((prev) => ({ ...prev, [id]: v }));
                    }}
                    style={{
                      width: "60px",
                      padding: "4px 6px",
                      background: "#0a0e1a",
                      border: "1px solid #0f3460",
                      color: "#e0e0e0",
                      "border-radius": "3px",
                      "font-size": "12px",
                      "font-family": "inherit",
                    }}
                  />
                  <span style={{ "font-size": "11px", color: "#888" }}>total</span>
                </div>
              </Show>
              {(() => {
                const isRunning = () => preparingPluginId() === sceneSource();
                const isReady = () => Boolean(sceneStatus()?.artifacts?.[sceneSource()]);
                const runningStageText = () => {
                  const job = sceneStatus()?.job;
                  if (!job || !job.running) return "Running...";
                  // Only show stage progress for multi-stage pipelines (i.e. more
                  // than one pipeline step); single-step plugins just report "Running".
                  const plugin = SCENE_PLUGINS_BY_ID[sceneSource()];
                  return plugin && plugin.pipeline.length > 1
                    ? `Running (${job.stage ?? "..."})`
                    : "Running...";
                };
                const [hovered, setHovered] = createSignal(false);
                // Plugins that consume another plugin's cameras.json (e.g.
                // InfiniDepth) can't run until at least one upstream source
                // is ready.
                const needsCameraSource = () =>
                  Boolean(SCENE_PLUGINS_BY_ID[sceneSource()]?.requiresCameraSource);
                const cameraSourceMissing = () =>
                  needsCameraSource() && !effectiveCameraSource();
                return (
                  <button
                    style={isRunning()
                      ? { ...cancellableRunningStyle(), width: "100%" }
                      : { ...accentBtnStyle(!!videoSrc() && availablePlugins().length > 0 && !cameraSourceMissing(), isReady()), width: "100%" }}
                    onMouseEnter={() => setHovered(true)}
                    onMouseLeave={() => setHovered(false)}
                    onClick={() => (isRunning() ? cancelScene() : runScenePlugin(sceneSource()))}
                    disabled={!videoSrc() || availablePlugins().length === 0 || cameraSourceMissing()}
                    title={isRunning()
                      ? "Click to cancel the running scene-prep pipeline"
                      : isReady()
                        ? "Outputs already exist for this plugin. Click to re-run from scratch (the plugin's output dir is wiped first). Status messages stream to analysis/<video>/_scene/<plugin>.log."
                        : "Run the selected scene-reconstruction pipeline. Writes camera poses (cameras.json) + per-frame depth maps under analysis/<video>/_scene/<plugin>/. Long-running (seconds–minutes); progress streams into the log file."}
                  >
                    {isRunning()
                      ? `${runningStageText()} (Click to Cancel)`
                      : isReady()
                        ? (hovered() ? "Re-Run Analysis" : "Analysis Ready")
                        : "Run Analysis"}
                  </button>
                );
              })()}
              <Show when={sceneStatus()?.job?.error && sceneStatus()?.job?.pluginId === sceneSource()}>
                <div style={{ "font-size": "11px", color: "#e94560", "margin-top": "4px" }}>error: {sceneStatus()!.job!.error}</div>
              </Show>
              <button
                style={{
                  ...accentBtnStyle((floorPoints().length >= 3 || savedWorldUp().length >= 3) && !aligning()),
                  width: "100%",
                  "margin-top": "6px",
                  ...(isAligned() ? { background: "#2ecc71" } : {}),
                }}
                onClick={alignScene}
                disabled={(floorPoints().length < 3 && savedWorldUp().length < 3) || aligning()}
                title="Apply a similarity transform that puts the picked floor at y=0 and rescales depth to metric units. Edits the active plugin's cameras.json in place. Required before 3D box lifting and mesh reconstruction give meaningful real-world coordinates. (You must Set World-Up Bounds in the Annotation Tab for this to work)."
              >
                {aligning() ? "Aligning..." : isAligned() ? "Aligned" : "Align Scene"}
              </button>
            </div>

            {/* Features (DinoV3 dense patch features) */}
            <div style={{ padding: "12px 16px" }}>
              <div style={{ "font-size": "11px", "text-transform": "uppercase", "letter-spacing": "0.5px", color: "#888", "margin-bottom": "8px" }}>
                Per-Frame Features (DinoV3)
              </div>
              <div
                style={{ display: "flex", "align-items": "center", gap: "6px", "margin-bottom": "6px" }}
                title="Use every Nth extracted frame as input to DinoV3. Higher N = faster + less disk but coarser temporal coverage."
              >
                <label style={{ "font-size": "11px", color: "#aaa" }}>Subsample every</label>
                <input
                  type="number"
                  min="1"
                  step="1"
                  value={dinov3Subsample()}
                  onInput={(e) => setDinov3Subsample(e.currentTarget.value)}
                  style={{
                    width: "60px",
                    padding: "4px 6px",
                    background: "#0a0e1a",
                    border: "1px solid #0f3460",
                    color: "#e0e0e0",
                    "border-radius": "3px",
                    "font-size": "12px",
                    "font-family": "inherit",
                  }}
                />
                <span style={{ "font-size": "11px", color: "#888" }}>frames</span>
              </div>
              <div
                style={{ display: "flex", "align-items": "center", gap: "6px", "margin-bottom": "6px" }}
                title="Resize each frame to this fraction of source resolution before patching. 100% = full source size. Smaller = faster + smaller files but coarser patch grid. Each axis is rounded to a multiple of 16."
              >
                <label style={{ "font-size": "11px", color: "#aaa" }}>Scaling</label>
                <select
                  value={dinov3Scaling()}
                  onChange={(e) => setDinov3Scaling(e.currentTarget.value)}
                  style={{
                    padding: "4px 6px",
                    background: "#0a0e1a",
                    border: "1px solid #0f3460",
                    color: "#e0e0e0",
                    "border-radius": "3px",
                    "font-size": "12px",
                    "font-family": "inherit",
                    cursor: "pointer",
                  }}
                >
                  <For each={DINOV3_SCALING_OPTIONS}>
                    {(o) => <option value={o}>{Number(o) * 100}%</option>}
                  </For>
                </select>
              </div>
              {(() => {
                const running = () => Boolean(dinov3Status()?.job?.running);
                const ready = () => dinov3ReadyMatches();
                const [hovered, setHovered] = createSignal(false);
                const disabled = () => !videoSrc() || !dinov3Available();
                const tip = () => {
                  if (!dinov3Available()) {
                    return "DinoV3 is not installed. Run `python setup/plugin_dinov3.py` from the project root (requires Hugging Face access to facebook/dinov3-vitl16-pretrain-lvd1689m) and refresh.";
                  }
                  if (running()) return "Click to cancel the running DinoV3 feature run";
                  if (ready()) {
                    const m = dinov3Status()!.meta!;
                    return `DinoV3 features ready (grid ${m.grid_width}×${m.grid_height}, subsample=${m.subsample_every}, scaling=${m.scaling}). Click to re-run; current outputs will be wiped first.`;
                  }
                  const m = dinov3Status()?.meta;
                  if (m) {
                    return `Stored result was computed with subsample=${m.subsample_every}, scaling=${m.scaling} — current selection differs. Click to re-run with the new options.`;
                  }
                  return "Run DinoV3 ViT-L/16 on every Nth frame, producing dense patch features (fp16 .npz per frame) under analysis/<video>/_scene/dinov3/.";
                };
                return (
                  <button
                    style={running()
                      ? { ...cancellableRunningStyle(), width: "100%" }
                      : { ...accentBtnStyle(!disabled(), ready()), width: "100%" }}
                    onMouseEnter={() => setHovered(true)}
                    onMouseLeave={() => setHovered(false)}
                    onClick={() => (running() ? cancelDinov3() : runDinov3())}
                    disabled={!running() && disabled()}
                    title={tip()}
                  >
                    {running()
                      ? "Running... (Click to Cancel)"
                      : ready()
                        ? (hovered() ? "Re-Run Features" : "Features Ready")
                        : "Detect Features (DinoV3)"}
                  </button>
                );
              })()}
              <Show when={dinov3Status()?.job?.error}>
                <div style={{ "font-size": "11px", color: "#e94560", "margin-top": "4px" }}>
                  error: {dinov3Status()!.job!.error}
                </div>
              </Show>
            </div>

            {/* Object Segmentation */}
            <div style={{ padding: "12px 16px" }}>
              <div style={{ "font-size": "11px", "text-transform": "uppercase", "letter-spacing": "0.5px", color: "#888", "margin-bottom": "8px" }}>
                Object Segmentation (SAMV2/V3)
              </div>
              <div
                style={{ display: "flex", "align-items": "center", gap: "6px", "margin-bottom": "8px" }}
                title="What you're looking for in the video (e.g. 'chair', 'red mug', 'person'). Used as the SAM3 text prompt to disambiguate which object the seed point refers to, and also names the analysis folder ('chair_1', 'chair_2', ...)."
              >
                <label style={{ "font-size": "11px", color: "#888", "white-space": "nowrap" }}>Object Label</label>
                <input
                  type="text"
                  value={detectLabel()}
                  onInput={(e) => setDetectLabel(e.currentTarget.value)}
                  placeholder="e.g. chair"
                  style={{
                    flex: "1",
                    padding: "5px 8px",
                    background: "#1a1a2e",
                    border: "1px solid #0f3460",
                    color: "#e0e0e0",
                    "border-radius": "3px",
                    "font-size": "12px",
                    "font-family": "inherit",
                    "box-sizing": "border-box",
                    "min-width": "0",
                  }}
                />
              </div>
              <button
                style={{
                  ...accentBtnStyle(!!videoSrc()),
                  width: "100%",
                  background: settingSeed() ? "#3498db" : seedPoint() ? "#2ecc71" : (videoSrc() ? "#e94560" : "#555"),
                }}
                title={seedPoint()
                  ? `Seed point set at (${seedPoint()!.x}, ${seedPoint()!.y}) on frame 0. Click to pick a new one.`
                  : "Snap to frame 0 and arm the next click on the video as a seed point. The point + the Object Label tell SAM3 which instance to segment when multiple objects match the label."}
                onClick={() => {
                  if (!videoSrc()) return;
                  if (videoEl) {
                    videoEl.pause();
                    videoEl.currentTime = 0;
                    setPlaying(false);
                    stopTimeTracking();
                    setCurrentTime(0);
                    setCurrentFrame(0);
                  }
                  setSettingSeed(true);
                  setStatus("Click on the video to set a seed point");
                }}
                disabled={!videoSrc()}
              >
                {settingSeed() ? "Click on video..." : "Set Seed Location"}
              </button>
              <button
                style={detecting()
                  ? { ...cancellableRunningStyle(), width: "100%", "margin-top": "6px" }
                  : {
                      ...accentBtnStyle(!!seedPoint() && !detecting() && sam3Available(), !!detection()),
                      width: "100%",
                      "margin-top": "6px",
                    }}
                title={detecting()
                  ? "Click to cancel SAM3 detection"
                  : [
                      sam3DisabledReason(),
                      detection()
                        ? `Last detection: ${detection()!.label} (conf ${detection()!.confidence.toFixed(2)}) bbox=[${detection()!.bbox.join(", ")}]. Click to re-run.`
                        : "Run SAM3 on frame 0 using the seed point + Object Label. Produces a 2D bounding box and segmentation mask, and creates a new analysis folder ('<label>_<N>') that holds every downstream artifact.",
                    ].filter(Boolean).join("\n\n")}
                onClick={() => (detecting() ? cancelDetect() : detectObject())}
                disabled={!detecting() && (!seedPoint() || !sam3Available())}
              >
                {detecting() ? "Detecting (Click to Cancel)" : "Detect Object In Frame 0 (SAM3)"}
              </button>
              <button
                style={tracking()
                  ? { ...cancellableRunningStyle(), width: "100%", "margin-top": "6px" }
                  : {
                      ...accentBtnStyle(!!currentAnalysis() && !tracking() && sam2Available(), !!trackData()),
                      width: "100%",
                      "margin-top": "6px",
                    }}
                title={tracking()
                  ? "Click to cancel SAM2 tracking"
                  : [
                      sam2DisabledReason(),
                      trackData()
                        ? `Tracked across ${trackData()!.frames.length} frames. Click to re-run.`
                        : "Run SAM2 video tracking starting from the frame-0 detection mask. Produces a per-frame mask sequence (track.json) used by every downstream step — 3D box lifting, WildDet3D, and mesh reconstruction.",
                    ].filter(Boolean).join("\n\n")}
                onClick={() => (tracking() ? cancelTrack() : trackThroughVideo())}
                disabled={!tracking() && (!currentAnalysis() || !sam2Available())}
              >
                {tracking() ? "Tracking (Click to Cancel)" : "Track Through Video (SAM2)"}
              </button>
            </div>

            {/* Object Placement */}
            <div style={{ padding: "12px 16px" }}>
              <div style={{ "font-size": "11px", "text-transform": "uppercase", "letter-spacing": "0.5px", color: "#888", "margin-bottom": "8px" }}>
                3D Object Bounds
              </div>
              <select
                value={availableBoxSolvers().length === 0 ? "" : boxSolverId()}
                disabled={availableBoxSolvers().length === 0}
                onChange={(e) => {
                  const id = e.currentTarget.value;
                  setBoxSolverId(id);
                  // Refetch result for this analysis under the newly selected solver.
                  const v = videoName();
                  const a = currentAnalysis();
                  if (v && a) refreshBoxResult(v, a, id);
                  else setBoxResult(null);
                }}
                title="Pick which 3D-box solver runs when you click Compute Boxes. Boxer fits an oriented box from depth + masked point clouds (needs depth, supports per-frame or fused). WildDet3D is a neural detector that predicts 3D boxes directly from the image (no depth required, optionally takes K/depth as priors). Only solvers whose setup script has been run appear here."
                style={{
                  width: "100%",
                  padding: "6px 8px",
                  "margin-bottom": "6px",
                  background: "#0a0e1a",
                  border: "1px solid #0f3460",
                  color: "#e0e0e0",
                  "border-radius": "3px",
                  "font-size": "12px",
                  "font-family": "inherit",
                  cursor: availableBoxSolvers().length === 0 ? "not-allowed" : "pointer",
                }}
              >
                <Show
                  when={availableBoxSolvers().length > 0}
                  fallback={<option value="">(none available)</option>}
                >
                  <For each={availableBoxSolvers()}>
                    {(s) => <option value={s.id}>{s.label}</option>}
                  </For>
                </Show>
              </select>
              <Show when={BOX_SOLVER_PLUGINS_BY_ID[boxSolverId()]?.options.length}>
                <div style={{ display: "flex", "flex-wrap": "wrap", gap: "4px 12px", "margin-bottom": "6px" }}>
                  <For each={BOX_SOLVER_PLUGINS_BY_ID[boxSolverId()]?.options ?? []}>
                    {(opt) => {
                      const value = () => boxSolverOptions()[boxSolverId()]?.[opt.key] ?? false;
                      return (
                        <label
                          style={{
                            display: "flex",
                            "align-items": "center",
                            gap: "6px",
                            "font-size": "12px",
                            color: "#aaa",
                            cursor: "pointer",
                          }}
                          title={opt.description}
                        >
                          <input
                            type="checkbox"
                            checked={value()}
                            onChange={(e) => {
                              const id = boxSolverId();
                              const checked = e.currentTarget.checked;
                              setBoxSolverOptions((prev) => ({
                                ...prev,
                                [id]: { ...(prev[id] ?? {}), [opt.key]: checked },
                              }));
                            }}
                            style={{ cursor: "pointer" }}
                          />
                          {opt.label}
                        </label>
                      );
                    }}
                  </For>
                </div>
              </Show>
              {(() => {
                const solver = () => BOX_SOLVER_PLUGINS_BY_ID[boxSolverId()];
                const ready = () => !!boxResult();
                const enabled = () => {
                  if (availableBoxSolvers().length === 0) return false;
                  if (!trackData() || boxRunning()) return false;
                  if (solver()?.requiresDepth && depthFrames().length === 0) return false;
                  return true;
                };
                const tip = () => {
                  if (availableBoxSolvers().length === 0) {
                    return "No 3D-box solver is installed. Run `python setup/plugin_boxer.py` (or `setup/plugin_wilddet3d.py`) to enable this.";
                  }
                  if (boxRunning()) return `Click to cancel ${solver().label}`;
                  if (!trackData()) return "Run tracking first";
                  if (solver().requiresDepth && depthFrames().length === 0) {
                    return `${solver().label} needs depth maps — run a scene plugin first`;
                  }
                  if (boxResult()?.num_frames_with_boxes) {
                    return `${solver().label} produced 3D boxes on ${boxResult()!.num_frames_with_boxes} frames. Click to re-run.`;
                  }
                  return `Run ${solver().label} 3D bounding box lifting on the tracked frames.`;
                };
                const style = boxRunning()
                  ? { ...cancellableRunningStyle(), width: "100%" }
                  : { ...accentBtnStyle(enabled(), ready()), width: "100%" };
                return (
                  <button
                    style={style}
                    title={tip()}
                    onClick={() => (boxRunning() ? cancelBoxSolver() : runBoxSolver())}
                    disabled={!boxRunning() && !enabled()}
                  >
                    {boxRunning() ? "Running (Click to Cancel)" : "Compute Boxes"}
                  </button>
                );
              })()}
            </div>

            {/* Object Point Cloud */}
            <div style={{ padding: "12px 16px" }}>
              <div style={{ "font-size": "11px", "text-transform": "uppercase", "letter-spacing": "0.5px", color: "#888", "margin-bottom": "8px" }}>
                Object Point Cloud
              </div>
              <div
                style={{ display: "flex", "align-items": "center", gap: "6px", "margin-bottom": "6px" }}
                title="Erode the per-frame mask by this many pixels at the mask's native (source) resolution, before downsampling to the depth map. Depth networks produce interpolated values right at silhouettes, so the outermost mask pixels often unproject to fly-away points behind the object; eroding peels those off. 0 = off. Bump it up if you see a wispy halo behind the cloud, but too high will shave thin features (table legs, fingers)."
              >
                <label style={{ "font-size": "11px", color: "#aaa" }}>Mask erode</label>
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={objectErode()}
                  onInput={(e) => setObjectErode(e.currentTarget.value)}
                  disabled={objectPointmapRunning()}
                  style={{
                    width: "60px",
                    padding: "4px 6px",
                    background: "#0a0e1a",
                    border: "1px solid #0f3460",
                    color: "#e0e0e0",
                    "border-radius": "3px",
                    "font-size": "12px",
                    "font-family": "inherit",
                  }}
                />
                <span style={{ "font-size": "11px", color: "#888" }}>px</span>
              </div>
              {(() => {
                const enabled = () =>
                  !!trackData() && depthFrames().length > 0 && !objectPointmapRunning();
                const tip = () => {
                  if (!trackData()) return "Run tracking first";
                  if (depthFrames().length === 0) {
                    return `Source '${SCENE_PLUGINS_BY_ID[sceneSource()]?.label ?? sceneSource()}' has no depth — run a scene plugin first`;
                  }
                  if (objectPointmapReady()) {
                    return "Object point cloud is ready — open the 3D (Object) tab. Click to rebuild from current depth + masks.";
                  }
                  return "For each frame, take the depth points inside the tracking mask, lift to world space, and fuse across frames into a single point cloud.";
                };
                return (
                  <button
                    style={{
                      ...accentBtnStyle(enabled(), objectPointmapReady()),
                      width: "100%",
                    }}
                    title={tip()}
                    onClick={runObjectPointmap}
                    disabled={!enabled()}
                  >
                    {objectPointmapRunning()
                      ? "Building..."
                      : objectPointmapReady()
                        ? "Object Cloud Ready"
                        : "Build Object Cloud"}
                  </button>
                );
              })()}
            </div>
          </div>
        </div>

        {/* Right column: viewport + bottom status bar */}
        <div style={{ display: "flex", "flex-direction": "column", flex: "1", "min-width": "0" }}>
        {/* ── Main Viewport ── */}
        <div
          style={{
            flex: "1",
            position: "relative",
            display: "flex",
            "flex-direction": "column",
            "min-width": "0",
            "min-height": "0",
          }}
        >
          {/* Tab bar */}
          <Show when={videoSrc()}>
            <div style={{ display: "flex", background: "#16213e", "border-bottom": "1px solid #0f3460" }}>
              <For each={[["source", "Source"], ["depth", "Depth"], ["3d", "3D (Per-Frame)"], ["3d-scene", "3D (Scene)"], ["3d-object", "3D (Object)"], ["dinov3", "DinoV3"]] as [ViewTab, string][]}>
                {([id, label]) => (
                  <Show when={
                    (id !== "3d-scene" || !!SCENE_PLUGINS_BY_ID[sceneSource()]?.features?.scenePointmap)
                    && (id !== "3d-object" || objectPointmapReady())
                    && (id !== "dinov3" || dinov3Status()?.meta != null)
                  }>
                    <button
                      onClick={() => setViewTab(id)}
                      style={{
                        padding: "6px 16px",
                        background: viewTab() === id ? "#1a1a2e" : "transparent",
                        border: "none",
                        "border-bottom": viewTab() === id ? "2px solid #e94560" : "2px solid transparent",
                        color: viewTab() === id ? "#e0e0e0" : "#888",
                        "font-size": "12px",
                        "font-family": "inherit",
                        cursor: "pointer",
                      }}
                    >
                      {label}
                    </button>
                  </Show>
                )}
              </For>
              <Show when={viewTab() === "source"}>
                {(() => {
                  const tbtn = (active: boolean) => ({
                    padding: "3px 8px",
                    background: active ? "#e94560" : "#0f3460",
                    border: "1px solid #0f3460",
                    "border-radius": "3px",
                    color: "#e0e0e0",
                    "font-size": "10px",
                    "font-family": "inherit",
                    cursor: "pointer",
                  });
                  return (
                    <div style={{ "margin-left": "auto", display: "flex", gap: "4px", "align-items": "center", "margin-right": "8px" }}>
                      <button
                        style={tbtn(showSourceBbox())}
                        onClick={() => setShowSourceBbox(!showSourceBbox())}
                        title="Toggle bbox overlay"
                      >
                        Bbox
                      </button>
                      <button
                        style={tbtn(showSourceMask())}
                        onClick={() => setShowSourceMask(!showSourceMask())}
                        title="Toggle mask overlay"
                      >
                        Mask
                      </button>
                    </div>
                  );
                })()}
              </Show>
              <Show when={viewTab() === "3d" || viewTab() === "3d-scene" || viewTab() === "3d-object"}>
                {(() => {
                  const tbtn = (active: boolean, disabled = false) => ({
                    padding: "3px 8px",
                    background: active ? "#e94560" : "#0f3460",
                    border: "1px solid #0f3460",
                    "border-radius": "3px",
                    color: disabled ? "#555" : "#e0e0e0",
                    "font-size": "10px",
                    "font-family": "inherit",
                    cursor: disabled ? "not-allowed" : "pointer",
                    opacity: disabled ? "0.5" : "1",
                  });
                  return (
                    <div style={{ "margin-left": "auto", display: "flex", gap: "4px", "align-items": "center", "margin-right": "8px" }}>
                      <Show when={viewTab() === "3d"}>
                        <select
                          value={String(meshSubsample())}
                          onChange={(e) => setMeshSubsample(Number(e.currentTarget.value))}
                          title="Mesh / pointmap sampling stride (every Nth pixel)"
                          style={{
                            padding: "2px 4px",
                            background: "#0f3460",
                            border: "1px solid #0f3460",
                            "border-radius": "3px",
                            color: "#e0e0e0",
                            "font-size": "10px",
                            "font-family": "inherit",
                            cursor: "pointer",
                          }}
                        >
                          <option value="1">1/1</option>
                          <option value="2">1/2</option>
                          <option value="4">1/4</option>
                        </select>
                      </Show>
                      <button
                        style={tbtn(showCameraPath())}
                        onClick={() => setShowCameraPath(!showCameraPath())}
                        title="Toggle camera path visibility"
                      >
                        Path
                      </button>
                      <Show when={viewTab() === "3d"}>
                        {(() => {
                          const pointmapOk = () => !!SCENE_PLUGINS_BY_ID[sceneSource()]?.features?.pointmap;
                          return (
                            <button
                              style={tbtn(pointmapView(), !pointmapOk())}
                              onClick={() => { if (pointmapOk()) setPointmapView(!pointmapView()); }}
                              disabled={!pointmapOk()}
                              title="Toggle pointmap view (plugins that publish pointmaps only)"
                            >
                              Pointmap
                            </button>
                          );
                        })()}
                        <button
                          style={tbtn(false)}
                          onClick={() => threeViewerActions?.snapCamera()}
                          title="Snap to current camera pose (F)"
                        >
                          Focus
                        </button>
                      </Show>
                      <button
                        style={tbtn(false)}
                        onClick={() => threeViewerActions?.fitAll()}
                        title="Fit view to entire scene"
                      >
                        Reset
                      </button>
                    </div>
                  );
                })()}
              </Show>
              <Show when={viewTab() === "dinov3"}>
                <div
                  style={{
                    "margin-left": "auto",
                    display: "flex",
                    gap: "8px",
                    "align-items": "center",
                    "margin-right": "8px",
                    color: "#aaa",
                    "font-size": "10px",
                    "font-family": "inherit",
                  }}
                >
                  <span title="Heatmap opacity over the frame image. 0% = image only, 100% = heatmap only.">Heatmap</span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={dinov3Opacity()}
                    onInput={(e) => setDinov3Opacity(Number(e.currentTarget.value))}
                    style={{ width: "140px", cursor: "pointer" }}
                    title="Heatmap opacity over the frame image. 0% = image only, 100% = heatmap only."
                  />
                  <span style={{ width: "32px", "text-align": "right" }}>
                    {Math.round(dinov3Opacity() * 100)}%
                  </span>
                </div>
              </Show>
            </div>
          </Show>

          {/* Viewport content */}
          <div
            style={{
              flex: "1",
              display: "flex",
              "align-items": "center",
              "justify-content": "center",
              background: "#111",
              position: "relative",
              overflow: "hidden",
            }}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <Show when={videoSrc()} fallback={
              <div
                style={{
                  display: "flex",
                  "flex-direction": "column",
                  "align-items": "center",
                  gap: "12px",
                  color: dragOver() ? "#e94560" : "#555",
                  "font-size": "16px",
                  border: `2px dashed ${dragOver() ? "#e94560" : "#333"}`,
                  padding: "60px 80px",
                  "border-radius": "12px",
                  transition: "all 0.2s",
                }}
              >
                <span style={{ "font-size": "48px" }}>&#x1F4F9;</span>
                <span>Drop a video file here</span>
                <span style={{ "font-size": "12px", color: "#444" }}>or select from the sidebar</span>
              </div>
            }>
              {/* Depth view */}
              <Show when={viewTab() === "depth"}>
                <Show when={depthFrames().length > 0} fallback={
                  <div style={{ color: "#555", "font-size": "14px" }}>
                    No depth maps available. Run "Prepare Scene" first.
                  </div>
                }>
                  <canvas
                    ref={(el) => setDepthCanvas(el)}
                    style={{ "max-width": "100%", "max-height": "100%", display: "block" }}
                  />
                  <Show when={depthLoading()}>
                    <div style={{
                      position: "absolute", inset: "0",
                      display: "flex", "align-items": "center", "justify-content": "center",
                      background: "rgba(0,0,0,0.5)", color: "#aaa", "font-size": "14px",
                    }}>
                      Loading depth...
                    </div>
                  </Show>
                </Show>
              </Show>
              {/* Depth right-side control panel: active color range + reset.
                  Mirrors the DinoV3 panel below. Hidden via DEPTH_RANGE_UI_ENABLED
                  for now (kept intact, not deleted). */}
              <Show when={DEPTH_RANGE_UI_ENABLED && viewTab() === "depth" && depthFrames().length > 0}>
                <div
                  style={{
                    position: "absolute",
                    right: "16px",
                    top: "16px",
                    background: "rgba(20, 20, 20, 0.85)",
                    border: "1px solid #333",
                    "border-radius": "6px",
                    padding: "12px",
                    display: "flex",
                    "flex-direction": "column",
                    gap: "10px",
                    "min-width": "180px",
                    "max-width": "220px",
                    color: "#e0e0e0",
                    "font-size": "10px",
                    "font-family": "inherit",
                    "z-index": "15",
                  }}
                >
                  <div style={{ color: "#aaa", "line-height": "1.45" }}>
                    Click-drag a line on the depth map to rescale the color map
                    between the depth at its two endpoints.
                  </div>
                  <div style={{ display: "flex", "flex-direction": "column", gap: "4px" }}>
                    <span style={{ color: "#aaa" }}>Color range</span>
                    <Show
                      when={depthDrag()}
                      fallback={<span>Auto (per-frame min/max)</span>}
                    >
                      {(d) => (
                        <div
                          style={{
                            display: "flex",
                            "flex-direction": "column",
                            gap: "2px",
                            "font-variant-numeric": "tabular-nums",
                          }}
                        >
                          <span>
                            <span style={{ color: "#3ad29f" }}>Start</span>{" "}
                            {formatDepth(d().d1)}
                          </span>
                          <span>
                            <span style={{ color: "#e94560" }}>End</span>{" "}
                            {formatDepth(d().d2)}
                          </span>
                          <Show when={depthRange()}>
                            {(r) => (
                              <span style={{ color: "#888", "margin-top": "2px" }}>
                                range {formatDepth(r().min)} – {formatDepth(r().max)}
                              </span>
                            )}
                          </Show>
                        </div>
                      )}
                    </Show>
                  </div>
                  <button
                    onClick={resetDepthRange}
                    disabled={!depthDrag() && !depthRange()}
                    title="Restore automatic per-frame color scaling"
                    style={{
                      padding: "4px 8px",
                      background: "#0f3460",
                      border: "1px solid #0f3460",
                      "border-radius": "3px",
                      color: "#e0e0e0",
                      "font-size": "10px",
                      "font-family": "inherit",
                      cursor: (depthDrag() || depthRange()) ? "pointer" : "default",
                      opacity: (depthDrag() || depthRange()) ? 1 : 0.5,
                    }}
                  >
                    Reset
                  </button>
                </div>
              </Show>
              {/* DinoV3 dense-patch cosine-sim viewer — always mounted so its
                  per-frame feature cache survives tab switches. */}
              <DinoV3Viewer
                videoName={videoName()}
                currentFrame={currentFrame()}
                visible={viewTab() === "dinov3"}
                heatmapOpacity={dinov3Opacity()}
                dataVersion={dinov3DataVersion()}
                resetVersion={dinov3ResetVersion()}
                mode={dinov3Mode()}
                threshold={dinov3Threshold()}
                onMeta={setDinov3Meta}
              />
              {/* DinoV3 right-side control panel: viz mode + (contour) threshold + reset. */}
              <Show when={viewTab() === "dinov3"}>
                <div
                  style={{
                    position: "absolute",
                    right: "16px",
                    top: "16px",
                    background: "rgba(20, 20, 20, 0.85)",
                    border: "1px solid #333",
                    "border-radius": "6px",
                    padding: "12px",
                    display: "flex",
                    "flex-direction": "column",
                    gap: "10px",
                    "min-width": "180px",
                    color: "#e0e0e0",
                    "font-size": "10px",
                    "font-family": "inherit",
                    "z-index": "15",
                  }}
                >
                  <label style={{ display: "flex", "flex-direction": "column", gap: "4px" }}>
                    <span style={{ color: "#aaa" }}>Mode</span>
                    <select
                      value={dinov3Mode()}
                      onChange={(e) => setDinov3Mode(e.currentTarget.value === "contour" ? "contour" : "heatmap")}
                      style={{
                        padding: "3px 4px",
                        background: "#0f3460",
                        border: "1px solid #0f3460",
                        "border-radius": "3px",
                        color: "#e0e0e0",
                        "font-size": "10px",
                        "font-family": "inherit",
                        cursor: "pointer",
                      }}
                    >
                      <option value="heatmap">Heatmap</option>
                      <option value="contour">Contour</option>
                    </select>
                  </label>
                  <Show when={dinov3Mode() === "contour"}>
                    <label style={{ display: "flex", "flex-direction": "column", gap: "4px" }}>
                      <span style={{ color: "#aaa" }}>
                        Threshold <span style={{ color: "#e0e0e0" }}>{dinov3Threshold().toFixed(2)}</span>
                      </span>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={dinov3Threshold()}
                        onInput={(e) => setDinov3Threshold(Number(e.currentTarget.value))}
                        style={{ width: "100%", cursor: "pointer" }}
                        title="Cosine-sim cutoff: patches at or above are filled, below are transparent."
                      />
                    </label>
                  </Show>
                  <button
                    onClick={() => setDinov3ResetVersion(dinov3ResetVersion() + 1)}
                    title="Discard the captured feature set"
                    style={{
                      padding: "4px 8px",
                      background: "#0f3460",
                      border: "1px solid #0f3460",
                      "border-radius": "3px",
                      color: "#e0e0e0",
                      "font-size": "10px",
                      "font-family": "inherit",
                      cursor: "pointer",
                    }}
                  >
                    Reset features
                  </button>
                </div>
              </Show>
              {/* 3D view — always mounted, hidden via CSS */}
              <ThreeDepthViewer
                videoName={videoName()}
                currentFrame={currentFrame()}
                depthFrames={depthFrames()}
                depthStem={depthStem()}
                cameras={cameras()}
                visible={viewTab() === "3d" || viewTab() === "3d-scene" || viewTab() === "3d-object"}
                boxResult={boxResult()}
                boxSolverId={boxSolverId()}
                sceneSource={sceneSource()}
                usePointmap={pointmapView()}
                scenePointmapMode={viewTab() === "3d-scene"}
                objectPointmapMode={viewTab() === "3d-object"}
                objectPointmapUrl={(() => {
                  const v = videoName();
                  const a = currentAnalysis();
                  return v && a ? objectPointmapUrl(v, a, sceneSource()) : null;
                })()}
                dataVersion={dataVersion()}
                showCameraPath={showCameraPath()}
                downsample={meshSubsample()}
                onReady={(actions) => { threeViewerActions = actions; }}
                onScenePointmapStatus={(s) => {
                  setScenePmLoading(s.loading);
                  setScenePmProgress(s.progress);
                  setScenePmPoints(s.pointCount);
                  setScenePmChunksLoaded(s.chunksLoaded);
                  setScenePmTotalChunks(s.totalChunks);
                }}
                onObjectPointmapStatus={(s) => {
                  setObjectPmLoading(s.loading);
                  setObjectPmProgress(s.progress);
                  setObjectPmPoints(s.pointCount);
                  setObjectPmChunksLoaded(s.chunksLoaded);
                  setObjectPmTotalChunks(s.totalChunks);
                }}
              />
              {/* Source video view */}
              <div
                ref={videoContainerEl!}
                style={{
                  position: "relative", display: viewTab() === "source" ? "inline-block" : "none",
                  "max-width": "100%", height: "100%",
                }}
              >
                <video
                  ref={videoEl!}
                  src={videoSrc()!}
                  onLoadedMetadata={handleVideoLoaded}
                  onEnded={() => { setPlaying(false); stopTimeTracking(); }}
                  onClick={handleVideoClick}
                  style={{
                    "max-width": "100%",
                    "max-height": "100%",
                    display: "block",
                    cursor: (settingSeed() || settingFloor()) ? "crosshair" : "default",
                  }}
                />
                {/* Detection mask overlay — show on first frame */}
                <Show when={showSourceMask() && currentFrame() === 0 && detection()}>
                  {(d) => (
                    <img
                      src={d().maskDataUrl}
                      style={{
                        position: "absolute",
                        left: "0",
                        top: "0",
                        width: "100%",
                        height: "100%",
                        "pointer-events": "none",
                      }}
                    />
                  )}
                </Show>
                {/* Detection bbox overlay — show on first frame */}
                <Show when={detection() && currentFrame() === 0 && showSourceBbox()}>
                  {(_) => {
                    const r = () => bboxOverlayRect();
                    return (
                      <Show when={r()}>
                        {(rr) => (
                          <div
                            style={{
                              position: "absolute",
                              left: `${rr().left}px`,
                              top: `${rr().top}px`,
                              width: `${rr().width}px`,
                              height: `${rr().height}px`,
                              border: "2px solid #2ecc71",
                              "box-sizing": "border-box",
                              "pointer-events": "none",
                            }}
                          />
                        )}
                      </Show>
                    );
                  }}
                </Show>
                {/* Tracked mask overlay — show on every frame once tracked */}
                <Show when={trackData() && showSourceMask()}>
                  {(_) => {
                    const url = () => trackMaskUrl();
                    return (
                      <Show when={url()}>
                        {(u) => (
                          <img
                            src={u()}
                            style={{
                              position: "absolute",
                              left: "0",
                              top: "0",
                              width: "100%",
                              height: "100%",
                              "pointer-events": "none",
                            }}
                          />
                        )}
                      </Show>
                    );
                  }}
                </Show>
                {/* Tracked bbox overlay — show on every frame once tracked */}
                <Show when={trackData() && showSourceBbox()}>
                  {(_) => {
                    const r = () => trackBboxRect();
                    return (
                      <Show when={r()}>
                        {(rr) => (
                          <div
                            style={{
                              position: "absolute",
                              left: `${rr().left}px`,
                              top: `${rr().top}px`,
                              width: `${rr().width}px`,
                              height: `${rr().height}px`,
                              border: "2px solid #f1c40f",
                              "box-sizing": "border-box",
                              "pointer-events": "none",
                            }}
                          />
                        )}
                      </Show>
                    );
                  }}
                </Show>
                {/* Seed point overlay — show on first frame */}
                <Show when={seedPoint() && currentFrame() === 0}>
                  {(_) => {
                    const pos = () => seedOverlayPos();
                    return (
                      <Show when={pos()}>
                        {(p) => (
                          <div
                            style={{
                              position: "absolute",
                              left: `${p().left - 8}px`,
                              top: `${p().top - 8}px`,
                              width: "16px",
                              height: "16px",
                              "border-radius": "50%",
                              background: "rgba(46, 204, 113, 0.6)",
                              border: "2px solid #2ecc71",
                              "pointer-events": "none",
                            }}
                          />
                        )}
                      </Show>
                    );
                  }}
                </Show>
                {/* World-up point overlays — full opacity on the exact frame
                    the point was picked on, drop to 0.5 once we step off, then
                    linearly fade to 0 by FADE_FRAMES away. */}
                <Show when={floorPoints().length > 0}>
                  <For each={floorPoints()}>
                    {(pt) => {
                      const FADE_FRAMES = 5;
                      const alpha = () => {
                        const d = Math.abs(pt.frame - currentFrame());
                        if (d === 0) return 1;
                        if (d >= FADE_FRAMES) return 0;
                        return 0.5 * (1 - (d - 1) / (FADE_FRAMES - 1));
                      };
                      const pos = () => {
                        if (!videoEl) return null;
                        const rect = videoEl.getBoundingClientRect();
                        const scaleX = rect.width / videoEl.videoWidth;
                        const scaleY = rect.height / videoEl.videoHeight;
                        return { left: pt.x * scaleX, top: pt.y * scaleY };
                      };
                      return (
                        <Show when={pos() && alpha() > 0 ? pos() : null}>
                          {(p) => (
                            <div
                              style={{
                                position: "absolute",
                                left: `${p()!.left - 6}px`,
                                top: `${p()!.top - 6}px`,
                                width: "12px",
                                height: "12px",
                                "border-radius": "50%",
                                background: "#fff",
                                border: "2px solid #000",
                                opacity: `${alpha()}`,
                                "pointer-events": "none",
                              }}
                            />
                          )}
                        </Show>
                      );
                    }}
                  </For>
                </Show>
              </div>
              {/* Point cloud chunk download indicator (3D Scene / 3D Object tabs) */}
              {(() => {
                const active = () => {
                  if (viewTab() === "3d-scene" && scenePmLoading()) {
                    return { loaded: scenePmChunksLoaded(), total: scenePmTotalChunks() };
                  }
                  if (viewTab() === "3d-object" && objectPmLoading()) {
                    return { loaded: objectPmChunksLoaded(), total: objectPmTotalChunks() };
                  }
                  return null;
                };
                return (
                  <Show when={active()}>
                    {(s) => (
                      <div
                        style={{
                          position: "absolute",
                          right: "8px",
                          bottom: "8px",
                          padding: "4px 8px",
                          background: "rgba(0, 0, 0, 0.65)",
                          color: "#e0e0e0",
                          "border-radius": "4px",
                          font: "11px/1.4 ui-monospace, Consolas, monospace",
                          "pointer-events": "none",
                          "user-select": "none",
                        }}
                      >
                        {s().total !== null
                          ? `chunks ${s().loaded ?? 0}/${s().total}`
                          : "loading manifest…"}
                      </div>
                    )}
                  </Show>
                );
              })()}
              {/* Active scene-analysis plugin name overlay (all tabs except Source) */}
              <Show when={SHOW_SCENE_NAME_OVERLAY && viewTab() !== "source"}>
                <div
                  style={{
                    position: "absolute",
                    right: "20px",
                    bottom: "16px",
                    color: "rgba(255, 255, 255, 0.22)",
                    "font-size": "72px",
                    "font-weight": "700",
                    "font-family": "inherit",
                    "letter-spacing": "0.02em",
                    "line-height": "1",
                    "text-shadow": "0 2px 12px rgba(0, 0, 0, 0.5)",
                    "pointer-events": "none",
                    "user-select": "none",
                    "z-index": "10",
                  }}
                >
                  {viewTab() === "dinov3" ? "DinoV3" : getScenePluginOrDefault(sceneSource()).label}
                </div>
              </Show>
              {/* Drop overlay when dragging over video */}
              <Show when={dragOver()}>
                <div
                  style={{
                    position: "absolute",
                    inset: "0",
                    background: "rgba(233, 69, 96, 0.15)",
                    display: "flex",
                    "align-items": "center",
                    "justify-content": "center",
                    "font-size": "18px",
                    color: "#e94560",
                    "pointer-events": "none",
                  }}
                >
                  Drop to replace video
                </div>
              </Show>
            </Show>
          </div>

          {/* Timeline + playback controls */}
          <Show when={videoSrc()}>
            <div style={{ background: "#16213e", "border-top": "1px solid #0f3460", padding: "6px 12px" }}>
              <div style={{ display: "flex", "align-items": "center", gap: "8px" }}>
                <button style={{ ...btnStyle(true), padding: "4px 8px", "font-size": "12px" }} onClick={() => stepFrame(-1)}>&lt;</button>
                <button style={{ ...accentBtnStyle(true), padding: "4px 10px", "font-size": "12px" }} onClick={togglePlay}>
                  {playing() ? "Pause" : "Play"}
                </button>
                <button style={{ ...btnStyle(true), padding: "4px 8px", "font-size": "12px" }} onClick={() => stepFrame(1)}>&gt;</button>
                <input
                  type="range"
                  min="0"
                  max={duration()}
                  step="0.001"
                  value={currentTime()}
                  onInput={(e) => {
                    const t = parseFloat(e.currentTarget.value);
                    // On the dinov3 tab, snap to the nearest frame that
                    // actually has features on disk — scrubbing between
                    // subsampled frames would just show the same frame.
                    if (viewTab() === "dinov3") {
                      const m = dinov3Meta();
                      if (m && m.frame_indices.length > 0) {
                        const targetFrame = t * fps();
                        let best = m.frame_indices[0];
                        let bestDist = Math.abs(targetFrame - best);
                        for (let i = 1; i < m.frame_indices.length; i++) {
                          const d = Math.abs(targetFrame - m.frame_indices[i]);
                          if (d < bestDist) { best = m.frame_indices[i]; bestDist = d; }
                        }
                        seek(best / fps());
                        return;
                      }
                    }
                    seek(t);
                  }}
                  style={{ flex: "1", cursor: "pointer", "accent-color": "#e94560" }}
                />
              </div>
              <div style={{ display: "flex", "justify-content": "space-between", "font-size": "11px", color: "#888", "margin-top": "4px" }}>
                <span>Time: {formatTime(currentTime())} / {formatTime(duration())}</span>
                <div style={{ display: "flex", gap: "12px" }}>
                  {(() => {
                    const tab = viewTab();
                    const size = tab === "source"
                      ? videoSize()
                      : (cameras() ? { w: cameras()!.width, h: cameras()!.height } : null);
                    let kfCount: number | null = null;
                    if (tab === "depth") {
                      kfCount = depthFrames().length;
                    } else if (tab === "3d" || tab === "3d-scene") {
                      const cam = cameras();
                      kfCount = cam ? cam.frames.filter((f) => f.registered).length : 0;
                    }
                    return (
                      <>
                        <Show when={tab === "3d-scene" && scenePmPoints() !== null}>
                          <span>Points: {(scenePmPoints()! / 1_000_000).toFixed(2)}m</span>
                        </Show>
                        <Show when={tab === "3d-object" && objectPmPoints() !== null}>
                          <span>Points: {(objectPmPoints()! / 1_000_000).toFixed(2)}m</span>
                        </Show>
                        <span>Resolution: {size ? `${size.w}x${size.h}` : "—"}</span>
                        <Show when={tab === "dinov3" && dinov3Meta()}>
                          {(m) => <span>Patch Grid: {m().grid_width}x{m().grid_height}</span>}
                        </Show>
                        <Show when={kfCount !== null}>
                          <span>Keyframes: {kfCount}</span>
                        </Show>
                      </>
                    );
                  })()}
                  <span>Frame: {currentFrame()} / {totalFrames()}</span>
                </div>
              </div>
            </div>
          </Show>
        </div>

      {/* ── Bottom status bar ── */}
      <div style={{
        background: "#16213e",
        "border-top": "1px solid #0f3460",
        padding: "10px 16px",
        display: "flex",
        gap: "12px",
        "align-items": "stretch",
        "min-height": "0",
      }}>
        <div style={{ flex: "1", display: "flex", "flex-direction": "column", gap: "4px", "min-height": "0" }}>
          <div style={{ flex: "1", display: "flex", gap: "6px", "align-items": "stretch", "min-height": "0" }}>
            <div
              style={{
                flex: "1",
                padding: "8px",
                background: "#0a0e1a",
                border: "1px solid #0f3460",
                "border-radius": "4px",
                "font-size": "12px",
                color: "#888",
                "min-height": "36px",
                "overflow-y": "auto",
                "white-space": "pre-wrap",
              }}
            >
              {status()}
            </div>
            <button
              type="button"
              onClick={() => setStatusLogOpen(true)}
              title="Open full status log"
              style={{
                padding: "6px 12px",
                background: "#0f3460",
                color: "#e2e2e2",
                border: "1px solid #1f5fa0",
                "border-radius": "4px",
                "font-size": "12px",
                cursor: "pointer",
                "white-space": "nowrap",
                "align-self": "stretch",
              }}
            >
              Log ({statusLog().length})
            </button>
          </div>
          {(() => {
            const g = gpuStatus();
            const pct = g && g.total > 0 ? Math.min(100, (g.used / g.total) * 100) : 0;
            const barColor = pct > 90 ? "#e94560" : pct > 70 ? "#f0ad4e" : "#2ecc71";
            return (
              <div
                style={{
                  display: "flex",
                  "align-items": "center",
                  "justify-content": "flex-end",
                  gap: "8px",
                  padding: "2px 4px",
                  "font-size": "11px",
                  color: "#888",
                  "font-family": "monospace",
                }}
                title="GPU memory (nvidia-smi, polled every 2s)"
              >
                <span>GPU:</span>
                <Show when={g} fallback={<span style={{ color: "#555" }}>n/a</span>}>
                  <div style={{
                    width: "120px", height: "6px", background: "#0a0e1a",
                    border: "1px solid #0f3460", "border-radius": "3px", overflow: "hidden",
                  }}>
                    <div style={{ width: `${pct}%`, height: "100%", background: barColor }} />
                  </div>
                  <span>{g!.used} / {g!.total} MiB</span>
                  <Show when={g!.util !== null}>
                    <span style={{ color: "#666" }}>· {g!.util}%</span>
                  </Show>
                </Show>
              </div>
            );
          })()}
        </div>
      </div>
      </div>

      {/* ── Status log popup ── */}
      <Show when={statusLogOpen()}>
        <div
          onClick={() => setStatusLogOpen(false)}
          style={{
            position: "fixed",
            inset: "0",
            background: "rgba(0,0,0,0.6)",
            display: "flex",
            "align-items": "center",
            "justify-content": "center",
            "z-index": "1000",
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: "min(900px, 90vw)",
              height: "min(700px, 85vh)",
              background: "#16213e",
              border: "1px solid #0f3460",
              "border-radius": "6px",
              display: "flex",
              "flex-direction": "column",
              overflow: "hidden",
            }}
          >
            <div style={{
              display: "flex",
              "align-items": "center",
              "justify-content": "space-between",
              padding: "10px 14px",
              "border-bottom": "1px solid #0f3460",
              background: "#1a2547",
            }}>
              <div style={{ "font-size": "14px", "font-weight": "600", color: "#e2e2e2" }}>
                Status log <span style={{ color: "#888", "font-weight": "400" }}>({statusLog().length} entries)</span>
              </div>
              <div style={{ display: "flex", gap: "6px" }}>
                <button
                  type="button"
                  onClick={copyStatusLog}
                  disabled={statusLog().length === 0}
                  style={{
                    padding: "5px 12px",
                    background: statusLogCopied() ? "#2ecc71" : "#0f3460",
                    color: "#e2e2e2",
                    border: "1px solid #1f5fa0",
                    "border-radius": "4px",
                    "font-size": "12px",
                    cursor: statusLog().length === 0 ? "not-allowed" : "pointer",
                    opacity: statusLog().length === 0 ? "0.5" : "1",
                  }}
                >
                  {statusLogCopied() ? "Copied!" : "Copy"}
                </button>
                <button
                  type="button"
                  onClick={clearStatusLog}
                  disabled={statusLog().length === 0}
                  title="Clear log"
                  style={{
                    padding: "5px 12px",
                    background: "#0f3460",
                    color: "#e2e2e2",
                    border: "1px solid #1f5fa0",
                    "border-radius": "4px",
                    "font-size": "12px",
                    cursor: statusLog().length === 0 ? "not-allowed" : "pointer",
                    opacity: statusLog().length === 0 ? "0.5" : "1",
                  }}
                >
                  Clear
                </button>
                <button
                  type="button"
                  onClick={() => setStatusLogOpen(false)}
                  style={{
                    padding: "5px 12px",
                    background: "#0f3460",
                    color: "#e2e2e2",
                    border: "1px solid #1f5fa0",
                    "border-radius": "4px",
                    "font-size": "12px",
                    cursor: "pointer",
                  }}
                >
                  Close
                </button>
              </div>
            </div>
            <div style={{
              flex: "1",
              overflow: "auto",
              padding: "10px 14px",
              background: "#0a0e1a",
              "font-family": "ui-monospace, SFMono-Regular, Menlo, monospace",
              "font-size": "12px",
              color: "#cfd6e4",
              "white-space": "pre-wrap",
              "word-break": "break-word",
            }}>
              <Show
                when={statusLog().length > 0}
                fallback={<div style={{ color: "#666", "font-style": "italic" }}>No status messages yet.</div>}
              >
                <For each={statusLog()}>
                  {(line) => <div>{line}</div>}
                </For>
              </Show>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
