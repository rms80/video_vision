import { createSignal, createEffect, For, Show, onMount, onCleanup, Switch, Match } from "solid-js";
import { parseNpz } from "./npz";
import type { CamerasJson } from "./depthMesh";
import ThreeDepthViewer, { type BoxerResult } from "./ThreeDepthViewer";
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
  const [status, setStatus] = createSignal("Drop a video file to begin");
  const [prompt, setPrompt] = createSignal("");
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
    job: { pluginId?: string; stage: string; running: boolean; error: string | null; cancelled?: boolean } | null;
  } | null>(null);
  // Which plugin (if any) we have locally initiated a prepare for. The
  // backend is authoritative via /api/scene/status.job.running, but we
  // optimistically set this to flip UI state immediately on click.
  const [preparingPluginId, setPreparingPluginId] = createSignal<string | null>(null);
  const savedSceneSource = localStorage.getItem("segviewer:sceneSource");
  const [sceneSource, setSceneSource] = createSignal<string>(
    savedSceneSource && SCENE_PLUGINS_BY_ID[savedSceneSource] ? savedSceneSource : DEFAULT_SCENE_PLUGIN_ID,
  );
  let scenePollTimer: number | undefined;
  type ViewTab = "source" | "depth" | "3d" | "3d-scene" | "3d-object";
  const storedTab = localStorage.getItem("segviewer:viewTab") as ViewTab | null;
  const [viewTab, setViewTab] = createSignal<ViewTab>(
    storedTab && ["source", "depth", "3d", "3d-scene", "3d-object"].includes(storedTab) ? storedTab : "source"
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

  // VGGT target total-frame count (shown only when VGGT is selected).
  // Sent to the backend as --num-frames; the backend picks exactly that
  // many evenly-spaced anchors. After a run, sync the input from
  // num_registered so switching to an existing analysis shows what was
  // computed; reset to 15 when no VGGT analysis exists for the video.
  const [vggtNumFrames, setVggtNumFrames] = createSignal<string>("15");
  createEffect(() => {
    if (sceneSource() !== "vggt") return;
    const cam = cameras();
    if (cam && typeof cam.num_registered === "number" && cam.num_registered > 0) {
      setVggtNumFrames(String(cam.num_registered));
    } else {
      setVggtNumFrames("15");
    }
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
      const res = await fetch("/api/object-pointmap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video, analysis, source }),
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
              const elapsed = ((s.job.finishedAt - s.job.startedAt) / 1000).toFixed(1);
              setStatus(`Object cloud built in ${elapsed}s`);
              setDataVersion((v) => v + 1);  // force viewer to refetch
            }
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
    // Find min/max for normalization (skip non-finite)
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (Number.isFinite(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
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
  }

  // Re-render depth when frame changes, tab switches, or canvas mounts
  createEffect(() => {
    const _frame = currentFrame();
    const _tab = viewTab();
    const _canvas = depthCanvas();
    if (_tab === "depth" && _canvas) renderDepthFrame();
  });

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
          setStatus(`${label}: ${data.job.stage}...`);
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
            setStatus("Scene prep complete.");
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
      if (pluginId === "vggt") {
        const n = Math.max(1, Math.round(Number(vggtNumFrames())));
        if (Number.isFinite(n)) options.numFrames = n;
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

  function handlePromptKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      // TODO: send prompt to backend
      setStatus(`Prompt: ${prompt()}`);
    }
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
    padding: "12px 16px",
    display: "flex",
    "align-items": "center",
    "justify-content": "space-between",
    "border-bottom": "1px solid #0f3460",
  } as const;

  const titleStyle = {
    "font-size": "14px",
    "font-weight": "700",
    "letter-spacing": "0.5px",
    color: "#e94560",
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
    <div style={{ display: "flex", "flex-direction": "column", width: "100%", height: "100%" }}>
      {/* Top row: sidebar + viewport */}
      <div style={{ display: "flex", flex: "1", "min-height": "0" }}>
        {/* ── Sidebar ── */}
        <div style={sidebarStyle}>
          <div style={headerStyle}>
            <span style={titleStyle}>Seg Viewer</span>
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
                3D Scene Analysis
              </div>
              <select
                value={sceneSource()}
                onChange={(e) => {
                  setSceneSource(e.currentTarget.value);
                  setBoxResult(null);
                  const v = videoName();
                  if (v) refreshDepthFrames(v);
                }}
                title="Pick which scene-reconstruction method produces camera poses + per-frame depth. COLMAP is classical SfM (slow, robust, geometry-only). The neural plugins (CUT3R, VGGT, Pi3, MapAnything, WorldMirror, DA3) infer poses + depth in one feed-forward pass — usually faster and don't depend on COLMAP. Pi3 / MapAnything / WorldMirror also produce a global scene pointmap for the 3D (Scene) tab. WildDet3D additionally runs 3D object detection."
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
                  cursor: "pointer",
                }}
              >
                <For each={SCENE_PLUGINS}>
                  {(p) => <option value={p.id}>{p.label}</option>}
                </For>
              </select>
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
              <Show when={sceneSource() === "vggt"}>
                <div
                  style={{ display: "flex", "align-items": "center", gap: "6px", "margin-bottom": "6px" }}
                  title="VGGT processes a fixed total number of anchor frames sampled evenly across the video (independent of subsample). Higher = better coverage / longer trajectory, but more VRAM."
                >
                  <label style={{ "font-size": "11px", color: "#aaa" }}>Target frames</label>
                  <input
                    type="number"
                    min="1"
                    step="1"
                    value={vggtNumFrames()}
                    onInput={(e) => setVggtNumFrames(e.currentTarget.value)}
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
                return (
                  <button
                    style={isRunning()
                      ? { ...cancellableRunningStyle(), width: "100%" }
                      : { ...accentBtnStyle(!!videoSrc(), isReady()), width: "100%" }}
                    onMouseEnter={() => setHovered(true)}
                    onMouseLeave={() => setHovered(false)}
                    onClick={() => (isRunning() ? cancelScene() : runScenePlugin(sceneSource()))}
                    disabled={!videoSrc()}
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
              <Show when={sceneStatus()?.job?.error}>
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
                title="Apply a similarity transform that puts the picked floor at y=0 and rescales depth to metric units. Edits the active plugin's cameras.json in place. Required before 3D box lifting and mesh reconstruction give meaningful real-world coordinates."
              >
                {aligning() ? "Aligning..." : isAligned() ? "Aligned" : "Align Scene"}
              </button>
            </div>

            {/* Object Segmentation */}
            <div style={{ padding: "12px 16px" }}>
              <div style={{ "font-size": "11px", "text-transform": "uppercase", "letter-spacing": "0.5px", color: "#888", "margin-bottom": "8px" }}>
                Object Segmentation
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
                      ...accentBtnStyle(!!seedPoint() && !detecting(), !!detection()),
                      width: "100%",
                      "margin-top": "6px",
                    }}
                title={detecting()
                  ? "Click to cancel SAM3 detection"
                  : detection()
                  ? `Last detection: ${detection()!.label} (conf ${detection()!.confidence.toFixed(2)}) bbox=[${detection()!.bbox.join(", ")}]. Click to re-run.`
                  : "Run SAM3 on frame 0 using the seed point + Object Label. Produces a 2D bounding box and segmentation mask, and creates a new analysis folder ('<label>_<N>') that holds every downstream artifact."}
                onClick={() => (detecting() ? cancelDetect() : detectObject())}
                disabled={!detecting() && !seedPoint()}
              >
                {detecting() ? "Detecting (Click to Cancel)" : "Detect Object In Frame 0 (SAM3)"}
              </button>
              <button
                style={tracking()
                  ? { ...cancellableRunningStyle(), width: "100%", "margin-top": "6px" }
                  : {
                      ...accentBtnStyle(!!currentAnalysis() && !tracking(), !!trackData()),
                      width: "100%",
                      "margin-top": "6px",
                    }}
                title={tracking()
                  ? "Click to cancel SAM2 tracking"
                  : trackData()
                  ? `Tracked across ${trackData()!.frames.length} frames. Click to re-run.`
                  : "Run SAM2 video tracking starting from the frame-0 detection mask. Produces a per-frame mask sequence (track.json) used by every downstream step — 3D box lifting, WildDet3D, and mesh reconstruction."}
                onClick={() => (tracking() ? cancelTrack() : trackThroughVideo())}
                disabled={!tracking() && !currentAnalysis()}
              >
                {tracking() ? "Tracking (Click to Cancel)" : "Track Through Video (SAM2)"}
              </button>
            </div>

            {/* Object Placement */}
            <div style={{ padding: "12px 16px" }}>
              <div style={{ "font-size": "11px", "text-transform": "uppercase", "letter-spacing": "0.5px", color: "#888", "margin-bottom": "8px" }}>
                Object Placement
              </div>
              <select
                value={boxSolverId()}
                onChange={(e) => {
                  const id = e.currentTarget.value;
                  setBoxSolverId(id);
                  // Refetch result for this analysis under the newly selected solver.
                  const v = videoName();
                  const a = currentAnalysis();
                  if (v && a) refreshBoxResult(v, a, id);
                  else setBoxResult(null);
                }}
                title="Pick which 3D-box solver runs when you click Compute Boxes. Boxer fits an oriented box from depth + masked point clouds (needs depth, supports per-frame or fused). WildDet3D is a neural detector that predicts 3D boxes directly from the image (no depth required, optionally takes K/depth as priors)."
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
                  cursor: "pointer",
                }}
              >
                <For each={BOX_SOLVER_PLUGINS}>
                  {(s) => <option value={s.id}>{s.label}</option>}
                </For>
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
                  if (!trackData() || boxRunning()) return false;
                  if (solver().requiresDepth && depthFrames().length === 0) return false;
                  return true;
                };
                const tip = () => {
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

        {/* ── Main Viewport ── */}
        <div
          style={{
            flex: "1",
            position: "relative",
            display: "flex",
            "flex-direction": "column",
            "min-width": "0",
          }}
        >
          {/* Tab bar */}
          <Show when={videoSrc()}>
            <div style={{ display: "flex", background: "#16213e", "border-bottom": "1px solid #0f3460" }}>
              <For each={[["source", "Source"], ["depth", "Depth"], ["3d", "3D (Per-Frame)"], ["3d-scene", "3D (Scene)"], ["3d-object", "3D (Object)"]] as [ViewTab, string][]}>
                {([id, label]) => (
                  <Show when={
                    (id !== "3d-scene" || !!SCENE_PLUGINS_BY_ID[sceneSource()]?.features?.scenePointmap)
                    && (id !== "3d-object" || objectPointmapReady())
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
                  onInput={(e) => seek(parseFloat(e.currentTarget.value))}
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
      </div>

      {/* ── Bottom prompt/status bar ── */}
      <div style={{
        background: "#16213e",
        "border-top": "1px solid #0f3460",
        padding: "10px 16px",
        display: "flex",
        gap: "12px",
        "align-items": "flex-start",
      }}>
        <textarea
          value={prompt()}
          onInput={(e) => setPrompt(e.currentTarget.value)}
          onKeyDown={handlePromptKeyDown}
          placeholder="Enter prompt... (Enter to send)"
          rows={2}
          style={{
            flex: "1",
            padding: "8px",
            background: "#0a0e1a",
            border: "1px solid #0f3460",
            color: "#e0e0e0",
            "border-radius": "4px",
            "font-size": "13px",
            "font-family": "inherit",
            resize: "vertical",
          }}
        />
        <div style={{ flex: "1", display: "flex", "flex-direction": "column", gap: "4px" }}>
          <div
            style={{
              padding: "8px",
              background: "#0a0e1a",
              border: "1px solid #0f3460",
              "border-radius": "4px",
              "font-size": "12px",
              color: "#888",
              "min-height": "36px",
              "max-height": "120px",
              "overflow-y": "auto",
              "white-space": "pre-wrap",
            }}
          >
            {status()}
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
  );
}
