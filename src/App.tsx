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
    job: { pluginId?: string; stage: string; running: boolean; error: string | null } | null;
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
  type ViewTab = "source" | "depth" | "3d" | "3d-scene";
  const storedTab = localStorage.getItem("segviewer:viewTab") as ViewTab | null;
  const [viewTab, setViewTab] = createSignal<ViewTab>(
    storedTab && ["source", "depth", "3d", "3d-scene"].includes(storedTab) ? storedTab : "source"
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
  const [boxerResult, setBoxerResult] = createSignal<BoxerResult | null>(null);
  const [boxerRunning, setBoxerRunning] = createSignal(false);
  const [boxerFuse, setBoxerFuse] = createSignal(true);
  const [wilddetResult, setWilddetResult] = createSignal<BoxerResult | null>(null);
  const [wilddetRunning, setWilddetRunning] = createSignal(false);
  const [wilddetUseIntrinsics, setWilddetUseIntrinsics] = createSignal(false);
  const [wilddetUseDepth, setWilddetUseDepth] = createSignal(false);
  const [reconstructRunning, setReconstructRunning] = createSignal(false);
  const [reconstructMesh, setReconstructMesh] = createSignal<string | null>(null);
  // Available .glb meshes for the current analysis (filenames like "000063.glb")
  const [reconstructMeshes, setReconstructMeshes] = createSignal<string[]>([]);
  const [selectedMesh, setSelectedMesh] = createSignal<string | null>(null);
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

  // Pi3 subsample rate (shown only when Pi3 is selected).
  const [pi3Subsample, setPi3Subsample] = createSignal<string>("2");
  // Restore from cameras.json whenever Pi3 is the active source and its
  // cameras load (refreshDepthFrames fetches them for the current plugin).
  createEffect(() => {
    if (sceneSource() !== "pi3") return;
    const cam = cameras();
    const n = cam?.subsample_every;
    if (typeof n === "number" && n > 0) setPi3Subsample(String(n));
  });

  // DA3 subsample rate (shown only when DA3 is selected).
  const [da3Subsample, setDa3Subsample] = createSignal<string>("2");
  createEffect(() => {
    if (sceneSource() !== "da3") return;
    const cam = cameras();
    const n = cam?.subsample_every;
    if (typeof n === "number" && n > 0) setDa3Subsample(String(n));
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
    setStatus(`Uploading and re-encoding ${file.name} for smooth scrubbing...`);
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
      setBoxerResult(null);
      setWilddetResult(null);
      // Try to load an existing track result for this analysis
      try {
        const tr = await fetch(`/api/track-result?video=${encodeURIComponent(video)}&name=${encodeURIComponent(name)}`);
        if (tr.ok) {
          const td = await tr.json();
          setTrackData({ imageWidth: td.image_width, imageHeight: td.image_height, frames: td.frames });
        }
      } catch {}
      // Try to load existing boxer result
      try {
        const br = await fetch(`/api/boxer-result?video=${encodeURIComponent(video)}&name=${encodeURIComponent(name)}`);
        if (br.ok) {
          setBoxerResult(await br.json());
        }
      } catch {}
      // Try to load existing wilddet3d result
      try {
        const wd = await fetch(`/api/wilddet3d-result?video=${encodeURIComponent(video)}&name=${encodeURIComponent(name)}`);
        if (wd.ok) {
          setWilddetResult(await wd.json());
        }
      } catch {}
      setReconstructMesh(null);
      setSelectedMesh(null);
      await refreshReconstructMeshes(video, name);
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
    setBoxerResult(null);
    setWilddetResult(null);
    setStatus(`Tracking through video with SAM2 (this can take a while)...`);
    try {
      const res = await fetch("/api/track", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video, analysis }),
      });
      const data = await res.json();
      if (data.error) { setStatus(`Tracking failed: ${data.error}`); return; }
      setTrackData({ imageWidth: data.image_width, imageHeight: data.image_height, frames: data.frames });
      setStatus(`Tracked ${data.frame_count} frames with ${data.model}`);
    } catch (err: any) {
      setStatus(`Tracking error: ${err.message}`);
    } finally {
      setTracking(false);
    }
  }

  async function runBoxer() {
    const video = videoName();
    const analysis = currentAnalysis();
    if (!video || !analysis || !trackData()) {
      setStatus("Run tracking first before running Boxer");
      return;
    }
    setBoxerRunning(true);
    setBoxerResult(null);
    setWilddetResult(null);
    setStatus("Running Boxer 3D bounding box lifting...");
    try {
      const det = detection();
      const res = await fetch("/api/boxer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video, analysis, label: det?.label ?? "object", fuse: boxerFuse(), source: sceneSource() }),
      });
      const data = await res.json();
      if (data.error) { setStatus(`Boxer failed: ${data.error}`); return; }
      // Poll for result
      setStatus("Boxer running — waiting for result...");
      const poll = setInterval(async () => {
        try {
          const r = await fetch(`/api/boxer-result?video=${encodeURIComponent(video)}&name=${encodeURIComponent(analysis)}`);
          if (r.ok) {
            clearInterval(poll);
            const result = await r.json();
            setBoxerResult(result);
            setBoxerRunning(false);
            setStatus(`Boxer: ${result.num_frames_with_boxes} frames with 3D boxes`);
          }
        } catch {}
      }, 2000);
      // Timeout after 10 minutes
      setTimeout(() => {
        clearInterval(poll);
        if (boxerRunning()) {
          setBoxerRunning(false);
          setStatus("Boxer timed out");
        }
      }, 600000);
    } catch (err: any) {
      setStatus(`Boxer error: ${err.message}`);
      setBoxerRunning(false);
    }
  }

  async function runWilddet() {
    const video = videoName();
    const analysis = currentAnalysis();
    if (!video || !analysis || !trackData()) {
      setStatus("Run tracking first before running WildDet3D");
      return;
    }
    setWilddetRunning(true);
    setBoxerResult(null);
    setWilddetResult(null);
    setStatus("Running WildDet3D 3D bounding box lifting...");
    try {
      const det = detection();
      const res = await fetch("/api/wilddet3d", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video, analysis,
          label: det?.label ?? "object",
          source: sceneSource(),
          useIntrinsics: wilddetUseIntrinsics(),
          useDepth: wilddetUseDepth(),
        }),
      });
      const data = await res.json();
      if (data.error) { setStatus(`WildDet3D failed: ${data.error}`); setWilddetRunning(false); return; }
      // Poll for result
      setStatus("WildDet3D running — waiting for result...");
      const poll = setInterval(async () => {
        try {
          const r = await fetch(`/api/wilddet3d-result?video=${encodeURIComponent(video)}&name=${encodeURIComponent(analysis)}`);
          if (r.ok) {
            clearInterval(poll);
            const result = await r.json();
            setWilddetResult(result);
            setWilddetRunning(false);
            setStatus(`WildDet3D: ${result.num_frames_with_boxes} frames with 3D boxes`);
          }
        } catch {}
      }, 3000);
      // Timeout after 10 minutes
      setTimeout(() => {
        clearInterval(poll);
        if (wilddetRunning()) {
          setWilddetRunning(false);
          setStatus("WildDet3D timed out");
        }
      }, 600000);
    } catch (err: any) {
      setStatus(`WildDet3D error: ${err.message}`);
      setWilddetRunning(false);
    }
  }

  function meshUrlFor(video: string, analysis: string, relPath: string): string {
    const stem = video.replace(/\.[^.]+$/, "");
    // relPath is "<modelDir>/<file>.glb" — encode each segment separately so the slash survives.
    const encoded = relPath.split("/").map(encodeURIComponent).join("/");
    return `/analysis/${encodeURIComponent(stem)}/${encodeURIComponent(analysis)}/${encoded}`;
  }

  async function refreshReconstructMeshes(video: string, analysis: string) {
    try {
      const r = await fetch(`/api/reconstruct3d-status?video=${encodeURIComponent(video)}&analysis=${encodeURIComponent(analysis)}`);
      if (!r.ok) { setReconstructMeshes([]); return; }
      const s = await r.json();
      const list: string[] = s.meshes ?? [];
      setReconstructMeshes(list);
      // If the current selection is gone, drop it; if no selection and list is non-empty, pick the latest.
      if (selectedMesh() && !list.includes(selectedMesh()!)) setSelectedMesh(null);
      if (!selectedMesh() && list.length > 0) setSelectedMesh(list[list.length - 1]);
    } catch { setReconstructMeshes([]); }
  }

  async function runReconstruct() {
    const video = videoName();
    const analysis = currentAnalysis();
    if (!video || !analysis || !trackData()) {
      setStatus("Run tracking first before reconstructing 3D mesh");
      return;
    }
    const plugin = SCENE_PLUGINS_BY_ID[sceneSource()];
    if (!plugin?.features?.pointmap) {
      setStatus(`Source '${plugin?.label ?? sceneSource()}' has no per-frame pointmap; switch to Pi3 or MapAnything`);
      return;
    }
    setReconstructRunning(true);
    setReconstructMesh(null);
    const frame = currentFrame();
    setStatus(`Reconstructing 3D mesh (Hunyuan3D-Omni) at frame ${frame}...`);
    try {
      const res = await fetch("/api/reconstruct3d", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video, analysis, frame, source: sceneSource() }),
      });
      const data = await res.json();
      if (data.error) {
        setStatus(`Reconstruct failed: ${data.error}`);
        setReconstructRunning(false);
        return;
      }
      const usedFrame = data.state?.frame ?? frame;
      if (usedFrame !== frame) {
        setStatus(`Frame ${frame} has no pointmap; snapped to ${usedFrame}`);
      }
      const stem = video.replace(/\.[^.]+$/, "");
      const meshPad = String(usedFrame).padStart(3, "0");
      const meshUrl = `/analysis/${encodeURIComponent(stem)}/${encodeURIComponent(analysis)}/hunyuan_omni/${meshPad}.glb`;
      const poll = setInterval(async () => {
        try {
          const r = await fetch(`/api/reconstruct3d-status?video=${encodeURIComponent(video)}&analysis=${encodeURIComponent(analysis)}`);
          if (!r.ok) return;
          const s = await r.json();
          if (s.job && !s.job.running) {
            clearInterval(poll);
            setReconstructRunning(false);
            if (s.job.error) {
              setStatus(`Reconstruct failed: ${s.job.error}`);
            } else {
              setReconstructMesh(meshUrl);
              const elapsed = ((s.job.finishedAt - s.job.startedAt) / 1000).toFixed(1);
              setStatus(`Reconstruct done in ${elapsed}s -> ${meshUrl}`);
              await refreshReconstructMeshes(video, analysis);
              setSelectedMesh(`hunyuan_omni/${meshPad}.glb`);
            }
          }
        } catch {}
      }, 3000);
      setTimeout(() => {
        clearInterval(poll);
        if (reconstructRunning()) {
          setReconstructRunning(false);
          setStatus("Reconstruct timed out");
        }
      }, 1200000);  // 20 min ceiling
    } catch (err: any) {
      setStatus(`Reconstruct error: ${err.message}`);
      setReconstructRunning(false);
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
    setBoxerResult(null);
    setWilddetResult(null);
    setReconstructMesh(null);
    setReconstructMeshes([]);
    setSelectedMesh(null);
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
    setBoxerResult(null);
    setWilddetResult(null);
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
    setBoxerResult(null);
    setWilddetResult(null);
    setStatus(`Starting ${plugin.label}...`);
    try {
      const options: Record<string, unknown> = {};
      if (pluginId === "pi3") {
        const n = Math.max(1, Math.round(Number(pi3Subsample())));
        if (Number.isFinite(n)) options.subsample = n;
      }
      if (pluginId === "da3") {
        const n = Math.max(1, Math.round(Number(da3Subsample())));
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
              <select
                value={videoName() ?? ""}
                onChange={(e) => {
                  const v = e.currentTarget.value;
                  if (v) loadVideo(v);
                }}
                style={{
                  width: "100%",
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
              <Show when={videoSrc()}>
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
                  style={{
                    width: "100%",
                    padding: "6px 8px",
                    "margin-top": "6px",
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
                  setBoxerResult(null);
                  setWilddetResult(null);
                  const v = videoName();
                  if (v) refreshDepthFrames(v);
                }}
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
              <Show when={sceneSource() === "pi3"}>
                <div style={{ display: "flex", "align-items": "center", gap: "6px", "margin-bottom": "6px" }}>
                  <label style={{ "font-size": "11px", color: "#aaa" }}>Subsample every</label>
                  <input
                    type="number"
                    min="1"
                    step="1"
                    value={pi3Subsample()}
                    onInput={(e) => setPi3Subsample(e.currentTarget.value)}
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
              <Show when={sceneSource() === "da3"}>
                <div style={{ display: "flex", "align-items": "center", gap: "6px", "margin-bottom": "6px" }}>
                  <label style={{ "font-size": "11px", color: "#aaa" }}>Subsample every</label>
                  <input
                    type="number"
                    min="1"
                    step="1"
                    value={da3Subsample()}
                    onInput={(e) => setDa3Subsample(e.currentTarget.value)}
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
                <div style={{ display: "flex", "align-items": "center", gap: "6px", "margin-bottom": "6px" }}>
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
                    style={{
                      ...accentBtnStyle(!!videoSrc() && !isRunning(), isReady()),
                      width: "100%",
                    }}
                    onMouseEnter={() => setHovered(true)}
                    onMouseLeave={() => setHovered(false)}
                    onClick={() => runScenePlugin(sceneSource())}
                    disabled={!videoSrc() || isRunning()}
                  >
                    {isRunning()
                      ? runningStageText()
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
              >
                {aligning() ? "Aligning..." : isAligned() ? "Aligned" : "Align Scene"}
              </button>
            </div>

            {/* Object Segmentation */}
            <div style={{ padding: "12px 16px" }}>
              <div style={{ "font-size": "11px", "text-transform": "uppercase", "letter-spacing": "0.5px", color: "#888", "margin-bottom": "8px" }}>
                Object Segmentation
              </div>
              <div style={{ display: "flex", "align-items": "center", gap: "6px", "margin-bottom": "8px" }}>
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
                title={seedPoint() ? `Seed: (${seedPoint()!.x}, ${seedPoint()!.y})` : "Click to set a seed point on frame 0"}
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
                style={{
                  ...accentBtnStyle(!!seedPoint() && !detecting(), !!detection()),
                  width: "100%",
                  "margin-top": "6px",
                }}
                title={detection() ? `${detection()!.label} (conf ${detection()!.confidence.toFixed(2)}) [${detection()!.bbox.join(", ")}]` : undefined}
                onClick={detectObject}
                disabled={!seedPoint() || detecting()}
              >
                {detecting() ? "Detecting..." : "Detect Object In Frame 0 (SAM3)"}
              </button>
              <button
                style={{
                  ...accentBtnStyle(!!currentAnalysis() && !tracking(), !!trackData()),
                  width: "100%",
                  "margin-top": "6px",
                }}
                title={trackData() ? `Tracked ${trackData()!.frames.length} frames` : undefined}
                onClick={trackThroughVideo}
                disabled={!currentAnalysis() || tracking()}
              >
                {tracking() ? "Tracking..." : "Track Through Video (SAM2)"}
              </button>
            </div>

            {/* Object Placement */}
            <div style={{ padding: "12px 16px" }}>
              <div style={{ "font-size": "11px", "text-transform": "uppercase", "letter-spacing": "0.5px", color: "#888", "margin-bottom": "8px" }}>
                Object Placement
              </div>
              <div style={{ display: "flex", gap: "4px" }}>
                <button
                  style={{
                    ...accentBtnStyle(!!trackData() && depthFrames().length > 0 && !boxerRunning(), !!boxerResult()),
                    flex: "1",
                  }}
                  title={boxerResult()?.num_frames_with_boxes ? `${boxerResult()!.num_frames_with_boxes} frames with 3D boxes` : undefined}
                  onClick={runBoxer}
                  disabled={!trackData() || depthFrames().length === 0 || boxerRunning()}
                >
                  {boxerRunning() ? "Running..." : "Lift to 3D (Boxer)"}
                </button>
                <button
                  style={{
                    ...btnStyle(true),
                    background: boxerFuse() ? "#2ecc71" : "#333",
                    color: boxerFuse() ? "#000" : "#777",
                    "font-weight": "600",
                    "font-size": "10px",
                    padding: "4px 8px",
                  }}
                  onClick={() => setBoxerFuse(!boxerFuse())}
                >
                  Fuse
                </button>
              </div>
              <div style={{ display: "flex", gap: "4px", "margin-top": "6px" }}>
                <button
                  style={{
                    ...accentBtnStyle(!!trackData() && !wilddetRunning(), !!wilddetResult()),
                    flex: "1",
                  }}
                  title={wilddetResult()?.num_frames_with_boxes ? `${wilddetResult()!.num_frames_with_boxes} frames with 3D boxes` : undefined}
                  onClick={runWilddet}
                  disabled={!trackData() || wilddetRunning()}
                >
                  {wilddetRunning() ? "Running..." : "Lift to 3D (WildDet3D)"}
                </button>
                <button
                  style={{
                    ...btnStyle(true),
                    background: wilddetUseIntrinsics() ? "#2ecc71" : "#333",
                    color: wilddetUseIntrinsics() ? "#000" : "#777",
                    "font-weight": "600",
                    "font-size": "10px",
                    padding: "4px 8px",
                  }}
                  title="Pass camera intrinsics to WildDet3D"
                  onClick={() => setWilddetUseIntrinsics(!wilddetUseIntrinsics())}
                >
                  K
                </button>
                <button
                  style={{
                    ...btnStyle(true),
                    background: wilddetUseDepth() ? "#2ecc71" : "#333",
                    color: wilddetUseDepth() ? "#000" : "#777",
                    "font-weight": "600",
                    "font-size": "10px",
                    padding: "4px 8px",
                  }}
                  title="Pass depth maps as input to WildDet3D"
                  onClick={() => setWilddetUseDepth(!wilddetUseDepth())}
                >
                  D
                </button>
              </div>
            </div>

            {/* Mesh Reconstruction */}
            <div style={{ padding: "12px 16px" }}>
              <div style={{ "font-size": "11px", "text-transform": "uppercase", "letter-spacing": "0.5px", color: "#888", "margin-bottom": "8px" }}>
                Mesh Reconstruction
              </div>
              {(() => {
                const plugin = () => SCENE_PLUGINS_BY_ID[sceneSource()];
                const hasPointmap = () => !!plugin()?.features?.pointmap;
                const enabled = () => !!trackData() && hasPointmap() && !reconstructRunning();
                const tip = () => {
                  if (!trackData()) return "Run tracking first";
                  if (!hasPointmap()) return `Source '${plugin()?.label ?? sceneSource()}' has no per-frame pointmap; switch to Pi3 or MapAnything`;
                  if (reconstructMesh()) return reconstructMesh() ?? undefined;
                  return `Reconstruct frame ${currentFrame()} with Hunyuan3D-Omni (~3-5 min on 12GB VRAM)`;
                };
                return (
                  <button
                    style={{
                      ...accentBtnStyle(enabled(), !!reconstructMesh()),
                      width: "100%",
                    }}
                    title={tip()}
                    onClick={runReconstruct}
                    disabled={!enabled()}
                  >
                    {reconstructRunning()
                      ? "Reconstructing..."
                      : reconstructMesh()
                        ? `Reconstructed frame ${currentFrame()}`
                        : `Reconstruct 3D (frame ${currentFrame()})`}
                  </button>
                );
              })()}
              <Show when={reconstructMeshes().length > 0}>
                <div style={{ display: "flex", gap: "4px", "margin-top": "6px" }}>
                  <select
                    value={selectedMesh() ?? ""}
                    onChange={(e) => setSelectedMesh(e.currentTarget.value || null)}
                    style={{
                      flex: "1",
                      padding: "5px 8px",
                      background: "#0a0e1a",
                      border: "1px solid #0f3460",
                      color: "#e0e0e0",
                      "border-radius": "3px",
                      "font-size": "12px",
                      "font-family": "inherit",
                      "min-width": "0",
                      cursor: "pointer",
                    }}
                  >
                    <For each={reconstructMeshes()}>
                      {(p) => <option value={p}>{p}</option>}
                    </For>
                  </select>
                  {(() => {
                    const v = videoName();
                    const a = currentAnalysis();
                    const m = selectedMesh();
                    const href = v && a && m ? meshUrlFor(v, a, m) : "#";
                    const downloadName = m ? m.split("/").pop() ?? m : "";
                    return (
                      <a
                        href={href}
                        download={downloadName}
                        style={{
                          ...accentBtnStyle(!!m),
                          padding: "5px 10px",
                          "text-decoration": "none",
                          display: "inline-flex",
                          "align-items": "center",
                          ...(m ? {} : { "pointer-events": "none", opacity: "0.5" }),
                        }}
                        title={m ? `Download ${m}` : "No mesh selected"}
                      >
                        Download
                      </a>
                    );
                  })()}
                </div>
              </Show>
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
              <For each={[["source", "Source"], ["depth", "Depth"], ["3d", "3D (Per-Frame)"], ["3d-scene", "3D (Scene)"]] as [ViewTab, string][]}>
                {([id, label]) => (
                  <Show when={id !== "3d-scene" || !!SCENE_PLUGINS_BY_ID[sceneSource()]?.features?.scenePointmap}>
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
              <Show when={viewTab() === "3d" || viewTab() === "3d-scene"}>
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
                visible={viewTab() === "3d" || viewTab() === "3d-scene"}
                boxerResult={boxerResult()}
                wilddetResult={wilddetResult()}
                sceneSource={sceneSource()}
                usePointmap={pointmapView()}
                scenePointmapMode={viewTab() === "3d-scene"}
                dataVersion={dataVersion()}
                showCameraPath={showCameraPath()}
                downsample={meshSubsample()}
                onReady={(actions) => { threeViewerActions = actions; }}
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
                {/* World-up point overlays */}
                <Show when={floorPoints().length > 0}>
                  <For each={floorPoints()}>
                    {(pt) => {
                      const pos = () => {
                        if (!videoEl) return null;
                        const rect = videoEl.getBoundingClientRect();
                        const scaleX = rect.width / videoEl.videoWidth;
                        const scaleY = rect.height / videoEl.videoHeight;
                        return { left: pt.x * scaleX, top: pt.y * scaleY };
                      };
                      return (
                        <Show when={pos()}>
                          {(p) => (
                            <div
                              style={{
                                position: "absolute",
                                left: `${p().left - 6}px`,
                                top: `${p().top - 6}px`,
                                width: "12px",
                                height: "12px",
                                "border-radius": "50%",
                                background: "rgba(52, 152, 219, 0.7)",
                                border: "2px solid #3498db",
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
                <span>Frame: {currentFrame()} / {totalFrames()}</span>
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
