// Scene-analysis plugin registry.
//
// Each entry describes one method for producing per-frame camera poses +
// depth (and optional extras like CUT3R pointmaps). Both the Vite-middleware
// backend and the Solid frontend import this file and drive their behavior
// off the registry, so adding a new method is a single-entry change here
// plus a python script under `scripts/`.

export interface PipelineStep {
  /** Human-readable stage name surfaced via /api/scene/status. */
  stage: string;
  /** Python script filename under segviewer/scripts/. */
  script: string;
  /**
   * Positional args for the script. The tokens "$VIDEO" and "$SCENE" are
   * replaced at dispatch time with the uploaded video path and the
   * per-video `_scene/` directory, respectively.
   */
  args: string[];
}

export interface ScenePlugin {
  id: string;
  label: string;
  /** Subdirectory under `_scene/` holding cameras.json for this method. */
  camerasDir: string;
  /** Subdirectory under `_scene/` holding per-frame NNNNNN.npz depth maps. */
  depthDir: string;
  /**
   * Value to pass as `--source` to downstream scripts (align_scene,
   * run_boxer, run_wilddet3d). `null` means omit the flag — those scripts
   * then fall back to their built-in default (COLMAP).
   */
  sourceFlag: string | null;
  /** Log filename under `_scene/` where pipeline stdout+stderr is appended. */
  logFile: string;
  /**
   * Paths (relative to `_scene/`) that must all exist for this plugin to
   * be considered "ready". Drives `/api/scene/status` artifacts.
   */
  readyMarkers: string[];
  /** Pipeline steps run sequentially when preparing this scene. */
  pipeline: PipelineStep[];
  /**
   * Subdirectory under `_scene/` to wipe before re-running. Only set on
   * plugins whose output lives in a single folder — COLMAP's pipeline
   * writes to several sibling dirs and handles its own cleanup.
   */
  cleanDir?: string;
  /**
   * Whether `frames.json` must already exist before this plugin can run
   * (true for plugins that consume extracted frames rather than the raw
   * video).
   */
  requiresFrames: boolean;
  features?: {
    /** Plugin publishes per-frame pointmap .npz files (for point-cloud view). */
    pointmap?: boolean;
    /** Plugin publishes a single scene_pointmap.npz with the global reconstruction. */
    scenePointmap?: boolean;
  };
  /**
   * If set, the runner script takes a "use every Nth frame" int and the
   * UI shows a "Subsample every N frames" input. The number is the
   * script's own default (kept in sync so the UI matches a no-options run).
   */
  subsampleDefault?: number;
  /**
   * CLI flag name for the subsample value. Defaults to `--subsample`;
   * COLMAP uses `--every`. Only consulted when subsampleDefault is set.
   */
  subsampleFlag?: string;
  /**
   * Restrict the subsample flag to a specific pipeline step (matched by
   * step.script). Required for multi-step plugins where only one step
   * accepts the flag (e.g. COLMAP's run_colmap.py). Unset = apply to
   * every step in the pipeline.
   */
  subsampleScript?: string;
}

export const SCENE_PLUGINS: ScenePlugin[] = [
  {
    id: "colmap",
    label: "COLMAP + DepthAnythingV2",
    camerasDir: "colmap",
    depthDir: "depthanythingv2",
    sourceFlag: null,
    logFile: "prepare.log",
    readyMarkers: ["colmap/cameras.json", "depthanythingv2/meta.json"],
    pipeline: [
      { stage: "frames", script: "extract_frames.py", args: ["$VIDEO", "$SCENE"] },
      { stage: "colmap", script: "run_colmap.py", args: ["$SCENE", "--max-size", "1920"] },
      { stage: "depth", script: "run_depth.py", args: ["$SCENE"] },
    ],
    requiresFrames: false,
    subsampleDefault: 3,
    subsampleFlag: "--every",
    subsampleScript: "run_colmap.py",
  },
  {
    id: "cut3r",
    label: "CUT3R",
    camerasDir: "cut3r",
    depthDir: "cut3r/depth",
    sourceFlag: "cut3r",
    logFile: "cut3r.log",
    readyMarkers: ["cut3r/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_cut3r.py", args: ["$SCENE"] },
    ],
    cleanDir: "cut3r",
    requiresFrames: true,
    features: { pointmap: true },
    subsampleDefault: 2,
  },
  {
    id: "vggt",
    label: "VGGT",
    camerasDir: "vggt",
    depthDir: "vggt/depth",
    sourceFlag: "vggt",
    logFile: "vggt.log",
    readyMarkers: ["vggt/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_vggt.py", args: ["$SCENE", "--anchors-only"] },
    ],
    cleanDir: "vggt",
    requiresFrames: true,
    features: { pointmap: true },
  },
  {
    id: "da3",
    label: "Depth-Anything-3 Metric (Large)",
    camerasDir: "da3",
    depthDir: "da3/depth",
    sourceFlag: "da3",
    logFile: "da3.log",
    readyMarkers: ["da3/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_da3.py", args: ["$SCENE"] },
    ],
    cleanDir: "da3",
    requiresFrames: true,
    features: { pointmap: true },
    subsampleDefault: 2,
  },
  {
    id: "pi3",
    label: "Pi3",
    camerasDir: "pi3",
    depthDir: "pi3/depth",
    sourceFlag: "pi3",
    logFile: "pi3.log",
    readyMarkers: ["pi3/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_pi3.py", args: ["$SCENE"] },
    ],
    cleanDir: "pi3",
    requiresFrames: true,
    features: { pointmap: true, scenePointmap: true },
    subsampleDefault: 2,
  },
  {
    id: "mapanything",
    label: "MapAnything",
    camerasDir: "mapanything",
    depthDir: "mapanything/depth",
    sourceFlag: "mapanything",
    logFile: "mapanything.log",
    readyMarkers: ["mapanything/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_mapanything.py", args: ["$SCENE"] },
    ],
    cleanDir: "mapanything",
    requiresFrames: true,
    features: { pointmap: true, scenePointmap: true },
    subsampleDefault: 3,
  },
  {
    id: "worldmirror2",
    label: "HunyuanWorld-Mirror 2.0",
    camerasDir: "worldmirror2",
    depthDir: "worldmirror2/depth",
    sourceFlag: "worldmirror2",
    logFile: "worldmirror2.log",
    readyMarkers: ["worldmirror2/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_worldmirror2.py", args: ["$SCENE"] },
    ],
    cleanDir: "worldmirror2",
    requiresFrames: true,
    features: { pointmap: true, scenePointmap: true },
    subsampleDefault: 3,
  },
  {
    id: "worldmirror",
    label: "HunyuanWorld-Mirror",
    camerasDir: "worldmirror",
    depthDir: "worldmirror/depth",
    sourceFlag: "worldmirror",
    logFile: "worldmirror.log",
    readyMarkers: ["worldmirror/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_worldmirror.py", args: ["$SCENE"] },
    ],
    cleanDir: "worldmirror",
    requiresFrames: true,
    features: { pointmap: true, scenePointmap: true },
    subsampleDefault: 3,
  },
  {
    id: "wilddet3d",
    label: "WildDet3D (depth + K)",
    camerasDir: "wilddet3d",
    depthDir: "wilddet3d/depth",
    sourceFlag: "wilddet3d",
    logFile: "wilddet3d_scene.log",
    readyMarkers: ["wilddet3d/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_wilddet3d_scene.py", args: ["$SCENE"] },
    ],
    cleanDir: "wilddet3d",
    requiresFrames: true,
    subsampleDefault: 10,
  },
];

export const SCENE_PLUGINS_BY_ID: Record<string, ScenePlugin> =
  Object.fromEntries(SCENE_PLUGINS.map((p) => [p.id, p]));

export const DEFAULT_SCENE_PLUGIN_ID = "colmap";

export function getScenePlugin(id: string | null | undefined): ScenePlugin | undefined {
  if (!id) return undefined;
  return SCENE_PLUGINS_BY_ID[id];
}

export function getScenePluginOrDefault(id: string | null | undefined): ScenePlugin {
  return getScenePlugin(id) ?? SCENE_PLUGINS_BY_ID[DEFAULT_SCENE_PLUGIN_ID];
}
