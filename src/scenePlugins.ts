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
   * Subdirectory under `_scene/` holding per-frame NNNNNN.npz pointmaps.
   * Set iff features.pointmap is true. Passed to downstream scripts as
   * `--pointmap-dir`; if unset, pointmap-aware scripts skip pointmap I/O.
   */
  pointmapDir?: string;
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
  /**
   * If set, the runner script takes a "process exactly N frames" int and
   * the UI shows a "Target frames" input. Used by plugins that select a
   * fixed-cardinality frame set rather than a stride (VGGT, VGGT-Omega).
   * The number is the script's own default, kept in sync.
   */
  targetFramesDefault?: number;
  /**
   * If set, the UI shows an "Upscale" dropdown listing these float
   * multipliers. The runner gets `--upscale <factor>`. Recorded in
   * cameras.json as `upscale` so the dropdown can sync to what was
   * computed. Used by plugins whose output resolution scales with an
   * arbitrary factor (e.g. InfiniDepth's neural implicit field).
   */
  upscaleOptions?: number[];
  upscaleDefault?: number;
  /**
   * If true, this plugin consumes another scene plugin's `cameras.json` as
   * input. The UI shows a "Camera source" dropdown of ready scene plugins
   * (excluding this one); the dev server resolves the selection to the
   * absolute path of that plugin's cameras.json and appends it to the
   * runner's args as `--source-cameras-json <abs path>`. The selection is
   * sent in the prepare body as `options.cameraSource: "<pluginId>"`.
   */
  requiresCameraSource?: boolean;
  /**
   * Checks the backend runs to decide whether this plugin is installed
   * (i.e. its setup script has been run). Plugin is "available" iff every
   * listed file/dir path exists, every command resolves on PATH, and
   * every HF repo is present in the local HuggingFace cache. Unset =
   * always available (no install needed).
   */
  availability?: {
    /** Filesystem paths, relative to the repo root. */
    paths?: string[];
    /** Binary names that must resolve via $PATH. */
    commands?: string[];
    /** HuggingFace repo IDs ("org/name") that must be present in the HF cache. */
    hfRepos?: string[];
  };
}

export const SCENE_PLUGINS: ScenePlugin[] = [
  {
    id: "colmap",
    label: "COLMAP + DepthAnythingV2",
    camerasDir: "colmap",
    depthDir: "depthanythingv2",
    logFile: "prepare.log",
    readyMarkers: ["colmap/cameras.json", "depthanythingv2/meta.json"],
    pipeline: [
      { stage: "colmap", script: "run_colmap.py", args: ["$SCENE", "--max-size", "1920"] },
      { stage: "depth", script: "run_depth.py", args: ["$SCENE"] },
    ],
    requiresFrames: true,
    subsampleDefault: 3,
    subsampleFlag: "--every",
    subsampleScript: "run_colmap.py",
    availability: {
      commands: ["colmap"],
      hfRepos: ["depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"],
    },
  },
  {
    id: "cut3r",
    label: "CUT3R",
    camerasDir: "cut3r",
    depthDir: "cut3r/depth",
    pointmapDir: "cut3r/pointmap",
    logFile: "cut3r.log",
    readyMarkers: ["cut3r/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_cut3r.py", args: ["$SCENE"] },
    ],
    cleanDir: "cut3r",
    requiresFrames: true,
    features: { pointmap: true },
    subsampleDefault: 2,
    availability: {
      paths: ["models/external/cut3r/src/cut3r_512_dpt_4_64.pth"],
    },
  },
  {
    id: "vggt",
    label: "VGGT",
    camerasDir: "vggt",
    depthDir: "vggt/depth",
    pointmapDir: "vggt/pointmap",
    logFile: "vggt.log",
    readyMarkers: ["vggt/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_vggt.py", args: ["$SCENE", "--anchors-only"] },
    ],
    cleanDir: "vggt",
    requiresFrames: true,
    features: { pointmap: true },
    targetFramesDefault: 15,
    availability: {
      paths: ["models/external/vggt"],
      hfRepos: ["facebook/VGGT-1B"],
    },
  },
  {
    id: "vggtomega",
    label: "VGGT-Omega",
    camerasDir: "vggtomega",
    depthDir: "vggtomega/depth",
    pointmapDir: "vggtomega/pointmap",
    logFile: "vggtomega.log",
    readyMarkers: ["vggtomega/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_vggtomega.py", args: ["$SCENE"] },
    ],
    cleanDir: "vggtomega",
    requiresFrames: true,
    features: { pointmap: true },
    targetFramesDefault: 50,
    availability: {
      paths: ["models/external/vggt-omega"],
      hfRepos: ["facebook/VGGT-Omega"],
    },
  },
  {
    id: "da3",
    label: "Depth-Anything-3 Metric (Large)",
    camerasDir: "da3",
    depthDir: "da3/depth",
    pointmapDir: "da3/pointmap",
    logFile: "da3.log",
    readyMarkers: ["da3/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_da3.py", args: ["$SCENE"] },
    ],
    cleanDir: "da3",
    requiresFrames: true,
    features: { pointmap: true },
    subsampleDefault: 2,
    availability: {
      paths: ["models/external/depth-anything-3"],
    },
  },
  {
    id: "pi3",
    label: "Pi3",
    camerasDir: "pi3",
    depthDir: "pi3/depth",
    pointmapDir: "pi3/pointmap",
    logFile: "pi3.log",
    readyMarkers: ["pi3/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_pi3.py", args: ["$SCENE"] },
    ],
    cleanDir: "pi3",
    requiresFrames: true,
    features: { pointmap: true, scenePointmap: true },
    subsampleDefault: 2,
    availability: {
      hfRepos: ["yyfz233/Pi3X"],
    },
  },
  {
    id: "mapanything",
    label: "MapAnything",
    camerasDir: "mapanything",
    depthDir: "mapanything/depth",
    pointmapDir: "mapanything/pointmap",
    logFile: "mapanything.log",
    readyMarkers: ["mapanything/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_mapanything.py", args: ["$SCENE"] },
    ],
    cleanDir: "mapanything",
    requiresFrames: true,
    features: { pointmap: true, scenePointmap: true },
    subsampleDefault: 3,
    availability: {
      hfRepos: ["facebook/map-anything"],
    },
  },
  {
    id: "worldmirror2",
    label: "HunyuanWorld-Mirror 2.0",
    camerasDir: "worldmirror2",
    depthDir: "worldmirror2/depth",
    pointmapDir: "worldmirror2/pointmap",
    logFile: "worldmirror2.log",
    readyMarkers: ["worldmirror2/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_worldmirror2.py", args: ["$SCENE"] },
    ],
    cleanDir: "worldmirror2",
    requiresFrames: true,
    features: { pointmap: true, scenePointmap: true },
    subsampleDefault: 3,
    availability: {
      paths: ["models/external/hy-world-2.0"],
      hfRepos: ["tencent/HY-World-2.0"],
    },
  },
  {
    id: "worldmirror",
    label: "HunyuanWorld-Mirror",
    camerasDir: "worldmirror",
    depthDir: "worldmirror/depth",
    pointmapDir: "worldmirror/pointmap",
    logFile: "worldmirror.log",
    readyMarkers: ["worldmirror/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_worldmirror.py", args: ["$SCENE"] },
    ],
    cleanDir: "worldmirror",
    requiresFrames: true,
    features: { pointmap: true, scenePointmap: true },
    subsampleDefault: 3,
    availability: {
      paths: ["models/external/hunyuanworld-mirror"],
      hfRepos: ["tencent/HunyuanWorld-Mirror"],
    },
  },
  {
    id: "infinidepth",
    label: "InfiniDepth",
    camerasDir: "infinidepth",
    depthDir: "infinidepth/depth",
    pointmapDir: "infinidepth/pointmap",
    logFile: "infinidepth.log",
    readyMarkers: ["infinidepth/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_infinidepth.py", args: ["$SCENE"] },
    ],
    cleanDir: "infinidepth",
    requiresFrames: true,
    features: { pointmap: true },
    requiresCameraSource: true,
    upscaleOptions: [1, 1.5, 2],
    upscaleDefault: 1,
    availability: {
      paths: [
        "models/external/infinidepth",
        "models/weights/infinidepth/depth/infinidepth.ckpt",
      ],
    },
  },
  {
    id: "wilddet3d",
    label: "WildDet3D (depth + K)",
    camerasDir: "wilddet3d",
    depthDir: "wilddet3d/depth",
    logFile: "wilddet3d_scene.log",
    readyMarkers: ["wilddet3d/cameras.json"],
    pipeline: [
      { stage: "running", script: "run_wilddet3d_scene.py", args: ["$SCENE"] },
    ],
    cleanDir: "wilddet3d",
    requiresFrames: true,
    subsampleDefault: 10,
    availability: {
      paths: ["models/external/wilddet3d"],
    },
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
