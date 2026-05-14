// Box-solver plugin registry.
//
// Each entry describes one method for lifting a tracked 2D bbox into 3D
// inside an object-level analysis. Both the Vite-middleware backend and
// the Solid frontend import this file and drive their behavior off the
// registry, so adding a new solver is a single-entry change here plus a
// python script under `scripts/`.
//
// Outputs land under `<analysis>/<plugin.subdir>/`, keyed by solver id,
// so multiple solvers can co-exist for the same analysis run.

export interface BoxSolverOption {
  /** Key sent in the `options` map of /api/box-solver. */
  key: string;
  /** Short label rendered next to the toggle. */
  label: string;
  /** Tooltip explaining the trade-off. */
  description: string;
  /** Initial value when no localStorage override is present. */
  defaultValue: boolean;
}

export interface BoxSolverPlugin {
  id: string;
  label: string;
  /** Subdir under the analysis run dir where this solver writes its outputs. */
  subdir: string;
  /** Python script under scripts/ that runs the solver. */
  script: string;
  /** Result filename inside `subdir`. */
  resultFile: string;
  /** Log filename inside `subdir`. */
  logFile: string;
  /** Whether this solver needs per-frame depth maps from the active scene plugin. */
  requiresDepth: boolean;
  /** Boolean toggles surfaced in the UI under the dropdown. */
  options: BoxSolverOption[];
  /**
   * Backend installation check. Same semantics as ScenePlugin.availability:
   * solver is "available" iff every path exists, every command resolves on
   * PATH, and every HF repo is present in the local HuggingFace cache.
   */
  availability?: {
    paths?: string[];
    commands?: string[];
    hfRepos?: string[];
  };
}

export const BOX_SOLVER_PLUGINS: BoxSolverPlugin[] = [
  {
    id: "boxer",
    label: "Boxer",
    subdir: "boxer",
    script: "run_boxer.py",
    resultFile: "boxes.json",
    logFile: "boxer.log",
    requiresDepth: true,
    options: [
      {
        key: "fuse",
        label: "Fuse",
        description:
          "Fuse all frames' masked point clouds into a single shared 3D box (more stable, slower). When off, fits an independent box per frame (jittery but tracks pose changes).",
        defaultValue: true,
      },
    ],
    availability: {
      paths: ["models/external/boxer/ckpts/boxernet_hw960in4x6d768-wssxpf9p.ckpt"],
    },
  },
  {
    id: "wilddet3d",
    label: "WildDet3D",
    subdir: "wilddet3d",
    script: "run_wilddet3d.py",
    resultFile: "boxes.json",
    logFile: "wilddet3d.log",
    requiresDepth: false,
    options: [
      {
        key: "useIntrinsics",
        label: "Use Cameras",
        description:
          "Pass camera intrinsics (K) from the active scene plugin to WildDet3D as a prior. Improves 3D box scale/orientation when poses are well-calibrated; turn off to let the model estimate K itself.",
        defaultValue: false,
      },
      {
        key: "useDepth",
        label: "Use Depth",
        description:
          "Pass per-frame depth maps from the active scene plugin to WildDet3D as a prior. Anchors box depth more reliably; turn off to let the model predict depth from the RGB alone.",
        defaultValue: false,
      },
    ],
    availability: {
      paths: ["models/external/wilddet3d"],
    },
  },
];

export const BOX_SOLVER_PLUGINS_BY_ID: Record<string, BoxSolverPlugin> =
  Object.fromEntries(BOX_SOLVER_PLUGINS.map((p) => [p.id, p]));

export const DEFAULT_BOX_SOLVER_ID = "boxer";

export function getBoxSolverPluginOrDefault(id: string | null | undefined): BoxSolverPlugin {
  if (id && BOX_SOLVER_PLUGINS_BY_ID[id]) return BOX_SOLVER_PLUGINS_BY_ID[id];
  return BOX_SOLVER_PLUGINS_BY_ID[DEFAULT_BOX_SOLVER_ID];
}
