import { defineConfig, Plugin } from "vite";
import solidPlugin from "vite-plugin-solid";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn, type ChildProcess } from "node:child_process";
import ffmpegPath from "ffmpeg-static";
import {
  SCENE_PLUGINS,
  getScenePluginOrDefault,
  DEFAULT_SCENE_PLUGIN_ID,
  type ScenePlugin,
  type PipelineStep,
} from "./src/scenePlugins";
import {
  BOX_SOLVER_PLUGINS,
  BOX_SOLVER_PLUGINS_BY_ID,
} from "./src/boxSolverPlugins";

const UPLOADS_DIR = path.resolve(__dirname, "uploads");
const TMP_DIR = path.resolve(__dirname, "tmp");
const ANALYSIS_DIR = path.resolve(__dirname, "analysis");
const SCRIPTS_DIR = path.resolve(__dirname, "scripts");
// Project-level venv with CUDA-enabled torch. On Windows the launcher lives
// at <venv>/Scripts/python.exe; on Linux/macOS it's <venv>/bin/python.
const VENV_PYTHON = path.resolve(
  __dirname,
  "models",
  ".venv",
  process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
);

type SceneJobState = {
  video: string;
  pluginId: string;
  stage: string;
  running: boolean;
  error: string | null;
  cancelled?: boolean;
  startedAt: number;
  finishedAt?: number;
  // Last `[progress] ...` line printed by the currently-running script.
  progress?: string | null;
};
const sceneJobs = new Map<string, SceneJobState>(); // key: video filename
// Tracks the *currently executing* python subprocess for a scene-prep job so
// the DELETE handler can kill mid-pipeline. The pipeline runs steps
// sequentially (e.g. COLMAP then depth), so only one entry per video is live
// at a time; the run loop swaps it out per step.
const sceneActiveJobs = new Map<string, CancellableJob>(); // key: video filename

type ObjectPointmapJobState = {
  running: boolean;
  source: string;
  startedAt: number;
  finishedAt?: number;
  error: string | null;
  progress?: string | null;
};
// key: `${video}::${analysis}::${source}` — one job per (analysis, source) pair
const objectPointmapJobs = new Map<string, ObjectPointmapJobState>();

type DinoV3JobState = {
  video: string;
  running: boolean;
  error: string | null;
  cancelled?: boolean;
  startedAt: number;
  finishedAt?: number;
  progress?: string | null;
  // Echo of the inputs so the UI can tell whether on-disk meta matches the
  // last request even before the run completes.
  subsample: number;
  scaling: number;
};
const dinov3Jobs = new Map<string, DinoV3JobState>();           // key: video filename
const dinov3ActiveJobs = new Map<string, CancellableJob>();    // key: video filename

// Live child processes that support cancellation. The cancel endpoint
// looks up the entry by key, sets `killed=true`, and tears down the
// process tree; the spawning request's await rejects, and the killed
// flag lets that handler downgrade the resulting non-zero exit into a
// clean log line and a "cancelled" response instead of "error".
type CancellableJob = { proc: ChildProcess; killed: boolean };
const boxSolverJobs = new Map<string, CancellableJob>();   // `${video}::${analysis}::${solverId}`
const detectJobs    = new Map<string, CancellableJob>();   // video filename
const trackJobs     = new Map<string, CancellableJob>();   // `${video}::${analysis}`

/**
 * Kill a child process tree. On Windows the python interpreter forks
 * worker subprocesses (CUDA, dataloader workers) and `proc.kill()` only
 * targets the direct child, leaving the workers orphaned and continuing
 * to chew GPU memory. `taskkill /F /T /PID` walks the tree.
 *
 * Resolves once taskkill (or proc.kill) has issued the termination —
 * NOT once the child has finished exiting. Callers that need to know
 * the child is gone should also await the spawn's close event.
 */
function killProcessTree(proc: ChildProcess, tag = ""): Promise<void> {
  return new Promise((resolve) => {
    if (!proc.pid) { console.warn(`[kill${tag}] no pid for child process`); resolve(); return; }
    if (process.platform === "win32") {
      const tk = spawn("taskkill", ["/F", "/T", "/PID", String(proc.pid)]);
      let stderr = "";
      let stdout = "";
      tk.stdout.on("data", (c) => (stdout += c.toString()));
      tk.stderr.on("data", (c) => (stderr += c.toString()));
      tk.on("close", (code) => {
        const msg = (stdout.trim() || stderr.trim()).replace(/\s+/g, " ");
        if (code === 0) console.log(`[kill${tag}] taskkill pid=${proc.pid}: ${msg}`);
        else console.warn(`[kill${tag}] taskkill pid=${proc.pid} exit=${code}: ${msg}`);
        resolve();
      });
      tk.on("error", (err) => {
        console.warn(`[kill${tag}] taskkill spawn error: ${err.message}; falling back to proc.kill()`);
        try { proc.kill(); } catch {}
        resolve();
      });
    } else {
      try { proc.kill("SIGTERM"); } catch (e) { console.warn(`[kill${tag}] kill failed:`, e); }
      resolve();
    }
  });
}

/**
 * Cancel a registered job: mark killed, fire the tree-kill, await both
 * the kill issuance and the child's actual exit before returning. Returns
 * true if the child exited within `timeoutMs`, false otherwise.
 */
async function cancelJob(job: CancellableJob, tag: string, timeoutMs = 8000): Promise<boolean> {
  job.killed = true;
  await killProcessTree(job.proc, tag);
  if (job.proc.exitCode !== null || job.proc.signalCode !== null) return true;
  return await new Promise<boolean>((resolve) => {
    let done = false;
    const onClose = () => { if (!done) { done = true; resolve(true); } };
    job.proc.once("close", onClose);
    setTimeout(() => {
      if (!done) {
        done = true;
        job.proc.off("close", onClose);
        console.warn(`[kill${tag}] pid=${job.proc.pid} did not exit within ${timeoutMs}ms`);
        resolve(false);
      }
    }, timeoutMs);
  });
}

const OBJECT_POINTMAP_DIR = "object_pointmap";

/**
 * The object cloud is sharded by build_object_pointmap.py into
 * <source>_chunks.json + <source>_NNN.npz. The .npz path passed to that
 * script is just used as the basename ("<source>"); the manifest is what
 * the client fetches first and what we use as the readiness marker.
 */
function objectPointmapBasePath(video: string, analysis: string): string {
  const stem = path.basename(video, path.extname(video));
  return path.join(ANALYSIS_DIR, stem, analysis, OBJECT_POINTMAP_DIR);
}

function objectPointmapOutArg(video: string, analysis: string, source: string): string {
  // Python derives the chunk basename from `args.out.stem`.
  return path.join(objectPointmapBasePath(video, analysis), `${source}.npz`);
}

function objectPointmapManifestPath(video: string, analysis: string, source: string): string {
  return path.join(objectPointmapBasePath(video, analysis), `${source}_chunks.json`);
}

function deleteObjectPointmapChunks(dir: string, source: string) {
  if (!fs.existsSync(dir)) return;
  const manifest = path.join(dir, `${source}_chunks.json`);
  if (fs.existsSync(manifest)) fs.rmSync(manifest, { force: true });
  const chunkRe = new RegExp(`^${source.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}_\\d{3}\\.npz$`);
  for (const name of fs.readdirSync(dir)) {
    if (chunkRe.test(name)) fs.rmSync(path.join(dir, name), { force: true });
  }
}

/**
 * Wipe any object-point-cloud files that depended on the given source's
 * cameras / depth (because that source was just re-prepared or aligned).
 * Other sources' files in the same analysis dir are untouched.
 */
function deleteObjectPointmapsForSource(video: string, source: string) {
  const stem = path.basename(video, path.extname(video));
  const videoDir = path.join(ANALYSIS_DIR, stem);
  if (!fs.existsSync(videoDir)) return;
  for (const name of fs.readdirSync(videoDir)) {
    if (name.startsWith("_")) continue;
    const dir = path.join(videoDir, name, OBJECT_POINTMAP_DIR);
    deleteObjectPointmapChunks(dir, source);
  }
}

function sceneDir(video: string) {
  const stem = path.basename(video, path.extname(video));
  return path.join(ANALYSIS_DIR, stem, "_scene");
}

function deleteBoxSolverResults(video: string) {
  const stem = path.basename(video, path.extname(video));
  const videoDir = path.join(ANALYSIS_DIR, stem);
  if (!fs.existsSync(videoDir)) return;
  for (const name of fs.readdirSync(videoDir)) {
    if (name.startsWith("_")) continue;
    for (const solver of BOX_SOLVER_PLUGINS) {
      const solverDir = path.join(videoDir, name, solver.subdir);
      if (fs.existsSync(solverDir)) {
        fs.rmSync(solverDir, { recursive: true, force: true });
        console.log(`[box-solver] deleted ${solverDir}`);
      }
    }
  }
}

function sceneArtifactState(video: string): Record<string, boolean> {
  const sd = sceneDir(video);
  const artifacts: Record<string, boolean> = {
    frames: fs.existsSync(path.join(sd, "frames.json")),
  };
  for (const plugin of SCENE_PLUGINS) {
    artifacts[plugin.id] = plugin.readyMarkers.every((m) =>
      fs.existsSync(path.join(sd, m)),
    );
  }
  return artifacts;
}

/**
 * Resolve the source=... query/body param sent by downstream endpoints
 * (align, boxer, wilddet3d) into a scene plugin. Unknown values fall
 * back to the default (COLMAP) for backwards compatibility.
 */
function resolveSourcePlugin(source: unknown): ScenePlugin {
  return getScenePluginOrDefault(typeof source === "string" ? source : null);
}

/**
 * Build the per-scene-plugin path args every downstream consumer script
 * needs: where to find cameras.json, depth maps, and (optionally) per-frame
 * pointmaps. Paths are scene-relative; consumers join them against the
 * scene dir. Centralizing here means adding a new plugin doesn't require
 * touching any script's hardcoded {id -> subdir} table.
 */
function scenePathArgs(plugin: ScenePlugin): string[] {
  const args = ["--cameras-dir", plugin.camerasDir, "--depth-dir", plugin.depthDir];
  if (plugin.pointmapDir) args.push("--pointmap-dir", plugin.pointmapDir);
  return args;
}

// Lines prefixed with this marker are surfaced live to the UI status bar
// (see scripts/_progress.py). The marker is stripped before the message
// is stored on job state, but the full line is still written to the log.
const PROGRESS_PREFIX = "[progress] ";
const PROGRESS_PREFIX_BUF = Buffer.from(PROGRESS_PREFIX, "utf8");
// Cap on the trailing partial-line buffer. A real progress line is short
// (well under 1 KiB); if a chunk arrives without a newline and the tail
// grows past this, drop the oldest bytes. Protects against tqdm-style
// `\r`-only progress bars or any other unterminated stdout from
// accumulating unboundedly across a long-running job.
const PROGRESS_TAIL_CAP = 64 * 1024;

function runPython(
  script: string,
  args: string[],
  logPath: string,
  pythonExe: string = VENV_PYTHON,
  onSpawn?: (proc: ChildProcess) => void,
  onProgress?: (msg: string) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const fd = fs.openSync(logPath, "a");
    fs.writeSync(fd, `\n===== ${new Date().toISOString()} ${pythonExe} ${script} ${args.join(" ")} =====\n`);
    // Spawn failures (e.g. ENOENT for a missing venv) emit BOTH 'error' and 'close',
    // so guard against double-settle and double-close — otherwise closeSync on a
    // freed fd throws EBADF and crashes the dev server.
    let settled = false;
    const finish = (fn: () => void) => {
      if (settled) return;
      settled = true;
      try { fs.closeSync(fd); } catch {}
      fn();
    };
    // Strip Python-related env vars inherited from the parent shell so the
    // spawned interpreter resolves its standard library and DLLs against
    // its own venv. A stale VIRTUAL_ENV / PYTHONHOME from the launching
    // shell can otherwise cause STATUS_DLL_INIT_FAILED (0xC0000142) on
    // Windows when the wrong pythonXY.dll is found first.
    // Force HF cache to be treated as offline-only at runtime: setup scripts
    // populated the cache, the runner just needs to read it. This avoids a
    // Windows-specific `socket.getaddrinfo` access violation seen on some
    // machines when transformers/huggingface_hub does its HEAD-revalidate
    // call. Safe because every model the runners use is preloaded by the
    // matching setup/plugin_*.py script.
    const cleanEnv: NodeJS.ProcessEnv = {
      ...process.env,
      PYTHONIOENCODING: "utf-8",
      HF_HUB_OFFLINE: "1",
      TRANSFORMERS_OFFLINE: "1",
    };
    delete cleanEnv.VIRTUAL_ENV;
    delete cleanEnv.PYTHONHOME;
    delete cleanEnv.PYTHONPATH;
    delete cleanEnv.PYTHONSTARTUP;
    const py = spawn(pythonExe, [script, ...args], { stdio: ["ignore", "pipe", "pipe"], env: cleanEnv });
    onSpawn?.(py);
    // Stdout: write every byte to the log fd verbatim. If onProgress is
    // wired, *also* scan for newline-delimited `[progress] ...` lines —
    // but on raw Buffers, without decoding the whole chunk or doing
    // string concat. Decoding only happens for lines that actually start
    // with the marker, which keeps verbose scripts cheap on the Node
    // side. The trailing partial-line buffer is capped (PROGRESS_TAIL_CAP)
    // so a chunk without `\n` (e.g. a tqdm `\r` bar) can't grow forever.
    let progressTail: Buffer = Buffer.alloc(0);
    py.stdout.on("data", (c: Buffer) => {
      try { fs.writeSync(fd, c); } catch {}
      if (!onProgress) return;
      // Concat the leftover partial line with the new chunk. `progressTail`
      // is typically a small subarray view from the previous chunk, so
      // this concat is bounded.
      const buf = progressTail.length === 0 ? c : Buffer.concat([progressTail, c]);
      let cursor = 0;
      while (true) {
        const nl = buf.indexOf(0x0a, cursor); // '\n'
        if (nl < 0) break;
        // Line bytes are [cursor, lineEnd); strip trailing '\r' if any.
        const lineEnd = nl > cursor && buf[nl - 1] === 0x0d ? nl - 1 : nl;
        const lineLen = lineEnd - cursor;
        if (lineLen >= PROGRESS_PREFIX_BUF.length &&
            buf.compare(
              PROGRESS_PREFIX_BUF, 0, PROGRESS_PREFIX_BUF.length,
              cursor, cursor + PROGRESS_PREFIX_BUF.length,
            ) === 0) {
          onProgress(buf.toString("utf8",
            cursor + PROGRESS_PREFIX_BUF.length, lineEnd));
        }
        cursor = nl + 1;
      }
      if (cursor >= buf.length) {
        progressTail = Buffer.alloc(0);
      } else if (buf.length - cursor > PROGRESS_TAIL_CAP) {
        // Trailing partial line is longer than any plausible progress
        // line — drop the oldest bytes so we don't accumulate forever.
        progressTail = buf.subarray(buf.length - PROGRESS_TAIL_CAP);
      } else {
        progressTail = buf.subarray(cursor);
      }
    });
    py.stderr.on("data", (c: Buffer) => { try { fs.writeSync(fd, c); } catch {} });
    py.on("close", (code) => finish(() => {
      if (code === 0) resolve();
      else reject(new Error(`${path.basename(script)} exit ${code} (see ${logPath})`));
    }));
    py.on("error", (err: NodeJS.ErrnoException) => finish(() => {
      const msg = err?.code === "ENOENT"
        ? `python not found at ${pythonExe} — check the venv setup`
        : err?.message ?? String(err);
      reject(new Error(msg));
    }));
  });
}

function hfCacheRoot(): string {
  if (process.env.HF_HUB_CACHE) return process.env.HF_HUB_CACHE;
  if (process.env.HF_HOME) return path.join(process.env.HF_HOME, "hub");
  return path.join(os.homedir(), ".cache", "huggingface", "hub");
}

function existsOnPath(cmd: string): boolean {
  const dirs = (process.env.PATH ?? "").split(path.delimiter);
  const exts = process.platform === "win32"
    ? (process.env.PATHEXT ?? ".EXE;.CMD;.BAT").split(";")
    : [""];
  for (const d of dirs) {
    if (!d) continue;
    for (const ext of exts) {
      try { if (fs.existsSync(path.join(d, cmd + ext))) return true; } catch {}
    }
  }
  return false;
}

type AvailabilitySpec = {
  paths?: string[];
  commands?: string[];
  hfRepos?: string[];
};

function isAvailable(a: AvailabilitySpec | undefined): boolean {
  if (!a) return true;
  for (const p of a.paths ?? []) {
    if (!fs.existsSync(path.resolve(__dirname, p))) return false;
  }
  for (const cmd of a.commands ?? []) {
    if (existsOnPath(cmd)) continue;
    // Windows: setup scripts install bundled standalone builds under
    // models/tools/<cmd>/ (e.g. models/tools/colmap/COLMAP.bat). The
    // runner scripts resolve those paths directly without touching PATH.
    if (process.platform === "win32") {
      const bundled = path.resolve(__dirname, "models", "tools", cmd);
      if (fs.existsSync(bundled)) continue;
    }
    return false;
  }
  if (a.hfRepos?.length) {
    const cache = hfCacheRoot();
    for (const repo of a.hfRepos) {
      const dir = path.join(cache, "models--" + repo.replace(/\//g, "--"));
      if (!fs.existsSync(dir)) return false;
    }
  }
  return true;
}

const SAM2_WEIGHTS = path.resolve(__dirname, "models", "weights", "sam2.1_l.pt");
const SAM3_WEIGHTS = path.resolve(__dirname, "models", "weights", "sam3.pt");

function segViewerPlugin(): Plugin {
  return {
    name: "seg-viewer",
    configureServer(server) {
      // Ensure uploads/tmp directories exist
      if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });
      if (!fs.existsSync(TMP_DIR)) fs.mkdirSync(TMP_DIR, { recursive: true });

      // GET /api/gpu-status — current used/total GPU memory (MiB) via nvidia-smi.
      // Returns { used, total, util } or { error } if nvidia-smi is unavailable.
      //
      // nvidia-smi can take multiple seconds on WSL2 (interop bridge). The
      // client polls every 2s, so without coalescing we'd accumulate
      // concurrent nvidia-smi spawns whenever one call ran longer than the
      // poll interval — each spawn compounds CPU on the bridge and makes
      // the next one even slower. Two defenses:
      //   1. In-flight coalescing: while one nvidia-smi is running, every
      //      additional caller awaits the *same* promise. Only one process
      //      ever runs at a time.
      //   2. Short freshness cache: if we have a result newer than
      //      GPU_CACHE_TTL_MS, serve it directly and skip the spawn.
      type GpuStatus =
        | { used: number; total: number; util: number | null }
        | { error: string };
      const GPU_CACHE_TTL_MS = 1500;
      let gpuInFlight: Promise<GpuStatus> | null = null;
      let gpuLast: { ts: number; value: GpuStatus } | null = null;

      const fetchGpuStatus = (): Promise<GpuStatus> => {
        if (gpuInFlight) return gpuInFlight;
        if (gpuLast && Date.now() - gpuLast.ts < GPU_CACHE_TTL_MS) {
          return Promise.resolve(gpuLast.value);
        }
        gpuInFlight = new Promise<GpuStatus>((resolve) => {
          const proc = spawn("nvidia-smi", [
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
          ]);
          let out = "";
          let settled = false;
          const done = (value: GpuStatus) => {
            if (settled) return;
            settled = true;
            resolve(value);
          };
          proc.stdout.on("data", (c: Buffer) => (out += c.toString()));
          proc.on("error", (err) => done({ error: err.message }));
          proc.on("close", () => {
            const line = out.split(/\r?\n/).find((l) => l.trim()) ?? "";
            const [u, t, g] = line.split(",").map((s) => Number(s.trim()));
            if (!Number.isFinite(u) || !Number.isFinite(t)) {
              done({ error: "Could not parse nvidia-smi output" });
              return;
            }
            done({ used: u, total: t, util: Number.isFinite(g) ? g : null });
          });
        }).then((value) => {
          gpuLast = { ts: Date.now(), value };
          gpuInFlight = null;
          return value;
        });
        return gpuInFlight;
      };

      server.middlewares.use("/api/gpu-status", (req, res, next) => {
        if (req.method !== "GET") return next();
        fetchGpuStatus().then((value) => {
          res.setHeader("Content-Type", "application/json");
          if ("error" in value) res.statusCode = 500;
          res.end(JSON.stringify(value));
        });
      });

      // GET /api/availability — which models/plugins have their setup installed.
      // Drives whether each scene plugin / box solver appears in the UI and
      // whether the SAM2/SAM3 buttons are enabled.
      server.middlewares.use("/api/availability", (req, res, next) => {
        if (req.method !== "GET") return next();
        const scenePlugins: Record<string, boolean> = {};
        for (const p of SCENE_PLUGINS) scenePlugins[p.id] = isAvailable(p.availability);
        const boxSolvers: Record<string, boolean> = {};
        for (const p of BOX_SOLVER_PLUGINS) boxSolvers[p.id] = isAvailable(p.availability);
        const dinov3 = isAvailable({
          paths: ["models/external/dinov3"],
          hfRepos: ["facebook/dinov3-vitl16-pretrain-lvd1689m"],
        });
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({
          scenePlugins,
          boxSolvers,
          sam2: fs.existsSync(SAM2_WEIGHTS),
          sam3: fs.existsSync(SAM3_WEIGHTS),
          dinov3,
        }));
      });

      // POST /api/upload — accept a video file upload (skips if same name+size exists)
      server.middlewares.use("/api/upload", (req, res, next) => {
        if (req.method !== "POST") return next();
        const filename = decodeURIComponent(req.headers["x-filename"] as string || "video.mp4");
        const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, "_");
        const filepath = path.join(UPLOADS_DIR, safeName);

        // Duplicate detection: same filename → skip upload
        if (fs.existsSync(filepath)) {
          req.resume();
          req.on("end", () => {
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ ok: true, filename: safeName, skipped: true }));
          });
          return;
        }

        const rawPath = path.join(TMP_DIR, `raw_${safeName}`);
        const ws = fs.createWriteStream(rawPath);
        req.pipe(ws);
        let responded = false;
        const respond = (status: number, body: object) => {
          if (responded) return;
          responded = true;
          res.statusCode = status;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(body));
        };
        ws.on("finish", () => {
          // Re-encode with keyframes every half second (-g 15 at 30fps)
          console.log(`[upload] re-encoding ${safeName} with half-second keyframes...`);
          const ff = spawn(ffmpegPath!, [
            "-y", "-i", rawPath,
            "-g", "15", "-c:v", "libx264", "-crf", "18", "-c:a", "copy",
            filepath,
          ]);
          let stderr = "";
          ff.stderr.on("data", (c: Buffer) => (stderr += c.toString()));
          ff.on("close", async (code) => {
            try { fs.unlinkSync(rawPath); } catch {}
            if (responded) return;
            if (code !== 0) {
              console.error(`[upload] ffmpeg error: ${stderr}`);
              respond(500, { error: `Re-encode failed: ${stderr.slice(-200)}` });
              return;
            }
            console.log(`[upload] re-encoded ${safeName} successfully`);
            // Extract frames now so downstream plugins (CUT3R, VGGT, Pi3, ...)
            // can run without first having to run COLMAP. Best-effort: if it
            // fails, /api/scene/prepare will retry on demand.
            const sd = sceneDir(safeName);
            fs.mkdirSync(sd, { recursive: true });
            console.log(`[upload] extracting frames for ${safeName}...`);
            try {
              await runPython(
                path.join(SCRIPTS_DIR, "extract_frames.py"),
                [filepath, sd],
                path.join(sd, "extract_frames.log"),
              );
              console.log(`[upload] frames extracted for ${safeName}`);
              respond(200, { ok: true, filename: safeName });
            } catch (err: any) {
              console.error(`[upload] extract_frames failed: ${err?.message ?? err}`);
              respond(200, { ok: true, filename: safeName, framesExtracted: false });
            }
          });
          ff.on("error", (err) => {
            try { fs.unlinkSync(rawPath); } catch {}
            respond(500, { error: err.message });
          });
        });
        ws.on("error", (err) => {
          respond(500, { error: err.message });
        });
      });

      // POST   /api/scene/prepare — run the pipeline defined by a scene plugin
      //   body: { video, pluginId? }  (pluginId defaults to COLMAP for backcompat)
      // DELETE /api/scene/prepare?video= — kill the running pipeline mid-step.
      //   The currently executing python subprocess is tree-killed; the run
      //   loop sees `killed` and surfaces a `cancelled` flag on the job state.
      server.middlewares.use("/api/scene/prepare", (req, res, next) => {
        if (req.method === "DELETE") {
          (async () => {
            const url = new URL(req.url!, `http://${req.headers.host}`);
            const video = url.searchParams.get("video");
            res.setHeader("Content-Type", "application/json");
            if (!video) {
              res.statusCode = 400;
              res.end(JSON.stringify({ error: "Missing video" }));
              return;
            }
            const job = sceneActiveJobs.get(video);
            if (!job) {
              res.statusCode = 404;
              res.end(JSON.stringify({ error: "No running scene prep for this video" }));
              return;
            }
            console.log(`[scene] cancel requested for ${video}`);
            const exited = await cancelJob(job, ":scene");
            res.end(JSON.stringify({ ok: true, exited }));
          })();
          return;
        }
        if (req.method !== "POST") return next();
        let body = "";
        req.on("data", (c: Buffer) => (body += c.toString()));
        req.on("end", async () => {
          try {
            const { video, pluginId = DEFAULT_SCENE_PLUGIN_ID, options = {} } = JSON.parse(body);
            if (!video) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Missing video" }));
              return;
            }
            const plugin = getScenePluginOrDefault(pluginId);
            if (plugin.id !== pluginId) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: `Unknown scene plugin: ${pluginId}` }));
              return;
            }
            const videoPath = path.join(UPLOADS_DIR, video);
            if (!fs.existsSync(videoPath)) {
              res.statusCode = 404;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Video not found" }));
              return;
            }
            const existing = sceneJobs.get(video);
            if (existing && existing.running) {
              res.statusCode = 409;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Scene prep already running", state: existing }));
              return;
            }
            const sd = sceneDir(video);
            fs.mkdirSync(sd, { recursive: true });
            // Frames are extracted at upload time; auto-bootstrap covers
            // pre-existing uploads and any failed upload-time extraction.
            const pipeline: PipelineStep[] = [...plugin.pipeline];
            if (!fs.existsSync(path.join(sd, "frames.json"))) {
              pipeline.unshift({
                stage: "frames",
                script: "extract_frames.py",
                args: ["$VIDEO", "$SCENE"],
              });
            }
            deleteBoxSolverResults(video);
            deleteObjectPointmapsForSource(video, plugin.id);
            if (plugin.cleanDir) {
              const d = path.join(sd, plugin.cleanDir);
              if (fs.existsSync(d)) fs.rmSync(d, { recursive: true, force: true });
            }
            const logPath = path.join(sd, plugin.logFile);
            const state: SceneJobState = {
              video,
              pluginId: plugin.id,
              stage: pipeline[0]?.stage ?? "running",
              running: true,
              error: null,
              startedAt: Date.now(),
              progress: null,
            };
            sceneJobs.set(video, state);
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ ok: true, state }));

            (async () => {
              try {
                for (const step of pipeline) {
                  state.stage = step.stage;
                  // Clear stale progress so the UI doesn't show the previous
                  // stage's last line while the new one is starting up.
                  state.progress = null;
                  const args = step.args.map((a) =>
                    a === "$VIDEO" ? videoPath : a === "$SCENE" ? sd : a,
                  );
                  if (plugin.subsampleDefault !== undefined
                      && (!plugin.subsampleScript || step.script === plugin.subsampleScript)
                      && options && Number.isFinite(Number(options.subsample))) {
                    const n = Math.max(1, Math.round(Number(options.subsample)));
                    args.push(plugin.subsampleFlag ?? "--subsample", String(n));
                  }
                  if (plugin.targetFramesDefault !== undefined
                      && options && Number.isFinite(Number(options.numFrames))) {
                    const T = Math.max(1, Math.round(Number(options.numFrames)));
                    args.push("--num-frames", String(T));
                    console.log(`[scene:${plugin.id}] targeting ${T} frames`);
                  }
                  if (plugin.requiresCameraSource) {
                    const sourceId = typeof options?.cameraSource === "string"
                      ? options.cameraSource : "";
                    const sourcePlugin = SCENE_PLUGINS.find((p) => p.id === sourceId);
                    if (!sourcePlugin || sourcePlugin.id === plugin.id) {
                      throw new Error(`Missing or invalid cameraSource for ${plugin.id}`);
                    }
                    const srcCamPath = path.join(sd, sourcePlugin.camerasDir, "cameras.json");
                    if (!fs.existsSync(srcCamPath)) {
                      throw new Error(
                        `Camera source ${sourcePlugin.id} has no cameras.json at ${srcCamPath}`,
                      );
                    }
                    const srcDepthDir = path.join(sd, sourcePlugin.depthDir);
                    if (!fs.existsSync(srcDepthDir)) {
                      throw new Error(
                        `Camera source ${sourcePlugin.id} has no depth dir at ${srcDepthDir}`,
                      );
                    }
                    args.push("--source-cameras-json", srcCamPath);
                    args.push("--source-depth-dir", srcDepthDir);
                    console.log(`[scene:${plugin.id}] camera source = ${sourcePlugin.id}`);
                  }
                  if (plugin.upscaleDefault !== undefined
                      && options && Number.isFinite(Number(options.upscale))) {
                    const u = Number(options.upscale);
                    args.push("--upscale", String(u));
                    console.log(`[scene:${plugin.id}] upscale = ${u}`);
                  }
                  console.log(`[scene:${plugin.id}] ${video}: ${step.stage} (${step.script})`);
                  // Already-cancelled before spawning the next step? Bail.
                  if (sceneActiveJobs.get(video)?.killed) throw new Error("__cancelled__");
                  let stepJob: CancellableJob | null = null;
                  try {
                    await runPython(
                      path.join(SCRIPTS_DIR, step.script), args, logPath, undefined,
                      (proc) => {
                        stepJob = { proc, killed: false };
                        sceneActiveJobs.set(video, stepJob);
                      },
                      (msg) => { state.progress = msg; },
                    );
                  } catch (err) {
                    // If cancelJob set killed=true on this step's job, surface
                    // that to the outer catch as a cancellation, not an error.
                    if (stepJob && (stepJob as CancellableJob).killed) throw new Error("__cancelled__");
                    throw err;
                  } finally {
                    if (stepJob && sceneActiveJobs.get(video) === stepJob) {
                      sceneActiveJobs.delete(video);
                    }
                  }
                }
                console.log(`[scene:${plugin.id}] ${video}: done`);
              } catch (err: any) {
                if (err?.message === "__cancelled__") {
                  state.cancelled = true;
                  console.log(`[scene:${plugin.id}] ${video}: cancelled`);
                } else {
                  state.error = err?.message ?? String(err);
                  console.error(`[scene:${plugin.id}] ${video}: ${state.error}`);
                }
              } finally {
                state.running = false;
                state.finishedAt = Date.now();
                state.progress = null;
                sceneActiveJobs.delete(video);
              }
            })();
          } catch (err: any) {
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: err?.message ?? String(err) }));
          }
        });
      });

      // POST /api/scene/align — align scene to user-defined floor plane
      // body: { video, points: [{x, y, frame}, ...], source?: "colmap"|"cut3r" }
      server.middlewares.use("/api/scene/align", (req, res, next) => {
        if (req.method !== "POST") return next();
        let body = "";
        req.on("data", (c: Buffer) => (body += c.toString()));
        req.on("end", async () => {
          try {
            const { video, points, source } = JSON.parse(body);
            if (!video || !points || points.length < 3) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Need video and at least 3 floor points" }));
              return;
            }
            const sd = sceneDir(video);
            const plugin = resolveSourcePlugin(source);
            const camerasPath = path.join(sd, plugin.camerasDir, "cameras.json");
            if (!fs.existsSync(camerasPath)) {
              res.statusCode = 404;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: `cameras.json not found in ${plugin.camerasDir}` }));
              return;
            }

            const scriptPath = path.join(SCRIPTS_DIR, "align_scene.py");
            const pointsJson = JSON.stringify(points);
            const logPath = path.join(sd, "align.log");
            deleteBoxSolverResults(video);
            // Read worldup ID to stamp into cameras.json
            const wuPath = path.join(sd, "worldup.json");
            let worldupId = "";
            if (fs.existsSync(wuPath)) {
              try { worldupId = JSON.parse(fs.readFileSync(wuPath, "utf-8")).id ?? ""; } catch {}
            }
            const extraArgs = worldupId ? ["--worldup-id", worldupId] : [];
            // Aligning rewrites this source's cameras.json + depth scale, which
            // invalidates any prior per-object cloud built from it.
            deleteObjectPointmapsForSource(video, plugin.id);
            // Pass the active plugin's scene-relative dirs so align_scene
            // reads cameras.json + depth maps from the right place.
            console.log(`[align] running on ${video} with ${points.length} floor points`);
            try {
              await runPython(scriptPath, [sd, pointsJson, ...scenePathArgs(plugin), ...extraArgs], logPath);
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ ok: true }));
            } catch (err: any) {
              res.statusCode = 500;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: `align_scene.py failed: ${err.message}` }));
            }
          } catch (err: any) {
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: err.message }));
          }
        });
      });

      // GET /api/scene/worldup?video=<filename> — load saved world-up points
      server.middlewares.use("/api/scene/worldup", (req, res, next) => {
        if (req.method === "GET") {
          const url = new URL(req.url!, `http://${req.headers.host}`);
          const video = url.searchParams.get("video");
          if (!video) {
            res.statusCode = 400;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: "Missing video" }));
            return;
          }
          const sd = sceneDir(video);
          const wuPath = path.join(sd, "worldup.json");
          if (fs.existsSync(wuPath)) {
            res.setHeader("Content-Type", "application/json");
            res.end(fs.readFileSync(wuPath, "utf-8"));
          } else {
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ points: [] }));
          }
          return;
        }
        // POST /api/scene/worldup — save world-up points
        if (req.method === "POST") {
          let body = "";
          req.on("data", (c: Buffer) => (body += c.toString()));
          req.on("end", () => {
            try {
              const { video, points } = JSON.parse(body);
              if (!video || !points) {
                res.statusCode = 400;
                res.setHeader("Content-Type", "application/json");
                res.end(JSON.stringify({ error: "Missing video or points" }));
                return;
              }
              const sd = sceneDir(video);
              if (!fs.existsSync(sd)) fs.mkdirSync(sd, { recursive: true });
              const wuPath = path.join(sd, "worldup.json");
              const id = crypto.randomUUID();
              fs.writeFileSync(wuPath, JSON.stringify({ id, points }, null, 2));
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ ok: true }));
            } catch (err: any) {
              res.statusCode = 500;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: err.message }));
            }
          });
          return;
        }
        next();
      });

      // GET /api/scene/status?video=<filename> — artifact + job state
      server.middlewares.use("/api/scene/status", (req, res, next) => {
        if (req.method !== "GET") return next();
        const url = new URL(req.url!, `http://${req.headers.host}`);
        const video = url.searchParams.get("video");
        if (!video) {
          res.statusCode = 400;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: "Missing video" }));
          return;
        }
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({
          artifacts: sceneArtifactState(video),
          job: sceneJobs.get(video) ?? null,
        }));
      });

      // POST   /api/dinov3/prepare  — start a DINOv3 feature run
      //   body: { video, subsample?, scaling? }
      // DELETE /api/dinov3/prepare?video=  — cancel a running job
      server.middlewares.use("/api/dinov3/prepare", (req, res, next) => {
        if (req.method === "DELETE") {
          (async () => {
            const url = new URL(req.url!, `http://${req.headers.host}`);
            const video = url.searchParams.get("video");
            res.setHeader("Content-Type", "application/json");
            if (!video) {
              res.statusCode = 400;
              res.end(JSON.stringify({ error: "Missing video" }));
              return;
            }
            const job = dinov3ActiveJobs.get(video);
            if (!job) {
              res.statusCode = 404;
              res.end(JSON.stringify({ error: "No running dinov3 job for this video" }));
              return;
            }
            console.log(`[dinov3] cancel requested for ${video}`);
            const exited = await cancelJob(job, ":dinov3");
            res.end(JSON.stringify({ ok: true, exited }));
          })();
          return;
        }
        if (req.method !== "POST") return next();
        let body = "";
        req.on("data", (c: Buffer) => (body += c.toString()));
        req.on("end", async () => {
          try {
            const { video, subsample, scaling } = JSON.parse(body || "{}");
            if (!video) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Missing video" }));
              return;
            }
            const videoPath = path.join(UPLOADS_DIR, video);
            if (!fs.existsSync(videoPath)) {
              res.statusCode = 404;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Video not found" }));
              return;
            }
            const existing = dinov3Jobs.get(video);
            if (existing && existing.running) {
              res.statusCode = 409;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "dinov3 already running", state: existing }));
              return;
            }
            const sd = sceneDir(video);
            fs.mkdirSync(sd, { recursive: true });
            // Bootstrap frames if /api/upload's extract step failed previously.
            if (!fs.existsSync(path.join(sd, "frames.json"))) {
              await runPython(
                path.join(SCRIPTS_DIR, "extract_frames.py"),
                [videoPath, sd],
                path.join(sd, "extract_frames.log"),
              );
            }
            const subN = Math.max(1, Math.round(Number(subsample) || 2));
            const scN = Number.isFinite(Number(scaling)) ? Number(scaling) : 0.5;
            // Wipe prior output so a partial/failed previous run can't be
            // mistaken for ready.
            const outDir = path.join(sd, "dinov3");
            if (fs.existsSync(outDir)) fs.rmSync(outDir, { recursive: true, force: true });
            const logPath = path.join(sd, "dinov3.log");
            const state: DinoV3JobState = {
              video,
              running: true,
              error: null,
              startedAt: Date.now(),
              progress: null,
              subsample: subN,
              scaling: scN,
            };
            dinov3Jobs.set(video, state);
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ ok: true, state }));

            (async () => {
              let stepJob: CancellableJob | null = null;
              try {
                await runPython(
                  path.join(SCRIPTS_DIR, "run_dinov3.py"),
                  [sd, "--subsample", String(subN), "--scaling", String(scN)],
                  logPath,
                  undefined,
                  (proc) => {
                    stepJob = { proc, killed: false };
                    dinov3ActiveJobs.set(video, stepJob);
                  },
                  (msg) => { state.progress = msg; },
                );
                console.log(`[dinov3] ${video}: done`);
              } catch (err: any) {
                if (stepJob && (stepJob as CancellableJob).killed) {
                  state.cancelled = true;
                  console.log(`[dinov3] ${video}: cancelled`);
                } else {
                  state.error = err?.message ?? String(err);
                  console.error(`[dinov3] ${video}: ${state.error}`);
                }
              } finally {
                state.running = false;
                state.finishedAt = Date.now();
                state.progress = null;
                if (stepJob && dinov3ActiveJobs.get(video) === stepJob) {
                  dinov3ActiveJobs.delete(video);
                }
              }
            })();
          } catch (err: any) {
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: err?.message ?? String(err) }));
          }
        });
      });

      // GET /api/dinov3/status?video=<filename>
      // Returns: { meta: <meta.json contents or null>, job: <DinoV3JobState or null> }
      server.middlewares.use("/api/dinov3/status", (req, res, next) => {
        if (req.method !== "GET") return next();
        const url = new URL(req.url!, `http://${req.headers.host}`);
        const video = url.searchParams.get("video");
        res.setHeader("Content-Type", "application/json");
        if (!video) {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: "Missing video" }));
          return;
        }
        const sd = sceneDir(video);
        const metaPath = path.join(sd, "dinov3", "meta.json");
        let meta: unknown = null;
        if (fs.existsSync(metaPath)) {
          try { meta = JSON.parse(fs.readFileSync(metaPath, "utf-8")); }
          catch { meta = null; }
        }
        res.end(JSON.stringify({
          meta,
          job: dinov3Jobs.get(video) ?? null,
        }));
      });

      // GET /api/scene/camera-sources?video=<filename>&self=<pluginId>
      // List scene plugins that already have a cameras.json for this video.
      // Used by plugins with `requiresCameraSource: true` (e.g. InfiniDepth)
      // to populate their "Camera source" dropdown. The `self` plugin is
      // excluded from the list. Additionally returns `currentSource`: the
      // upstream plugin id recorded in self's own cameras.json (if any), so
      // the dropdown can reflect what was actually computed, not just the
      // last in-memory selection.
      server.middlewares.use("/api/scene/camera-sources", (req, res, next) => {
        if (req.method !== "GET") return next();
        const url = new URL(req.url!, `http://${req.headers.host}`);
        const video = url.searchParams.get("video");
        const self = url.searchParams.get("self");
        if (!video) {
          res.statusCode = 400;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: "Missing video" }));
          return;
        }
        const sd = sceneDir(video);
        const sources = SCENE_PLUGINS
          .filter((p) => p.id !== self && !p.requiresCameraSource)
          .map((p) => ({
            id: p.id,
            label: p.label,
            ready: fs.existsSync(path.join(sd, p.camerasDir, "cameras.json")),
          }))
          .filter((s) => s.ready);
        let currentSource: string | null = null;
        const selfPlugin = SCENE_PLUGINS.find((p) => p.id === self);
        if (selfPlugin) {
          const selfCamPath = path.join(sd, selfPlugin.camerasDir, "cameras.json");
          if (fs.existsSync(selfCamPath)) {
            try {
              const data = JSON.parse(fs.readFileSync(selfCamPath, "utf-8"));
              const srcPath: unknown = data?.source_cameras;
              if (typeof srcPath === "string" && srcPath) {
                const dirName = path.basename(path.dirname(srcPath));
                const match = SCENE_PLUGINS.find((p) => p.camerasDir === dirName);
                if (match) currentSource = match.id;
              }
            } catch { /* ignore unparseable cameras.json */ }
          }
        }
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ sources, currentSource }));
      });

      // GET /api/scene/cameras?video=<filename> — return cameras.json
      server.middlewares.use("/api/scene/cameras", (req, res, next) => {
        if (req.method !== "GET") return next();
        const url = new URL(req.url!, `http://${req.headers.host}`);
        const video = url.searchParams.get("video");
        if (!video) {
          res.statusCode = 400;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: "Missing video" }));
          return;
        }
        const jsonPath = path.join(sceneDir(video), "colmap", "cameras.json");
        if (!fs.existsSync(jsonPath)) {
          res.statusCode = 404;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: "cameras.json not found" }));
          return;
        }
        res.setHeader("Content-Type", "application/json");
        fs.createReadStream(jsonPath).pipe(res);
      });

      // GET /api/videos — list uploaded videos
      // DELETE /api/videos?name=<filename> — remove the upload + its analysis dir
      server.middlewares.use("/api/videos", (req, res, next) => {
        if (req.method === "GET") {
          const files = fs.existsSync(UPLOADS_DIR)
            ? fs.readdirSync(UPLOADS_DIR).filter((f) => /\.(mp4|webm|mov|avi|mkv)$/i.test(f))
            : [];
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ videos: files }));
          return;
        }
        if (req.method === "DELETE") {
          const url = new URL(req.url!, `http://${req.headers.host}`);
          const name = url.searchParams.get("name");
          res.setHeader("Content-Type", "application/json");
          if (!name) {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: "Missing name" }));
            return;
          }
          // Reject path traversal attempts
          if (name.includes("/") || name.includes("\\") || name.includes("..")) {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: "Invalid name" }));
            return;
          }
          const job = sceneJobs.get(name);
          if (job && job.running) {
            res.statusCode = 409;
            res.end(JSON.stringify({ error: "Scene prep is running for this video" }));
            return;
          }
          const filepath = path.join(UPLOADS_DIR, name);
          const stem = path.basename(name, path.extname(name));
          const analysisDir = path.join(ANALYSIS_DIR, stem);
          try {
            if (fs.existsSync(filepath)) fs.unlinkSync(filepath);
            if (fs.existsSync(analysisDir)) fs.rmSync(analysisDir, { recursive: true, force: true });
            sceneJobs.delete(name);
            for (const key of Array.from(objectPointmapJobs.keys())) {
              if (key.startsWith(`${name}::`)) objectPointmapJobs.delete(key);
            }
            console.log(`[delete] removed video ${name} and ${analysisDir}`);
            res.end(JSON.stringify({ ok: true }));
          } catch (err: any) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: err?.message ?? String(err) }));
          }
          return;
        }
        next();
      });

      // POST   /api/detect — extract frame 0 and run SAM3 detection on it
      //   body: { video, x, y, label }
      // DELETE /api/detect?video= — kill the running detection for this video.
      server.middlewares.use("/api/detect", (req, res, next) => {
        if (req.method === "DELETE") {
          (async () => {
            const url = new URL(req.url!, `http://${req.headers.host}`);
            const video = url.searchParams.get("video");
            res.setHeader("Content-Type", "application/json");
            if (!video) {
              res.statusCode = 400;
              res.end(JSON.stringify({ error: "Missing video" }));
              return;
            }
            const job = detectJobs.get(video);
            if (!job) {
              res.statusCode = 404;
              res.end(JSON.stringify({ error: "No running detection for this video" }));
              return;
            }
            console.log(`[detect] cancel requested for ${video}`);
            const exited = await cancelJob(job, ":detect");
            res.end(JSON.stringify({ ok: true, exited }));
          })();
          return;
        }
        if (req.method !== "POST") return next();
        let body = "";
        req.on("data", (chunk: Buffer) => (body += chunk.toString()));
        req.on("end", async () => {
          try {
            const { video, x, y, label } = JSON.parse(body);
            if (!video || typeof x !== "number" || typeof y !== "number" || !label) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Missing video, x, y, or label" }));
              return;
            }
            const videoPath = path.join(UPLOADS_DIR, video);
            if (!fs.existsSync(videoPath)) {
              res.statusCode = 404;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Video not found" }));
              return;
            }
            if (detectJobs.has(video)) {
              res.statusCode = 409;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Detection already running for this video" }));
              return;
            }

            const stem = path.basename(video, path.extname(video));
            const safeLabel = label.replace(/[^a-zA-Z0-9._-]/g, "_");
            const videoAnalysisDir = path.join(ANALYSIS_DIR, stem);
            // Find next unique ID for this label
            fs.mkdirSync(videoAnalysisDir, { recursive: true });
            const existing = fs.readdirSync(videoAnalysisDir).filter((d) => d.startsWith(`${safeLabel}_`));
            const nextId = existing.reduce((max, d) => {
              const m = d.match(/_(\d+)$/);
              return m ? Math.max(max, parseInt(m[1], 10) + 1) : max;
            }, 1);
            const runDir = path.join(videoAnalysisDir, `${safeLabel}_${nextId}`);
            fs.mkdirSync(runDir, { recursive: true });

            const framePath = path.join(runDir, "frame0.png");
            const outputJsonPath = path.join(runDir, "detect.json");

            // 1. Extract frame 0 with ffmpeg
            console.log(`[detect] extracting frame 0 of ${video}`);
            await new Promise<void>((resolve, reject) => {
              const ff = spawn(ffmpegPath!, ["-y", "-i", videoPath, "-vframes", "1", "-update", "1", framePath]);
              let stderr = "";
              ff.stderr.on("data", (c) => (stderr += c.toString()));
              ff.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg exit ${code}: ${stderr}`))));
              ff.on("error", reject);
            });

            // 2. Run python detection script (registered for cancellation).
            const scriptPath = path.join(SCRIPTS_DIR, "detect_object.py");
            console.log(`[detect] running ${scriptPath} on ${framePath} click=(${x},${y}) label="${label}"`);
            const job: CancellableJob = {
              proc: spawn(VENV_PYTHON, [scriptPath, framePath, String(x), String(y), label, outputJsonPath]),
              killed: false,
            };
            detectJobs.set(video, job);
            try {
              await new Promise<void>((resolve, reject) => {
                let stderr = "";
                job.proc.stderr?.on("data", (c) => (stderr += c.toString()));
                job.proc.stdout?.on("data", (c) => process.stdout.write(`[detect_object] ${c.toString()}`));
                job.proc.on("close", (code) => {
                  if (code === 0) resolve();
                  else if (job.killed) reject(new Error("__cancelled__"));
                  else reject(new Error(`python exit ${code}: ${stderr}`));
                });
                job.proc.on("error", reject);
              });
            } finally {
              detectJobs.delete(video);
            }

            // 3. Read and return result
            const result = JSON.parse(fs.readFileSync(outputJsonPath, "utf-8"));
            result.analysis = `${safeLabel}_${nextId}`;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(result));
          } catch (err: any) {
            const cancelled = err?.message === "__cancelled__";
            if (cancelled) {
              console.log(`[detect] cancelled`);
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ cancelled: true }));
            } else {
              console.error("[detect] error:", err);
              res.statusCode = 500;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: err.message ?? String(err) }));
            }
          }
        });
      });

      // POST   /api/track — run SAM2 video tracking seeded from a previous detection
      //   body: { video, analysis }  (analysis is the run folder name, e.g. "chair_1")
      // DELETE /api/track?video=&analysis= — kill the running tracker.
      server.middlewares.use("/api/track", (req, res, next) => {
        if (req.method === "DELETE") {
          (async () => {
            const url = new URL(req.url!, `http://${req.headers.host}`);
            const video = url.searchParams.get("video");
            const analysis = url.searchParams.get("analysis");
            res.setHeader("Content-Type", "application/json");
            if (!video || !analysis) {
              res.statusCode = 400;
              res.end(JSON.stringify({ error: "Missing video or analysis" }));
              return;
            }
            const job = trackJobs.get(`${video}::${analysis}`);
            if (!job) {
              res.statusCode = 404;
              res.end(JSON.stringify({ error: "No running tracker for this analysis" }));
              return;
            }
            console.log(`[track] cancel requested for ${video} (${analysis})`);
            const exited = await cancelJob(job, ":track");
            res.end(JSON.stringify({ ok: true, exited }));
          })();
          return;
        }
        if (req.method !== "POST") return next();
        let body = "";
        req.on("data", (chunk: Buffer) => (body += chunk.toString()));
        req.on("end", async () => {
          try {
            const { video, analysis } = JSON.parse(body);
            if (!video || !analysis) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Missing video or analysis" }));
              return;
            }
            const videoPath = path.join(UPLOADS_DIR, video);
            if (!fs.existsSync(videoPath)) {
              res.statusCode = 404;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Video not found" }));
              return;
            }
            const stem = path.basename(video, path.extname(video));
            const runDir = path.join(ANALYSIS_DIR, stem, analysis);
            const detectJsonPath = path.join(runDir, "detect.json");
            if (!fs.existsSync(detectJsonPath)) {
              res.statusCode = 404;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "detect.json not found for this analysis" }));
              return;
            }
            const jobKey = `${video}::${analysis}`;
            if (trackJobs.has(jobKey)) {
              res.statusCode = 409;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Tracker already running for this analysis" }));
              return;
            }

            const scriptPath = path.join(SCRIPTS_DIR, "track_object.py");
            // Re-tracking invalidates any prior 3D-box solver outputs for
            // this analysis (boxes are anchored to the previous track), so
            // wipe both solver subdirs. The per-object cloud is masked by
            // the same track and is just as stale, so wipe it too.
            for (const solver of BOX_SOLVER_PLUGINS) {
              const solverDir = path.join(runDir, solver.subdir);
              if (fs.existsSync(solverDir)) {
                fs.rmSync(solverDir, { recursive: true, force: true });
                console.log(`[track] cleared ${solverDir}`);
              }
            }
            const objDir = path.join(runDir, OBJECT_POINTMAP_DIR);
            if (fs.existsSync(objDir)) {
              fs.rmSync(objDir, { recursive: true, force: true });
              console.log(`[track] cleared ${objDir}`);
            }
            console.log(`[track] running ${scriptPath} on ${video} (${analysis})`);
            const job: CancellableJob = {
              proc: spawn(VENV_PYTHON, [scriptPath, videoPath, detectJsonPath, runDir]),
              killed: false,
            };
            trackJobs.set(jobKey, job);
            try {
              await new Promise<void>((resolve, reject) => {
                let stderr = "";
                job.proc.stderr?.on("data", (c) => (stderr += c.toString()));
                job.proc.stdout?.on("data", (c) => process.stdout.write(`[track_object] ${c.toString()}`));
                job.proc.on("close", (code) => {
                  if (code === 0) resolve();
                  else if (job.killed) reject(new Error("__cancelled__"));
                  else reject(new Error(`python exit ${code}: ${stderr}`));
                });
                job.proc.on("error", reject);
              });
            } finally {
              trackJobs.delete(jobKey);
            }

            const trackJsonPath = path.join(runDir, "track.json");
            const result = JSON.parse(fs.readFileSync(trackJsonPath, "utf-8"));
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(result));
          } catch (err: any) {
            const cancelled = err?.message === "__cancelled__";
            if (cancelled) {
              console.log(`[track] cancelled`);
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ cancelled: true }));
            } else {
              console.error("[track] error:", err);
              res.statusCode = 500;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: err.message ?? String(err) }));
            }
          }
        });
      });

      // GET /api/track-result?video=<filename>&name=<analysis> — load previous tracking result
      server.middlewares.use("/api/track-result", (req, res, next) => {
        if (req.method !== "GET") return next();
        const url = new URL(req.url!, `http://${req.headers.host}`);
        const video = url.searchParams.get("video");
        const name = url.searchParams.get("name");
        if (!video || !name) { res.statusCode = 400; res.end(JSON.stringify({ error: "Missing video or name" })); return; }
        const stem = path.basename(video, path.extname(video));
        const jsonPath = path.join(ANALYSIS_DIR, stem, name, "track.json");
        if (!fs.existsSync(jsonPath)) { res.statusCode = 404; res.end(JSON.stringify({ error: "Track result not found" })); return; }
        const result = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify(result));
      });

      // GET /api/analyses?video=<filename> — list analysis runs for a video
      // DELETE /api/analyses?video=<filename>&name=<analysis> — remove one run
      server.middlewares.use("/api/analyses", (req, res, next) => {
        if (req.method === "GET") {
          const url = new URL(req.url!, `http://${req.headers.host}`);
          const video = url.searchParams.get("video");
          if (!video) { res.statusCode = 400; res.end(JSON.stringify({ error: "Missing video" })); return; }
          const stem = path.basename(video, path.extname(video));
          const videoDir = path.join(ANALYSIS_DIR, stem);
          const analyses = fs.existsSync(videoDir)
            ? fs.readdirSync(videoDir).filter((d) => !d.startsWith("_") && fs.statSync(path.join(videoDir, d)).isDirectory()).sort()
            : [];
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ analyses }));
          return;
        }
        if (req.method === "DELETE") {
          const url = new URL(req.url!, `http://${req.headers.host}`);
          const video = url.searchParams.get("video");
          const name = url.searchParams.get("name");
          res.setHeader("Content-Type", "application/json");
          if (!video || !name) {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: "Missing video or name" }));
            return;
          }
          if (name.startsWith("_") || name.includes("/") || name.includes("\\") || name.includes("..")) {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: "Invalid analysis name" }));
            return;
          }
          const stem = path.basename(video, path.extname(video));
          const runDir = path.join(ANALYSIS_DIR, stem, name);
          try {
            if (fs.existsSync(runDir)) fs.rmSync(runDir, { recursive: true, force: true });
            for (const key of Array.from(objectPointmapJobs.keys())) {
              if (key.startsWith(`${video}::${name}::`)) objectPointmapJobs.delete(key);
            }
            console.log(`[delete] removed analysis ${runDir}`);
            res.end(JSON.stringify({ ok: true }));
          } catch (err: any) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: err?.message ?? String(err) }));
          }
          return;
        }
        next();
      });

      // GET /api/analysis-result?video=<filename>&name=<name> — load a previous result
      server.middlewares.use("/api/analysis-result", (req, res, next) => {
        if (req.method !== "GET") return next();
        const url = new URL(req.url!, `http://${req.headers.host}`);
        const video = url.searchParams.get("video");
        const name = url.searchParams.get("name");
        if (!video || !name) { res.statusCode = 400; res.end(JSON.stringify({ error: "Missing video or name" })); return; }
        const stem = path.basename(video, path.extname(video));
        const jsonPath = path.join(ANALYSIS_DIR, stem, name, "detect.json");
        if (!fs.existsSync(jsonPath)) { res.statusCode = 404; res.end(JSON.stringify({ error: "Analysis not found" })); return; }
        const result = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify(result));
      });

      // POST   /api/box-solver — run a 3D bbox lifting solver (Boxer or WildDet3D)
      //   body: { video, analysis, solverId, label, source, options }
      //   Result lands at <analysis>/<solver.subdir>/<solver.resultFile>; both
      //   solvers can co-exist for the same analysis run.
      // DELETE /api/box-solver?video=&analysis=&solverId= — kill the running
      //   solver process for this (video, analysis, solverId) tuple.
      server.middlewares.use("/api/box-solver", (req, res, next) => {
        if (req.method === "DELETE") {
          (async () => {
            const url = new URL(req.url!, `http://${req.headers.host}`);
            const video = url.searchParams.get("video");
            const analysis = url.searchParams.get("analysis");
            const solverId = url.searchParams.get("solverId");
            res.setHeader("Content-Type", "application/json");
            if (!video || !analysis || !solverId) {
              res.statusCode = 400;
              res.end(JSON.stringify({ error: "Missing video, analysis, or solverId" }));
              return;
            }
            const jobKey = `${video}::${analysis}::${solverId}`;
            const job = boxSolverJobs.get(jobKey);
            if (!job) {
              res.statusCode = 404;
              res.end(JSON.stringify({ error: "No running solver for this analysis" }));
              return;
            }
            console.log(`[${solverId}] cancel requested for ${video} (${analysis})`);
            const exited = await cancelJob(job, `:${solverId}`);
            res.end(JSON.stringify({ ok: true, exited }));
          })();
          return;
        }
        if (req.method !== "POST") return next();
        let body = "";
        req.on("data", (chunk: Buffer) => (body += chunk.toString()));
        req.on("end", async () => {
          try {
            const { video, analysis, solverId, label, source, options = {} } = JSON.parse(body);
            if (!video || !analysis || !solverId) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Missing video, analysis, or solverId" }));
              return;
            }
            const solver = BOX_SOLVER_PLUGINS_BY_ID[solverId];
            if (!solver) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: `Unknown solverId: ${solverId}` }));
              return;
            }
            const stem = path.basename(video, path.extname(video));
            const sd = sceneDir(video);
            const runDir = path.join(ANALYSIS_DIR, stem, analysis);
            const trackPath = path.join(runDir, "track.json");
            if (!fs.existsSync(trackPath)) {
              res.statusCode = 404;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "track.json not found — run tracking first" }));
              return;
            }
            const plugin = resolveSourcePlugin(source);
            const camerasPath = path.join(sd, plugin.camerasDir, "cameras.json");
            if (!fs.existsSync(camerasPath)) {
              res.statusCode = 404;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: `cameras.json not found in ${plugin.camerasDir}` }));
              return;
            }

            const jobKey = `${video}::${analysis}::${solverId}`;
            const existing = boxSolverJobs.get(jobKey);
            if (existing) {
              res.statusCode = 409;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Solver already running for this analysis" }));
              return;
            }

            // Wipe this solver's output dir so polling doesn't see a stale result.
            // Other solvers' outputs are independent and untouched.
            const outDir = path.join(runDir, solver.subdir);
            if (fs.existsSync(outDir)) {
              fs.rmSync(outDir, { recursive: true, force: true });
              console.log(`[${solver.id}] cleared ${outDir}`);
            }
            fs.mkdirSync(outDir, { recursive: true });

            // Translate the options map into CLI flags. Only the keys declared
            // in the plugin's options are honored; truthy → emit `--<key-kebab>`.
            const optionFlags: string[] = [];
            for (const opt of solver.options) {
              if (options[opt.key]) {
                optionFlags.push(`--${opt.key.replace(/[A-Z]/g, (m) => "-" + m.toLowerCase())}`);
              }
            }

            const scriptPath = path.join(SCRIPTS_DIR, solver.script);
            const pyArgs = [
              sd, runDir,
              "--out-dir", outDir,
              ...(label ? ["--label", label] : []),
              ...scenePathArgs(plugin),
              ...optionFlags,
            ];
            const flagSummary = optionFlags.length ? ` ${optionFlags.join(" ")}` : "";
            console.log(`[${solver.id}] running ${scriptPath} on ${video} (${analysis})${flagSummary}`);
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ ok: true, status: "running" }));

            const logPath = path.join(outDir, solver.logFile);
            runPython(scriptPath, pyArgs, logPath, undefined, (proc) => {
              boxSolverJobs.set(jobKey, { proc, killed: false });
            })
              .then(() => console.log(`[${solver.id}] done: ${video} (${analysis})`))
              .catch((err) => {
                const job = boxSolverJobs.get(jobKey);
                if (job?.killed) {
                  console.log(`[${solver.id}] cancelled: ${video} (${analysis})`);
                } else {
                  console.error(`[${solver.id}] failed: ${err}`);
                }
              })
              .finally(() => boxSolverJobs.delete(jobKey));
          } catch (err: any) {
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: err.message }));
          }
        });
      });

      // GET /api/box-solver-result?video=<f>&name=<analysis>&solverId=<id> — load result
      server.middlewares.use("/api/box-solver-result", (req, res, next) => {
        if (req.method !== "GET") return next();
        const url = new URL(req.url!, `http://${req.headers.host}`);
        const video = url.searchParams.get("video");
        const name = url.searchParams.get("name");
        const solverId = url.searchParams.get("solverId");
        if (!video || !name || !solverId) {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: "Missing video, name, or solverId" }));
          return;
        }
        const solver = BOX_SOLVER_PLUGINS_BY_ID[solverId];
        if (!solver) {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: `Unknown solverId: ${solverId}` }));
          return;
        }
        const stem = path.basename(video, path.extname(video));
        const jsonPath = path.join(ANALYSIS_DIR, stem, name, solver.subdir, solver.resultFile);
        if (!fs.existsSync(jsonPath)) {
          res.statusCode = 404;
          res.end(JSON.stringify({ error: `${solver.label} result not found` }));
          return;
        }
        const result = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify(result));
      });

      // POST /api/object-pointmap — build a per-object world-space cloud by
      // unprojecting the per-frame depth maps, masked by the per-frame SAM2
      // tracks, into world coordinates and concatenating across frames.
      // body: { video, analysis, source }
      // Output: <analysis>/object_pointmap/<source>.npz
      server.middlewares.use("/api/object-pointmap", (req, res, next) => {
        if (req.method !== "POST") return next();
        let body = "";
        req.on("data", (c: Buffer) => (body += c.toString()));
        req.on("end", () => {
          try {
            const { video, analysis, source, options = {} } = JSON.parse(body);
            if (!video || !analysis) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Missing video or analysis" }));
              return;
            }
            const plugin = resolveSourcePlugin(source);
            const stem = path.basename(video, path.extname(video));
            const sd = sceneDir(video);
            const runDir = path.join(ANALYSIS_DIR, stem, analysis);
            const camerasPath = path.join(sd, plugin.camerasDir, "cameras.json");
            if (!fs.existsSync(camerasPath)) {
              res.statusCode = 404;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({
                error: `cameras.json not found in ${plugin.camerasDir} — run the scene plugin first`,
              }));
              return;
            }
            const depthDir = path.join(sd, plugin.depthDir);
            if (!fs.existsSync(depthDir)) {
              res.statusCode = 404;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({
                error: `No depth maps in ${plugin.depthDir} — run the scene plugin first`,
              }));
              return;
            }
            const trackPath = path.join(runDir, "track.json");
            if (!fs.existsSync(trackPath)) {
              res.statusCode = 404;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "track.json not found — run tracking first" }));
              return;
            }

            const jobKey = `${video}::${analysis}::${plugin.id}`;
            const existing = objectPointmapJobs.get(jobKey);
            if (existing && existing.running) {
              res.statusCode = 409;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Object point cloud already running", state: existing }));
              return;
            }

            const outPath = objectPointmapOutArg(video, analysis, plugin.id);
            const outDir = objectPointmapBasePath(video, analysis);
            fs.mkdirSync(outDir, { recursive: true });
            // Wipe any prior chunks/manifest for this source so the readiness
            // check doesn't see a stale manifest while the rebuild is in flight.
            deleteObjectPointmapChunks(outDir, plugin.id);
            const logPath = path.join(outDir, `${plugin.id}.log`);

            const state: ObjectPointmapJobState = {
              running: true,
              source: plugin.id,
              startedAt: Date.now(),
              error: null,
              progress: null,
            };
            objectPointmapJobs.set(jobKey, state);

            const scriptPath = path.join(SCRIPTS_DIR, "build_object_pointmap.py");
            const pyArgs = [
              sd, runDir,
              "--cameras-dir", plugin.camerasDir,
              "--depth-dir", plugin.depthDir,
              "--out", outPath,
            ];
            // Mask erosion (depth-map space) — peels off boundary pixels
            // where interpolated depth would produce fly-aways behind the
            // object. 0 = off; UI defaults to 1.
            const erode = Number(options?.erode);
            if (Number.isFinite(erode) && erode > 0) {
              pyArgs.push("--erode", String(Math.max(0, Math.round(erode))));
            }
            console.log(`[object-pointmap] running on ${video} (${analysis}) source=${plugin.id}`
              + (erode > 0 ? ` erode=${erode}px` : ""));
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ ok: true, state }));

            runPython(scriptPath, pyArgs, logPath, undefined, undefined,
              (msg) => { state.progress = msg; })
              .then(() => {
                state.running = false;
                state.finishedAt = Date.now();
                state.progress = null;
                console.log(`[object-pointmap] done: ${outPath}`);
              })
              .catch((err) => {
                state.running = false;
                state.finishedAt = Date.now();
                state.progress = null;
                state.error = err?.message ?? String(err);
                console.error(`[object-pointmap] failed: ${state.error}`);
              });
          } catch (err: any) {
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: err?.message ?? String(err) }));
          }
        });
      });

      // GET /api/object-pointmap-status?video=<f>&analysis=<a>&source=<id>
      // Returns { job, ready } where ready=true iff the .npz exists.
      server.middlewares.use("/api/object-pointmap-status", (req, res, next) => {
        if (req.method !== "GET") return next();
        const url = new URL(req.url!, `http://${req.headers.host}`);
        const video = url.searchParams.get("video");
        const analysis = url.searchParams.get("analysis");
        const sourceParam = url.searchParams.get("source");
        if (!video || !analysis) {
          res.statusCode = 400;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: "Missing video or analysis" }));
          return;
        }
        const plugin = resolveSourcePlugin(sourceParam);
        const job = objectPointmapJobs.get(`${video}::${analysis}::${plugin.id}`) ?? null;
        const ready = fs.existsSync(objectPointmapManifestPath(video, analysis, plugin.id));
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ job, ready, source: plugin.id }));
      });

      // GET /api/depth-frames?video=<filename>&source=<pluginId> — list available depth map frames
      server.middlewares.use("/api/depth-frames", (req, res, next) => {
        if (req.method !== "GET") return next();
        const url = new URL(req.url!, `http://${req.headers.host}`);
        const video = url.searchParams.get("video");
        if (!video) { res.statusCode = 400; res.end(JSON.stringify({ error: "Missing video" })); return; }
        const plugin = resolveSourcePlugin(url.searchParams.get("source"));
        const stem = path.basename(video, path.extname(video));
        const depthDir = path.join(ANALYSIS_DIR, stem, "_scene", plugin.depthDir);
        if (!fs.existsSync(depthDir)) {
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ frames: [] }));
          return;
        }
        const frames = fs.readdirSync(depthDir)
          .filter((f) => f.endsWith(".npz"))
          .map((f) => parseInt(f.replace(".npz", ""), 10))
          .sort((a, b) => a - b);
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ stem, frames }));
      });

      // Serve per-frame track mask PNGs: /analysis/<stem>/<runName>/masks/<NNNNNN>.png
      // Also serves any other file under analysis/ (useful for debug).
      server.middlewares.use("/analysis", (req, res, next) => {
        // Strip any query string (e.g. ?v=<cache-bust>) before resolving
        // against the filesystem; without this, callers using a cache-bust
        // suffix get a 404 because we'd try to open `file.ext?v=1`.
        const raw = req.url?.replace(/^\//, "").split("?")[0] ?? "";
        const rel = decodeURIComponent(raw);
        if (!rel) return next();
        // Block path traversal
        const filepath = path.normalize(path.join(ANALYSIS_DIR, rel));
        if (!filepath.startsWith(ANALYSIS_DIR)) {
          res.statusCode = 400;
          return res.end("Invalid path");
        }
        if (!fs.existsSync(filepath) || !fs.statSync(filepath).isFile()) {
          res.statusCode = 404;
          return res.end("Not found");
        }
        const stat = fs.statSync(filepath);
        const ext = path.extname(filepath).toLowerCase();
        const mime: Record<string, string> = {
          ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
          ".json": "application/json", ".glb": "model/gltf-binary",
          ".ply": "application/octet-stream",
        };
        res.setHeader("Content-Type", mime[ext] || "application/octet-stream");
        // Set Content-Length so the browser can compute download progress
        // (the scene/object pointmap loaders use it to drive the progress bar).
        res.setHeader("Content-Length", String(stat.size));
        // Mask PNGs are immutable once written; other files (cameras.json, depth npz)
        // may be regenerated, so use ETag-based revalidation instead of long max-age.
        if (ext === ".png") {
          res.setHeader("Cache-Control", "public, max-age=3600");
        } else {
          res.setHeader("Cache-Control", "no-cache");
        }
        fs.createReadStream(filepath).pipe(res);
      });

      // Serve uploaded video files
      server.middlewares.use("/uploads", (req, res, next) => {
        const filename = decodeURIComponent(req.url?.replace(/^\//, "") ?? "");
        if (!filename) return next();
        const filepath = path.join(UPLOADS_DIR, filename);
        if (!fs.existsSync(filepath)) {
          res.statusCode = 404;
          return res.end("Not found");
        }
        const ext = path.extname(filename).toLowerCase();
        const mime: Record<string, string> = {
          ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
          ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
        };
        const stat = fs.statSync(filepath);

        // Support range requests for video seeking
        const range = req.headers.range;
        if (range) {
          const parts = range.replace(/bytes=/, "").split("-");
          const start = parseInt(parts[0], 10);
          const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;
          res.writeHead(206, {
            "Content-Range": `bytes ${start}-${end}/${stat.size}`,
            "Accept-Ranges": "bytes",
            "Content-Length": end - start + 1,
            "Content-Type": mime[ext] || "application/octet-stream",
          });
          fs.createReadStream(filepath, { start, end }).pipe(res);
        } else {
          res.writeHead(200, {
            "Content-Length": stat.size,
            "Content-Type": mime[ext] || "application/octet-stream",
            "Accept-Ranges": "bytes",
          });
          fs.createReadStream(filepath).pipe(res);
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [solidPlugin(), segViewerPlugin()],
  server: {
    host: true,
    port: 4444,
    strictPort: true,
    allowedHosts: ["basement", "basement-wsl"],
    fs: { allow: [".."] },
  },
});
