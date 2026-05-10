import { defineConfig, Plugin } from "vite";
import solidPlugin from "vite-plugin-solid";
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import ffmpegPath from "ffmpeg-static";
import {
  SCENE_PLUGINS,
  getScenePluginOrDefault,
  DEFAULT_SCENE_PLUGIN_ID,
  type ScenePlugin,
} from "./src/scenePlugins";

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
// Hunyuan3D-Omni needs Python 3.10 + torch 2.5.1, so it lives in a sibling
// venv at hunyuan3d-omni/.venv (set up via uv) — not the project venv.
const HUNYUAN_VENV_PYTHON = path.resolve(
  __dirname,
  "..",
  "hunyuan3d-omni",
  ".venv",
  process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
);

type SceneJobState = {
  video: string;
  pluginId: string;
  stage: string;
  running: boolean;
  error: string | null;
  startedAt: number;
};
const sceneJobs = new Map<string, SceneJobState>(); // key: video filename

type ReconstructJobState = {
  running: boolean;
  frame: number;          // the frame actually used (post-snap)
  requestedFrame: number; // what the client asked for
  source: string;
  startedAt: number;
  finishedAt?: number;
  error: string | null;
  outPath?: string;  // absolute path to the .glb on disk
};
// key: `${video}::${analysis}` — one job per analysis
const reconstructJobs = new Map<string, ReconstructJobState>();

function sceneDir(video: string) {
  const stem = path.basename(video, path.extname(video));
  return path.join(ANALYSIS_DIR, stem, "_scene");
}

function deleteBoxerResults(video: string) {
  const stem = path.basename(video, path.extname(video));
  const videoDir = path.join(ANALYSIS_DIR, stem);
  if (!fs.existsSync(videoDir)) return;
  for (const name of fs.readdirSync(videoDir)) {
    if (name.startsWith("_")) continue;
    const boxerPath = path.join(videoDir, name, "boxer.json");
    if (fs.existsSync(boxerPath)) {
      fs.unlinkSync(boxerPath);
      console.log(`[boxer] deleted ${boxerPath}`);
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

/** Build the ["--source", X] args, or [] if the plugin has no source flag. */
function sourceFlagArgs(plugin: ScenePlugin): string[] {
  return plugin.sourceFlag ? ["--source", plugin.sourceFlag] : [];
}

function runPython(script: string, args: string[], logPath: string, pythonExe: string = VENV_PYTHON): Promise<void> {
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
    const cleanEnv: NodeJS.ProcessEnv = { ...process.env, PYTHONIOENCODING: "utf-8" };
    delete cleanEnv.VIRTUAL_ENV;
    delete cleanEnv.PYTHONHOME;
    delete cleanEnv.PYTHONPATH;
    delete cleanEnv.PYTHONSTARTUP;
    const py = spawn(pythonExe, [script, ...args], { stdio: ["ignore", "pipe", "pipe"], env: cleanEnv });
    py.stdout.on("data", (c: Buffer) => { try { fs.writeSync(fd, c); } catch {} });
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

function segViewerPlugin(): Plugin {
  return {
    name: "seg-viewer",
    configureServer(server) {
      // Ensure uploads/tmp directories exist
      if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });
      if (!fs.existsSync(TMP_DIR)) fs.mkdirSync(TMP_DIR, { recursive: true });

      // GET /api/gpu-status — current used/total GPU memory (MiB) via nvidia-smi.
      // Returns { used, total, util } or { error } if nvidia-smi is unavailable.
      server.middlewares.use("/api/gpu-status", (req, res, next) => {
        if (req.method !== "GET") return next();
        const proc = spawn("nvidia-smi", [
          "--query-gpu=memory.used,memory.total,utilization.gpu",
          "--format=csv,noheader,nounits",
        ]);
        let out = "";
        proc.stdout.on("data", (c: Buffer) => (out += c.toString()));
        proc.on("error", (err) => {
          res.statusCode = 500;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: err.message }));
        });
        proc.on("close", () => {
          res.setHeader("Content-Type", "application/json");
          const line = out.split(/\r?\n/).find((l) => l.trim()) ?? "";
          const [u, t, g] = line.split(",").map((s) => Number(s.trim()));
          if (!Number.isFinite(u) || !Number.isFinite(t)) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: "Could not parse nvidia-smi output" }));
            return;
          }
          res.end(JSON.stringify({ used: u, total: t, util: Number.isFinite(g) ? g : null }));
        });
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
          ff.on("close", (code) => {
            // Clean up raw file
            try { fs.unlinkSync(rawPath); } catch {}
            if (code === 0) {
              console.log(`[upload] re-encoded ${safeName} successfully`);
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ ok: true, filename: safeName }));
            } else {
              console.error(`[upload] ffmpeg error: ${stderr}`);
              res.statusCode = 500;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: `Re-encode failed: ${stderr.slice(-200)}` }));
            }
          });
          ff.on("error", (err) => {
            try { fs.unlinkSync(rawPath); } catch {}
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: err.message }));
          });
        });
        ws.on("error", (err) => {
          res.statusCode = 500;
          res.end(JSON.stringify({ error: err.message }));
        });
      });

      // POST /api/scene/prepare — run the pipeline defined by a scene plugin
      // body: { video, pluginId? }  (pluginId defaults to COLMAP for backcompat)
      server.middlewares.use("/api/scene/prepare", (req, res, next) => {
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
            if (plugin.requiresFrames && !fs.existsSync(path.join(sd, "frames.json"))) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Extract frames first (run COLMAP pipeline)" }));
              return;
            }
            deleteBoxerResults(video);
            if (plugin.cleanDir) {
              const d = path.join(sd, plugin.cleanDir);
              if (fs.existsSync(d)) fs.rmSync(d, { recursive: true, force: true });
            }
            const logPath = path.join(sd, plugin.logFile);
            const state: SceneJobState = {
              video,
              pluginId: plugin.id,
              stage: plugin.pipeline[0]?.stage ?? "running",
              running: true,
              error: null,
              startedAt: Date.now(),
            };
            sceneJobs.set(video, state);
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ ok: true, state }));

            (async () => {
              try {
                for (const step of plugin.pipeline) {
                  state.stage = step.stage;
                  const args = step.args.map((a) =>
                    a === "$VIDEO" ? videoPath : a === "$SCENE" ? sd : a,
                  );
                  if (plugin.id === "pi3" && step.script === "run_pi3.py"
                      && options && Number.isFinite(Number(options.subsample))) {
                    const n = Math.max(1, Math.round(Number(options.subsample)));
                    args.push("--subsample", String(n));
                  }
                  if (plugin.id === "vggt" && step.script === "run_vggt.py"
                      && options && Number.isFinite(Number(options.numFrames))) {
                    const T = Math.max(1, Math.round(Number(options.numFrames)));
                    args.push("--num-frames", String(T));
                    console.log(`[scene:vggt] targeting ${T} frames`);
                  }
                  console.log(`[scene:${plugin.id}] ${video}: ${step.stage} (${step.script})`);
                  await runPython(path.join(SCRIPTS_DIR, step.script), args, logPath);
                }
                console.log(`[scene:${plugin.id}] ${video}: done`);
              } catch (err: any) {
                state.error = err?.message ?? String(err);
                console.error(`[scene:${plugin.id}] ${video}: ${state.error}`);
              } finally {
                state.running = false;
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
            deleteBoxerResults(video);
            // Read worldup ID to stamp into cameras.json
            const wuPath = path.join(sd, "worldup.json");
            let worldupId = "";
            if (fs.existsSync(wuPath)) {
              try { worldupId = JSON.parse(fs.readFileSync(wuPath, "utf-8")).id ?? ""; } catch {}
            }
            const extraArgs = worldupId ? ["--worldup-id", worldupId] : [];
            // align_scene.py always wants --source=<camerasDir> so it knows where to read/write
            console.log(`[align] running on ${video} with ${points.length} floor points`);
            try {
              await runPython(scriptPath, [sd, pointsJson, "--source", plugin.camerasDir, ...extraArgs], logPath);
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
      server.middlewares.use("/api/videos", (_req, res) => {
        const files = fs.existsSync(UPLOADS_DIR)
          ? fs.readdirSync(UPLOADS_DIR).filter((f) => /\.(mp4|webm|mov|avi|mkv)$/i.test(f))
          : [];
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ videos: files }));
      });

      // POST /api/detect — extract frame 0 and run SAM3 detection on it
      // body: { video, x, y, label }
      server.middlewares.use("/api/detect", (req, res, next) => {
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

            // 2. Run python detection script
            const scriptPath = path.join(SCRIPTS_DIR, "detect_object.py");
            console.log(`[detect] running ${scriptPath} on ${framePath} click=(${x},${y}) label="${label}"`);
            await new Promise<void>((resolve, reject) => {
              const py = spawn(VENV_PYTHON, [scriptPath, framePath, String(x), String(y), label, outputJsonPath]);
              let stderr = "";
              py.stderr.on("data", (c) => (stderr += c.toString()));
              py.stdout.on("data", (c) => process.stdout.write(`[detect_object] ${c.toString()}`));
              py.on("close", (code) => {
                if (code === 0) resolve();
                else reject(new Error(`python exit ${code}: ${stderr}`));
              });
              py.on("error", reject);
            });

            // 3. Read and return result
            const result = JSON.parse(fs.readFileSync(outputJsonPath, "utf-8"));
            result.analysis = `${safeLabel}_${nextId}`;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(result));
          } catch (err: any) {
            console.error("[detect] error:", err);
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: err.message ?? String(err) }));
          }
        });
      });

      // POST /api/track — run SAM2 video tracking seeded from a previous detection
      // body: { video, analysis }  (analysis is the run folder name, e.g. "chair_1")
      server.middlewares.use("/api/track", (req, res, next) => {
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

            const scriptPath = path.join(SCRIPTS_DIR, "track_object.py");
            const boxerPath = path.join(runDir, "boxer.json");
            if (fs.existsSync(boxerPath)) { fs.unlinkSync(boxerPath); console.log(`[track] deleted ${boxerPath}`); }
            console.log(`[track] running ${scriptPath} on ${video} (${analysis})`);
            await new Promise<void>((resolve, reject) => {
              const py = spawn(VENV_PYTHON, [scriptPath, videoPath, detectJsonPath, runDir]);
              let stderr = "";
              py.stderr.on("data", (c) => (stderr += c.toString()));
              py.stdout.on("data", (c) => process.stdout.write(`[track_object] ${c.toString()}`));
              py.on("close", (code) => {
                if (code === 0) resolve();
                else reject(new Error(`python exit ${code}: ${stderr}`));
              });
              py.on("error", reject);
            });

            const trackJsonPath = path.join(runDir, "track.json");
            const result = JSON.parse(fs.readFileSync(trackJsonPath, "utf-8"));
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify(result));
          } catch (err: any) {
            console.error("[track] error:", err);
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: err.message ?? String(err) }));
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
      server.middlewares.use("/api/analyses", (req, res, next) => {
        if (req.method !== "GET") return next();
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

      // POST /api/boxer — run Boxer 3D bbox lifting
      // body: { video, analysis, label }
      server.middlewares.use("/api/boxer", (req, res, next) => {
        if (req.method !== "POST") return next();
        let body = "";
        req.on("data", (chunk: Buffer) => (body += chunk.toString()));
        req.on("end", async () => {
          try {
            const { video, analysis, label, fuse, source } = JSON.parse(body);
            if (!video || !analysis) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Missing video or analysis" }));
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

            // Delete stale results so polling doesn't see old runs
            for (const stale of ["boxer.json", "wilddet3d.json"]) {
              const p = path.join(runDir, stale);
              if (fs.existsSync(p)) { fs.unlinkSync(p); console.log(`[boxer] deleted ${p}`); }
            }

            const scriptPath = path.join(SCRIPTS_DIR, "run_boxer.py");
            const pyArgs = [sd, runDir, ...(label ? ["--label", label] : []), ...(fuse ? ["--fuse"] : []), ...sourceFlagArgs(plugin)];
            console.log(`[boxer] running ${scriptPath} on ${video} (${analysis})${fuse ? " +fuse" : ""}`);
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ ok: true, status: "running" }));

            const logPath = path.join(runDir, "boxer.log");
            runPython(scriptPath, pyArgs, logPath)
              .then(() => console.log(`[boxer] done: ${video} (${analysis})`))
              .catch((err) => console.error(`[boxer] failed: ${err}`));
          } catch (err: any) {
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: err.message }));
          }
        });
      });

      // GET /api/boxer-result?video=<filename>&name=<analysis> — load 3D bbox result
      server.middlewares.use("/api/boxer-result", (req, res, next) => {
        if (req.method !== "GET") return next();
        const url = new URL(req.url!, `http://${req.headers.host}`);
        const video = url.searchParams.get("video");
        const name = url.searchParams.get("name");
        if (!video || !name) { res.statusCode = 400; res.end(JSON.stringify({ error: "Missing video or name" })); return; }
        const stem = path.basename(video, path.extname(video));
        const jsonPath = path.join(ANALYSIS_DIR, stem, name, "boxer.json");
        if (!fs.existsSync(jsonPath)) { res.statusCode = 404; res.end(JSON.stringify({ error: "Boxer result not found" })); return; }
        const result = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify(result));
      });

      // POST /api/wilddet3d — run WildDet3D 3D bbox lifting
      // body: { video, analysis, label, source, useIntrinsics, useDepth }
      server.middlewares.use("/api/wilddet3d", (req, res, next) => {
        if (req.method !== "POST") return next();
        let body = "";
        req.on("data", (chunk: Buffer) => (body += chunk.toString()));
        req.on("end", async () => {
          try {
            const { video, analysis, label, source, useIntrinsics, useDepth } = JSON.parse(body);
            if (!video || !analysis) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Missing video or analysis" }));
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

            // Delete stale results so polling doesn't see old runs
            for (const stale of ["boxer.json", "wilddet3d.json"]) {
              const p = path.join(runDir, stale);
              if (fs.existsSync(p)) { fs.unlinkSync(p); console.log(`[wilddet3d] deleted ${p}`); }
            }

            const scriptPath = path.join(SCRIPTS_DIR, "run_wilddet3d.py");
            const intrinsicsFlag = useIntrinsics ? ["--use-intrinsics"] : [];
            const depthFlag = useDepth ? ["--use-depth"] : [];
            const pyArgs = [sd, runDir, ...(label ? ["--label", label] : []), ...sourceFlagArgs(plugin), ...intrinsicsFlag, ...depthFlag];
            console.log(`[wilddet3d] running ${scriptPath} on ${video} (${analysis})${useIntrinsics ? " +K" : ""}${useDepth ? " +depth" : ""}`);
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ ok: true, status: "running" }));

            const logPath = path.join(runDir, "wilddet3d.log");
            runPython(scriptPath, pyArgs, logPath)
              .then(() => console.log(`[wilddet3d] done: ${video} (${analysis})`))
              .catch((err) => console.error(`[wilddet3d] failed: ${err}`));
          } catch (err: any) {
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: err.message }));
          }
        });
      });

      // GET /api/wilddet3d-result?video=<filename>&name=<analysis> — load WildDet3D result
      server.middlewares.use("/api/wilddet3d-result", (req, res, next) => {
        if (req.method !== "GET") return next();
        const url = new URL(req.url!, `http://${req.headers.host}`);
        const video = url.searchParams.get("video");
        const name = url.searchParams.get("name");
        if (!video || !name) { res.statusCode = 400; res.end(JSON.stringify({ error: "Missing video or name" })); return; }
        const stem = path.basename(video, path.extname(video));
        const jsonPath = path.join(ANALYSIS_DIR, stem, name, "wilddet3d.json");
        if (!fs.existsSync(jsonPath)) { res.statusCode = 404; res.end(JSON.stringify({ error: "WildDet3D result not found" })); return; }
        const result = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify(result));
      });

      // POST /api/reconstruct3d — run Hunyuan3D-Omni on segmented per-frame pointmap
      // body: { video, analysis, frame, source }
      // Spawns segviewer/scripts/run_hunyuan_omni.py in the hunyuan venv (separate
      // Python 3.10 install). Output: <analysis>/hunyuan_omni/<NNNNNN>.glb.
      server.middlewares.use("/api/reconstruct3d", (req, res, next) => {
        if (req.method !== "POST") return next();
        let body = "";
        req.on("data", (c: Buffer) => (body += c.toString()));
        req.on("end", () => {
          try {
            const { video, analysis, frame, source } = JSON.parse(body);
            if (!video || !analysis || typeof frame !== "number") {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Missing video, analysis, or frame" }));
              return;
            }
            const plugin = resolveSourcePlugin(source);
            if (!plugin.features?.pointmap) {
              res.statusCode = 400;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({
                error: `Source '${plugin.id}' has no per-frame pointmap; pick Pi3 or MapAnything`,
              }));
              return;
            }
            const stem = path.basename(video, path.extname(video));
            const sd = sceneDir(video);
            const runDir = path.join(ANALYSIS_DIR, stem, analysis);
            // Pointmaps are subsampled (e.g. Pi3 every 3rd frame); the user's
            // current_frame may not have one. Snap to the nearest pointmap
            // frame that *also* has a mask + a frame jpg.
            const pmDir = path.join(sd, plugin.id, "pointmap");
            const framesDir = path.join(sd, "frames");
            const masksDir = path.join(runDir, "masks");
            if (!fs.existsSync(pmDir)) {
              res.statusCode = 404;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: `No pointmaps in ${pmDir} — re-run scene analysis` }));
              return;
            }
            const available = fs.readdirSync(pmDir)
              .filter((f) => f.endsWith(".npz"))
              .map((f) => parseInt(f.replace(".npz", ""), 10))
              .filter((n) => Number.isFinite(n))
              .filter((n) => {
                const pad = String(n).padStart(6, "0");
                return fs.existsSync(path.join(framesDir, `${pad}.jpg`))
                    && fs.existsSync(path.join(masksDir, `${pad}.png`));
              })
              .sort((a, b) => a - b);
            if (available.length === 0) {
              res.statusCode = 404;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "No frame has all of {pointmap, frame jpg, mask}" }));
              return;
            }
            const snappedFrame = available.reduce(
              (best, n) => Math.abs(n - frame) < Math.abs(best - frame) ? n : best,
              available[0],
            );
            if (snappedFrame !== frame) {
              console.log(`[hunyuan-omni] snapped frame ${frame} -> ${snappedFrame}`);
            }
            const padded = String(snappedFrame).padStart(6, "0");
            const meshPad = String(snappedFrame).padStart(3, "0");

            const jobKey = `${video}::${analysis}`;
            const existing = reconstructJobs.get(jobKey);
            if (existing && existing.running) {
              res.statusCode = 409;
              res.setHeader("Content-Type", "application/json");
              res.end(JSON.stringify({ error: "Reconstruction already running", state: existing }));
              return;
            }

            const outDir = path.join(runDir, "hunyuan_omni");
            fs.mkdirSync(outDir, { recursive: true });
            const outPath = path.join(outDir, `${meshPad}.glb`);
            const logPath = path.join(outDir, "hunyuan_omni.log");

            const state: ReconstructJobState = {
              running: true,
              frame: snappedFrame,
              requestedFrame: frame,
              source: plugin.id,
              startedAt: Date.now(),
              error: null,
              outPath,
            };
            reconstructJobs.set(jobKey, state);

            const scriptPath = path.join(SCRIPTS_DIR, "run_hunyuan_omni.py");
            const pyArgs = [
              sd, runDir,
              "--frame", String(snappedFrame),
              "--source", plugin.id,
              "--out", outPath,
            ];
            console.log(`[hunyuan-omni] running on ${video} (${analysis}) frame=${snappedFrame} source=${plugin.id}`);
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ ok: true, state }));

            runPython(scriptPath, pyArgs, logPath, HUNYUAN_VENV_PYTHON)
              .then(() => {
                state.running = false;
                state.finishedAt = Date.now();
                console.log(`[hunyuan-omni] done: ${outPath}`);
              })
              .catch((err) => {
                state.running = false;
                state.finishedAt = Date.now();
                state.error = err?.message ?? String(err);
                console.error(`[hunyuan-omni] failed: ${state.error}`);
              });
          } catch (err: any) {
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: err?.message ?? String(err) }));
          }
        });
      });

      // GET /api/reconstruct3d-status?video=<f>&analysis=<a> — poll job + artifact state
      server.middlewares.use("/api/reconstruct3d-status", (req, res, next) => {
        if (req.method !== "GET") return next();
        const url = new URL(req.url!, `http://${req.headers.host}`);
        const video = url.searchParams.get("video");
        const analysis = url.searchParams.get("analysis");
        if (!video || !analysis) {
          res.statusCode = 400;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: "Missing video or analysis" }));
          return;
        }
        const job = reconstructJobs.get(`${video}::${analysis}`) ?? null;
        const stem = path.basename(video, path.extname(video));
        const analysisDir = path.join(ANALYSIS_DIR, stem, analysis);
        // Walk all model subdirs (any direct child dir of the analysis that
        // contains .glb files) — not hardcoded to "hunyuan_omni" so future
        // model dirs show up automatically.
        const meshes: string[] = [];
        if (fs.existsSync(analysisDir)) {
          for (const entry of fs.readdirSync(analysisDir, { withFileTypes: true })) {
            if (!entry.isDirectory()) continue;
            const subdir = path.join(analysisDir, entry.name);
            for (const f of fs.readdirSync(subdir)) {
              if (f.endsWith(".glb")) meshes.push(`${entry.name}/${f}`);
            }
          }
          meshes.sort();
        }
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ job, meshes }));
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
        const rel = decodeURIComponent(req.url?.replace(/^\//, "") ?? "");
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
        const ext = path.extname(filepath).toLowerCase();
        const mime: Record<string, string> = {
          ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
          ".json": "application/json", ".glb": "model/gltf-binary",
          ".ply": "application/octet-stream",
        };
        res.setHeader("Content-Type", mime[ext] || "application/octet-stream");
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
    allowedHosts: ["basement"],
    fs: { allow: [".."] },
  },
});
