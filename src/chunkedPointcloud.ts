import { parseNpz } from "./npz";

export interface ChunkManifestEntry {
  file: string;
  points: number;
  bytes: number;
}

export interface ChunkManifest {
  version: number;
  totalPoints: number;
  chunkSize: number;
  chunks: ChunkManifestEntry[];
}

export interface ChunkedPointcloudChunk {
  index: number;
  pts3d: Float32Array;
  rgb: Uint8Array;
  conf: Float32Array;
}

/**
 * Stream a chunked point cloud from a manifest URL. Each yielded chunk is
 * a parsed (pts3d, rgb, conf) triple ready to feed into THREE.BufferGeometry.
 *
 * Sequential by design: the viewer creates one THREE.Points per chunk and
 * pushes vertex data to the GPU as each chunk lands. Parallel fetches would
 * race the GPU upload queue with no real wall-clock win on a single-server
 * dev setup.
 *
 * `onProgress` is called whenever bytes accumulate or a chunk finishes
 * parsing — `bytes` is total bytes received across all chunks so far,
 * `total` is the sum of `bytes` from the manifest.
 */
export async function* streamChunkedPointcloud(
  manifestUrl: string,
  onProgress?: (bytes: number, total: number) => void,
  onManifest?: (manifest: ChunkManifest) => void,
): AsyncGenerator<ChunkedPointcloudChunk, ChunkManifest, void> {
  const manifestResp = await fetch(manifestUrl);
  if (!manifestResp.ok) {
    throw new Error(`manifest fetch failed: ${manifestResp.status} ${manifestUrl}`);
  }
  const manifest: ChunkManifest = await manifestResp.json();
  onManifest?.(manifest);
  const baseUrl = manifestUrl.slice(0, manifestUrl.lastIndexOf("/") + 1);
  const total = manifest.chunks.reduce((s, c) => s + c.bytes, 0);
  let bytesSoFar = 0;
  onProgress?.(0, total);

  for (let i = 0; i < manifest.chunks.length; i++) {
    const meta = manifest.chunks[i];
    const url = baseUrl + encodeURIComponent(meta.file);
    const resp = await fetch(url);
    if (!resp.ok || !resp.body) {
      throw new Error(`chunk fetch failed: ${resp.status} ${url}`);
    }
    // Stream the chunk body so progress updates land while it's still
    // downloading, not just on completion.
    const reader = resp.body.getReader();
    const parts: Uint8Array[] = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      parts.push(value);
      received += value.length;
      onProgress?.(bytesSoFar + received, total);
    }
    bytesSoFar += received;
    const merged = new Uint8Array(received);
    let pos = 0;
    for (const p of parts) { merged.set(p, pos); pos += p.length; }

    const arrays = await parseNpz(merged.buffer);
    const pts3d = arrays["pts3d"];
    const rgb = arrays["rgb"];
    const conf = arrays["conf"];
    if (!pts3d || !rgb || !conf) {
      throw new Error(`chunk ${meta.file} missing pts3d/rgb/conf`);
    }
    yield {
      index: i,
      pts3d: new Float32Array(pts3d.data),
      rgb: new Uint8Array(rgb.data),
      conf: new Float32Array(conf.data),
    };
  }
  return manifest;
}
