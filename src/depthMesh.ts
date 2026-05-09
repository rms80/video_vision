/**
 * Build a triangle mesh from a depth map, positioned in world space.
 *
 * For each sampled pixel (u,v) with depth z:
 *   1. Unproject to camera space via intrinsics: x=z*(u-cx)/f, y=z*(v-cy)/f, z=z
 *   2. Transform to world space: X_world = Rᵀ(X_cam - t)
 *   3. Flip Y/Z for Three.js: (x, -y, -z) — COLMAP is Y-down/Z-forward,
 *      Three.js is Y-up/Z-back.
 *
 * Returns typed arrays in Three.js coordinates, ready for BufferGeometry.
 */

export interface CamerasJson {
  model: string;
  width: number;
  height: number;
  source_width: number;
  source_height: number;
  scale_factor: number;
  K: number[][];
  num_registered: number;
  frames: CameraFrame[];
  gravity_aligned?: boolean;
  worldup_id?: string;
  subsample_every?: number;
}

export interface CameraFrame {
  idx: number;
  name: string;
  registered: boolean;
  R: number[][] | null;
  t: number[] | null;
  sparse_obs: number[][];
}

export interface DepthMeshResult {
  positions: Float32Array;
  uvs: Float32Array;
  indices: Uint32Array;
  vertexCount: number;
  triangleCount: number;
}

/**
 * Build a depth mesh for a single frame.
 *
 * @param depthData  - Float32Array of shape [height * width], row-major (metric depth,
 *                     already aligned to COLMAP scale via affine fit)
 * @param width      - depth map width
 * @param height     - depth map height
 * @param K          - 3x3 intrinsics matrix
 * @param R          - 3x3 rotation (camera-from-world)
 * @param t          - 3-element translation (camera-from-world)
 * @param downsample - sample every Nth pixel (default 4)
 * @param depthRatioThreshold - max depth ratio between adjacent vertices before
 *                              skipping a triangle (avoids rubber-sheet at edges)
 */
export function buildDepthMesh(
  depthData: Float32Array,
  width: number,
  height: number,
  K: number[][],
  R: number[][],
  t: number[],
  downsample = 4,
  depthRatioThreshold = 1.05,
): DepthMeshResult {
  const f = K[0][0];
  const cx = K[0][2];
  const cy = K[1][2];

  // Precompute Rᵀ for world-from-camera transform
  const RT = transpose3(R);

  // Grid dimensions after downsampling
  const gw = Math.floor((width - 1) / downsample) + 1;
  const gh = Math.floor((height - 1) / downsample) + 1;
  const maxVerts = gw * gh;

  const positions = new Float32Array(maxVerts * 3);
  const uvs = new Float32Array(maxVerts * 2);
  // Track which grid cells have valid depth
  const valid = new Uint8Array(maxVerts);
  const depthValues = new Float32Array(maxVerts);

  let vi = 0;
  for (let gy = 0; gy < gh; gy++) {
    const v = Math.min(gy * downsample, height - 1);
    for (let gx = 0; gx < gw; gx++) {
      const u = Math.min(gx * downsample, width - 1);
      const z = depthData[v * width + u];
      const idx = gy * gw + gx;

      if (!Number.isFinite(z) || z <= 0) {
        valid[idx] = 0;
        depthValues[idx] = 0;
        // Still need placeholder positions
        positions[idx * 3] = 0;
        positions[idx * 3 + 1] = 0;
        positions[idx * 3 + 2] = 0;
      } else {
        valid[idx] = 1;
        depthValues[idx] = z;

        // Unproject to camera space
        const xc = z * (u - cx) / f;
        const yc = z * (v - cy) / f;
        const zc = z;

        // Camera-space point minus t
        const dx = xc - t[0];
        const dy = yc - t[1];
        const dz = zc - t[2];

        // World space: Rᵀ @ (X_cam - t), then flip Y/Z for Three.js coords
        const wx = RT[0][0] * dx + RT[0][1] * dy + RT[0][2] * dz;
        const wy = RT[1][0] * dx + RT[1][1] * dy + RT[1][2] * dz;
        const wz = RT[2][0] * dx + RT[2][1] * dy + RT[2][2] * dz;
        positions[idx * 3] = wx;
        positions[idx * 3 + 1] = -wy;
        positions[idx * 3 + 2] = -wz;
      }

      uvs[idx * 2] = u / width;
      uvs[idx * 2 + 1] = 1 - v / height;
      vi++;
    }
  }

  // Build index buffer — two triangles per quad, skip invalid / discontinuous
  const maxTris = (gw - 1) * (gh - 1) * 2;
  const indices = new Uint32Array(maxTris * 3);
  let ti = 0;

  for (let gy = 0; gy < gh - 1; gy++) {
    for (let gx = 0; gx < gw - 1; gx++) {
      const i00 = gy * gw + gx;
      const i10 = i00 + 1;
      const i01 = (gy + 1) * gw + gx;
      const i11 = i01 + 1;

      if (!valid[i00] || !valid[i10] || !valid[i01] || !valid[i11]) continue;

      const d00 = depthValues[i00];
      const d10 = depthValues[i10];
      const d01 = depthValues[i01];
      const d11 = depthValues[i11];

      // Check depth discontinuity for triangle 1: i00, i01, i10
      if (
        isDepthContinuous(d00, d01, depthRatioThreshold) &&
        isDepthContinuous(d00, d10, depthRatioThreshold) &&
        isDepthContinuous(d01, d10, depthRatioThreshold)
      ) {
        indices[ti * 3] = i00;
        indices[ti * 3 + 1] = i01;
        indices[ti * 3 + 2] = i10;
        ti++;
      }

      // Triangle 2: i10, i01, i11
      if (
        isDepthContinuous(d10, d01, depthRatioThreshold) &&
        isDepthContinuous(d10, d11, depthRatioThreshold) &&
        isDepthContinuous(d01, d11, depthRatioThreshold)
      ) {
        indices[ti * 3] = i10;
        indices[ti * 3 + 1] = i01;
        indices[ti * 3 + 2] = i11;
        ti++;
      }
    }
  }

  return {
    positions,
    uvs,
    indices: indices.slice(0, ti * 3),
    vertexCount: maxVerts,
    triangleCount: ti,
  };
}

function isDepthContinuous(a: number, b: number, threshold: number): boolean {
  const ratio = a > b ? a / b : b / a;
  return ratio < threshold;
}

function transpose3(m: number[][]): number[][] {
  return [
    [m[0][0], m[1][0], m[2][0]],
    [m[0][1], m[1][1], m[2][1]],
    [m[0][2], m[1][2], m[2][2]],
  ];
}
