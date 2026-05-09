/**
 * Build a point cloud from a CUT3R pointmap, positioned in world space.
 *
 * Each pixel (u,v) has a 3D point in camera space from pts3d (H,W,3).
 * Transform to world space: X_world = Rt @ (X_cam - t), then flip Y/Z
 * for Three.js (same convention as depthMesh.ts).
 *
 * Returns flat Float32Arrays for THREE.Points (positions + colors).
 */

export interface PointCloudResult {
  positions: Float32Array;
  colors: Float32Array;
  pointCount: number;
}

/**
 * @param pts3dData - Float32Array of shape [H * W * 3], row-major (camera-space XYZ)
 * @param confData  - Float32Array of shape [H * W], confidence per pixel
 * @param width     - pointmap width
 * @param height    - pointmap height
 * @param R         - 3x3 rotation (camera-from-world)
 * @param t         - 3-element translation (camera-from-world)
 * @param downsample - sample every Nth pixel (default 2)
 * @param confThreshold - minimum confidence to include a point (default 1.0)
 */
export function buildPointCloud(
  pts3dData: Float32Array,
  confData: Float32Array,
  width: number,
  height: number,
  R: number[][],
  t: number[],
  downsample = 2,
  confThreshold = 1.0,
): PointCloudResult {
  // Precompute Rt for world-from-camera transform
  const RT = [
    [R[0][0], R[1][0], R[2][0]],
    [R[0][1], R[1][1], R[2][1]],
    [R[0][2], R[1][2], R[2][2]],
  ];

  // Grid dimensions after downsampling
  const gw = Math.floor((width - 1) / downsample) + 1;
  const gh = Math.floor((height - 1) / downsample) + 1;
  const maxPoints = gw * gh;

  const positions = new Float32Array(maxPoints * 3);
  const colors = new Float32Array(maxPoints * 3);
  let count = 0;

  // Find confidence range for coloring
  let confMin = Infinity;
  let confMax = -Infinity;
  for (let gy = 0; gy < gh; gy++) {
    const v = Math.min(gy * downsample, height - 1);
    for (let gx = 0; gx < gw; gx++) {
      const u = Math.min(gx * downsample, width - 1);
      const ci = v * width + u;
      const c = confData[ci];
      if (c >= confThreshold && Number.isFinite(c)) {
        if (c < confMin) confMin = c;
        if (c > confMax) confMax = c;
      }
    }
  }
  const confRange = confMax > confMin ? confMax - confMin : 1;

  for (let gy = 0; gy < gh; gy++) {
    const v = Math.min(gy * downsample, height - 1);
    for (let gx = 0; gx < gw; gx++) {
      const u = Math.min(gx * downsample, width - 1);
      const pi = (v * width + u) * 3;
      const ci = v * width + u;

      const conf = confData[ci];
      if (conf < confThreshold) continue;

      const xc = pts3dData[pi];
      const yc = pts3dData[pi + 1];
      const zc = pts3dData[pi + 2];

      if (!Number.isFinite(xc) || !Number.isFinite(yc) || !Number.isFinite(zc)) continue;

      // Camera-space point minus t
      const dx = xc - t[0];
      const dy = yc - t[1];
      const dz = zc - t[2];

      // World space: Rt @ (X_cam - t), then flip Y/Z for Three.js
      const wx = RT[0][0] * dx + RT[0][1] * dy + RT[0][2] * dz;
      const wy = RT[1][0] * dx + RT[1][1] * dy + RT[1][2] * dz;
      const wz = RT[2][0] * dx + RT[2][1] * dy + RT[2][2] * dz;

      positions[count * 3] = wx;
      positions[count * 3 + 1] = -wy;
      positions[count * 3 + 2] = -wz;

      // Color from confidence: low=blue, high=white
      const norm = (conf - confMin) / confRange;
      colors[count * 3] = 0.3 + 0.7 * norm;
      colors[count * 3 + 1] = 0.3 + 0.7 * norm;
      colors[count * 3 + 2] = 1.0;

      count++;
    }
  }

  return {
    positions: positions.slice(0, count * 3),
    colors: colors.slice(0, count * 3),
    pointCount: count,
  };
}

/**
 * Build a point cloud from pre-computed world-space points (e.g. Pi3 global
 * reconstruction). Points are already in Three.js convention (x, -y, -z)
 * and coloured by their source-frame RGB.
 *
 * @param pts3dData - Float32Array of shape [M * 3], world-space XYZ
 * @param rgbData   - Uint8Array  of shape [M * 3], per-point RGB (0-255)
 * @param confData  - Float32Array of shape [M], confidence per point
 * @param downsample - keep every Nth point (default 1 = all)
 * @param confThreshold - minimum confidence to include (default 0)
 */
export function buildScenePointCloud(
  pts3dData: Float32Array,
  rgbData: Uint8Array,
  confData: Float32Array,
  downsample = 1,
  confThreshold = 0,
): PointCloudResult {
  const totalPoints = confData.length;
  const maxPoints = Math.ceil(totalPoints / downsample);

  const positions = new Float32Array(maxPoints * 3);
  const colors = new Float32Array(maxPoints * 3);
  let count = 0;

  for (let i = 0; i < totalPoints; i += downsample) {
    if (confData[i] < confThreshold) continue;

    const pi = i * 3;
    const x = pts3dData[pi];
    const y = pts3dData[pi + 1];
    const z = pts3dData[pi + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;

    positions[count * 3] = x;
    positions[count * 3 + 1] = y;
    positions[count * 3 + 2] = z;

    colors[count * 3] = rgbData[pi] / 255;
    colors[count * 3 + 1] = rgbData[pi + 1] / 255;
    colors[count * 3 + 2] = rgbData[pi + 2] / 255;

    count++;
  }

  return {
    positions: positions.slice(0, count * 3),
    colors: colors.slice(0, count * 3),
    pointCount: count,
  };
}
