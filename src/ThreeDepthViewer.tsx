import { onMount, onCleanup, createEffect, on } from "solid-js";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { parseNpz } from "./npz";
import { buildDepthMesh, type CamerasJson, type CameraFrame } from "./depthMesh";
import { buildPointCloud, buildScenePointCloud } from "./pointcloudMesh";
import { getScenePluginOrDefault } from "./scenePlugins";
import { streamChunkedPointcloud } from "./chunkedPointcloud";

export interface BoxerBox {
  center: number[];
  size: number[];
  R: number[][];
  t: number[];
  corners: number[][];
  confidence: number;
}

export interface BoxerFrame {
  frame: number;
  colmap_frame: number;
  boxes: BoxerBox[];
}

export interface BoxerResult {
  label: string;
  gravity: number[];
  frames: BoxerFrame[];
  fused_boxes?: BoxerBox[];
  num_frames_with_boxes?: number;
}

export interface ThreeDepthViewerProps {
  videoName: string | null;
  currentFrame: number;
  depthFrames: number[];
  depthStem: string;
  cameras: CamerasJson | null;
  visible: boolean;
  downsample?: number;
  /** 3D-box result for the currently selected solver, or null if none. */
  boxResult: BoxerResult | null;
  /** ID of the solver that produced boxResult — drives the box wireframe color. */
  boxSolverId: string;
  dataVersion?: number;
  sceneSource?: string;
  usePointmap?: boolean;
  scenePointmapMode?: boolean;
  /** When true, render the per-object world-space cloud at objectPointmapUrl. */
  objectPointmapMode?: boolean;
  /** URL of the .npz produced by build_object_pointmap.py for the current analysis + source. */
  objectPointmapUrl?: string | null;
  showCameraPath?: boolean;
  onReady?: (actions: { snapCamera: () => void; fitAll: () => void }) => void;
  /**
   * Status callback for the global scene_pointmap fetch — fires while the
   * cloud is downloading (chunks stream in one at a time) and when the
   * cloud is disposed. Used by the parent to drive the loading badge and
   * the point-count readout in the bottom bar.
   *   progress: null until the manifest reports a non-zero total byte size
   *   pointCount: null until at least one chunk has been added to the scene
   *   chunksLoaded / totalChunks: null until the manifest has been fetched
   */
  onScenePointmapStatus?: (state: PointcloudStreamStatus) => void;
  /** Same contract as onScenePointmapStatus, for the per-object cloud fetch. */
  onObjectPointmapStatus?: (state: PointcloudStreamStatus) => void;
}

export interface PointcloudStreamStatus {
  loading: boolean;
  progress: number | null;
  pointCount: number | null;
  chunksLoaded: number | null;
  totalChunks: number | null;
}

interface CachedEntry {
  object: THREE.Mesh | THREE.Points;
  frameIdx: number;
  isPointCloud: boolean;
}

export default function ThreeDepthViewer(props: ThreeDepthViewerProps) {
  let containerEl!: HTMLDivElement;
  let renderer: THREE.WebGLRenderer | null = null;
  let scene: THREE.Scene | null = null;
  let camera: THREE.PerspectiveCamera | null = null;
  let controls: OrbitControls | null = null;
  let animId: number | null = null;
  let resizeObs: ResizeObserver | null = null;

  // Mesh cache — maps depth frame index to cached entry (mesh or point cloud).
  // The cache and meshGroup.children are kept in lockstep: every cached entry
  // is a child of meshGroup, and disposeMeshes wipes both together.
  const meshCache = new Map<number, CachedEntry>();
  let meshGroup: THREE.Group | null = null;
  let currentShownFrame: number | null = null;
  // Track loading to avoid double-fetching
  const loadingFrames = new Set<number>();

  // Scene-level pointmap (global reconstruction). Sharded into chunks at
  // write time (scene_pointmap_NNN.npz + manifest); each chunk becomes its
  // own THREE.Points inside scenePointmapGroup so the cloud renders
  // progressively as it streams in.
  let scenePointmapGroup: THREE.Group | null = null;
  let scenePointmapLoading = false;
  // Aborts an in-flight stream when we leave/dispose the cloud mid-fetch.
  let scenePointmapAbortToken = 0;

  // Per-object cloud: same data layout as scene_pointmap.npz (pts3d/rgb/conf
  // already in Three.js convention), built by build_object_pointmap.py from
  // depth ∩ track-mask. The object-mode effect disposes on URL change.
  let objectPointmapGroup: THREE.Group | null = null;
  let objectPointmapLoading = false;
  let objectPointmapAbortToken = 0;

  // Camera path visualization
  let cameraPathLine: THREE.Line | null = null;
  let cameraMarker: THREE.Mesh | null = null;
  let cameraFrustums: THREE.LineSegments[] = [];
  // Map frame idx → index into the path points array
  let cameraPathFrameIndices: Map<number, number> = new Map();
  let cameraWorldPositions: THREE.Vector3[] = [];

  function nearestDepthFrame(frame: number): number | null {
    const frames = props.depthFrames;
    if (!frames.length) return null;
    let best = frames[0];
    let bestDist = Math.abs(frame - best);
    for (const f of frames) {
      const d = Math.abs(frame - f);
      if (d < bestDist) { best = f; bestDist = d; }
    }
    return best;
  }

  function findCameraFrame(idx: number): CameraFrame | null {
    if (!props.cameras) return null;
    return props.cameras.frames.find((f) => f.idx === idx && f.registered && f.R && f.t) ?? null;
  }

  async function loadAndShowFrame(depthIdx: number) {
    if (!scene || !meshGroup || !props.cameras) return;

    // Already cached?
    const cached = meshCache.get(depthIdx);
    if (cached) {
      showOnlyMesh(depthIdx);
      return;
    }

    if (loadingFrames.has(depthIdx)) return;
    loadingFrames.add(depthIdx);

    try {
      const stem = props.depthStem;
      const padded = String(depthIdx).padStart(6, "0");
      const camFrame = findCameraFrame(depthIdx);
      if (!camFrame || !camFrame.R || !camFrame.t) return;

      const plugin = getScenePluginOrDefault(props.sceneSource);
      if (props.usePointmap && plugin.features?.pointmap) {
        // --- Pointmap path: load raw 3D points, render as THREE.Points ---
        // Pointmap .npz lives next to the plugin's cameras.json, in a
        // "pointmap" sibling of its depthDir.
        const pmResp = await fetch(`/analysis/${stem}/_scene/${plugin.camerasDir}/pointmap/${padded}.npz`);
        if (!pmResp.ok) return;
        const pmBuf = await pmResp.arrayBuffer();
        const arrays = await parseNpz(pmBuf);
        const pts3dArr = arrays["pts3d"];
        const confArr = arrays["conf"];
        if (!pts3dArr || !confArr) return;

        const [h, w] = confArr.shape;
        const downsample = props.downsample ?? 2;
        const result = buildPointCloud(
          new Float32Array(pts3dArr.data),
          new Float32Array(confArr.data),
          w, h,
          camFrame.R,
          camFrame.t,
          downsample,
        );

        if (result.pointCount === 0) return;

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(result.positions, 3));
        geometry.setAttribute("color", new THREE.BufferAttribute(result.colors, 3));
        geometry.computeBoundingBox();

        // Scale point size relative to the cloud extent
        const bbox = geometry.boundingBox!;
        const extent = bbox.getSize(new THREE.Vector3());
        const pointSize = Math.max(extent.x, extent.y, extent.z) * 0.003;

        const material = new THREE.PointsMaterial({
          size: pointSize,
          vertexColors: true,
          sizeAttenuation: true,
        });
        const points = new THREE.Points(geometry, material);
        points.visible = false;

        // Bounding box wireframe
        const boxHelper = new THREE.Box3Helper(bbox, new THREE.Color(0x888888));
        points.add(boxHelper);

        points.userData.frameIdx = depthIdx;
        meshGroup!.add(points);
        meshCache.set(depthIdx, { object: points, frameIdx: depthIdx, isPointCloud: true });
      } else {
        // --- Depth mesh path: load depth, unproject through K, render as triangle mesh ---
        const depthResp = await fetch(`/analysis/${stem}/_scene/${plugin.depthDir}/${padded}.npz`);
        if (!depthResp.ok) return;
        const depthBuf = await depthResp.arrayBuffer();
        const arrays = await parseNpz(depthBuf);
        const depthArr = arrays["depth"];
        if (!depthArr) return;
        const [h, w] = depthArr.shape;

        const downsample = props.downsample ?? 4;
        const result = buildDepthMesh(
          new Float32Array(depthArr.data),
          w, h,
          props.cameras.K,
          camFrame.R,
          camFrame.t,
          downsample,
        );

        if (result.triangleCount === 0) return;

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(result.positions, 3));
        geometry.setAttribute("uv", new THREE.BufferAttribute(result.uvs, 2));
        geometry.setIndex(new THREE.BufferAttribute(result.indices, 1));
        geometry.computeVertexNormals();

        const texUrl = `/analysis/${stem}/_scene/frames/${padded}.jpg`;
        const texture = await new Promise<THREE.Texture>((resolve) => {
          new THREE.TextureLoader().load(texUrl, (tex) => {
            tex.colorSpace = THREE.SRGBColorSpace;
            tex.minFilter = THREE.LinearMipmapLinearFilter;
            tex.magFilter = THREE.LinearFilter;
            resolve(tex);
          });
        });

        const material = new THREE.MeshBasicMaterial({
          map: texture,
          side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.visible = false;

        mesh.userData.frameIdx = depthIdx;
        meshGroup!.add(mesh);
        meshCache.set(depthIdx, { object: mesh, frameIdx: depthIdx, isPointCloud: false });
      }

      showOnlyMesh(depthIdx);
      if (needsFit) {
        needsFit = false;
        fitCameraToScene();
      }
    } finally {
      loadingFrames.delete(depthIdx);
    }
  }

  function showOnlyMesh(depthIdx: number) {
    // meshGroup is the single source of truth for per-frame meshes.
    // Iterating its children (rather than meshCache) defends against any
    // divergence between cache state and what's actually in the scene.
    if (meshGroup) {
      for (const child of meshGroup.children) {
        const idx = child.userData.frameIdx as number | undefined;
        child.visible = idx === depthIdx;
      }
    }
    currentShownFrame = depthIdx;
  }



  function buildCameraPath(cameras: CamerasJson) {
    if (!scene) return;
    disposeCameraPath();

    const registeredFrames = cameras.frames
      .filter((f) => f.registered && f.R && f.t)
      .sort((a, b) => a.idx - b.idx);

    if (registeredFrames.length < 2) return;

    const points: THREE.Vector3[] = [];
    cameraPathFrameIndices = new Map();
    cameraWorldPositions = [];

    // Frustum geometry: 4 corner rays in camera space at unit depth,
    // scaled by frustumDepth to set the visible size
    const K = cameras.K;
    const fx = K[0][0], fy = K[1][1], cxK = K[0][2], cyK = K[1][2];
    const w = cameras.width, h = cameras.height;
    // Four image corners unprojected to camera space at z=1
    const corners = [
      [(0 - cxK) / fx, (0 - cyK) / fy, 1],       // top-left
      [(w - cxK) / fx, (0 - cyK) / fy, 1],       // top-right
      [(w - cxK) / fx, (h - cyK) / fy, 1],       // bottom-right
      [(0 - cxK) / fx, (h - cyK) / fy, 1],       // bottom-left
    ];

    // We'll set frustum depth after computing all camera positions
    // to scale it relative to the path extent
    const allPositions: THREE.Vector3[] = [];

    for (const f of registeredFrames) {
      const R = f.R!;
      const t = f.t!;
      // Camera center in COLMAP world: c = -Rᵀ t, then flip Y/Z for Three.js
      const cx = -(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2]);
      const cy = -(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2]);
      const cz = -(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2]);
      const pos = new THREE.Vector3(cx, -cy, -cz);
      cameraPathFrameIndices.set(f.idx, points.length);
      cameraWorldPositions.push(pos);
      allPositions.push(pos);
      points.push(pos);
    }

    // Compute frustum depth as a fraction of the path extent
    const pathBox = new THREE.Box3().setFromPoints(allPositions);
    const pathSize = pathBox.getSize(new THREE.Vector3());
    const frustumDepth = Math.max(pathSize.x, pathSize.y, pathSize.z) * 0.04;

    // Build frustums
    const frustumMat = new THREE.LineBasicMaterial({ color: 0x4488cc, opacity: 0.6, transparent: true });
    const fwdMat = new THREE.LineBasicMaterial({ color: 0x4444ff }); // blue = forward
    const upMat = new THREE.LineBasicMaterial({ color: 0x44ff44 });  // green = up
    for (let fi = 0; fi < registeredFrames.length; fi++) {
      const f = registeredFrames[fi];
      const R = f.R!;
      const t = f.t!;

      // Transform a camera-space point to Three.js coords: Rᵀ(p - t), flip Y/Z
      const toThreeJS = (px: number, py: number, pz: number): THREE.Vector3 => {
        const dx = px - t[0], dy = py - t[1], dz = pz - t[2];
        const wx = R[0][0] * dx + R[1][0] * dy + R[2][0] * dz;
        const wy = R[0][1] * dx + R[1][1] * dy + R[2][1] * dz;
        const wz = R[0][2] * dx + R[1][2] * dy + R[2][2] * dz;
        return new THREE.Vector3(wx, -wy, -wz);
      };

      const origin = allPositions[fi];
      const far = corners.map(([cx, cy, cz]) =>
        toThreeJS(cx * frustumDepth, cy * frustumDepth, cz * frustumDepth)
      );

      // Camera axes in Three.js coords (at frustumDepth length)
      const axisLen = frustumDepth * 1.2;
      const fwd = toThreeJS(0, 0, axisLen);   // +Z camera = forward (look direction)
      const up = toThreeJS(0, -axisLen, 0);   // -Y camera = up (COLMAP Y is down)

      // 8 line segments for frustum wireframe
      const verts = new Float32Array(8 * 2 * 3);
      let vi = 0;
      const put = (v: THREE.Vector3) => { verts[vi++] = v.x; verts[vi++] = v.y; verts[vi++] = v.z; };
      for (let c = 0; c < 4; c++) { put(origin); put(far[c]); }
      for (let c = 0; c < 4; c++) { put(far[c]); put(far[(c + 1) % 4]); }

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
      const frustum = new THREE.LineSegments(geo, frustumMat);
      scene.add(frustum);
      cameraFrustums.push(frustum);

      // Forward axis (blue)
      const fwdVerts = new Float32Array([origin.x, origin.y, origin.z, fwd.x, fwd.y, fwd.z]);
      const fwdGeo = new THREE.BufferGeometry();
      fwdGeo.setAttribute("position", new THREE.BufferAttribute(fwdVerts, 3));
      const fwdLine = new THREE.LineSegments(fwdGeo, fwdMat);
      scene.add(fwdLine);
      cameraFrustums.push(fwdLine);

      // Up axis (green)
      const upVerts = new Float32Array([origin.x, origin.y, origin.z, up.x, up.y, up.z]);
      const upGeo = new THREE.BufferGeometry();
      upGeo.setAttribute("position", new THREE.BufferAttribute(upVerts, 3));
      const upLine = new THREE.LineSegments(upGeo, upMat);
      scene.add(upLine);
      cameraFrustums.push(upLine);
    }

    // Camera path line
    const showPath = props.showCameraPath !== false;
    const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
    const lineMat = new THREE.LineBasicMaterial({ color: 0xe94560, linewidth: 2 });
    cameraPathLine = new THREE.Line(lineGeo, lineMat);
    cameraPathLine.visible = showPath;
    scene.add(cameraPathLine);

    // Current camera marker (small sphere)
    const markerGeo = new THREE.SphereGeometry(frustumDepth * 0.3, 8, 8);
    const markerMat = new THREE.MeshBasicMaterial({ color: 0x2ecc71 });
    cameraMarker = new THREE.Mesh(markerGeo, markerMat);
    cameraMarker.visible = false; // shown by updateCameraMarker
    scene.add(cameraMarker);

    // Apply visibility to frustums
    for (const f of cameraFrustums) f.visible = showPath;
  }

  function updateCameraMarker(frameIdx: number) {
    if (!cameraMarker) return;
    // Find nearest registered frame in the path
    let bestIdx = -1;
    let bestDist = Infinity;
    for (const [fIdx, pIdx] of cameraPathFrameIndices) {
      const d = Math.abs(fIdx - frameIdx);
      if (d < bestDist) { bestDist = d; bestIdx = pIdx; }
    }
    if (bestIdx >= 0 && bestIdx < cameraWorldPositions.length) {
      cameraMarker.position.copy(cameraWorldPositions[bestIdx]);
      cameraMarker.visible = props.showCameraPath !== false;
    }
  }

  /** Set the Three.js camera to match the current COLMAP camera pose */
  function snapToColmapCamera() {
    if (!camera || !controls || !props.cameras) return;
    const depthIdx = nearestDepthFrame(props.currentFrame);
    if (depthIdx == null) return;
    const camFrame = findCameraFrame(depthIdx);
    if (!camFrame || !camFrame.R || !camFrame.t) return;

    const R = camFrame.R;
    const t = camFrame.t;

    // Camera center in COLMAP world space: c = -Rᵀ t
    const cx = -(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2]);
    const cy = -(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2]);
    const cz = -(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2]);

    // Camera look direction in COLMAP world space: Rᵀ @ [0,0,1] (camera +Z axis)
    const lookX = R[2][0]; // Rᵀ column 2 = R row 2
    const lookY = R[2][1];
    const lookZ = R[2][2];

    // Set FOV from COLMAP intrinsics: vfov = 2 * atan(h / (2*fy))
    const fy = props.cameras.K[1][1];
    const imgH = props.cameras.height;
    const vfovDeg = 2 * Math.atan(imgH / (2 * fy)) * (180 / Math.PI);
    camera.fov = vfovDeg;
    camera.updateProjectionMatrix();

    // Flip Y/Z to convert COLMAP world → Three.js coords
    const posTJS = new THREE.Vector3(cx, -cy, -cz);
    const lookTJS = new THREE.Vector3(lookX, -lookY, -lookZ).normalize();

    camera.position.copy(posTJS);
    controls.target.copy(posTJS.clone().add(lookTJS.multiplyScalar(2)));
    controls.update();

    // Adjust near plane to the closest vertex of the currently-shown mesh
    // (measured along the view axis) so nothing snapped-in-front clips.
    const cached = meshCache.get(depthIdx);
    const posAttr = cached?.object.geometry.getAttribute("position");
    if (posAttr) {
      const forward = controls.target.clone().sub(camera.position).normalize();
      const cp = camera.position;
      const arr = posAttr.array as ArrayLike<number>;
      let minDist = Infinity;
      for (let i = 0; i < arr.length; i += 3) {
        const dx = arr[i] - cp.x, dy = arr[i + 1] - cp.y, dz = arr[i + 2] - cp.z;
        const d = dx * forward.x + dy * forward.y + dz * forward.z;
        if (d > 0 && d < minDist) minDist = d;
      }
      if (Number.isFinite(minDist)) {
        camera.near = Math.max(minDist * 0.5, 1e-4);
        camera.updateProjectionMatrix();
      }
    }
  }

  function disposeCameraPath() {
    if (cameraPathLine) {
      cameraPathLine.geometry.dispose();
      (cameraPathLine.material as THREE.LineBasicMaterial).dispose();
      scene?.remove(cameraPathLine);
      cameraPathLine = null;
    }
    if (cameraMarker) {
      cameraMarker.geometry.dispose();
      (cameraMarker.material as THREE.MeshBasicMaterial).dispose();
      scene?.remove(cameraMarker);
      cameraMarker = null;
    }
    // Collect unique materials before disposing
    const mats = new Set<THREE.Material>();
    for (const f of cameraFrustums) {
      f.geometry.dispose();
      const m = f.material;
      if (m instanceof THREE.Material) mats.add(m);
      scene?.remove(f);
    }
    for (const m of mats) m.dispose();
    cameraFrustums = [];
    cameraPathFrameIndices = new Map();
    cameraWorldPositions = [];
  }

  // 3D bounding box visualization (one solver active at a time).
  // Per-frame groups so we can show/hide by frame.
  let boxesGroup: THREE.Group | null = null;
  let boxFrameGroups: Map<number, THREE.Group> = new Map();
  let currentBoxFrame: number | null = null;

  // Wireframe colors per solver. Boxer keeps the established orange/green-fused
  // palette; WildDet3D stays cyan so the two are visually distinct when the
  // user toggles between them.
  const SOLVER_COLORS: Record<string, { perFrame: number; fused: number }> = {
    boxer: { perFrame: 0xff8800, fused: 0x00ff88 },
    wilddet3d: { perFrame: 0x00ccff, fused: 0x00ccff },
  };

  function makeTextSprite(text: string, color: string): THREE.Sprite {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d")!;
    const fontSize = 48;
    ctx.font = `bold ${fontSize}px monospace`;
    const metrics = ctx.measureText(text);
    const pad = 8;
    canvas.width = Math.ceil(metrics.width) + pad * 2;
    canvas.height = fontSize + pad * 2;
    // Re-set font after resize
    ctx.font = `bold ${fontSize}px monospace`;
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = color;
    ctx.textBaseline = "middle";
    ctx.fillText(text, pad, canvas.height / 2);
    const tex = new THREE.CanvasTexture(canvas);
    tex.minFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
    const sprite = new THREE.Sprite(mat);
    // Scale so the label is readable but not huge
    sprite.scale.set(canvas.width / canvas.height * 0.0375, 0.0375, 1);
    return sprite;
  }

  function buildBoxWireframe(box: BoxerBox, parent: THREE.Object3D, mat: THREE.LineBasicMaterial, showDims = false) {
    // box.corners is 8 corners in COLMAP world coords
    // Convert to Three.js: (x, -y, -z)
    const c = box.corners.map(
      (p) => new THREE.Vector3(p[0], -p[1], -p[2])
    );

    // 12 edges of a box
    const edgeIndices = [
      0,1, 1,2, 2,3, 3,0,
      4,5, 5,6, 6,7, 7,4,
      0,4, 1,5, 2,6, 3,7,
    ];

    const verts = new Float32Array(edgeIndices.length * 3);
    for (let i = 0; i < edgeIndices.length; i++) {
      const p = c[edgeIndices[i]];
      verts[i * 3] = p.x;
      verts[i * 3 + 1] = p.y;
      verts[i * 3 + 2] = p.z;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    const lines = new THREE.LineSegments(geo, mat);
    parent.add(lines);

    // Dimension labels along one edge per axis
    if (showDims) {
      const color = "#" + mat.color.getHexString();
      // Three unique edges from corner 0: 0→1, 0→3, 0→4
      const dimEdges: [number, number][] = [[0, 1], [0, 3], [0, 4]];
      for (const [a, b] of dimEdges) {
        const len = c[a].distanceTo(c[b]);
        const label = `${(len * 100).toFixed(1)}cm`;
        const sprite = makeTextSprite(label, color);
        const mid = c[a].clone().add(c[b]).multiplyScalar(0.5);
        sprite.position.copy(mid);
        // Scale label size relative to box extent
        const maxDim = Math.max(c[0].distanceTo(c[1]), c[0].distanceTo(c[3]), c[0].distanceTo(c[4]));
        const s = maxDim * 0.0375;
        sprite.scale.set(sprite.scale.x / 0.0375 * s, s, 1);
        parent.add(sprite);
      }
    }
  }

  function buildBoxes(result: BoxerResult, solverId: string) {
    if (!scene) return;
    disposeBoxes();

    boxesGroup = new THREE.Group();
    scene.add(boxesGroup);

    const colors = SOLVER_COLORS[solverId] ?? SOLVER_COLORS.boxer;
    const boxMat = new THREE.LineBasicMaterial({ color: colors.perFrame, linewidth: 2 });
    const fusedMat = new THREE.LineBasicMaterial({ color: colors.fused, linewidth: 2 });

    const hasFused = !!(result.fused_boxes && result.fused_boxes.length > 0);

    // If fused boxes exist, show them as always-visible static boxes with dimensions
    if (hasFused) {
      for (const box of result.fused_boxes!) {
        buildBoxWireframe(box, boxesGroup, fusedMat, true);
      }
    }

    // Also build per-frame boxes (shown per-frame); fade them when fused
    // boxes are the primary readout.
    const perFrameMat = hasFused
      ? new THREE.LineBasicMaterial({ color: colors.perFrame, opacity: 0.6, transparent: true })
      : boxMat;

    for (const frame of result.frames) {
      if (frame.boxes.length === 0) continue;
      const frameGroup = new THREE.Group();
      frameGroup.visible = false;
      boxesGroup.add(frameGroup);
      boxFrameGroups.set(frame.frame, frameGroup);

      for (const box of frame.boxes) {
        buildBoxWireframe(box, frameGroup, perFrameMat, !hasFused);
      }
    }

    // Show the per-frame box for the current frame immediately
    currentBoxFrame = null;
    updateBoxFrame(props.currentFrame);
  }

  function updateBoxFrame(frameIdx: number) {
    if (!boxesGroup || boxFrameGroups.size === 0) return;

    // Find nearest box frame
    let bestFrame = -1;
    let bestDist = Infinity;
    for (const fIdx of boxFrameGroups.keys()) {
      const d = Math.abs(fIdx - frameIdx);
      if (d < bestDist) { bestDist = d; bestFrame = fIdx; }
    }

    if (bestFrame === currentBoxFrame) return;
    currentBoxFrame = bestFrame;

    for (const [fIdx, group] of boxFrameGroups) {
      group.visible = fIdx === bestFrame;
    }
  }

  function disposeBoxes() {
    if (boxesGroup && scene) {
      boxesGroup.traverse((obj) => {
        if (obj instanceof THREE.LineSegments) {
          obj.geometry.dispose();
          if (obj.material instanceof THREE.Material) obj.material.dispose();
        } else if (obj instanceof THREE.Sprite) {
          const mat = obj.material as THREE.SpriteMaterial;
          mat.map?.dispose();
          mat.dispose();
        }
      });
      scene.remove(boxesGroup);
    }
    boxesGroup = null;
    boxFrameGroups = new Map();
    currentBoxFrame = null;
  }

  function disposeMeshes() {
    for (const [, cached] of meshCache) {
      cached.object.geometry.dispose();
      if (cached.isPointCloud) {
        (cached.object.material as THREE.PointsMaterial).dispose();
      } else {
        const mat = cached.object.material as THREE.MeshBasicMaterial;
        mat.map?.dispose();
        mat.dispose();
      }
    }
    // Wipe meshGroup wholesale — covers both cached entries and any orphans.
    if (meshGroup) {
      while (meshGroup.children.length) meshGroup.remove(meshGroup.children[0]);
    }
    meshCache.clear();
    currentShownFrame = null;
    loadingFrames.clear();
  }

  function disposeChunkedGroup(group: THREE.Group | null): boolean {
    if (!group || !scene) return false;
    for (const child of group.children) {
      if (child instanceof THREE.Points) {
        child.geometry.dispose();
        (child.material as THREE.PointsMaterial).dispose();
      }
    }
    scene.remove(group);
    return true;
  }

  function disposeScenePointmap() {
    // Bump the abort token so any in-flight stream's per-chunk check bails
    // before adding more children to a stale group.
    scenePointmapAbortToken++;
    const wiped = disposeChunkedGroup(scenePointmapGroup);
    scenePointmapGroup = null;
    if (wiped) {
      // Only emit the cleared status when we actually wiped a cloud — this
      // path also runs as a no-op on cleanup, where firing would clobber a
      // freshly-loaded count from a different analysis.
      props.onScenePointmapStatus?.(emptyStreamStatus());
    }
  }

  function emptyStreamStatus(): PointcloudStreamStatus {
    return { loading: false, progress: null, pointCount: null, chunksLoaded: null, totalChunks: null };
  }

  /**
   * Stream a chunked point cloud from a manifest URL into a new THREE.Group
   * of THREE.Points, one per chunk. Each chunk is added to the scene as
   * soon as it's parsed so the cloud appears progressively.
   *
   * Point size is fixed across all chunks of a given cloud, computed from
   * the first chunk's extent — a chunk-by-chunk recompute would make
   * earlier points visibly resize as later, larger-extent chunks arrive.
   */
  async function streamCloudIntoGroup(opts: {
    manifestUrl: string;
    pointSizeFactor: number;
    abortToken: () => number;
    initialAbortToken: number;
    onStatus: (s: PointcloudStreamStatus) => void;
    onFirstChunk?: () => void;
  }): Promise<{ group: THREE.Group; totalPoints: number } | null> {
    if (!scene) return null;
    const group = new THREE.Group();
    let totalPoints = 0;
    let pointSize: number | null = null;
    let material: THREE.PointsMaterial | null = null;
    let totalBytes = 0;
    let totalChunks: number | null = null;
    let chunksLoaded = 0;

    const emit = (bytes: number) => {
      opts.onStatus({
        loading: true,
        progress: totalBytes > 0 ? Math.min(1, bytes / totalBytes) : null,
        pointCount: totalPoints || null,
        chunksLoaded,
        totalChunks,
      });
    };

    const stream = streamChunkedPointcloud(
      opts.manifestUrl,
      (bytes, total) => {
        totalBytes = total;
        emit(bytes);
      },
      (manifest) => {
        totalChunks = manifest.chunks.length;
        emit(0);
      },
    );

    let added = false;
    for (;;) {
      const next = await stream.next();
      if (opts.abortToken() !== opts.initialAbortToken) {
        // Caller disposed mid-stream; drop everything we've built so far
        // and clear the loading indicator (dispose can't always do this
        // because the cloud was never installed into scenePointmapGroup).
        for (const child of group.children) {
          if (child instanceof THREE.Points) {
            child.geometry.dispose();
            (child.material as THREE.PointsMaterial).dispose();
          }
        }
        if (added) scene.remove(group);
        opts.onStatus(emptyStreamStatus());
        return null;
      }
      if (next.done) break;
      const chunk = next.value;

      const result = buildScenePointCloud(chunk.pts3d, chunk.rgb, chunk.conf);
      chunksLoaded++;
      if (result.pointCount === 0) {
        emit(totalBytes);
        continue;
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(result.positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(result.colors, 3));
      geometry.computeBoundingBox();

      if (material === null) {
        const bbox = geometry.boundingBox!;
        const extent = bbox.getSize(new THREE.Vector3());
        pointSize = Math.max(extent.x, extent.y, extent.z) * opts.pointSizeFactor;
        material = new THREE.PointsMaterial({
          size: pointSize,
          vertexColors: true,
          sizeAttenuation: true,
        });
      }

      const points = new THREE.Points(geometry, material);
      group.add(points);
      if (!added) {
        scene.add(group);
        added = true;
        opts.onFirstChunk?.();
      }
      totalPoints += result.pointCount;
      emit(totalBytes); // refresh point count readout
    }

    if (!added) {
      opts.onStatus({
        loading: false, progress: null, pointCount: 0,
        chunksLoaded, totalChunks,
      });
      return { group, totalPoints: 0 };
    }
    opts.onStatus({
      loading: false, progress: 1, pointCount: totalPoints,
      chunksLoaded, totalChunks,
    });
    return { group, totalPoints };
  }

  async function loadScenePointmap() {
    if (!scene || !props.depthStem || scenePointmapLoading) return;
    const plugin = getScenePluginOrDefault(props.sceneSource);
    if (!plugin.features?.scenePointmap) return;

    scenePointmapLoading = true;
    const myToken = ++scenePointmapAbortToken;
    const emit = (s: PointcloudStreamStatus) => props.onScenePointmapStatus?.(s);
    emit({ loading: true, progress: 0, pointCount: null, chunksLoaded: null, totalChunks: null });
    try {
      const manifestUrl = `/analysis/${props.depthStem}/_scene/${plugin.camerasDir}/scene_pointmap_chunks.json`;
      const result = await streamCloudIntoGroup({
        manifestUrl,
        pointSizeFactor: 0.002,
        abortToken: () => scenePointmapAbortToken,
        initialAbortToken: myToken,
        onStatus: emit,
        onFirstChunk: () => fitCameraToScene(),
      });
      if (result && result.totalPoints > 0) {
        scenePointmapGroup = result.group;
      }
    } catch (err) {
      console.warn("[scene-pointmap] stream failed:", err);
      emit(emptyStreamStatus());
    } finally {
      scenePointmapLoading = false;
    }
  }

  function disposeObjectPointmap() {
    objectPointmapAbortToken++;
    const wiped = disposeChunkedGroup(objectPointmapGroup);
    objectPointmapGroup = null;
    if (wiped) {
      props.onObjectPointmapStatus?.(emptyStreamStatus());
    }
  }

  async function loadObjectPointmap(manifestUrl: string) {
    if (!scene || objectPointmapLoading) return;
    objectPointmapLoading = true;
    const myToken = ++objectPointmapAbortToken;
    const emit = (s: PointcloudStreamStatus) => props.onObjectPointmapStatus?.(s);
    emit({ loading: true, progress: 0, pointCount: null, chunksLoaded: null, totalChunks: null });
    try {
      const result = await streamCloudIntoGroup({
        manifestUrl,
        // Object clouds are typically a small fraction of scene extent; bias
        // the point size up slightly so single objects remain visible.
        pointSizeFactor: 0.004,
        abortToken: () => objectPointmapAbortToken,
        initialAbortToken: myToken,
        onStatus: emit,
      });
      if (result && result.totalPoints > 0) {
        objectPointmapGroup = result.group;
      }
    } catch (err) {
      console.warn("[object-pointmap] stream failed:", err);
      emit(emptyStreamStatus());
    } finally {
      objectPointmapLoading = false;
    }
  }

  function disposeAll() {
    disposeBoxes();
    disposeMeshes();
    disposeScenePointmap();
    disposeObjectPointmap();
    disposeCameraPath();
  }

  function resetCamera() {
    if (!camera || !controls) return;
    // Position camera looking at origin, offset along Z
    camera.position.set(0, 0, 5);
    controls.target.set(0, 0, 0);
    controls.update();
  }

  function fitCameraToScene() {
    if (!camera || !controls) return;
    // Fit to camera path positions (not the full mesh extent)
    if (cameraWorldPositions.length === 0) return;
    const box = new THREE.Box3();
    for (const p of cameraWorldPositions) {
      box.expandByPoint(p);
    }
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const dist = maxDim * 2;
    camera.position.copy(center).add(new THREE.Vector3(0, 0, dist));
    camera.near = dist * 0.0005;
    camera.far = dist * 200;
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.update();
  }

  onMount(() => {
    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x111111);
    containerEl.appendChild(renderer.domElement);

    // Scene
    scene = new THREE.Scene();
    meshGroup = new THREE.Group();
    scene.add(meshGroup);

    // Camera
    camera = new THREE.PerspectiveCamera(60, 1, 0.005, 1000);
    camera.position.set(0, 0, 5);

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;

    // Restore saved 3D view
    try {
      const saved = localStorage.getItem("segviewer:3dview");
      if (saved) {
        const v = JSON.parse(saved);
        camera.position.set(v.cx, v.cy, v.cz);
        controls.target.set(v.tx, v.ty, v.tz);
        if (v.fov) camera.fov = v.fov;
        if (v.near) camera.near = v.near;
        if (v.far) camera.far = v.far;
        camera.updateProjectionMatrix();
        controls.update();
      }
    } catch {}

    // Axes helper
    scene.add(new THREE.AxesHelper(1));

    // Size to container
    const resize = () => {
      if (!renderer || !camera) return;
      const w = containerEl.clientWidth;
      const h = containerEl.clientHeight;
      if (w === 0 || h === 0) return;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resizeObs = new ResizeObserver(resize);
    resizeObs.observe(containerEl);
    resize();

    // Track mouse position in NDC for raycasting
    const mouse = new THREE.Vector2();
    const raycaster = new THREE.Raycaster();
    const handleMouseMove = (e: MouseEvent) => {
      const rect = containerEl.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    };
    containerEl.addEventListener("mousemove", handleMouseMove);

    // Keyboard shortcuts
    const handleKey = (e: KeyboardEvent) => {
      if (!props.visible) return;
      if (e.key === "f") {
        snapToColmapCamera();
      } else if (e.key === "c" && camera && controls && scene) {
        raycaster.setFromCamera(mouse, camera);
        // Collect visible meshes for raycasting
        const targets: THREE.Object3D[] = [];
        for (const [, cached] of meshCache) {
          if (cached.object.visible && !cached.isPointCloud) targets.push(cached.object);
        }
        const hits = raycaster.intersectObjects(targets, false);
        if (hits.length > 0) {
          controls.target.copy(hits[0].point);
          controls.update();
        }
      }
    };
    window.addEventListener("keydown", handleKey);

    // Render loop — save 3D view to localStorage every ~2s
    let lastSaveTime = 0;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      controls?.update();
      if (renderer && scene && camera) {
        renderer.render(scene, camera);
      }
      const now = performance.now();
      if (now - lastSaveTime > 2000 && camera && controls) {
        lastSaveTime = now;
        const p = camera.position;
        const t = controls.target;
        localStorage.setItem("segviewer:3dview", JSON.stringify({
          cx: p.x, cy: p.y, cz: p.z,
          tx: t.x, ty: t.y, tz: t.z,
          fov: camera.fov, near: camera.near, far: camera.far,
        }));
      }
    };
    animate();

    onCleanup(() => {
      containerEl.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("keydown", handleKey);
      if (animId !== null) cancelAnimationFrame(animId);
      resizeObs?.disconnect();
      disposeAll();
      controls?.dispose();
      renderer?.dispose();
      if (renderer?.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    });

    // Expose actions to parent
    props.onReady?.({
      snapCamera: () => snapToColmapCamera(),
      fitAll: () => fitCameraToScene(),
    });
  });

  // Show/hide camera path
  createEffect(() => {
    const show = props.showCameraPath !== false;
    if (cameraPathLine) cameraPathLine.visible = show;
    if (cameraMarker) cameraMarker.visible = show;
    for (const f of cameraFrustums) f.visible = show;
  });

  // Build camera path when cameras data or data version changes
  createEffect(on(
    () => [props.cameras, props.dataVersion] as const,
    ([cameras]) => {
      if (cameras && scene) {
        buildCameraPath(cameras);
      }
    },
  ));

  // React to frame changes — update depth mesh and camera marker
  createEffect(on(
    () => [props.currentFrame, props.visible, props.depthFrames, props.cameras, props.usePointmap, props.scenePointmapMode, props.objectPointmapMode, props.dataVersion, props.downsample] as const,
    ([frame, visible, depthFrames, cameras]) => {
      if (!visible || !cameras) return;
      updateCameraMarker(frame);
      // In scene / object pointmap modes, don't load per-frame meshes
      if (props.scenePointmapMode || props.objectPointmapMode) return;
      if (!depthFrames.length) return;
      const depthIdx = nearestDepthFrame(frame);
      if (depthIdx == null || depthIdx === currentShownFrame) return;
      loadAndShowFrame(depthIdx);
    },
  ));

  // Rebuild box wireframes when the active solver result (or its solver id)
  // changes. The solver id drives wireframe color, so a solver switch must
  // also trigger a rebuild even if the underlying BoxerResult object happens
  // to be reference-equal across the swap.
  createEffect(on(
    () => [props.boxResult, props.boxSolverId] as const,
    ([result, solverId]) => {
      if (result && scene) {
        buildBoxes(result, solverId);
      } else {
        disposeBoxes();
      }
    },
  ));

  // Update visible per-frame box when the playhead moves
  createEffect(on(
    () => [props.currentFrame, props.visible, props.boxResult] as const,
    ([frame, visible]) => {
      if (!visible) return;
      updateBoxFrame(frame);
    },
  ));

  // React to video, source, pointmap, or data version change — dispose old meshes
  let lastVideo: string | null = null;
  let lastSource: string | undefined = undefined;
  let lastPointmap: boolean | undefined = undefined;
  let lastDataVersion: number | undefined = undefined;
  let lastDownsample: number | undefined = undefined;
  // Only auto-fit camera when loading a genuinely new video (not on reload/source switch)
  let needsFit = false;
  createEffect(on(
    () => [props.videoName, props.sceneSource, props.usePointmap, props.dataVersion, props.downsample] as const,
    ([videoName, source, usePointmap, dataVersion, downsample]) => {
      const videoChanged = videoName !== lastVideo;
      const sourceChanged = source !== lastSource;
      const pointmapChanged = usePointmap !== lastPointmap;
      const dataVersionChanged = dataVersion !== lastDataVersion;
      const downsampleChanged = lastDownsample !== undefined && downsample !== lastDownsample;
      const isInitial = lastVideo === null;
      lastVideo = videoName;
      lastSource = source;
      lastPointmap = usePointmap;
      lastDataVersion = dataVersion;
      lastDownsample = downsample;

      if (videoChanged) {
        disposeAll();
        if (!isInitial) {
          needsFit = true;
        } else if (!localStorage.getItem("segviewer:3dview")) {
          needsFit = true;
        }
      } else if (sourceChanged || dataVersionChanged) {
        // Source switch or data refresh (align, re-run) — dispose meshes only, camera path rebuilds via cameras effect
        disposeMeshes();
        disposeScenePointmap();
      } else if (pointmapChanged || downsampleChanged) {
        disposeMeshes();
      }
    },
  ));

  // Scene pointmap mode — load/dispose the global reconstruction
  let lastSceneMode: boolean | undefined = undefined;
  let lastSceneDataVersion: number | undefined = undefined;
  createEffect(on(
    () => [props.scenePointmapMode, props.visible, props.depthStem, props.sceneSource, props.dataVersion] as const,
    ([sceneMode, visible, , , dataVersion]) => {
      const changed = sceneMode !== lastSceneMode;
      const dataChanged = dataVersion !== lastSceneDataVersion;
      lastSceneMode = sceneMode;
      lastSceneDataVersion = dataVersion;

      if (sceneMode && visible) {
        // Entering scene pointmap mode: hide per-frame meshes, boxes, cameras
        if (meshGroup) meshGroup.visible = false;
        if (boxesGroup) boxesGroup.visible = false;
        if (cameraPathLine) cameraPathLine.visible = false;
        if (cameraMarker) cameraMarker.visible = false;
        for (const f of cameraFrustums) f.visible = false;
        // Re-fetch if data version changed (e.g. after align)
        if (dataChanged && (scenePointmapGroup || scenePointmapLoading)) {
          disposeScenePointmap();
        }
        if (!scenePointmapGroup && !scenePointmapLoading) loadScenePointmap();
      } else if (changed && !sceneMode) {
        // Leaving scene pointmap mode: dispose global cloud, restore everything
        disposeScenePointmap();
        if (meshGroup) meshGroup.visible = true;
        if (currentShownFrame != null) showOnlyMesh(currentShownFrame);
        if (boxesGroup) {
          boxesGroup.visible = true;
          currentBoxFrame = null;  // force re-evaluation
          updateBoxFrame(props.currentFrame);
        }
        const showPath = props.showCameraPath !== false;
        if (cameraPathLine) cameraPathLine.visible = showPath;
        if (cameraMarker) cameraMarker.visible = showPath;
        for (const f of cameraFrustums) f.visible = showPath;
      }
    },
  ));

  // Object pointmap mode — load/dispose the per-object world cloud
  let lastObjectMode: boolean | undefined = undefined;
  let lastObjectUrl: string | null | undefined = undefined;
  let lastObjectDataVersion: number | undefined = undefined;
  createEffect(on(
    () => [props.objectPointmapMode, props.visible, props.objectPointmapUrl, props.dataVersion] as const,
    ([objectMode, visible, url, dataVersion]) => {
      const changed = objectMode !== lastObjectMode;
      const urlChanged = url !== lastObjectUrl;
      const dataChanged = dataVersion !== lastObjectDataVersion;
      lastObjectMode = objectMode;
      lastObjectUrl = url;
      lastObjectDataVersion = dataVersion;

      if (objectMode && visible) {
        // Hide per-frame meshes + boxes + cameras (mirror scene-mode)
        if (meshGroup) meshGroup.visible = false;
        if (boxesGroup) boxesGroup.visible = false;
        if (cameraPathLine) cameraPathLine.visible = false;
        if (cameraMarker) cameraMarker.visible = false;
        for (const f of cameraFrustums) f.visible = false;
        // Refetch if URL or data version changed (e.g. rebuild after re-track)
        if ((urlChanged || dataChanged) && (objectPointmapGroup || objectPointmapLoading)) {
          disposeObjectPointmap();
        }
        if (!objectPointmapGroup && !objectPointmapLoading && url) loadObjectPointmap(url);
      } else if (changed && !objectMode) {
        // Leaving object mode — dispose cloud, restore per-frame view unless
        // scene mode is also active (which keeps things hidden).
        disposeObjectPointmap();
        if (props.scenePointmapMode) return;
        if (meshGroup) meshGroup.visible = true;
        if (currentShownFrame != null) showOnlyMesh(currentShownFrame);
        if (boxesGroup) {
          boxesGroup.visible = true;
          currentBoxFrame = null;
          updateBoxFrame(props.currentFrame);
        }
        const showPath = props.showCameraPath !== false;
        if (cameraPathLine) cameraPathLine.visible = showPath;
        if (cameraMarker) cameraMarker.visible = showPath;
        for (const f of cameraFrustums) f.visible = showPath;
      }
    },
  ));

  return (
    <div
      ref={containerEl!}
      style={{
        position: "absolute",
        inset: "0",
        display: props.visible ? "block" : "none",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: "2px",
          right: "8px",
          padding: "4px 8px",
          background: "rgba(0, 0, 0, 0.55)",
          color: "#888",
          font: "11px/1.4 ui-monospace, Consolas, monospace",
          "border-radius": "4px",
          "pointer-events": "none",
          "user-select": "none",
        }}
      >
        keys: [f]ocus  [c]enter
      </div>
    </div>
  );
}
