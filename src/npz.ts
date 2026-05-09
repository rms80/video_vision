/** Parse a .npy buffer (NumPy array format v1/v2). Returns typed array + shape. */
export function parseNpy(buf: ArrayBuffer): { data: Float32Array | Float64Array | Uint8Array | Int16Array | Uint16Array; shape: number[]; dtype: string } {
  const view = new DataView(buf);
  // Magic: \x93NUMPY
  const major = view.getUint8(6);
  let headerLen: number;
  let headerOff: number;
  if (major === 1) {
    headerLen = view.getUint16(8, true);
    headerOff = 10;
  } else {
    headerLen = view.getUint32(8, true);
    headerOff = 12;
  }
  const headerStr = new TextDecoder().decode(new Uint8Array(buf, headerOff, headerLen));
  const descrMatch = headerStr.match(/'descr'\s*:\s*'([^']+)'/);
  const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/);
  const dtype = descrMatch ? descrMatch[1] : "<f4";
  const shape = shapeMatch
    ? shapeMatch[1].split(",").map((s) => s.trim()).filter(Boolean).map(Number)
    : [];
  const dataOff = headerOff + headerLen;
  const raw = buf.slice(dataOff);
  // Support common dtypes
  const dtypeBase = dtype.replace(/[<>=|]/, "");
  let data: Float32Array | Float64Array | Uint8Array | Int16Array | Uint16Array;
  switch (dtypeBase) {
    case "f2": // float16 → convert to float32
      data = float16ToFloat32(new Uint16Array(raw));
      break;
    case "f4": data = new Float32Array(raw); break;
    case "f8": data = new Float64Array(raw); break;
    case "u1": data = new Uint8Array(raw); break;
    case "i2": data = new Int16Array(raw); break;
    case "u2": data = new Uint16Array(raw); break;
    default: data = new Float32Array(raw); break;
  }
  return { data, shape, dtype };
}

function float16ToFloat32(u16: Uint16Array): Float32Array {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sign = (h >> 15) & 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    if (exp === 0) {
      out[i] = (sign ? -1 : 1) * (frac / 1024) * Math.pow(2, -14);
    } else if (exp === 31) {
      out[i] = frac ? NaN : (sign ? -Infinity : Infinity);
    } else {
      out[i] = (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
    }
  }
  return out;
}

/** Parse a .npz (zip of .npy) and return named arrays. */
export async function parseNpz(buf: ArrayBuffer): Promise<Record<string, ReturnType<typeof parseNpy>>> {
  const view = new DataView(buf);
  const result: Record<string, ReturnType<typeof parseNpy>> = {};
  // Parse from central directory (at end of file) for reliable sizes
  // Find End of Central Directory record
  let eocdOff = buf.byteLength - 22;
  while (eocdOff >= 0 && view.getUint32(eocdOff, true) !== 0x06054b50) eocdOff--;
  if (eocdOff < 0) return result;

  let cdOffset: number;
  let cdCount: number;

  // Check for Zip64 EOCD locator
  const zip64LocOff = eocdOff - 20;
  if (zip64LocOff >= 0 && view.getUint32(zip64LocOff, true) === 0x07064b50) {
    // Zip64 end of central directory locator → read Zip64 EOCD
    const eocd64Off = Number(view.getBigUint64(zip64LocOff + 8, true));
    cdCount = Number(view.getBigUint64(eocd64Off + 32, true));
    cdOffset = Number(view.getBigUint64(eocd64Off + 48, true));
  } else {
    cdCount = view.getUint16(eocdOff + 10, true);
    cdOffset = view.getUint32(eocdOff + 16, true);
  }

  let pos = cdOffset;
  for (let i = 0; i < cdCount; i++) {
    if (view.getUint32(pos, true) !== 0x02014b50) break; // central dir signature
    const compMethod = view.getUint16(pos + 10, true);
    let compSize = view.getUint32(pos + 20, true);
    let uncompSize = view.getUint32(pos + 24, true);
    const nameLen = view.getUint16(pos + 28, true);
    const extraLen = view.getUint16(pos + 30, true);
    const commentLen = view.getUint16(pos + 32, true);
    let localHeaderOff = view.getUint32(pos + 42, true);
    const name = new TextDecoder().decode(new Uint8Array(buf, pos + 46, nameLen));

    // Parse Zip64 extra field if sizes are 0xFFFFFFFF
    if (compSize === 0xFFFFFFFF || uncompSize === 0xFFFFFFFF || localHeaderOff === 0xFFFFFFFF) {
      let eOff = pos + 46 + nameLen;
      const eEnd = eOff + extraLen;
      while (eOff + 4 <= eEnd) {
        const eId = view.getUint16(eOff, true);
        const eSize = view.getUint16(eOff + 2, true);
        if (eId === 0x0001) { // Zip64 extended info
          let fp = eOff + 4;
          if (uncompSize === 0xFFFFFFFF) { uncompSize = Number(view.getBigUint64(fp, true)); fp += 8; }
          if (compSize === 0xFFFFFFFF) { compSize = Number(view.getBigUint64(fp, true)); fp += 8; }
          if (localHeaderOff === 0xFFFFFFFF) { localHeaderOff = Number(view.getBigUint64(fp, true)); }
          break;
        }
        eOff += 4 + eSize;
      }
    }

    // Read data from local file header
    const localNameLen = view.getUint16(localHeaderOff + 26, true);
    const localExtraLen = view.getUint16(localHeaderOff + 28, true);
    const dataStart = localHeaderOff + 30 + localNameLen + localExtraLen;

    let entryBuf: ArrayBuffer;
    if (compMethod === 0) {
      entryBuf = buf.slice(dataStart, dataStart + compSize);
    } else {
      const compressed = new Uint8Array(buf, dataStart, compSize);
      const ds = new DecompressionStream("deflate-raw");
      const writer = ds.writable.getWriter();
      writer.write(compressed);
      writer.close();
      const reader = ds.readable.getReader();
      const chunks: Uint8Array[] = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
      }
      const total = chunks.reduce((a, c) => a + c.length, 0);
      const merged = new Uint8Array(total);
      let mPos = 0;
      for (const c of chunks) { merged.set(c, mPos); mPos += c.length; }
      entryBuf = merged.buffer.slice(merged.byteOffset, merged.byteOffset + merged.byteLength);
    }

    const key = name.replace(/\.npy$/, "");
    try { result[key] = parseNpy(entryBuf); } catch {}
    pos += 46 + nameLen + extraLen + commentLen;
  }
  return result;
}
