import type { SegmentationResult } from './onnx';

export function largest(mask: Uint8Array, n: number): Uint8Array {
  const seen = new Uint8Array(mask.length); let best: number[] = [];
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i] || seen[i]) continue;
    const component = [i]; seen[i] = 1;
    for (let k = 0; k < component.length; k++) {
      const p = component[k], x = p % n, y = Math.floor(p / n);
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy, q = yy * n + xx;
        if (xx >= 0 && xx < n && yy >= 0 && yy < n && mask[q] && !seen[q]) {
          seen[q] = 1; component.push(q);
        }
      }
    }
    if (component.length > best.length) best = component;
  }
  const out = new Uint8Array(mask.length); for (const p of best) out[p] = 1; return out;
}
function fillHoles(mask: Uint8Array, n: number): Uint8Array {
  const outside = new Uint8Array(mask.length), queue: number[] = [];
  const add = (p: number) => { if (!mask[p] && !outside[p]) { outside[p] = 1; queue.push(p); } };
  for (let i = 0; i < n; i++) { add(i); add((n - 1) * n + i); add(i * n); add(i * n + n - 1); }
  for (let i = 0; i < queue.length; i++) {
    const p = queue[i], x = p % n, y = Math.floor(p / n);
    if (x > 0) add(p - 1); if (x + 1 < n) add(p + 1);
    if (y > 0) add(p - n); if (y + 1 < n) add(p + n);
  }
  return mask.map((v, i) => v || !outside[i] ? 1 : 0);
}
function close(mask: Uint8Array, n: number, radius: number): Uint8Array {
  const morph = (input: Uint8Array, dilate: boolean) => input.map((_, p) => {
    const x = p % n, y = Math.floor(p / n); let value = dilate ? 0 : 1;
    for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) {
      const xx = x + dx, yy = y + dy;
      // OpenCV's neutral border: zero for dilation, one for erosion.
      const v = xx >= 0 && xx < n && yy >= 0 && yy < n ? input[yy * n + xx] : (dilate ? 0 : 1);
      value = dilate ? Math.max(value, v) : Math.min(value, v);
    }
    return value;
  });
  return morph(morph(mask, true), false);
}
export function postprocess(seg: SegmentationResult): SegmentationResult {
  const n = seg.size;
  const disc = close(fillHoles(largest(seg.argmax.map(v => v > 0 ? 1 : 0), n), n), n, 2);
  const pupil = close(largest(seg.argmax.map(v => v === 2 ? 1 : 0), n), n, 1);
  return { ...seg, argmax: disc.map((v, i) => v ? (pupil[i] ? 2 : 1) : 0) };
}
export function pupilCenter(seg: SegmentationResult): [number, number] | null {
  let x = 0, y = 0, count = 0;
  seg.argmax.forEach((v, i) => { if (v === 2) { x += i % seg.size + .5; y += Math.floor(i / seg.size) + .5; count++; } });
  return count ? [x / count, y / count] : null;
}
