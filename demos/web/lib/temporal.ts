import type { SegmentationResult } from './onnx';
export type Box = { x: number; y: number; w: number; h: number };
export class OneEuro {
  private value?: number; private raw = 0; private derivative = 0; private time = 0;
  constructor(private minCutoff = 1.5, private beta = .02) {}
  reset() { this.value = undefined; this.derivative = 0; }
  filter(x: number, t: number) {
    if (this.value === undefined) { this.value = this.raw = x; this.time = t; return x; }
    const dt = Math.max(.001, (t - this.time) / 1000);
    const alpha = (cutoff: number) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
    this.derivative += alpha(1) * ((x - this.raw) / dt - this.derivative);
    this.value += alpha(this.minCutoff + this.beta * Math.abs(this.derivative)) * (x - this.value);
    this.raw = x; this.time = t; return this.value;
  }
}
export function changed(a: Box, b: Box): boolean {
  return Math.hypot(a.x + a.w / 2 - b.x - b.w / 2, a.y + a.h / 2 - b.y - b.h / 2) > .5 * Math.max(b.w, b.h)
    || Math.max(a.w / b.w, b.w / a.w, a.h / b.h, b.h / a.h) > 1.35;
}
export class TemporalEye {
  private boxFilters = Array.from({ length: 4 }, () => new OneEuro());
  private centerFilters = [new OneEuro(1.2, .02), new OneEuro(1.2, .02)];
  private rawBox?: Box; private lastTime = 0;
  private previous?: { probs: Float32Array; box: Box; size: number; time: number };
  reset() { this.boxFilters.forEach(f => f.reset()); this.centerFilters.forEach(f => f.reset()); this.rawBox = undefined; this.previous = undefined; }
  box(box: Box, t: number): Box {
    if (this.rawBox && (changed(box, this.rawBox) || t - this.lastTime > 1000)) this.reset();
    this.rawBox = { ...box }; this.lastTime = t;
    const values = [box.x, box.y, box.w, box.h].map((v, i) => this.boxFilters[i].filter(v, t));
    return { x: values[0], y: values[1], w: values[2], h: values[3] };
  }
  mask(seg: SegmentationResult, box: Box, t: number): SegmentationResult {
    const prev = this.previous, n = seg.size, plane = n * n;
    const probs = seg.probs.slice();
    const hasPupil = seg.argmax.includes(2), hasEye = seg.argmax.some(v => v > 0);
    if (hasEye && prev && prev.size === n && t - prev.time <= 1000 && !changed(box, prev.box)) {
      const alpha = 1 - Math.exp(-Math.max(1, t - prev.time) / 80);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        // Map current pixel centers through source-image coordinates into the previous crop.
        const u = (box.x + (x + .5) * box.w / n - prev.box.x) * n / prev.box.w - .5;
        const v = (box.y + (y + .5) * box.h / n - prev.box.y) * n / prev.box.h - .5;
        if (u < 0 || v < 0 || u > n - 1 || v > n - 1) continue;
        const x0 = Math.floor(u), y0 = Math.floor(v), x1 = Math.min(n - 1, x0 + 1), y1 = Math.min(n - 1, y0 + 1);
        const fx = u - x0, fy = v - y0;
        for (let c = 0; c < seg.numClasses; c++) {
          const offset = c * plane;
          const old = (1 - fy) * ((1 - fx) * prev.probs[offset + y0 * n + x0] + fx * prev.probs[offset + y0 * n + x1])
            + fy * ((1 - fx) * prev.probs[offset + y1 * n + x0] + fx * prev.probs[offset + y1 * n + x1]);
          const i = offset + y * n + x; probs[i] = alpha * probs[i] + (1 - alpha) * old;
        }
      }
    }
    // An absent pupil in this frame cannot survive through temporal history.
    if (!hasPupil) for (let i = 0; i < plane; i++) { probs[i] += probs[2 * plane + i]; probs[2 * plane + i] = 0; }
    const argmax = new Uint8Array(plane);
    for (let i = 0; i < plane; i++) for (let c = 1; c < seg.numClasses; c++)
      if (probs[c * plane + i] > probs[argmax[i] * plane + i]) argmax[i] = c;
    this.previous = hasEye ? { probs, box: { ...box }, size: n, time: t } : undefined;
    return { ...seg, probs, argmax };
  }
  center(point: [number, number] | null, t: number): [number, number] | null {
    if (!point) { this.centerFilters.forEach(f => f.reset()); return null; }
    return point.map((v, i) => this.centerFilters[i].filter(v, t)) as [number, number];
  }
}
