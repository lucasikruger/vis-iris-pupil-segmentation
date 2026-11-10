import type { SegmentationResult } from './onnx';

/** Display-only probabilities. Keep soft scores except at pixels explicitly changed
 * by cleanup (or blink handling), where the chosen label becomes one-hot. */
export function displayProbabilities(seg: SegmentationResult): Float32Array {
  const out = seg.probs.slice(), n = seg.size * seg.size;
  for (let p = 0; p < n; p++) {
    let best = 0;
    for (let c = 1; c < seg.numClasses; c++) if (out[c * n + p] > out[best * n + p]) best = c;
    if (best !== seg.argmax[p]) for (let c = 0; c < seg.numClasses; c++) out[c * n + p] = Number(c === seg.argmax[p]);
  }
  return out;
}
/** OpenCV INTER_LINEAR pixel-center geometry, replicate border. `box` is the
 * destination crop rectangle relative to this raster, including fractional offsets. */
export function resizeProbabilities(seg: SegmentationResult, width: number, height: number,
  box = { x: 0, y: 0, w: width, h: height }): Float32Array {
  const input = displayProbabilities(seg), n = seg.size, plane = width * height;
  const out = new Float32Array(plane * seg.numClasses);
  for (let y = 0; y < height; y++) {
    const sy = Math.max(0, Math.min(n - 1, (y + .5 - box.y) * n / box.h - .5));
    const y0 = Math.floor(sy), y1 = Math.min(y0 + 1, n - 1), fy = sy - y0;
    for (let x = 0; x < width; x++) {
      const p = y * width + x;
      if (x + .5 < box.x || x + .5 >= box.x + box.w || y + .5 < box.y || y + .5 >= box.y + box.h) { out[p] = 1; continue; }
      const sx = Math.max(0, Math.min(n - 1, (x + .5 - box.x) * n / box.w - .5));
      const x0 = Math.floor(sx), x1 = Math.min(x0 + 1, n - 1), fx = sx - x0;
      for (let c = 0; c < seg.numClasses; c++) {
        const a = c * n * n;
        out[c * plane + p] = (input[a + y0*n+x0]*(1-fx)+input[a+y0*n+x1]*fx)*(1-fy)
          + (input[a+y1*n+x0]*(1-fx)+input[a+y1*n+x1]*fx)*fy;
      }
    }
  }
  return out;
}
export function displayLabels(seg: SegmentationResult, width: number, height: number,
  box = { x: 0, y: 0, w: width, h: height }): Uint8Array {
  const probs = resizeProbabilities(seg, width, height, box), plane = width * height;
  const labels = new Uint8Array(plane);
  for (let p = 0; p < plane; p++) for (let c = 1; c < seg.numClasses; c++) if (probs[c*plane+p] > probs[labels[p]*plane+p]) labels[p] = c;
  return labels;
}
