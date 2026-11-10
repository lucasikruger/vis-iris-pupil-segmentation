import { displayLabels } from './display';
// Render utilities for overlaying segmentation masks on crops.

import type { SegmentationResult } from "./onnx";
import { pupilCenter } from "./postprocess";

// 3-class palette: 0=bg (transparent), 1=iris (cyan), 2=pupil (magenta).
const PALETTE: Array<[number, number, number, number]> = [
  [0, 0, 0, 0],
  [55, 220, 245, 200],   // iris
  [245, 80, 200, 220],   // pupil
];

export type MaskStyle = 'fill' | 'outline' | 'fill-outline';
export function maskPixels(seg: SegmentationResult, style: MaskStyle = 'fill', width = 1): Uint8ClampedArray<ArrayBuffer> {
  const { argmax, size } = seg;
  const out = new Uint8ClampedArray(size * size * 4);
  const radius = Math.max(1, Math.min(5, Math.round(width)));
  const inside = (x: number, y: number, pupil: boolean) => x >= 0 && y >= 0 && x < size && y < size && (pupil ? argmax[y * size + x] === 2 : argmax[y * size + x] === 1 || argmax[y * size + x] === 2);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const i = y * size + x, cls = argmax[i];
    if (cls !== 1 && cls !== 2) continue;
    let edge = false;
    if (style !== 'fill') {
      for (let dy = -radius; dy <= radius && !edge; dy++) for (let dx = -radius; dx <= radius; dx++) {
        if (dx * dx + dy * dy <= radius * radius && !inside(x + dx, y + dy, cls === 2)) { edge = true; break; }
      }
    }
    if (style === 'outline' && !edge) continue;
    const color = PALETTE[cls];
    out.set([color[0], color[1], color[2], edge ? 255 : color[3]], 4 * i);
  }
  return out;
}
export function maskToRgba(seg: SegmentationResult, style?: MaskStyle, width?: number): ImageData {
  return new ImageData(maskPixels(seg, style, width), seg.size, seg.size);
}

// Distance to the outside of a class, in display pixels (two-pass chamfer).
function insideDistance(labels: Uint8Array, w: number, h: number, pupil: boolean): Float32Array {
  const d = Float32Array.from(labels, (v, p) => (pupil ? v === 2 : v > 0) ? Math.min(p % w + 1, w - p % w, Math.floor(p / w) + 1, h - Math.floor(p / w)) : 0);
  const diagonal = Math.SQRT2;
  for (let y=0;y<h;y++) for(let x=0;x<w;x++) {
    const p=y*w+x;
    if(x) d[p]=Math.min(d[p],d[p-1]+1);
    if(y) d[p]=Math.min(d[p],d[p-w]+1, x ? d[p-w-1]+diagonal : Infinity, x+1<w ? d[p-w+1]+diagonal : Infinity);
  }
  for (let y=h-1;y>=0;y--) for(let x=w-1;x>=0;x--) {
    const p=y*w+x;
    if(x+1<w) d[p]=Math.min(d[p],d[p+1]+1);
    if(y+1<h) d[p]=Math.min(d[p],d[p+w]+1, x ? d[p+w-1]+diagonal : Infinity, x+1<w ? d[p+w+1]+diagonal : Infinity);
  }
  return d;
}
export function displayPixels(seg: SegmentationResult, w: number, h: number, style: MaskStyle = 'outline', thickness = 2,
  box = { x: 0, y: 0, w, h }, fillAlpha = 1): Uint8ClampedArray<ArrayBuffer> {
  const labels = displayLabels(seg,w,h,box), out = new Uint8ClampedArray(w*h*4);
  const disc = style === 'fill' ? null : insideDistance(labels,w,h,false);
  const pupil = style === 'fill' ? null : insideDistance(labels,w,h,true);
  const radius = Math.max(1, thickness);
  for(let p=0;p<labels.length;p++) {
    const cls=labels[p]; if(!cls) continue;
    const edge = disc !== null && (cls===2 ? pupil![p] : disc[p]) <= radius;
    if(style==='outline' && !edge) continue;
    const color=PALETTE[cls];out.set([color[0],color[1],color[2],edge?255:Math.round(255 * fillAlpha)],p*4);
  }
  return out;
}

export interface RenderOptions {
  transparentBackground?: boolean;
  maskStyle?: MaskStyle;
  contourWidth?: number;
  show: "crop" | "mask" | "blend";
  center?: [number, number] | null;
  showBoxes?: boolean;
  blendAlpha?: number;  // 0..1, fill opacity, default 0.35; outlines stay opaque
}

export function renderCropWithMask(
  target: HTMLCanvasElement,
  crop: HTMLCanvasElement,
  seg: SegmentationResult,
  opts: RenderOptions,
): void {
  const displaySize = Math.max(1, Math.round((target.getBoundingClientRect().width || crop.width) * (globalThis.devicePixelRatio || 1)));
  target.width = target.height = displaySize;
  const ctx = target.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, target.width, target.height);
  if (opts.show !== "mask") ctx.drawImage(crop, 0, 0, target.width, target.height);
  if (opts.show === "crop") return;

  const maskData = new ImageData(displayPixels(seg, target.width, target.height, opts.maskStyle, (opts.contourWidth ?? 2) * (globalThis.devicePixelRatio || 1), undefined, opts.show === "mask" ? 1 : (opts.blendAlpha ?? .35)), target.width, target.height);
  const tmp = document.createElement("canvas");
  tmp.width = target.width;
  tmp.height = target.height;
  tmp.getContext("2d")!.putImageData(maskData, 0, 0);

  ctx.globalAlpha = 1;
  ctx.drawImage(tmp, 0, 0, target.width, target.height);
  ctx.globalAlpha = 1;
  if (opts.center) {
    const scale = target.width / seg.size;
    const [x, y] = opts.center.map(v => v * scale);
    ctx.strokeStyle = "white"; ctx.lineWidth = Math.max(1, scale);
    ctx.beginPath(); ctx.moveTo(x - 5 * scale, y); ctx.lineTo(x + 5 * scale, y);
    ctx.moveTo(x, y - 5 * scale); ctx.lineTo(x, y + 5 * scale); ctx.stroke();
  }
}

/** Render a captured photo and its masks in original-image coordinates. */
export function renderPhoto(
  target: HTMLCanvasElement,
  source: HTMLCanvasElement,
  eyes: Array<{ crop: import('./cropper').EyeCrop; result: SegmentationResult }>,
  opts: RenderOptions,
): void {
  target.width = source.width; target.height = source.height;
  const ctx = target.getContext('2d')!;
  if (opts.transparentBackground) ctx.clearRect(0, 0, target.width, target.height);
  else if (opts.show === 'mask') { ctx.fillStyle = 'black'; ctx.fillRect(0, 0, target.width, target.height); }
  else ctx.drawImage(source, 0, 0);
  if (opts.show === 'crop') {
    if (opts.showBoxes) drawCropBoxes(ctx, eyes.map(eye => eye.crop), source.width);
    return;
  }
  for (const { crop, result } of eyes) {
    const b = crop.bbox, x0 = Math.floor(b.x), y0 = Math.floor(b.y);
    const w = Math.ceil(b.x+b.w)-x0, h = Math.ceil(b.y+b.h)-y0;
    const mask = document.createElement('canvas'); mask.width=w; mask.height=h;
    const pixels = displayPixels(result,w,h,opts.maskStyle,(opts.contourWidth ?? 2) * source.width / (target.getBoundingClientRect().width || source.width),{x:b.x-x0,y:b.y-y0,w:b.w,h:b.h},opts.show === 'mask' ? 1 : (opts.blendAlpha ?? .35));
    mask.getContext('2d')!.putImageData(new ImageData(pixels,w,h),0,0);
    ctx.globalAlpha = 1;
    ctx.drawImage(mask,x0,y0); ctx.globalAlpha = 1;
    const center = pupilCenter(result);
    if (!center) continue;
    const x = b.x + center[0] * b.w / result.size, y = b.y + center[1] * b.h / result.size;
    ctx.strokeStyle = 'white'; ctx.lineWidth = Math.max(1, source.width / 1000);
    const radius = Math.max(4, b.w / 12);
    ctx.beginPath(); ctx.moveTo(x - radius, y); ctx.lineTo(x + radius, y);
    ctx.moveTo(x, y - radius); ctx.lineTo(x, y + radius); ctx.stroke();
  }
  if (opts.showBoxes) drawCropBoxes(ctx, eyes.map(eye => eye.crop), source.width);
}


export type CropOutline = Pick<import('./cropper').EyeCrop, 'side' | 'bbox'>;

export function drawCropBoxes(ctx: CanvasRenderingContext2D, boxes: CropOutline[], sourceWidth: number): void {
  ctx.save();
  ctx.globalAlpha = 1;
  const width = Math.max(1.5, sourceWidth / 450);
  for (const { bbox: b } of boxes) {
    ctx.strokeStyle = '#111111'; ctx.lineWidth = width + 2;
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    ctx.strokeStyle = '#ffd54f'; ctx.lineWidth = width;
    ctx.strokeRect(b.x, b.y, b.w, b.h);
  }
  ctx.restore();
}
