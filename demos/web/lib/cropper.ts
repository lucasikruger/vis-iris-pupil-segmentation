import type { ResilientLandmarker } from './mediapipe';
// Port of tools/prepare/eye_cropper.py — MediaPipe Face Landmarker → eye crops.
// Mismos índices, padding 0.12 por lado, square crop forzado. Sin rotación/alineación.

import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

export const LEFT_EYE_IDXS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
export const RIGHT_EYE_IDXS = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];

export interface EyeCropperOptions {
  padding?: number;       // default 0.12 por lado (el lado del recorte queda x1.24)
  minEyeSize?: number;    // default 30 px
  square?: boolean;       // default true
  outputSize?: number;    // default 96 (final resize)
}

export interface EyeCrop {
  side: "left" | "right";
  bbox: { x: number; y: number; w: number; h: number };  // in source coords
  openness: number;
  canvas: HTMLCanvasElement;  // resized to outputSize x outputSize
}

export class EyeCropper {
  private padding: number;
  private minEyeSize: number;
  private square: boolean;
  private outputSize: number;

  constructor(opts: EyeCropperOptions = {}) {
    // Deriva de la razón entre el diámetro del iris y la caja del detector
    // (r = 0.435 medido sobre FEI): p = (r/0.35 - 1)/2. La red se entrena con el
    // iris ocupando 0.35 del cuadro; con 0.4 quedaba en 0.23 y perdía el iris en
    // uno de cada cinco recortes. Ver §4.2 y §5.4.
    this.padding = opts.padding ?? 0.12;
    this.minEyeSize = opts.minEyeSize ?? 30;
    this.square = opts.square ?? true;
    this.outputSize = opts.outputSize ?? 96;
  }

  /** Detect faces and produce up to 2 eye crops. */
  async cropEyes(
    landmarker: ResilientLandmarker,
    source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    timestampMs?: number,
    transform?: (box: EyeCrop["bbox"], side: EyeCrop["side"]) => EyeCrop["bbox"],
  ): Promise<EyeCrop[]> {
    const W = "videoWidth" in source ? source.videoWidth : (source as HTMLImageElement).naturalWidth || source.width;
    const H = "videoHeight" in source ? source.videoHeight : (source as HTMLImageElement).naturalHeight || source.height;
    if (W === 0 || H === 0) return [];

    const result =
      source instanceof HTMLVideoElement
        ? await landmarker.detectForVideo(source, timestampMs ?? performance.now())
        : await landmarker.detect(source);

    if (!result.faceLandmarks || result.faceLandmarks.length === 0) return [];
    const landmarks = result.faceLandmarks[0];

    const crops: EyeCrop[] = [];
    for (const [indices, side] of [
      [LEFT_EYE_IDXS, "left" as const],
      [RIGHT_EYE_IDXS, "right" as const],
    ] as const) {
      let bbox = this.bboxFromLandmarks(landmarks, indices as number[], W, H);
      if (!bbox) continue;
      const points = (indices as number[]).map(i => landmarks[i]);
      const openness = (Math.max(...points.map(p => p.y)) - Math.min(...points.map(p => p.y))) * H
        / Math.max(1, (Math.max(...points.map(p => p.x)) - Math.min(...points.map(p => p.x))) * W);
      if (transform) bbox = transform(bbox, side);
      bbox = { x: Math.max(0, Math.min(W - 1, bbox.x)), y: Math.max(0, Math.min(H - 1, bbox.y)), w: bbox.w, h: bbox.h };
      bbox.w = Math.max(1, Math.min(W - bbox.x, bbox.w)); bbox.h = Math.max(1, Math.min(H - bbox.y, bbox.h));
      crops.push({ side, bbox, openness, canvas: this.cropAndResize(source, bbox) });
    }
    return crops;
  }

  private bboxFromLandmarks(
    landmarks: NormalizedLandmark[],
    indices: number[],
    W: number,
    H: number,
  ): { x: number; y: number; w: number; h: number } | null {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const i of indices) {
      const lm = landmarks[i];
      if (lm && Number.isFinite(lm.x) && Number.isFinite(lm.y)) {
        xs.push(lm.x * W);
        ys.push(lm.y * H);
      }
    }
    if (xs.length !== indices.length) return null;

    let x0 = Math.min(...xs);
    let x1 = Math.max(...xs);
    let y0 = Math.min(...ys);
    let y1 = Math.max(...ys);
    let w = x1 - x0;
    let h = y1 - y0;

    const px = w * this.padding;
    const py = h * this.padding;
    x0 -= px;
    y0 -= py;
    w += 2 * px;
    h += 2 * py;

    if (this.square) {
      const side = Math.max(w, h);
      const cx = x0 + w / 2;
      const cy = y0 + h / 2;
      x0 = cx - side / 2;
      y0 = cy - side / 2;
      w = side;
      h = side;
    }

    x0 = Math.max(0, Math.floor(x0));
    y0 = Math.max(0, Math.floor(y0));
    w = Math.min(W - x0, Math.floor(w));
    h = Math.min(H - y0, Math.floor(h));

    if (w < this.minEyeSize || h < this.minEyeSize) return null;
    return { x: x0, y: y0, w, h };
  }

  private cropAndResize(
    source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    bbox: { x: number; y: number; w: number; h: number },
  ): HTMLCanvasElement {
    const out = document.createElement("canvas");
    out.width = this.outputSize;
    out.height = this.outputSize;
    const ctx = out.getContext("2d");
    if (!ctx) throw new Error("2d context unavailable");
    ctx.drawImage(source, bbox.x, bbox.y, bbox.w, bbox.h, 0, 0, this.outputSize, this.outputSize);
    return out;
  }
}
