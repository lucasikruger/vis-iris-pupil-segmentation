// Preprocesamiento por modelo. Cada función toma un canvas RGBA (HxW)
// y devuelve un Float32Array NCHW listo para el ONNX runtime.

export type PreprocessName =
  | "grayscale_norm_01"
  | "rgb_imagenet"
  | "rgb_minus1_1"
  | "rgb_clahe_norm_01";

export interface PreprocessFn {
  channels: number;
  toTensor: (canvas: HTMLCanvasElement, size: number) => Float32Array;
}

function getImageData(canvas: HTMLCanvasElement, size: number): Uint8ClampedArray {
  if (canvas.width === size && canvas.height === size) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d context unavailable");
    return ctx.getImageData(0, 0, size, size).data;
  }
  const tmp = document.createElement("canvas");
  tmp.width = size;
  tmp.height = size;
  const ctx = tmp.getContext("2d");
  if (!ctx) throw new Error("2d context unavailable");
  ctx.drawImage(canvas, 0, 0, size, size);
  return ctx.getImageData(0, 0, size, size).data;
}

const grayscale_norm_01: PreprocessFn = {
  channels: 1,
  toTensor: (canvas, size) => {
    const px = getImageData(canvas, size);
    const out = new Float32Array(size * size);
    for (let i = 0, j = 0; i < px.length; i += 4, j++) {
      // Standard luma weights (Rec. 709-ish)
      const gray = (0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]) / 255.0;
      out[j] = gray;
    }
    return out;
  },
};

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

const rgb_imagenet: PreprocessFn = {
  channels: 3,
  toTensor: (canvas, size) => {
    const px = getImageData(canvas, size);
    const plane = size * size;
    const out = new Float32Array(3 * plane);
    for (let i = 0, p = 0; i < px.length; i += 4, p++) {
      out[p] = (px[i] / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      out[plane + p] = (px[i + 1] / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      out[2 * plane + p] = (px[i + 2] / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
    return out;
  },
};

const rgb_minus1_1: PreprocessFn = {
  channels: 3,
  toTensor: (canvas, size) => {
    const px = getImageData(canvas, size);
    const plane = size * size;
    const out = new Float32Array(3 * plane);
    for (let i = 0, p = 0; i < px.length; i += 4, p++) {
      out[p] = px[i] / 127.5 - 1.0;
      out[plane + p] = px[i + 1] / 127.5 - 1.0;
      out[2 * plane + p] = px[i + 2] / 127.5 - 1.0;
    }
    return out;
  },
};


// OpenCV uint8 LAB → CLAHE(L, 1.5, 8×8) → RGB → gamma 0.8.
// Use the native WASM implementation, including tile redistribution and bilinear
// interpolation. Rec. 601 luma is NOT equivalent (thesis: 0.631 vs 0.641).
let cv: any;
let initializing: Promise<void> | undefined;
export async function initializePreprocess(runtime?: any): Promise<void> {
  if (runtime) { cv = await runtime; return; }
  if (cv) return;
  if (!initializing) initializing = (async () => {
    await new Promise<void>((resolve, reject) => {
      const script = document.createElement('script');
      script.src = '/opencv/opencv.js';
      script.onload = () => resolve();
      script.onerror = () => reject(new Error('No se pudo cargar OpenCV'));
      document.head.appendChild(script);
    });
    cv = await (globalThis as any).cv;
    if (!cv?.Mat) throw new Error('OpenCV no disponible');
  })().catch(error => { initializing = undefined; throw error; });
  return initializing;
}
const gamma = Uint8Array.from({ length: 256 }, (_, i) => Math.floor(255 * Math.pow(i / 255, 1 / .8)));
export function preprocessRgb(rgb: Uint8Array, size: number): Uint8Array {
  if (!cv) throw new Error('initializePreprocess() first');
  if (rgb.length !== size * size * 3) throw new Error('Invalid RGB dimensions');
  const input = cv.matFromArray(size, size, cv.CV_8UC3, rgb);
  const lab = new cv.Mat(), l = new cv.Mat(size, size, cv.CV_8UC1);
  const equalized = new cv.Mat(), output = new cv.Mat();
  const clahe = new cv.CLAHE(1.5, new cv.Size(8, 8));
  try {
    cv.cvtColor(input, lab, cv.COLOR_RGB2Lab);
    for (let i = 0; i < size * size; i++) l.data[i] = lab.data[3 * i];
    clahe.apply(l, equalized);
    for (let i = 0; i < size * size; i++) lab.data[3 * i] = equalized.data[i];
    cv.cvtColor(lab, output, cv.COLOR_Lab2RGB);
    return Uint8Array.from(output.data, (v: number) => gamma[v]);
  } finally {
    for (const object of [input, lab, l, equalized, output, clahe]) object.delete();
  }
}
export function rgbToTensor(rgb: Uint8Array, size: number): Float32Array {
  const px = preprocessRgb(rgb, size), plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let p = 0; p < plane; p++) for (let c = 0; c < 3; c++) out[c * plane + p] = px[3 * p + c] / 255;
  return out;
}
const rgb_clahe_norm_01: PreprocessFn = {
  channels: 3,
  toTensor: (canvas, size) => {
    const rgba = getImageData(canvas, size), rgb = new Uint8Array(size * size * 3);
    for (let p = 0; p < size * size; p++) rgb.set(rgba.subarray(p * 4, p * 4 + 3), p * 3);
    return rgbToTensor(rgb, size);
  },
};

export const PREPROCESS: Record<PreprocessName, PreprocessFn> = {
  grayscale_norm_01,
  rgb_imagenet,
  rgb_minus1_1,
  rgb_clahe_norm_01,
};
