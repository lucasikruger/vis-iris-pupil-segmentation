// Separate IMAGE/VIDEO instances; remember CPU fallback on this device.
import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from '@mediapipe/tasks-vision';
export type RunningMode = 'IMAGE' | 'VIDEO';
export type Delegate = 'GPU' | 'CPU';
const STORAGE_KEY = 'iris-mediapipe-delegate';
let preferred: Delegate = 'GPU';
try { if (localStorage.getItem(STORAGE_KEY) === 'CPU') preferred = 'CPU'; } catch { /* storage may be disabled */ }
function rememberCPU() {
  preferred = 'CPU';
  try { localStorage.setItem(STORAGE_KEY, 'CPU'); } catch { /* session fallback still works */ }
}
async function createNative(mode: RunningMode, delegate: Delegate): Promise<FaceLandmarker> {
  const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: '/mediapipe/face_landmarker.task', delegate },
    numFaces: 1, minFaceDetectionConfidence: .2, minFacePresenceConfidence: .2, runningMode: mode,
  });
}
export class ResilientLandmarker {
  delegate: Delegate = preferred;
  private native: FaceLandmarker | null = null;
  failedFrames = 0;
  trackingResets = 0;
  private lastTimestamp = -Infinity;
  constructor(readonly mode: RunningMode) {}
  private async replace(delegate: Delegate) {
    this.discardTracking();
    this.delegate = delegate;
    this.native = await createNative(this.mode, delegate);
  }
  private discardTracking() {
    const old = this.native; this.native = null;
    if (!old) return;
    this.trackingResets++;
    try { old.close(); } catch { /* a broken graph can also fail to close */ }
  }
  async initialize() {
    if (this.native && !(preferred === 'CPU' && this.delegate === 'GPU')) return;
    try { await this.replace(preferred); }
    catch (error) {
      if (this.delegate !== 'GPU') throw error;
      rememberCPU(); await this.replace('CPU');
    }
  }
  private async run(call: (native: FaceLandmarker) => FaceLandmarkerResult): Promise<FaceLandmarkerResult> {
    await this.initialize();
    try {
      return call(this.native!);
    } catch (error) {
      // One failed frame, even if a GPU→CPU retry fails too. A damaged ROI must
      // never survive into the next detection, including isolated CPU errors.
      this.failedFrames++;
      this.discardTracking();
      if (this.delegate === 'GPU') {
        rememberCPU(); await this.replace('CPU');
        try { return call(this.native!); }
        catch (retryError) { this.discardTracking(); throw retryError; }
      }
      // Skip this frame. initialize() recreates CPU before the next valid frame.
      throw error;
    }
  }
  async detect(source: HTMLImageElement | HTMLCanvasElement) {
    const width = 'naturalWidth' in source ? source.naturalWidth : source.width;
    const height = 'naturalHeight' in source ? source.naturalHeight : source.height;
    if (this.mode !== 'IMAGE' || width <= 0 || height <= 0) throw new Error('Imagen sin dimensiones válidas');
    return this.run(native => native.detect(source));
  }
  async detectForVideo(source: HTMLVideoElement, _timestamp?: number) {
    if (this.mode !== 'VIDEO' || source.readyState < 2 || source.videoWidth <= 0 || source.videoHeight <= 0) throw new Error('Cuadro de video no disponible');
    return this.run(native => {
      // performance.now can repeat under timer precision reduction. Never reuse a timestamp,
      // including retries, photo/resume transitions and replacement of a failed graph.
      this.lastTimestamp = Math.max(performance.now(), this.lastTimestamp + .001);
      return native.detectForVideo(source, this.lastTimestamp);
    });
  }
}
const detectors: Partial<Record<RunningMode, ResilientLandmarker>> = {};
const loading: Partial<Record<RunningMode, Promise<ResilientLandmarker>>> = {};
export function loadFaceLandmarker(mode: RunningMode = 'IMAGE'): Promise<ResilientLandmarker> {
  if (loading[mode]) return loading[mode]!;
  const detector = detectors[mode] ??= new ResilientLandmarker(mode);
  const promise = detector.initialize().then(() => detector).finally(() => { delete loading[mode]; });
  loading[mode] = promise;
  return promise;
}
