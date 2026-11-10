import EyeComparison from './EyeComparison';
import { downloadComparison } from '@lib/download';
import ControlsPanel from './ControlsPanel';
import MaskControls from './MaskControls';
import type { MaskStyle } from '@lib/render';
import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { EyeCropper } from '@lib/cropper';
import { loadFaceLandmarker } from '@lib/mediapipe';
import { OnnxSegmenter, loadManifest } from '@lib/onnx';
import { renderCropWithMask, renderPhoto, type CropOutline } from '@lib/render';
import { postprocess, pupilCenter } from '@lib/postprocess';
import { TemporalEye } from '@lib/temporal';
import WebcamPhoto from './WebcamPhoto';
import CropBoxes from './CropBoxes';

export default function WebcamDemo({ active = false, onExit }: { active?: boolean; onExit: () => void }) {
  const [videoRatio, setVideoRatio] = useState(16 / 9);
  const [cameraNotice, setCameraNotice] = useState('');
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [maskStyle, setMaskStyle] = useState<MaskStyle>('outline');
  const [contourWidth, setContourWidth] = useState(2);
  const [showFullMask, setShowFullMask] = useState(false);
  const fullMaskRef = useRef<HTMLCanvasElement>(null);
  const [showBoxes, setShowBoxes] = useState(true);
  const [boxFrame, setBoxFrame] = useState<{ width: number; height: number; boxes: CropOutline[] }>({ width: 0, height: 0, boxes: [] });
  const [photo, setPhoto] = useState<HTMLCanvasElement | null>(null);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const paused = useRef(false), epoch = useRef(0);
  const segmenter = useRef<OnnxSegmenter | null>(null);
  const resumeTracking = useRef<() => void>(() => {});
  const videoRef = useRef<HTMLVideoElement>(null);
  const leftRef = useRef<HTMLCanvasElement>(null), rightRef = useRef<HTMLCanvasElement>(null);
  const cleanLeft = useRef<HTMLCanvasElement>(null), cleanRight = useRef<HTMLCanvasElement>(null);
  const [smooth, setSmooth] = useState(true), [clean, setClean] = useState(true);
  const [blendAlpha, setBlendAlpha] = useState(.35);
  const [show, setShow] = useState<'crop' | 'mask' | 'blend'>('blend');
  const options = useRef({ smooth, clean, show, blendAlpha, maskStyle, contourWidth, showFullMask }); options.current = { smooth, clean, show, blendAlpha, maskStyle, contourWidth, showFullMask };
  const [delegate, setDelegate] = useState('…');
  const [status, setStatus] = useState('Cámara apagada');
  const [frameStats, setFrameStats] = useState({ total: 0, failed: 0 });
  const [fps, setFps] = useState(0), [centers, setCenters] = useState<string[]>(['perdida', 'perdida']);
  useEffect(() => { setCameraEnabled(active); if (!active) setPhoto(null); }, [active]);
  useEffect(() => {
    setCameraReady(false); setFps(0); setCameraNotice(''); epoch.current++;
    let cancelled = false, raf = 0, stream: MediaStream | undefined;
    const tracks = { left: new TemporalEye(), right: new TemporalEye() };
    const reset = () => { tracks.left.reset(); tracks.right.reset(); };
    const clear = () => {
      for (const ref of [leftRef, rightRef, cleanLeft, cleanRight, fullMaskRef]) { const c = ref.current; c?.getContext('2d')?.clearRect(0, 0, c.width, c.height); }
      setCenters(['perdida', 'perdida']);
      setBoxFrame(previous => ({ ...previous, boxes: [] }));
    };
    if (!active || !cameraEnabled) { clear(); setDelegate('—'); return; }
    paused.current = false;
    setStatus('Pidiendo permiso para la cámara…');
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: { ideal: facingMode } }, audio: false });
        if (cancelled) { stream.getTracks().forEach(t => t.stop()); return; }
        if (facingMode === 'environment') {
          const track = stream.getVideoTracks()[0];
          const actual = track.getSettings().facingMode;
          const devices = await navigator.mediaDevices.enumerateDevices();
          if (cancelled) return;
          const supported = track.getCapabilities?.().facingMode ?? [];
          if ((actual && actual !== 'environment') || (!actual && devices.filter(d => d.kind === 'videoinput').length === 1 && !supported.includes('environment'))) {
            setCameraNotice('No se encontró una cámara trasera. Se está usando la cámara disponible.');
          }
        }
        setStatus('Cargando modelo…');
        const seg = segmenter.current ?? new OnnxSegmenter(); await seg.load((await loadManifest())[0]);
        if (cancelled) return;
        segmenter.current = seg;
        const detector = await loadFaceLandmarker('VIDEO');
        if (cancelled) return;
        setDelegate(detector.delegate);
        const video = videoRef.current!; video.srcObject = stream; await video.play();
        if (cancelled) return;
        setCameraReady(true);
        const cropper = new EyeCropper({ outputSize: 96 });
        let dimensions = '';
        let lastVideoTime = -1, timer = performance.now(), count = 0;
        resumeTracking.current = () => { reset(); lastVideoTime = -1; timer = performance.now(); count = 0; clear(); setFps(0); };
        let previousOptions = { ...options.current };
        const diagnostic = new URLSearchParams(location.search).has('diagnostics');
        const loop = async () => {
          if (cancelled) return;
          if (paused.current || video.paused || video.readyState < 2 || video.videoWidth <= 0 || video.videoHeight <= 0 || video.currentTime === lastVideoTime) { raf = requestAnimationFrame(loop); return; }
          const frameEpoch = epoch.current;
          lastVideoTime = video.currentTime;
          const currentDimensions = `${video.videoWidth}x${video.videoHeight}`;
          if (dimensions !== currentDimensions) { reset(); dimensions = currentDimensions; setVideoRatio(video.videoWidth / video.videoHeight); }
          const now = performance.now(), opts = { ...options.current };
          if (opts.smooth !== previousOptions.smooth || opts.clean !== previousOptions.clean) reset();
          previousOptions = opts;
          const failuresBefore = detector.failedFrames;
          let frameError: string | null = null;
          try {
            const crops = await cropper.cropEyes(detector, video, now, (box, side) => opts.smooth ? tracks[side].box(box, now) : box);
            if (cancelled) return;
            if (epoch.current !== frameEpoch) { raf = requestAnimationFrame(loop); return; }
            setDelegate(detector.delegate);
            setBoxFrame({ width: video.videoWidth, height: video.videoHeight, boxes: crops.map(({ side, bbox }) => ({ side, bbox })) });
            if (!crops.length) { reset(); clear(); setStatus('Sin rostro'); }
            const labels = ['perdida', 'perdida'];
            const frameEyes: Parameters<typeof renderPhoto>[2] = [];
            for (const side of ['left', 'right'] as const) {
              const crop = crops.find(c => c.side === side), track = tracks[side];
              const target = (side === 'left' ? leftRef : rightRef).current;
              if (!target) continue;
              const cleanTarget = (side === 'left' ? cleanLeft : cleanRight).current!;
              if (!crop) { cleanTarget.width = 96; track.reset(); target.getContext('2d')?.clearRect(0, 0, target.width, target.height); continue; }
              let result = await seg.run(crop.canvas);
              if (cancelled) return;
              if (epoch.current !== frameEpoch) { raf = requestAnimationFrame(loop); return; }
              const raw = pupilCenter(result), b = crop.bbox;
              const sourcePoint = (p: [number, number] | null): [number, number] | null => p ? [b.x + p[0] * b.w / result.size, b.y + p[1] * b.h / result.size] : null;
              const blink = crop.openness < .12;
              if (blink) {
                track.reset(); result = { ...result, argmax: new Uint8Array(result.argmax.length), probs: new Float32Array(result.probs.length) };
                result.probs.fill(1, 0, result.argmax.length);
              } else if (opts.smooth) result = track.mask(result, b, now);
              if (opts.clean) result = postprocess(result);
              const observed = sourcePoint(pupilCenter(result));
              let center = opts.smooth ? track.center(observed, now) : observed;
              if (center && (center[0] < b.x || center[0] >= b.x + b.w || center[1] < b.y || center[1] >= b.y + b.h)) {
                track.center(null, now); center = track.center(observed, now);
              }
              const local: [number, number] | null = center ? [(center[0] - b.x) * 96 / b.w, (center[1] - b.y) * 96 / b.h] : null;
              frameEyes.push({ crop, result });
              renderCropWithMask(cleanTarget, crop.canvas, result, { show: 'crop' });
              renderCropWithMask(target, crop.canvas, result, { maskStyle: opts.maskStyle, contourWidth: opts.contourWidth, show: opts.show, center: local, blendAlpha: opts.blendAlpha });
              labels[side === 'left' ? 0 : 1] = center ? `${center[0].toFixed(1)}, ${center[1].toFixed(1)} px` : 'perdida';
              if (diagnostic) {
                const win = window as any; win.__irisFrames ??= [];
                win.__irisFrames.push({ t: now, videoTime: lastVideoTime, side, smooth: opts.smooth, raw: sourcePoint(raw), center, blink, bbox: b });
                if (win.__irisFrames.length > 4000) win.__irisFrames.splice(0, 2);
              }
            }
            if (opts.showFullMask && fullMaskRef.current) {
              const source = document.createElement('canvas');
              source.width = video.videoWidth; source.height = video.videoHeight;
              renderPhoto(fullMaskRef.current, source, frameEyes, { transparentBackground: true, show: 'blend', maskStyle: opts.maskStyle, contourWidth: opts.contourWidth, blendAlpha: opts.blendAlpha });
            }
            setCenters(labels);
            // Clear any previous error on every successful frame, including no face.
            setStatus(crops.length ? `${crops.length}/2 ojos procesados` : 'Sin rostro');
            count++;

          } catch (error) {
            frameError = (error as Error).message;
            if (!cancelled && epoch.current === frameEpoch) {
              setDelegate(detector.delegate); reset(); clear();
              setStatus('Cuadro omitido. Reintentando con tracking nuevo…');
            }
          } finally {
            if (!cancelled) {
              const end = performance.now();
              if (end - timer >= 1000) { setFps(count * 1000 / (end - timer)); timer = end; count = 0; }
              const failed = frameError !== null || detector.failedFrames > failuresBefore;
              setFrameStats(previous => ({ total: previous.total + 1, failed: previous.failed + Number(failed) }));
              if (failed && diagnostic) {
                const win = window as any; win.__irisFrameErrors ??= [];
                win.__irisFrameErrors.push({ t: now, delegate: detector.delegate, error: frameError ?? 'Detección recuperada al cambiar a CPU', trackingResets: detector.trackingResets });
                if (win.__irisFrameErrors.length > 200) win.__irisFrameErrors.shift();
              }
            }
          }
          if (!cancelled) raf = requestAnimationFrame(loop);
        };
        raf = requestAnimationFrame(loop);
      } catch (error) { if (!cancelled) { stream?.getTracks().forEach(t => t.stop()); setCameraEnabled(false); setStatus(`No se pudo prender la cámara: ${(error as Error).message}. Podés volver a intentarlo.`); } }
    })();
    return () => { cancelled = true; cancelAnimationFrame(raf); stream?.getTracks().forEach(t => t.stop()); reset(); const video = videoRef.current; if (video) { video.pause(); video.srcObject = null; } };
  }, [facingMode, cameraEnabled, active]);
  function takePhoto() {
    const video = videoRef.current;
    if (!video || !segmenter.current || video.readyState < 2 || !video.videoWidth || !video.videoHeight) return;
    const frame = document.createElement('canvas');
    frame.width = video.videoWidth; frame.height = video.videoHeight;
    frame.getContext('2d')!.drawImage(video, 0, 0);
    paused.current = true; epoch.current++; video.pause(); setPhoto(frame); setFps(0);
  }
  function stopCamera() { setCameraEnabled(false); setCameraReady(false); setStatus('Cámara apagada'); onExit(); }
  async function resumeVideo() {
    if (!cameraEnabled) { setPhoto(null); setCameraEnabled(true); return; }
    epoch.current++; resumeTracking.current(); setPhoto(null); setStatus('Retomando video…');
    try { await videoRef.current?.play(); paused.current = false; }
    catch (error) { setStatus(`Error: ${(error as Error).message}`); }
  }
  const actions = <>
    <button onClick={() => cameraEnabled ? stopCamera() : setCameraEnabled(true)}>{cameraEnabled ? 'Apagar cámara' : 'Prender cámara'}</button>
    <button onClick={takePhoto} disabled={!cameraReady}>Tomar foto</button>
    <button disabled={!boxFrame.boxes.length} onClick={() => downloadComparison([cleanLeft.current!, cleanRight.current!], [leftRef.current!, rightRef.current!])}>Descargar grilla PNG</button>
    <button onClick={stopCamera}>Volver a ejemplos</button>
  </>;
  return <>
    {photo && segmenter.current && <WebcamPhoto cameraEnabled={cameraEnabled} onStop={stopCamera} maskStyle={maskStyle} contourWidth={contourWidth} setMaskStyle={setMaskStyle} setContourWidth={setContourWidth} frame={photo} segmenter={segmenter.current} initialClean={clean} showBoxes={showBoxes} onBoxesChange={setShowBoxes} onResume={resumeVideo} />}
    <div className="demo-grid" style={{ display: photo ? 'none' : undefined }}>
    <div className="panel">
      <div hidden={!cameraReady}><EyeComparison clean={[cleanLeft, cleanRight]} segmented={[leftRef, rightRef]} centers={centers} prefix="live" showClean={false} /></div>
      {!cameraReady && <div className="camera-wait" data-testid="camera-wait">
        <h2>{cameraEnabled ? 'Preparando cámara' : 'Cámara apagada'}</h2>
        <p>{cameraEnabled ? status : 'Prendé la cámara para empezar. También podés probar los ejemplos sin habilitarla.'}</p>
        {!cameraEnabled && <button onClick={() => setCameraEnabled(true)}>Prender cámara</button>}
      </div>}
      <div hidden={!cameraReady}>
      <div className="camera-frame" style={{ '--camera-ratio': videoRatio } as CSSProperties}>
        <video ref={videoRef} playsInline muted onLoadedMetadata={e => {
          const v = e.currentTarget; if (v.videoWidth > 0 && v.videoHeight > 0) setVideoRatio(v.videoWidth / v.videoHeight);
        }} />
        <canvas ref={fullMaskRef} data-testid="live-full-segmentation" aria-hidden="true" hidden={!showFullMask}
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }} />
        <CropBoxes {...boxFrame} visible={showBoxes} testId="live-crop-boxes" />
      </div>
      </div>
      <div className="photo-actions mobile-actions">{actions}</div>
    </div>
    <aside className="panel" style={{ display: 'grid', gap: 16 }}>
      <div className="photo-actions desktop-actions">{actions}</div>
      <strong data-testid="camera-state">{cameraEnabled ? cameraReady ? 'Cámara prendida' : 'Preparando cámara…' : 'Cámara apagada'}</strong>

      <p className="muted">La foto se procesa en este navegador y no se sube a ningún lado.</p>
      <div role="status">{status}</div><div data-testid="mediapipe-delegate">MediaPipe: {delegate}</div><div data-testid="fps">{fps.toFixed(1)} fps</div>
      <div data-testid="frame-errors">Cuadros con error: {frameStats.failed} / {frameStats.total}</div>
      {cameraNotice && <p role="note">{cameraNotice}</p>}
      <ControlsPanel>
      <label>Cámara <select aria-label="Cámara" value={facingMode} onChange={e => setFacingMode(e.target.value as typeof facingMode)}>
        <option value="user">Frontal</option><option value="environment">Trasera</option>
      </select></label>
      <MaskControls style={maskStyle} width={contourWidth} onStyle={setMaskStyle} onWidth={setContourWidth} />
      <label><input type="checkbox" checked={showFullMask} onChange={e => setShowFullMask(e.target.checked)} />Segmentación sobre la cámara completa</label>
      <label><input type="checkbox" checked={showBoxes} onChange={e => setShowBoxes(e.target.checked)} />Mostrar cajas de recorte</label>
      <p className="muted">Amarillo: recorte enviado a la red, con margen del 12 %. Con suavizado temporal se muestra la caja suavizada.</p>
      <label><input type="checkbox" checked={smooth} onChange={e => setSmooth(e.target.checked)} />Suavizado temporal</label>
      <label><input type="checkbox" checked={clean} onChange={e => setClean(e.target.checked)} />Postprocesado</label>
      <label>Vista <select value={show} onChange={e => setShow(e.target.value as typeof show)}>
        <option value="blend">Superposición</option><option value="mask">Máscara</option><option value="crop">Recorte</option>
      </select></label>
      {show === 'blend' && <label>Opacidad {Math.round(blendAlpha * 100)} %
        <input type="range" min="0" max="1" step="0.05" value={blendAlpha} onChange={e => setBlendAlpha(Number(e.target.value))} />
      </label>}
      <p className="muted">Las cruces marcan el centro pupilar. Si la pupila desaparece, se muestra como perdida. Todo se procesa en este navegador.</p>
      </ControlsPanel>
    </aside>
  </div></>;
}
