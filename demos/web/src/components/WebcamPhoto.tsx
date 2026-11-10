import EyeComparison from './EyeComparison';
import { downloadComparison, downloadCanvas } from '@lib/download';
import { useViewportRevision } from './useViewportRevision';
import ControlsPanel from './ControlsPanel';
import MaskControls from './MaskControls';
import type { MaskStyle } from '@lib/render';
import { useEffect, useRef, useState } from 'react';
import { EyeCropper, type EyeCrop } from '@lib/cropper';
import { loadFaceLandmarker } from '@lib/mediapipe';
import type { OnnxSegmenter, SegmentationResult } from '@lib/onnx';
import { postprocess, pupilCenter } from '@lib/postprocess';
import { renderCropWithMask, renderPhoto } from '@lib/render';

type Eye = { crop: EyeCrop; result: SegmentationResult };
export default function WebcamPhoto({ frame, segmenter, initialClean, maskStyle, contourWidth, setMaskStyle, setContourWidth, showBoxes, onBoxesChange, onResume, cameraEnabled, onStop }: {
  cameraEnabled: boolean; onStop: () => void;
  maskStyle: MaskStyle; contourWidth: number; setMaskStyle: (style: MaskStyle) => void; setContourWidth: (width: number) => void;
  frame: HTMLCanvasElement; segmenter: OnnxSegmenter; initialClean: boolean; showBoxes: boolean; onBoxesChange: (show: boolean) => void; onResume: () => void;
}) {
  const viewportRevision = useViewportRevision();
  const canvas = useRef<HTMLCanvasElement>(null);
  const left = useRef<HTMLCanvasElement>(null), right = useRef<HTMLCanvasElement>(null);
  const cleanLeft = useRef<HTMLCanvasElement>(null), cleanRight = useRef<HTMLCanvasElement>(null);
  const [eyes, setEyes] = useState<Eye[] | null>(null);
  const [clean, setClean] = useState(initialClean);
  const [show, setShow] = useState<'crop' | 'mask' | 'blend'>('blend');
  const [opacity, setOpacity] = useState(.35);
  const [delegate, setDelegate] = useState('…');
  const [status, setStatus] = useState('Procesando foto…');
  const [centers, setCenters] = useState<string[]>(['perdida', 'perdida']);
  useEffect(() => {
    let active = true;
    if (frame.width <= 0 || frame.height <= 0) { setStatus("Foto sin dimensiones válidas"); return; }
    const target = canvas.current!; target.width = frame.width; target.height = frame.height;
    target.getContext('2d')!.drawImage(frame, 0, 0);
    (async () => {
      try {
        const detector = await loadFaceLandmarker('IMAGE');
        setDelegate(detector.delegate);
        if (!active) return;
        const crops = await new EyeCropper({ outputSize: 96 }).cropEyes(detector, frame);
        if (!active) return;
        setDelegate(detector.delegate);
        const results: Eye[] = [];
        for (const crop of crops) {
          const result = await segmenter.run(crop.canvas);
          if (!active) return;
          results.push({ crop, result });
        }
        setEyes(results);
        setStatus(results.length ? `Foto procesada: ${results.length} ojo(s).` : 'Foto procesada: no se detectó un rostro.');
      } catch (error) { if (active) setStatus(`Error al procesar la foto: ${(error as Error).message}`); }
    })();
    return () => { active = false; };
  }, [frame, segmenter]);
  useEffect(() => {
    if (!eyes) return;
    const processed = eyes.map(eye => ({ ...eye, result: clean ? postprocess(eye.result) : eye.result }));
    renderPhoto(canvas.current!, frame, processed, { maskStyle, contourWidth, show: 'blend', blendAlpha: opacity, showBoxes });
    const labels = ['perdida', 'perdida'];
    for (const [i, ref] of [left, right].entries()) {
      const target = ref.current!;
      target.width = target.height = 96;
      const cleanTarget = [cleanLeft, cleanRight][i].current!; cleanTarget.width = cleanTarget.height = 96;
      const eye = processed.find(e => e.crop.side === (i === 0 ? 'left' : 'right'));
      if (!eye) continue;
      const center = pupilCenter(eye.result), b = eye.crop.bbox;
      renderCropWithMask(cleanTarget, eye.crop.canvas, eye.result, { show: 'crop' });
      renderCropWithMask(target, eye.crop.canvas, eye.result, { maskStyle, contourWidth, show, center, blendAlpha: opacity });
      if (center) labels[i] = `${(b.x + center[0] * b.w / eye.result.size).toFixed(1)}, ${(b.y + center[1] * b.h / eye.result.size).toFixed(1)} px`;
    }
    setCenters(labels);
  }, [eyes, clean, show, opacity, frame, showBoxes, maskStyle, contourWidth, viewportRevision]);
  function download() {
    if (!eyes) return;
    downloadCanvas(canvas.current!, 'foto-webcam-segmentada.png');
  }
  const actions = <>
    <button onClick={onStop}>Volver a ejemplos</button>
    <button onClick={cameraEnabled ? onStop : onResume}>{cameraEnabled ? 'Apagar cámara' : 'Prender cámara'}</button>
      <button onClick={() => downloadComparison([cleanLeft.current!, cleanRight.current!], [left.current!, right.current!])} disabled={!eyes?.length}>Descargar grilla PNG</button><button onClick={() => download()} disabled={eyes === null}>Descargar segmentación PNG</button><button onClick={onResume}>Volver al video en vivo</button></>;
  return <section className="demo-grid" aria-label="Foto de webcam">
    <div className="panel">
      <EyeComparison clean={[cleanLeft, cleanRight]} segmented={[left, right]} centers={centers} prefix="photo" />
      <h2>Foto capturada · segmentación y cajas</h2>
      <canvas ref={canvas} data-testid="webcam-photo" data-boxes={JSON.stringify(eyes?.map(({ crop: { side, bbox } }) => ({ side, bbox })) ?? [])} data-visible={showBoxes} style={{ width: '100%', background: '#000' }} />
      <div className="photo-actions mobile-actions">{actions}</div>
    </div>
    <aside className="panel" style={{ display: 'grid', gap: 16 }}>
      <div className="photo-actions desktop-actions">{actions}</div>
      <strong>{cameraEnabled ? 'Cámara prendida · foto congelada' : 'Cámara apagada · foto guardada en esta vista'}</strong>
      <div role="status">{status}</div><div data-testid="mediapipe-delegate">MediaPipe: {delegate}</div>
      <p>La foto se procesa en este navegador y no se sube a ningún lado.</p>
      <ControlsPanel>
      <label>Vista de la foto <select aria-label="Vista de la foto" value={show} onChange={e => setShow(e.target.value as typeof show)}>
        <option value="blend">Superposición</option><option value="mask">Máscara</option><option value="crop">Foto original</option>
      </select></label>
      <MaskControls style={maskStyle} width={contourWidth} onStyle={setMaskStyle} onWidth={setContourWidth} />
      <label><input type="checkbox" checked={showBoxes} onChange={e => onBoxesChange(e.target.checked)} />Mostrar cajas de recorte</label>
      <p className="muted">Amarillo: cajas usadas para segmentar esta foto, con margen del 12 %. La descarga respeta este interruptor.</p>
      <label><input type="checkbox" checked={clean} onChange={e => setClean(e.target.checked)} />Postprocesado de la foto</label>
      {show === 'blend' && <label>Opacidad del relleno {Math.round(opacity * 100)} %
        <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={e => setOpacity(Number(e.target.value))} />
      </label>}
      <p className="muted">Descarga la vista elegida a resolución original. Las cruces señalan los centros pupilares.</p>
      </ControlsPanel>
    </aside>
  </section>;
}
