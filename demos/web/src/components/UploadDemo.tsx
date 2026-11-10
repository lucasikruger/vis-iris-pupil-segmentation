import EyeComparison from './EyeComparison';
import { downloadComparison } from '@lib/download';
import { useViewportRevision } from './useViewportRevision';
import ControlsPanel from './ControlsPanel';
import MaskControls from './MaskControls';
import type { MaskStyle } from '@lib/render';
import { useEffect, useRef, useState } from 'react';
import { EyeCropper, type EyeCrop } from '@lib/cropper';
import { loadFaceLandmarker } from '@lib/mediapipe';
import { OnnxSegmenter, loadManifest } from '@lib/onnx';
import { renderCropWithMask, type CropOutline } from '@lib/render';
import { postprocess, pupilCenter } from '@lib/postprocess';
import CropBoxes from './CropBoxes';
import ExampleGallery, { type Example } from './ExampleGallery';
export default function UploadDemo({ active = true, autoSelect = false, onCamera }: { active?: boolean; autoSelect?: boolean; onCamera?: () => void }) {
  const img = useRef<HTMLImageElement>(null), left = useRef<HTMLCanvasElement>(null), right = useRef<HTMLCanvasElement>(null);
  const cleanLeft = useRef<HTMLCanvasElement>(null), cleanRight = useRef<HTMLCanvasElement>(null);
  const segmenter = useRef<OnnxSegmenter | null>(null), generation = useRef(0);
  const viewportRevision = useViewportRevision();
  const [maskStyle, setMaskStyle] = useState<MaskStyle>('outline');
  const [contourWidth, setContourWidth] = useState(2);
  const [showBoxes, setShowBoxes] = useState(true);
  const [boxFrame, setBoxFrame] = useState<{ width: number; height: number; boxes: CropOutline[] }>({ width: 0, height: 0, boxes: [] });
  const [ready, setReady] = useState(false), [url, setUrl] = useState<string | null>(null);
  const [examples, setExamples] = useState<Example[]>([]), [locals, setLocals] = useState<File[]>([]);
  const [clean, setClean] = useState(true), [eyeCrop, setEyeCrop] = useState(false);
  const [show, setShow] = useState<'crop' | 'mask' | 'blend'>('blend');
  const [status, setStatus] = useState('Cargando modelo…'), [loaded, setLoaded] = useState(0);
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const response = await fetch('/examples/examples.json'); if (!response.ok) throw new Error('No se pudo cargar la galería');
        const entries = await response.json(); if (active) setExamples(entries);
        const seg = new OnnxSegmenter(); await seg.load((await loadManifest())[0]);
        if (active) { segmenter.current = seg; setReady(true); setStatus('Elegí un ejemplo o una foto'); }
      } catch (e) { if (active) setStatus(`Error: ${(e as Error).message}`); }
    })();
    return () => { active = false; generation.current++; };
  }, []);
  useEffect(() => () => { if (url?.startsWith('blob:')) URL.revokeObjectURL(url); }, [url]);
  useEffect(() => {
    if (!active || !ready || !url || !img.current?.complete || !img.current.naturalWidth) return;
    const token = ++generation.current;
    (async () => {
      setStatus('Procesando…');
      setBoxFrame(previous => ({ ...previous, boxes: [] }));
      for (const ref of [left, right, cleanLeft, cleanRight]) { const c = ref.current; if (c) c.width = c.width; }
      try {
        const image = img.current!;
        let crops: EyeCrop[];
        if (eyeCrop) {
          const canvas = document.createElement('canvas'); canvas.width = canvas.height = 96;
          canvas.getContext('2d')!.drawImage(image, 0, 0, 96, 96);
          crops = [{ side: 'left', canvas, openness: 1, bbox: { x: 0, y: 0, w: image.naturalWidth, h: image.naturalHeight } }];
        } else {
          const detector = await loadFaceLandmarker('IMAGE');
          if (token !== generation.current) return;
          crops = await new EyeCropper({ outputSize: 96 }).cropEyes(detector, image);
        }
        if (token !== generation.current) return;
        setBoxFrame({ width: image.naturalWidth, height: image.naturalHeight, boxes: crops.map(({ side, bbox }) => ({ side, bbox })) });
        if (!crops.length) { setStatus('No se detectó un rostro. Para un ojo, activá «Ya es un recorte ocular».'); return; }
        let pupils = 0;
        for (const crop of crops) {
          let result = await segmenter.current!.run(crop.canvas);
          if (token !== generation.current) return;
          if (clean) result = postprocess(result);
          const center = pupilCenter(result); if (center) pupils++;
          const target = (crop.side === 'left' ? left : right).current!;
          renderCropWithMask((crop.side === 'left' ? cleanLeft : cleanRight).current!, crop.canvas, result, { show: 'crop' });
          renderCropWithMask(target, crop.canvas, result, { maskStyle, contourWidth, show, center });
        }
        setStatus(`${crops.length} ojo(s) procesado(s). ${pupils} pupila(s) detectada(s).`);
      } catch (e) { if (token === generation.current) setStatus(`Error: ${(e as Error).message}`); }
    })();
    return () => { generation.current++; };
  }, [active, ready, url, loaded, clean, eyeCrop, show, maskStyle, contourWidth, viewportRevision]);
  function choose(src: string) { if (src === url) return; generation.current++; setBoxFrame(previous => ({ ...previous, boxes: [] })); setUrl(src); setStatus('Cargando imagen…'); }
  function selectExample(example: Example) {
    setEyeCrop(example.eye_crop === true); choose(example.src!);
    document.getElementById('resultados')?.scrollIntoView({ block: 'start' });
  }
  useEffect(() => {
    if (!ready) return;
    const requested = new URLSearchParams(location.search).get('example');
    const example = examples.find(e => e.id === requested && e.src) ?? (autoSelect ? examples.find(e => e.src) : undefined);
    if (example) { setEyeCrop(example.eye_crop === true); choose(example.src!); }
  }, [ready]);
  return <>
    <ExampleGallery id="ejemplos" entries={examples} ready={ready} onChoose={selectExample} />
    <div id="resultados" className="demo-grid"><div className="panel">
      {examples.find(e => e.src === url)?.description && <p data-testid="example-description">{examples.find(e => e.src === url)!.description}</p>}
      {url ? <div style={{ position: 'relative', display: eyeCrop ? 'none' : undefined }}><img ref={img} src={url} alt="Imagen seleccionada" onLoad={() => { setBoxFrame(previous => ({ ...previous, boxes: [] })); setLoaded(n => n + 1); }} onError={() => setStatus('Error al cargar la imagen')} style={{ display: 'block', width: '100%' }} /><CropBoxes {...boxFrame} visible={showBoxes} testId="gallery-crop-boxes" /></div> : <p>Elegí un ejemplo o una foto.</p>}
      <EyeComparison clean={[cleanLeft, cleanRight]} segmented={[left, right]} prefix="gallery" single={eyeCrop} />
      <button disabled={!boxFrame.boxes.length} onClick={() => downloadComparison((eyeCrop ? [cleanLeft] : [cleanLeft, cleanRight]).map(r => r.current!), (eyeCrop ? [left] : [left, right]).map(r => r.current!))}>Descargar grilla PNG</button>
    </div><aside className="panel" style={{ display: 'grid', gap: 12 }}>
      {onCamera && <button onClick={onCamera}>Probar con la cámara</button>}
      <div role="status">{status}</div>
      <ControlsPanel>
      <label>Foto <input type="file" accept="image/*" onChange={e => { const f = e.target.files?.[0]; if (f) choose(URL.createObjectURL(f)); }} /></label>
      <MaskControls style={maskStyle} width={contourWidth} onStyle={setMaskStyle} onWidth={setContourWidth} />
      <label><input type="checkbox" checked={showBoxes} onChange={e => setShowBoxes(e.target.checked)} />Mostrar cajas de recorte</label>
      <p className="muted">Amarillo: recorte enviado a la red, con margen del 12 %. Si la entrada ya es un ojo, se usa la imagen completa.</p>
      <label><input type="checkbox" checked={eyeCrop} onChange={e => setEyeCrop(e.target.checked)} />Ya es un recorte ocular</label>
      <label><input type="checkbox" checked={clean} onChange={e => setClean(e.target.checked)} />Postprocesado</label>
      <label>Vista <select value={show} onChange={e => setShow(e.target.value as typeof show)}><option value="blend">Superposición</option><option value="mask">Máscara</option><option value="crop">Recorte</option></select></label>
      <details><summary>Carpeta local</summary><p>Para imágenes obtenidas con permiso, por ejemplo MOBIUS. Los archivos quedan en tu navegador.</p>
        <input type="file" aria-label="Carpeta local" {...{ webkitdirectory: '', directory: '' } as any} multiple onChange={e => setLocals(Array.from(e.target.files ?? []).filter(f => /\.(png|jpe?g|webp|bmp)$/i.test(f.name)))} />
        {locals.map((file, i) => <button key={i} onClick={() => choose(URL.createObjectURL(file))}>{file.name}</button>)}
      </details>
      </ControlsPanel>
    </aside></div>
  </>;
}
