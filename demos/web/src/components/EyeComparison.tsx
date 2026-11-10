import type { RefObject } from 'react';
export default function EyeComparison({ clean, segmented, centers, prefix, single = false, showClean = true }: {
  clean: RefObject<HTMLCanvasElement>[]; segmented: RefObject<HTMLCanvasElement>[];
  centers?: string[]; prefix: string; single?: boolean; showClean?: boolean;
}) {
  return <div className={single ? "eye-comparison side" : "eye-comparison"} data-testid={`${prefix}-comparison`}>
    <div hidden={!showClean}>
    <p>Recortes limpios</p>
    <div className="eyes-grid">{clean.map((ref, i) => <div key={i} hidden={single && i === 1}>
      {!single && <span>{i === 0 ? 'Izquierdo' : 'Derecho'}</span>}
      <canvas ref={ref} width="96" height="96" data-testid={`${prefix}-clean-${i}`} />
    </div>)}</div>
    </div>
    <div>
    <p>Segmentación</p>
    <div className="eyes-grid">{segmented.map((ref, i) => <div key={i} hidden={single && i === 1}>
      {!showClean && <span>{i === 0 ? 'Izquierdo' : 'Derecho'}</span>}
      <canvas ref={ref} width="96" height="96" data-testid={prefix === 'gallery' ? `segmentation-${i}` : `${prefix}-segmented-${i}`} />
      {centers && <p>Pupila: <span data-testid={prefix === 'live' ? `center-${i}` : `photo-center-${i}`}>{centers[i]}</span></p>}
    </div>)}</div>
    </div>
  </div>;
}
