import { useEffect, useState } from 'react';
export type Example = { description?: string; eye_crop?: boolean; id: string; label: string; src: string | null; thumbnail: string | null };
export default function ExampleGallery({ entries, onChoose, ready = true, id = 'ejemplos-final' }: {
  entries?: Example[]; onChoose?: (example: Example) => void; ready?: boolean; id?: string;
}) {
  const [loaded, setLoaded] = useState<Example[]>([]);
  useEffect(() => {
    if (entries) return;
    const abort = new AbortController();
    fetch('/examples/examples.json', { signal: abort.signal }).then(r => { if (!r.ok) throw new Error('Galería no disponible'); return r.json(); }).then(setLoaded).catch(() => {});
    return () => abort.abort();
  }, [entries]);
  return <section id={id} className="panel" aria-label="Probar con ejemplos" style={{ marginTop: 18, marginBottom: 18 }}>
    <h2>Probar con ejemplos</h2>
    <div className="example-grid">{(entries ?? loaded).map(example => {
      const content = <>{example.thumbnail && <img src={example.thumbnail} alt="" width="120" height="84" style={{ objectFit: 'cover' }} />}<span style={{ display: 'block' }}>{example.label}</span></>;
      return onChoose || !example.src
        ? <button key={example.id} data-example={example.id} disabled={!example.src || !ready} onClick={() => onChoose?.(example)}>{content}</button>
        : <a className="button-link" key={example.id} data-example={example.id} href={`/upload.html?example=${encodeURIComponent(example.id)}#resultados`}>{content}</a>;
    })}</div>
    <p className="muted">Recortes oculares de webcam y white-eyes. Las capturas originales de webcam no se distribuyen.</p>
  </section>;
}
