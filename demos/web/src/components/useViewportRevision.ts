import { useEffect, useState } from 'react';
/** Repaint frozen/uploaded canvases when CSS dimensions or device scale change. */
export function useViewportRevision() {
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let raf = 0, width = innerWidth, scale = devicePixelRatio;
    const resize = () => {
      // Mobile browser bars (and full-page screenshots) change height, not crop width.
      if (innerWidth === width && devicePixelRatio === scale) return;
      width = innerWidth; scale = devicePixelRatio;
      cancelAnimationFrame(raf); raf = requestAnimationFrame(() => setRevision(v => v + 1));
    };
    window.addEventListener('resize', resize);
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize); };
  }, []);
  return revision;
}
