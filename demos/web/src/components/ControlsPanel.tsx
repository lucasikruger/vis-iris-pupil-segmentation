import { useEffect, useState, type ReactNode } from 'react';
export default function ControlsPanel({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const media = matchMedia('(min-width: 961px)');
    const update = () => setOpen(media.matches);
    update(); media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);
  return <details className="controls-panel" open={open} onToggle={e => setOpen(e.currentTarget.open)}>
    <summary>Controles</summary><div className="controls-body">{children}</div>
  </details>;
}
