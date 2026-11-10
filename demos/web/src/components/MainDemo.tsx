import { useState } from 'react';
import UploadDemo from './UploadDemo';
import WebcamDemo from './WebcamDemo';

export default function MainDemo() {
  const [mode, setMode] = useState<'examples' | 'webcam'>('examples');
  function switchMode(next: typeof mode) {
    setMode(next);
    window.scrollTo({ top: 0, behavior: 'instant' });
  }
  return <>
    {mode === 'webcam' && <div className="mode-header">
      <h1>Modo webcam</h1>
      <p className="muted">La cámara y las fotos se procesan en este navegador.</p>
    </div>}
    <div hidden={mode !== 'examples'} data-testid="examples-mode">
      <UploadDemo active={mode === 'examples'} autoSelect onCamera={() => switchMode('webcam')} />
    </div>
    <div hidden={mode !== 'webcam'} data-testid="webcam-mode">
      <WebcamDemo active={mode === 'webcam'} onExit={() => switchMode('examples')} />
    </div>
  </>;
}
