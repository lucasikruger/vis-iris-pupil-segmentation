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
    <div className="mode-header">
      <h1>{mode === 'examples' ? 'Probar con ejemplos' : 'Modo webcam'}</h1>
      <p className="muted">{mode === 'examples' ? 'Elegí un ejemplo para ver su segmentación. La cámara está apagada.' : 'La cámara y las fotos se procesan en este navegador.'}</p>
      {mode === 'examples' && <button onClick={() => switchMode('webcam')}>Prender cámara</button>}
    </div>
    <div hidden={mode !== 'examples'} data-testid="examples-mode">
      <UploadDemo active={mode === 'examples'} autoSelect />
    </div>
    <div hidden={mode !== 'webcam'} data-testid="webcam-mode">
      <WebcamDemo active={mode === 'webcam'} onExit={() => switchMode('examples')} />
    </div>
  </>;
}
