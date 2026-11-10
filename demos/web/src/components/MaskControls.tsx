import type { MaskStyle } from '@lib/render';
export default function MaskControls({ style, width, onStyle, onWidth }: {
  style: MaskStyle; width: number; onStyle: (style: MaskStyle) => void; onWidth: (width: number) => void;
}) {
  return <div>
    <label>Representación de máscara <select value={style} onChange={e => onStyle(e.target.value as MaskStyle)}>
      <option value="fill">Relleno</option><option value="outline">Borde</option><option value="fill-outline">Relleno y borde</option>
    </select></label>
    <label>Grosor del borde: {width} px <input aria-label="Grosor del borde" type="range" min="1" max="5" step="1" value={width} disabled={style === 'fill'} onChange={e => onWidth(Number(e.target.value))} /></label>
    <p className="muted">Contorno exterior del iris y contorno de la pupila. Grosor en píxeles de pantalla. El PNG conserva la apariencia de la vista.</p>
  </div>;
}
