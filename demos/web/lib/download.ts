export function downloadCanvas(canvas: HTMLCanvasElement, filename: string) {
  canvas.toBlob(blob => {
    if (!blob) return;
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = filename; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, 'image/png');
}
/** Same aligned rows as the UI, preserving each canvas's backing resolution. */
export function downloadComparison(clean: HTMLCanvasElement[], segmented: HTMLCanvasElement[]) {
  const size = Math.max(...clean.concat(segmented).map(c => c.width));
  const canvas = document.createElement('canvas');
  const label = 32, gap = 12;
  canvas.width = size * clean.length; canvas.height = 2 * (size + label) + gap;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#171a20'; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.font = '16px sans-serif';
  for (const [row, items] of [clean, segmented].entries()) {
    const y = row * (size + label + gap);
    for (const [i, item] of items.entries()) {
      ctx.fillStyle = 'white'; ctx.fillText(`${row ? 'Segmentado' : 'Limpio'} · ${clean.length === 1 ? 'Recorte' : i === 0 ? 'Izquierdo' : 'Derecho'}`, i * size + 8, y + 22);
      ctx.drawImage(item, i * size, y + label, size, size);
    }
  }
  downloadCanvas(canvas, 'recortes-comparacion.png');
}
