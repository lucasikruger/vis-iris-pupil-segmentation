import { useEffect, useRef } from 'react';
import { drawCropBoxes, type CropOutline } from '@lib/render';

/** Transparent layer with the exact crop coordinates, scaled with its source image. */
export default function CropBoxes({ width, height, boxes, visible, testId }: {
  width: number; height: number; boxes: CropOutline[]; visible: boolean; testId: string;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const target = canvas.current!; target.width = width; target.height = height;
    if (visible && width && height) drawCropBoxes(target.getContext('2d')!, boxes, width);
  }, [width, height, boxes, visible]);
  return <canvas ref={canvas} aria-hidden="true" data-testid={testId}
    data-boxes={JSON.stringify(boxes)} data-visible={visible}
    style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }} />;
}
