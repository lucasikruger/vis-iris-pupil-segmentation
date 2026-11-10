# Pesos del modelo reportado

`model.pt` es copia íntegra de `best.pt` de la corrida
`res_unet_in_b2__mobius__b_cap_b2__20260731-024417`. Corresponde a la mejor validación, época 99, con EMA. `val_iou_mean=0.8529487329911185`. Entrada RGB 96×96, tres clases, sin TTA.
MOBIUS quedó fuera del entrenamiento. No es un checkpoint entrenado con todo el pool.

La tesis informa LODO con EMA de época 100 y webcam con este best de época 99.
El checkpoint inicial de comparación no se distribuye. No es necesario para la demo.

`demos/web/public/models/res_unet_in_b2__96.onnx` es la exportación usada por la web,
conservada sin alteraciones. La prueba comprueba coherencia con PyTorch. Sobre ruido
RGB, error máximo observado ~0,00011 en logits. Sobre negro constante, ~0,026 por
sensibilidad numérica de InstanceNorm, pero la máscara argmax coincide completamente.
El preprocesado no está dentro del ONNX: se aplica CLAHE + gamma antes de inferir.

Los SHA-256 están en `manifest.json`. Reexportación opcional:
`python scripts/tools/export_onnx.py --output runs/model.onnx`.

Los pesos auxiliares pix2pix viven en `auxiliary-package/`, separado del repositorio.
Ambos pesos son CC BY-NC 4.0: atribución y uso no comercial.
Texto completo en `LICENSE` de esta carpeta y del paquete auxiliar. El código es
GPL-3.0. MediaPipe y las bases de entrenamiento conservan sus propias condiciones.
