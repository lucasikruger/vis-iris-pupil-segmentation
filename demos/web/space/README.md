---
title: Iris y pupila
emoji: 👁️
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
license: gpl-3.0
pinned: false
---

Segmentación de iris y pupila con la webcam, dentro del navegador. MediaPipe recorta cada ojo con 12 % de margen por lado y una ResU-Net de 0,32 M de parámetros (entrada RGB de 96×96) segmenta iris y pupila con ONNX Runtime Web. Las imágenes no salen de la computadora del usuario.

- Código y entrenamiento: [github.com/lucasikruger/vis-iris-pupil-segmentation](https://github.com/lucasikruger/vis-iris-pupil-segmentation) (GPL-3.0)
- Datos propios: [lucasikruger/vis-iris-pupil-segmentation-data](https://huggingface.co/datasets/lucasikruger/vis-iris-pupil-segmentation-data) (CC BY-NC 4.0)
- Pesos del modelo: CC BY-NC 4.0

Tesis de licenciatura *Segmentación ocular de bajo costo con cámaras web convencionales*, Lucas Ivan Kruger, Universidad de Buenos Aires, 2026.
