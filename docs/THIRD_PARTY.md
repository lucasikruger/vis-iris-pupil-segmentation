# Procedencias y licencias

- `src/models/res_unet.py` conserva la implementación de investigación y su
  atribución a [HybridGNet](https://github.com/ngaggion/HybridGNet).
  El repositorio de origen publica [GPL-3.0](https://github.com/ngaggion/HybridGNet/blob/main/LICENSE),
  copiada en `docs/licenses/HybridGNet-GPL-3.0.txt`. El código del proyecto usa GPL-3.0 y conserva esta atribución.
- MediaPipe es un componente externo de Google, con
  [Apache-2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE).
  Se conserva el archivo Face Landmarker de la demo original. La procedencia exacta
  de ese binario y su correspondencia con el modelo descargable oficial deben
  confirmarse antes de redistribuirlo. No se aplica a él la licencia de datos propios.
- ONNX Runtime, Astro, React y las demás dependencias mantienen sus licencias.
  `demos/web/package-lock.json` fija las versiones instaladas. Los runtimes web se copian
  desde npm al compilar, no forman parte de los fuentes versionados.
- `src/nir2vis/models.py` y `data/prepare/nir_translate.py` proceden del código de la
  tesis para pix2pix. Sus pesos auxiliares fueron entrenados con PolyU. La licencia
  del código no concede derechos sobre sus datos de entrenamiento.
- Los siete assets externos de oclusión v3 no se incluyeron: no se encontró evidencia
  suficiente de autoría/licencia, incluido `glass_generated.png`. El entrenamiento
  público usa oclusión procedural por defecto. Los originales autorizados pueden
  añadirse en `src/assets/occlusion/` con las reglas de `src/assets/occlusion_spec.json`.
- Ver [data/DATA.md](../data/DATA.md) para las bases originales y las anotaciones.

Los datos propios y pesos del proyecto se distribuyen bajo CC BY-NC 4.0. No se
aplica esa licencia a los componentes de terceros enumerados aquí.

OpenCV.js 5.0.0 (@techstark/opencv-js 5.0.0-release.1): conversiones LAB uint8 y CLAHE en WASM. Apache 2.0, ver [licencia](licenses/OpenCV-Apache-2.0.txt).
