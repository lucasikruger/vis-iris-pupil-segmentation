---
license: cc-by-nc-4.0
language:
- es
- en
task_categories:
- image-segmentation
pretty_name: Anotaciones de iris y pupila y webcam propia
---

# Datos propios de segmentación ocular

Autor: Lucas Ivan Kruger. CC BY-NC 4.0: atribución y uso no comercial.
Texto completo en `LICENSE`. Citar la tesis e indicar modificaciones.
Dataset: [lucasikruger/vis-iris-pupil-segmentation-data](https://huggingface.co/datasets/lucasikruger/vis-iris-pupil-segmentation-data).
[Código GPL-3.0](https://github.com/lucasikruger/vis-iris-pupil-segmentation).

El acceso es **a pedido, con aprobación manual**. En la página del dataset,
completá el formulario con nombre, institución y uso previsto, y aceptá las
condiciones de uso no comercial y de no identificar a las personas. Esperá la
aprobación; luego iniciá sesión con **la misma cuenta aprobada**:

```sh
hf auth login
python scripts/tools/download_dataset.py --output dataset-package
```

Los comandos se ejecutan desde el repositorio de código.
El comando `hf` se instala con `pip install -r requirements.txt`. Usá un token
con permiso de lectura del dataset. Iniciar sesión no concede acceso por sí solo:
una solicitud pendiente o rechazada impide la descarga. El script explica estos
pasos si falta autenticación o autorización. Para fijar una versión, agregá
`--revision <commit-del-dataset>`.


- `webcam_manual_mobius-format/`: 218 recortes oculares de 31 personas
  y 207 máscaras manuales. Las capturas originales no se distribuyen. Los 11 descartes no tienen GT. Solo prueba.
  `crops.csv`, `bboxes.json`, `anotaciones.json`, `descartados.json` y `data.csv`
  conservan la geometría para evaluación con margen del 12 % por lado.
- `white_eyes_bg/` y `white_eyes_bg_test/`: 300 y 38 recortes negativos. Máscaras de fondo.
  `white-eyes-originals/` conserva las 150 y 19 imágenes generadas de origen.
- `polyu_zoomed_mobius-format/Masks/`: 5.749 máscaras propias proyectadas.
- `casia_pix2pix_rgb_mobius-format/Masks/`: 2.479 máscaras del proceso híbrido propio.
- `fei_manual_mobius-format/Masks/`: 210 máscaras manuales propias.
- `annotations/`: máscaras PolyU previas al zoom, pupila UBIRIS en geometría original
  y final (solo ese canal), cajas exactas FEI (210), webcam (218), white-eyes (300+38), CASIA (2.479),
  MOBIUS (3.542), PolyU (5.749) y UBIRIS (2.250). Polígonos/coordenadas estimadas
  MICHE de los 792 casos. Formato y uso: `DATA.md`.

No se incluyen imágenes de bases externas, sus versiones pix2pix, máscaras MOBIUS,
GT iris de UBIRIS, GT manual MICHE ni GT nativo TEyeD.
Las 31 personas del conjunto webcam dieron su consentimiento.

Máscaras MOBIUS: en RGB, rojo=fondo, verde=iris y azul=pupila. Si verde y azul están
ambos activos, prevalece pupila. Los PNG UBIRIS de `annotations/ubiris_pupil_*`
son binarios (255=pupila). No son máscaras de iris.

Uso: colocar este directorio como `dataset-package/` junto al código y ejecutar
`python scripts/eval/evaluate.py`. No entrenar ni seleccionar hiperparámetros con la webcam.
Los recortes almacenados tienen margen histórico 40 %. La evaluación reconstruye
12 % desde el contorno crudo. Redimensionarlos directamente no reproduce la tesis.

English: Author-created iris/pupil annotations, generated white-eye negatives,
and a consented webcam evaluation set (31 people, 207 annotated eye crops).
Third-party source photographs and native labels are excluded. Author-owned data
are licensed CC BY-NC 4.0.

Las cajas CASIA coinciden exactamente con los recortes NIR archivados.
