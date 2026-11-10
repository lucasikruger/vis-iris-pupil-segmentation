# v1.0 — Datos y código de la tesis de segmentación ocular

Materiales de la tesis *Segmentación ocular de bajo costo con cámaras web
convencionales*, Lucas Ivan Kruger, Universidad de Buenos Aires, 2026.

El aporte principal es la preparación y anotación de datos en un formato común de
tres clases, con oclusiones preservadas. Se incluyen preparadores, particiones y
anotaciones propias de PolyU, CASIA, FEI, UBIRIS y MICHE, negativas white-eyes y
un conjunto independiente de webcam: 207 ojos anotados de 31 personas.
Las imágenes y etiquetas nativas de bases externas se solicitan a sus titulares.

- [Tesis en PDF](https://github.com/lucasikruger/vis-iris-pupil-segmentation/releases/download/v1.0/tesis.pdf), adjunta a este release.
- [Dataset](https://huggingface.co/datasets/lucasikruger/vis-iris-pupil-segmentation-data): acceso a pedido y aprobación manual. Completar nombre, institución, uso previsto y aceptación de uso no comercial y no identificar. Una vez aprobado, ejecutar `hf auth login` y el descargador del repositorio.
- [Demo web](https://huggingface.co/spaces/lucasikruger/vis-iris-pupil-segmentation): ejemplos o webcam, procesamiento local en el navegador.
- [Reconstrucción y límites](https://github.com/lucasikruger/vis-iris-pupil-segmentation/blob/v1.0/data/DATA.md).

La ResU-Net distribuida obtuvo 0,641 mIoU sobre la webcam propia. Se incluyen pesos,
exportación ONNX, entrenamiento y evaluación. La reconstrucción desde los originales
de MICHE y TEyeD sigue sin validar; pix2pix CPU no reproduce exactamente el JPEG
archivado de CASIA. El entrenamiento público reemplaza las oclusiones externas de
la tesis por oclusiones procedurales. No se midió un entrenamiento completo con
ese reemplazo.

Código GPL-3.0. Datos y pesos propios CC BY-NC 4.0. Citar la tesis y conservar la
atribución. El acceso a las anotaciones no concede permisos sobre bases ajenas.
