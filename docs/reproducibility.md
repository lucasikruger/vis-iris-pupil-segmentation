# Reproducibilidad

Ejecutar los comandos desde la raíz del repositorio. El código compartido vive en
`src/`. La configuración de entrenamiento está en `src/configs/final.yaml` y las
reglas de oclusión, en `src/assets/occlusion_spec.json`.

## Datos y entrenamiento

El [catálogo](../data/DATA.md) detalla la reconstrucción por fuente. Los preparadores
están en `data/prepare/`. Usan las cajas guardadas, no vuelven a detectar los ojos.
El dataset requiere solicitud, aprobación manual y `hf auth login` antes de usar
`python scripts/tools/download_dataset.py`.

La webcam queda reservada para evaluación. El entrenamiento corto usa negativas
propias: `python scripts/eval/smoke_train.py`. Para la evaluación de los 207 recortes:
`python scripts/eval/evaluate.py --data-root dataset-package --output reports/webcam.json`.
El [resultado conservado](results/webcam.json) reproduce el protocolo de la tesis:
media de IoU por lote de 16, margen 12 %, sin ajustar el modelo con webcam.

La extracción NIR de CASIA coincide con sus cajas. La traducción pix2pix en CPU
no reproduce exactamente el JPEG archivado. Se conserva el
[informe de preparación](results/preparation-casia.json). La extracción completa
desde los originales de MICHE y TEyeD sigue sin validar.

Las siete imágenes externas de oclusión usadas en la tesis no se redistribuyen.
La receta pública usa párpados procedurales en su lugar. Los originales obtenidos
con permiso se pueden colocar en `src/assets/occlusion/`. No se midió la precisión
de un entrenamiento completo con el reemplazo procedural.

## Verificación

```sh
python -m pytest -q
python data/prepare/check_boxes.py
python scripts/eval/smoke_train.py
cd demos/web && npx tsc --noEmit && npm test && npm run build
```

El empaquetador `scripts/tools/build_space.sh` genera el Space estático en una
salida nueva y no publica ni sube archivos. Conserva las rutas necesarias para
modelos, ejemplos y licencias. No depende de la ubicación del clon original.

La [demo web](../demos/web/README.md) documenta las pruebas de navegador y paridad.
Los informes de ejecución, capturas privadas, videos de prueba y diagnósticos
quedan en `reports/` o `runs/`, excluidos de Git. Solo se conservan los dos JSON
citados arriba en `docs/results/`. No son necesarios para ejecutar la demo.
