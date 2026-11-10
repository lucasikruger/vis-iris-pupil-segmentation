# Datos y reconstrucción

El paquete público no incluye imágenes ni máscaras nativas de bases de terceros,
ni imágenes derivadas con pix2pix. `partitions.csv` contiene las 16.898 filas del pool,
rutas relativas a `data/mobius_format`, `dataset`, `source`, `dataset_format`, `subject_id`
y `split`. El CSV no contiene edad, género, condiciones médicas ni otras columnas personales.
La clave de sujeto es un identificador de particionado, no un nombre.

El split guardado corresponde al fold MOBIUS: train 11.605, val 1.751, test 3.542.
`src/splits.py` reconstruye los otros folds usando el mismo orden de filas, agrupación
por fuente, `RandomState(42)` y 15 % de identificadores. TEyeD se separa por video.
No se puede certificar separación por persona. Las 1.260 negativas solo van a train.
La webcam propia queda fuera de este CSV y se utiliza exclusivamente como prueba.

## Fuentes y permisos

| Fuente | Filas del pool | Obtención y licencia | Qué se entrega |
|---|---:|---|---|
| MOBIUS | 3.542 | [Universidad de Ljubljana](https://sclera.fri.uni-lj.si/datasets.html): formulario firmado, condiciones del titular | Índice y cajas. Imágenes y máscaras se piden al titular |
| PolyU Cross-Spectral | 5.749 | [PolyU](https://www4.comp.polyu.edu.hk/~csajaykr/polyuiris.htm): solicitud. Prohíbe distribución y uso comercial de la base | Máscaras propias proyectadas de NIR a VIS. Ninguna imagen |
| UBIRIS.v2 | 2.250 | [UBI](https://iris.di.ubi.pt/ubiris2.html): obtener imágenes y anotación original bajo sus condiciones. Licencia exacta pendiente de archivar | Solo canal pupilar generado/corregido por la tesis, no el iris de terceros |
| FEI | 210 | [FEI](https://fei.edu.br/~cet/facedatabase.html): solicitar imágenes y condiciones al titular | Máscaras manuales propias y cajas. Ninguna cara de FEI |
| MICHE-I | 792 | [Proyecto MICHE](http://biplab.unisa.it/MICHE/database/). Máscaras Wang cedidas por sus autores por correo, permiso no transferido | Polígonos palpebrales y coordenadas propias. No GT manual ajeno |
| CASIA Iris Distance | 2.479 | [CASIA](https://biometrics.idealtest.org/): solicitar la base y aceptar sus condiciones. Términos exactos pendientes | Máscaras propias del proceso híbrido. Ninguna imagen NIR ni traducida |
| TayedEyes / TEyeD | 616 | [Publicación de los autores](https://arxiv.org/abs/2102.02115): solicitar imágenes y GT. Licencia exacta pendiente | Índice por video. No máscaras nativas ni imágenes |
| Ojos cerrados | 960 | [Kaggle](https://www.kaggle.com/datasets/sehriyarmemmedli/open-closed-eyes-dataset): la tesis informa Apache-2.0. Falta archivar la licencia de esa versión | Huellas de píxeles para recuperar la selección curada, ninguna imagen |
| White-eyes | 300 | Rostros generados sin imágenes base | Rostros generados, recortes y máscaras de fondo |

Se incluyen además 38 negativas white-eyes de prueba y el conjunto propio de webcam:
218 recortes oculares de 31 personas, 207 máscaras manuales y 11 descartes.
Las capturas originales de webcam no se distribuyen. Las cajas y contornos
conservan sus coordenadas sobre esas capturas privadas.
Las 62 negativas de ojos cerrados de prueba no están incluidas en el paquete.
TEyeD conserva GT nativo, no es anotación propia.
CASIA usa en parte un pipeline híbrido (IrisParseNet + open-iris), según el índice original.

## Comandos

Python 3.11, dependencias de `requirements.txt`. Ejecutar desde la raíz.
Descargar/descomprimir las fuentes por cuenta propia, conservando sus condiciones.
Cada script usa el CSV como lista cerrada de ejemplos y escribe solo bajo `--output`.
No borra el directorio de entrada. `--limit N` permite verificar una muestra.

```sh
python data/prepare/prepare_mobius.py --raw originals/mobius-export
python data/prepare/prepare_polyu.py --raw originals/PolyU_Cross_Iris
python data/prepare/prepare_ubiris.py --raw originals/ubiris-images --masks originals/ubiris-masks
python data/prepare/prepare_fei.py --raw originals/fei
python data/prepare/prepare_miche.py --raw originals/wang-eyes --masks originals/wang-mobius-masks
python data/prepare/prepare_casia.py --raw originals/CASIA-Iris-Distance --generator auxiliary-package/pix2pix_rgb.pt
python data/prepare/prepare_tayed.py --raw originals/tayed-nir-eye-crops --masks originals/tayed-mobius-masks --generator auxiliary-package/pix2pix_rgb.pt
python data/prepare/prepare_closed_eyes.py --raw originals/closed-eyes-extracted
python data/prepare/prepare_white_eyes.py --raw dataset-package
python data/prepare/prepare_webcam.py --raw dataset-package
```

`--annotations dataset-package` es el valor predeterminado. Para copiar las máscaras
propias necesarias, los scripts las leen directamente del paquete separado.

- MOBIUS: entrada con `MOBIUS/{Images,Masks}/...`, como la exportación original.
  Selección por CSV, sin cambios de píxeles. No confundir los 3.559 ejemplos ofrecidos
  por el proveedor con los 3.542 usados en esta tesis.
- PolyU: indicar la carpeta de imágenes crudas de sesión 1 `PolyU_Cross_Iris`,
  sin las versiones normalizadas ni sesión 2 que repiten nombres. Proyección guardada,
  JPEG 95 intermedio, recorte 35 %, 480×360 AREA, JPEG 95. Un caso verificado byte a byte.
- UBIRIS: imágenes `C100_S1_I1.tiff`. GT de iris `OperatorA_C100_S1_I1.tiff`
  (blanco=iris). Se rellenan huecos del iris, se incorpora solo la pupila propia,
  JPEG 95 intermedio, recorte 35 %, 400×300. Un caso verificado byte a byte.
- FEI: originales `1-03.jpg`, cajas guardadas, resize LANCZOS4 a 512, JPEG 95,
  máscaras manuales. Dos casos verificados byte a byte.
- MICHE: el script implementa recorte palpebral y publicación, pero requiere los
  recortes Wang y GT ya convertido a MOBIUS a 512×512. Falta validar la transformación
  completa desde la distribución original. No está certificada la reconstrucción exacta.
- CASIA: extrae los ojos desde las capturas originales usando las 2.479 cajas
  recuperadas y luego aplica pix2pix. `--nir-only` reconstruye únicamente los PNG NIR
  y comprueba sus hashes archivados. Todos los recortes NIR coincidieron píxel a píxel
  con regiones de las capturas completas. La traducción CPU de
  un caso difiere del JPEG archivado. Ver `docs/results/preparation-casia.json`.
- TEyeD: todavía requiere recortes NIR previos y GT nativo convertido. Falta
  certificar la extracción desde la base original y la equivalencia de pix2pix.
- Cerrados: después de descomprimir, identifica los 960 elegidos por SHA-256 de
  píxeles BGR. Evita depender del orden del ZIP o repetir la curaduría manual.
  No se verificó aún sobre el ZIP completo.

`prepared_sha256.json` conserva las huellas de las 33.796 imágenes/máscaras del pool
original. Cada preparador informa diferencias y devuelve error si no coincide.
Las diferencias de códec también pueden producir hashes distintos.

## Paquete separado

`dataset-package/` está excluido del Git del código. Licencia de los datos propios: CC BY-NC 4.0.
Su README y el texto completo de licencia se
conservan también en `data/package/`. Dataset:
[lucasikruger/vis-iris-pupil-segmentation-data](https://huggingface.co/datasets/lucasikruger/vis-iris-pupil-segmentation-data).

El acceso es **a pedido, con aprobación manual**. En la página del dataset,
completá el formulario con nombre, institución y uso previsto, y aceptá las
condiciones de uso no comercial y de no identificar a las personas. Esperá la
aprobación. Luego iniciá sesión con **la misma cuenta aprobada**:

```sh
hf auth login
python scripts/tools/download_dataset.py --output dataset-package
```

El comando `hf` se instala con `pip install -r requirements.txt`. Usá un token
con permiso de lectura del dataset. Iniciar sesión no concede acceso por sí solo:
una solicitud pendiente o rechazada impide la descarga. El script explica estos
pasos si falta autenticación o autorización. Para fijar una versión, agregá
`--revision <commit-del-dataset>`.

Las anotaciones propias no trasladan permisos sobre las imágenes originales.

El CSV de particiones y las anotaciones propias están cubiertos por `data/LICENSE`.
Las fotografías y etiquetas nativas externas siguen sujetas a sus licencias.

## Cajas reproducibles (`annotations/*_boxes.json`)

El paquete incluye 5.749 máscaras propias de PolyU y 2.479 de CASIA.

Cada JSON tiene `schema_version: 1`, `coordinate_system: "pixel_xyxy_half_open"`,
`source` (procedencia) y `records`, indexado por el nombre del recorte sin extensión.
Las coordenadas son enteros en píxeles de la imagen original: origen arriba a la
izquierda, x hacia la derecha, y hacia abajo. `bbox_xyxy: [x0,y0,x1,y1]` incluye
el extremo inicial y excluye el final: `image[y0:y1, x0:x1]`, antes del resize.
`source_size` y `output_size` siempre son `[ancho,alto]`.

Cada registro identifica `source_image` relativa a `--raw`, `image_path` y
`mask_path` relativas al pool de salida, dimensiones, caja y `resize` (`none`,
`AREA`, `CUBIC` o `LANCZOS4`, OpenCV). `encoding`, cuando aparece, indica el códec
final. `before_crop: JPEG95_roundtrip` en PolyU/UBIRIS requiere codificar/decodificar
el original a JPEG 95 antes de recortar. Las máscaras usan vecino más cercano.
Webcam copia los 218 recortes archivados desde el paquete y conserva sus cajas.
No requiere las capturas privadas. La evaluación aplica esas coordenadas para
reencuadrar al 12 %. Los demás preparadores validan dimensiones y límites y consumen estas cajas. No vuelven
a detectar ojos ni a estimar la caja a partir de la máscara.

| Archivo en `annotations/` | Registros | Original relativo a `--raw` / resultado |
|---|---:|---|
| `fei_boxes.json` | 210 | `1-03.jpg`, etc. 512×512 LANCZOS4, JPEG 95 |
| `webcam_boxes.json` | 218 | Original privado no distribuido. Recortes 512×512 CUBIC, PNG |
| `white_eyes_train_boxes.json` | 300 | `white-eyes-originals/train/…`. 256×256 AREA, PNG |
| `white_eyes_test_boxes.json` | 38 | `white-eyes-originals/test/…`. 256×256 AREA, PNG |
| `casia_boxes.json` | 2.479 | `000/S4000D00.jpg`, etc. Ojo NIR sin resize, anterior a pix2pix |
| `mobius_boxes.json` | 3.542 | `MOBIUS/Images/…`. Cuadro completo del pool y caja de evaluación |
| `polyu_boxes.json` | 5.749 | `001/L/VIS/001_L_VIS_1.tiff`, etc. 480×360 AREA, JPEG 95 |
| `ubiris_boxes.json` | 2.250 | `C100_S1_I1.tiff`, etc. 400×300 AREA, JPEG 95 |

FEI conserva además `fei_bboxes.json`: sus claves cubren los mismos 210 casos.
Webcam conserva el `crops.csv` del laboratorio y su contorno de MediaPipe de 16
puntos: `raw_contour_crop_xy` está referido a la caja inicial
`raw_crop_bbox_xyxy`, y `mediapipe_contour_xy` está referido al original completo.
`bbox_xyxy` es la caja efectivamente usada para el PNG almacenado, con margen
histórico del 40 %. En tres casos se reencuadró respecto de la caja inicial:
`person14_4__f3_izq`, `person14_4__f3_der` y `person28_1__f2_der`.
Se preservan ambas geometrías para mantener la evaluación de la tesis al 12 %.
`has_mask` distingue los 207 anotados de los 11 descartados.

Las cajas CASIA ubican cada ojo NIR archivado dentro de su captura original
con igualdad de todos los píxeles BGR. Se guardan `source_sha256`, `crop_sha256`,
lado y procedencia v1/v2. Las cajas describen el NIR previo a la traducción. Las
rutas del pool identifican la imagen final pix2pix. Esto resuelve el recorte,
pero no la diferencia pendiente de runtime/códec de pix2pix.

En MOBIUS, `bbox_xyxy` abarca toda la imagen del pool: no había un recorte offline.
`evaluation_bbox_xyxy` y `evaluation_output_size` guardan el recorte determinista
usado con objetivo de iris 0,35 y salida 96×96. Para aplicarlo:
`python data/prepare/prepare_mobius.py --raw originals/mobius-export --stage evaluation --output runs/mobius-evaluation`.
La imagen usa LINEAR y la máscara NEAREST. El JPEG de salida es una exportación.
La evaluación normal redimensiona en memoria. El entrenamiento tiene jitter
aleatorio: no existe una única caja histórica por imagen para todas las épocas.

Validación: `python data/prepare/check_boxes.py` comprueba cobertura, límites,
contornos y conservación íntegra de las máscaras PolyU/CASIA. Ver
`reports/crop-boxes.json` y `reports/crop-reconstruction.json` como informes locales, fuera de Git. Ver [reproducibilidad](../docs/reproducibility.md).
MICHE conserva sus polígonos y coordenadas estimadas previas, pero no se certifica
una caja exacta sobre la captura completa. La extracción de TEyeD tampoco está validada.
