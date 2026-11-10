# Figuras

Figuras de la tesis de Lucas Ivan Kruger, bajo CC BY-NC 4.0. Las imágenes WebP
conservan el contenido de las figuras originales, con menor resolución y compresión.

| Archivo | Origen | Contenido |
|---|---|---|
| `pipeline.webp` | `fig_pipeline_eyetracking.png` | Diagrama propio de las etapas de seguimiento ocular. No contiene fotografías |
| `lodo.webp` | `fig_lodo_resultados.png` | Gráfico propio de métricas agregadas por fuente. No contiene imágenes de las bases |
| `margen.webp` | `fig_barrido_padding.png` | Gráfico propio del barrido de margen sobre webcam. Solo curvas y valores |
| `webcam.webp` | `fig_cualitativo_webcam.png` | Recortes del conjunto propio consentido, con predicciones y anotaciones manuales |
| `white-eyes.webp` | `fig_white_eyes_ejemplos.png` | Ojos de rostros generados por texto, sin fotografías de base |

Las figuras de webcam y white-eyes corresponden a los capítulos 5 y 3 de la tesis,
respectivamente. No contienen imágenes de MOBIUS, UBIRIS, MICHE, PolyU,
CASIA, TEyeD ni FEI.

La licencia de estas figuras es la de los [datos propios](../../data/LICENSE).

## Apertura del README

`datos-intro.png` conserva el diseño de dos filas de la figura 1.1, con cinco
recortes propios de webcam. No reutiliza las imágenes de terceros de esa figura.
La fila superior muestra los ojos; la inferior superpone las máscaras manuales
de referencia (iris verde con alfa 0,6; pupila azul opaca), no predicciones.

Se genera con `python scripts/tools/build_intro_figure.py --data-root dataset-package`.
El script adapta el diseño de `intro_seg_fig.py` de las herramientas de la tesis.
`datos-intro.json` registra los archivos de entrada, sus SHA-256 y las cajas al
12 % tomadas de la configuración de ejemplos. Solo lee `Images/` y `Masks/` del
conjunto propio de webcam. No lee capturas completas ni bases externas.
