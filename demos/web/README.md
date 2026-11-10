# Demo web

La página principal abre en modo ejemplos y procesa un recorte propio. «Prender
cámara» pasa a webcam sin recargar. «Apagar cámara» y «Volver a ejemplos» liberan
los tracks y recuperan el ejemplo seleccionado. Todo se procesa en el navegador.

Desde la raíz del repositorio:

```sh
cd demos/web
npm ci
npm run dev
```

Abrir `http://localhost:4323`. La cámara requiere HTTPS o localhost y permiso del
navegador, solicitado solo después del clic. La orientación frontal se pide como
preferencia (`facingMode: ideal`). Se puede elegir trasera.

En vivo se muestra una fila de ojos segmentados encima del video con cajas. La
segmentación sobre la imagen completa es opcional y comienza apagada. Una foto
muestra primero los ojos limpios, debajo los segmentados y al final la imagen
completa con cajas y segmentación. Se descargan la foto marcada o la grilla PNG.
En escritorio las acciones quedan arriba del panel derecho. En mobile quedan
debajo de la imagen. Los controles se pueden plegar.

El modo inicial es borde de 2 px. El relleno de la superposición tiene 35 % de
opacidad. La visualización interpola probabilidades por clase antes de argmax,
alineadas al centro del píxel. Las métricas se calculan sobre 96×96.

## Modelo y preprocesado

`public/models/manifest.json` describe el modelo ONNX. Se aplica RGB→LAB uint8,
CLAHE sobre L (clipLimit 1,5 y tiles 8×8), LAB→RGB y gamma 0,8 con OpenCV.js.
El procesamiento reproduce el de Python sobre los mismos recortes decodificados.
El postprocesado y el suavizado temporal se pueden desactivar.

MediaPipe intenta GPU y recuerda el fallback a CPU. Cada error de detección
reinicia el tracking. El cuadro se omite y el aviso se borra al recuperarse.
La foto usa una instancia IMAGE. El vivo usa VIDEO y timestamps crecientes.

## Pruebas

```sh
cd demos/web
npx tsc --noEmit
npm test
npm run build
```

Para comprobar paridad Python/Node, desde la raíz y con el dataset autorizado:

```sh
python scripts/eval/check_preprocess.py
node demos/web/scripts/check-preprocess.mjs
node demos/web/scripts/check-display.mjs
python scripts/eval/check_display.py
```

La prueba de navegador necesita Playwright, Chromium y ffmpeg:

```sh
python -m pip install playwright
python -m playwright install chromium
python tests/browser/make_test_video.py
python -m http.server 8765 --bind 127.0.0.1 --directory demos/web/dist
# En otra terminal, desde la raíz:
python tests/browser/check_main_modes.py
```

Usa un rostro generado white-eyes, nunca una captura real de webcam. Comprueba
390×844 y 1366×768, exclusión de modos, conservación del ejemplo, ausencia de
recarga y cierre de streams, incluso los recibidos después de salir de webcam.
Los informes y capturas generados se guardan en `reports/`, excluido de Git.

## Space estático

Desde la raíz: `scripts/tools/build_space.sh dist-static-space`. La salida debe
ser nueva. El script instala dependencias, compila y reúne páginas, modelos,
ejemplos propios y licencias. No publica nada. El README de HF está en
`space/README.md`. Usa `sdk: static` y `app_file: index.html`.

Ver [reproducibilidad](../../docs/reproducibility.md) y
[procedencias y licencias](../../docs/THIRD_PARTY.md).
