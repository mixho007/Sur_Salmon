# AppleVision Chile
Sistema inteligente de clasificación y conteo de manzanas para plantas de embalaje frutícola.
Proyecto MLY1101 — Machine Learning | DUOC UC

## Objetivo del primer paso
Construir un modelo de visión (YOLO) que **detecte y cuente manzanas sueltas** que pasan frente a una cámara, como en una cinta transportadora.

En esta etapa **no se califica** (calibre, color, sana/podrida): solo se identifica cada manzana y se cuenta. La calificación queda para la etapa siguiente, sobre recortes de cada manzana detectada.

## Introducción del caso
Chile exporta manzanas bajo estándares SAG/Asoex. En las plantas de embalaje, la clasificación y el conteo manual generan errores y cuellos de botella.

La idea es automatizarlo con cámara y visión por computador. Las manzanas avanzan por una cinta, el modelo detecta cada una, y un contador suma las que cruzan el campo de visión. Después, cada detección se recorta y se califica.

Un primer dataset de huerto (fotos de dron) se descartó porque no coincide con el escenario real: ver [Datos](#datos).

## Datos
Se usan dos datasets públicos de Roboflow Universe, en formato **YOLOv8** (una línea por objeto: `clase x y ancho alto`, o polígono). Ambos se unificaron en una sola clase: `apple`.

| | Dataset 1: conveyor_apple_sorting | Dataset 2: apple-sorting |
|---|---|---|
| Link | https://universe.roboflow.com/fruitgradechecking/conveyor_apple_sorting | https://universe.roboflow.com/arfiani-nur-sayidah-9lizr/apple-sorting-2bfhk |
| Tipo de fotos | Frames de un video de manzanas sueltas sobre una cinta industrial | Manzanas apiladas en cajas o montones, con oclusión |
| Imágenes | 61 (43 train / 12 valid / 6 test) | 697 (489 / 178 / 30), versión 2 |
| Anotación | Polígonos | Cajas |
| Clases originales | `apple`, `apple_a` | `apple`, `damaged_apple` |
| Tamaño | 640×300 | 416×416 (con bordes blancos) |
| Licencia | Public Domain | CC BY 4.0 (dar crédito al autor) |

**Limitaciones conocidas**
- Dataset 1: todos los frames salen de un solo video, así que son muy parecidos entre sí. Tiene marca de agua (`clideo.com`) y desenfoque por movimiento.
- Dataset 2: no es de cinta; sirve para que el modelo aprenda cómo se ve una manzana.
- Hay muchas más imágenes de montones (697) que de cinta (61). Las métricas de `valid` y `test` reflejan sobre todo montones, no la cinta.

**Dataset descartado:** *Apple Detection on split images* (fotos de dron sobre huertos, cortadas en cuadrículas y con casi todo augmentado). Fue eliminado del proyecto.

## Preparación realizada
- Se descargaron ambos datasets con la API de Roboflow.
- Se fusionaron en `images/`. Las imágenes del dataset 2 llevan el prefijo `as_`.
- Se unificaron las clases: `apple_a` y `damaged_apple` pasan a `apple`.
- El dataset 2 original, con sus 2 clases, se conserva en `data/01_raw/apple_sorting_original/`.

Resultado en `images/`: 758 imágenes (train 532 / valid 190 / test 36) y 2.223 manzanas en una clase.

## Estructura del proyecto
```
images/                       dataset actual (YOLO, 1 clase)
  train|valid|test/images     imágenes
  train|valid|test/labels     etiquetas
  data.yaml                   configuración del dataset
data/
  01_raw/                     datasets originales
    apple_sorting_original/   dataset 2 con sus 2 clases
  02_processed/               salida del pipeline (detecciones.csv)
  03_features/                features para ML
src/
  vision/                     detector, calibrador, clasificador
  ml/                         features, train, evaluate
  utils/                      config, s3_helper
notebooks/                    01_eda, 02_models, 03_optimization
scripts/                      descargar y comprimir dataset
models/                       pesos entrenados (best.pt)
reports/                      informes y gráficos
config.yaml                   calibre, HSV, px_por_mm, fuente de video
```

## Instalación
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # solo si se usará S3 o Roboflow
```
Si usas Roboflow, guarda la clave en `.env` (`ROBOFLOW_API_KEY=...`). No la compartas ni la subas a git.

## Uso
1. Entrenar el detector:
   ```bash
   yolo detect train data=images/data.yaml model=yolo26n.pt epochs=100 imgsz=640
   ```
2. Ajustar `config.yaml` (calibre, HSV, `px_por_mm`, fuente de video).
3. Generar el CSV: `python -m src.vision.detector --modelo models/best.pt --video ruta.mp4`
   (salida: `data/02_processed/detecciones.csv`).
4. Abrir `notebooks/01_eda.ipynb`.

## Resultados
*Pendiente: aún no se entrena el modelo.* Al terminar se completará con:
- mAP50 y mAP50-95, precision y recall sobre `valid` y `test`.
- Prueba separada solo con frames de cinta.
- Conteo de manzanas en un video de prueba, comparado con el conteo manual.

## Metodología CRISP-DM
1. **Comprensión del negocio:** automatizar el conteo y la clasificación de manzanas en plantas de embalaje. Primer paso: conteo.
2. **Comprensión de los datos:** ver [Datos](#datos). Referencia: MinneApple (Häni et al., 2020). *(Completar con hallazgos de `01_eda.ipynb`.)*
3. **Preparación de datos:** unificación de clases y fusión de datasets (hecho). Pendiente: features y split final.
4. **Modelamiento:** YOLO para detección. Después, Random Forest y XGBoost (supervisado) y K-Means (no supervisado) para calificar.
5. **Evaluación:** mAP, Precision y Recall para detección. F1, AUC, Silhouette y validación cruzada estratificada para calificación.
6. **Despliegue:** demo en vivo, modelos en S3 y tracking con MLflow. *(Pendiente.)*

## Alcance
Ver `proyecto_manzanas_ml.md`. No clasifica variedad, no calcula peso real, no controla actuadores.

## Licencias
YOLO (ultralytics): AGPL-3.0, uso académico.
Datos: Public Domain (dataset 1) y CC BY 4.0 (dataset 2, crédito a su autor en Roboflow).
