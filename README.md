# Modelo de Detección de Anomalías en Mamografías

Este repositorio corresponde al desarrollo de un sistema de detección de cáncer de mama basado en imágenes de mamografía, utilizando el modelo YOLO-v8 para identificar masas y calcificaciones.

## Descripción del Proyecto

El proyecto desarrolla un sistema de detección automática de anomalías en mamografías, específicamente enfocado en la identificación de masas y calcificaciones que podrían indicar la presencia de cáncer de mama. Utilizamos el modelo YOLO-v8 para el análisis de imágenes médicas, optimizado para maximizar la precisión en la detección.

## Exploración de Datos

El sistema utiliza la base de datos VinDr-Mammo, que contiene:
- 5,000 imágenes de casos de cáncer de mama 
- Cada examen fue sometido a doble lectura independiente
- Las discordancias fueron resueltas mediante arbitraje de un tercer radiólogo

De esta base, hemos extraído 1,659 casos específicos:
- 1,363 casos con masas y calcificaciones (BIRADS > 3)
- 296 casos sin anomalías detectables

## Preprocesamiento*

\*Los resultados de esta fase, previo al aumento de datos, pueden visualizarse en `notebooks/eda_annotations_processed.ipynb`.

### Filtrado y división de Datos
- Distribución: 70% de casos anómalos y 30% de casos no anómalos

  * La inclusión de imágenes sin anomalías ayuda a reducir falsos positivos
- División en 3 grupos para el entrenamiento de 3 modelos, cada uno con la misma distribución de casos anómalos y características relevantes, a modo de evitar sesgos y permitir que los modelos se complementen entre sí para la predicción.
- División por cada grupo: 70% entrenamiento, 15% validación (durante entrenamiento), 15% pruebas (después del entrenamiento)

### Ajustes de Anotaciones y Mamografías
1. Etiquetado: 
   - Clase 0: Con anomalías
   - Clase 1: Sin anomalías

2. Corrección de cajas delimitadoras: Ajustar coordenadas para que se delimitaran dentro de las dimensiones de la imagen.

4. Conversión a formato YOLO:
   - Transformación de coordenadas absolutas a normalizadas YOLO.
   - Fórmulas para el cálculo:
     - `bx (x_center) = (x_max - x_min) / (2 * ancho_imagen)`
     - `by (y_center) = (y_max - y_min) / (2 * alto_imagen)`
     - `bw = (x_max - x_min) / ancho_imagen`
     - `bh = (y_max - y_min) / alto_imagen`

5. Procesamiento de imágenes:
   - Conversión de DICOM a PNG
   - Estandarización: MONOCHROME1 (fondo blanco) a MONOCHROME2 (fondo negro)
   - Mejora de contraste: VOI LUT y filtro CLAHE
   - Redimensionamiento a 640x640 píxeles para compatibilidad con YOLO

### Aumento de Datos
- Generación de tres versiones aumentadas por imagen:
  - Inversión horizontal
  - Inversión vertical
  - Rotación de 90 grados

## Tuning

Se realizó un entrenamiento inicial reducido (x épocas, y iteraciones) para explorar:
- Tasa de aprendizaje inicial (`lr0`)
- Factor de tasa de aprendizaje final (`lrf`)
- Momento de SGD (`momentum`)
- Regularización L2 (`weight_decay`)
- Épocas de calentamiento (`warmup_epochs`)
- Momento inicial en fase de calentamiento (`warmup_momentum`)
- Peso de pérdida de caja delimitadora (`box`)
- Peso de pérdida de clasificación (`cls`)
Los mejores resultados son almacenados para su posterior uso en el entrenamiento.

## Entrenamiento
- Entrenamiento completo: 128 épocas (punto de convergencia óptimo de acuerdo a la literatura consultada)
- Se emplean los hiperparámetros obtenidos del proceso de tuning. Adicionalmente, se realizan modificaciones sobre los siguientes:
```
# Manage color augmentations for mammograms
hsv_h=0.0, 
hsv_s=0.0, 
hsv_v=0.2,
# Manage geometric augmentations for mammograms (flips already applied)
degrees=15.0,
fliplr=0.0,
flipud=0.0,
translate=0.1,
scale=0.2,
# Disable inappropriate augmentations for mammograms
shear=0.0,
mosaic=0.0,
mixup=0.0,
copy_paste=0.0,
# Remaining training parameters
epochs=epochs,
imgsz=640,
device=0,   # Use GPU 0
workers=1,
save_period=10, # Save every 10 epochs
patience=60, # Early stopping if there is no improvement
```
(Código de `src/model/run_model`)

## Validación

### Reporte de clasificación
**Precisión**: 
- Mide la proporción de detecciones positivas que son correctas.
- Fórmula: VP / (VP + FP)

**Recall (Sensibilidad)**:
- Mide la proporción de anomalías reales que son correctamente identificadas.
- Fórmula: VP / (VP + FN)

**F1-Score:**
- Media armónica entre precisión y recall, proporcionando un balance entre ambas métricas.
- Fórmula: 2 * (Precisión * Recall) / (Precisión + Recall)
  
**mAP50 (mean Average Precision con IoU de 0.5)**:
- Evalúa la precisión de las cajas delimitadoras cuando se considera un umbral de IoU (Intersection over Union) de 0.5.
- Una detección se considera correcta si la superposición entre la predicción y la anotación real supera el 50%.

**mAP50-95 (mean Average Precision promediada sobre múltiples umbrales de IoU)**:
- Promedio de mAP calculado en diferentes umbrales de IoU desde 0.5 hasta 0.95 con incrementos de 0.05.
- Proporciona una evaluación más robusta de la precisión de localización.

**Matriz de confusión**: 
  
- **Verdaderos Positivos (VP)**: Anomalías correctamente identificadas como anomalías.
- **Verdaderos Negativos (VN)**: Imágenes normales correctamente identificadas como normales.
- **Falsos Negativos (FN)**: Anomalías no detectadas por el modelo.
- **Falsos Positivos (FP)**: Detecciones incorrectas de anomalías en imágenes normales.

## Resultados

`TODO`

## Ejecución del Sistema

El programa principal se ejecuta desde `src/main.py` con los siguientes parámetros disponibles:

```bash
python src/main.py [opciones]
```

#### Opciones Disponibles

| Parámetro      | Descripción                                                        |
|----------------|--------------------------------------------------------------------|
| `--tuning`     | Ejecuta el proceso de tuning de hiperparámetros previo al entrenamiento completo |
| `--omit_setup` | Omite la fase de preprocesamiento (útil cuando ya se ha ejecutado anteriormente para ahorrar tiempo) |

## Estructura del Repositorio

```
.
├── output/                  # Directorio de archivos de salida
│   ├── data/                # Datos (anotaciones, resultados de tuning)
│   ├── images/              # Imágenes procesadas
│   └── validation/          # Resultados de la evaluación de los modelos (matrices de confusión, reportes de clasificación)
├── notebooks/               # Jupyter notebooks
├── src/                     # Código fuente
│   ├── preprocessing/       # Scripts de preprocesamiento
│   ├── augmentation/        # Scripts de aumento de datos
│   ├── tuning/              # Scripts de tuning/refinamiento de hiperparámetros
│   ├── model/               # Scripts de preparación para entrenamiento y entrenamiento
│   └── evaluation/          # Scripts de evaluación de rendimiento
│   └── config/              # Configuración
│   └── extras/              # Funciones auxiliares
│   └── main.py              # Script principal del programa
├── requirements.txt         # Dependencias del proyecto `TODO`
└── README.md                
```

## 📚 Referencias
- [Nguyen, H. T., Nguyen, H. Q., Pham, H. H., Lam, K., Le, L. T., Dao, M., \& Vu, V. (2023). VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography. *Scientific Data, 10*(1), 277.](https://doi.org/10.1038/s41597-023-02100-7)
- [Ultralytics. (17 de marzo, 2025). Configuración. *Ultralytics Yolo Docs.*](docs.ultralytics.com/es/usage/cfg/#augmentation-settings)
- [Ultralytics. (30 de marzo, 2025). Profundización en las métricas de rendimiento. *Ultralytics Yolo Docs.*](https://docs.ultralytics.com/es/guides/yolo-performance-metrics/)


## 📄 Licencias

Este proyecto utiliza datos bajo la PhysioNet Restricted Health Data License (Versión 1.5.0) del MIT Laboratory for Computational Physiology (LICENSE.TXT).
