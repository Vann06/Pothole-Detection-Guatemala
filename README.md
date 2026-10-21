# Pothole Detection Guatemala

Proyecto de Deep Learning para estudiar la **detección de presencia de baches (potholes)** en imágenes viales y analizar qué tan bien un modelo entrenado con datos internacionales generaliza a imágenes de **Guatemala**.

El proyecto utiliza **RDD2022** como dataset principal y compara enfoques de:

- **Transfer Learning** con ResNet18 preentrenada.
- ResNet18 entrenada **desde cero**.
- **Data augmentation** para reducir sensibilidad al cambio de dominio.
- **Variational Autoencoder (VAE)** para detección de anomalías.
- **XAI / Grad-CAM** para interpretar predicciones y errores.
- Evaluación **in-distribution**, **out-of-distribution (OOD)** y posteriormente en un conjunto externo de Guatemala.

---

## 1. Pregunta principal

> ¿Qué tan bien generaliza un modelo entrenado con imágenes de daños viales de otros países a calles de Guatemala, y qué técnicas ayudan a reducir el domain shift?

La tarea supervisada principal se define a nivel de imagen como:

- `has_D40 = 1`: la imagen contiene al menos un pothole `D40`.
- `has_D40 = 0`: la imagen no contiene `D40`, aunque pueda contener otro tipo de daño.

RDD2022 contiene anotaciones de objetos en formato Pascal VOC y conserva etiquetas adicionales que se mantienen en los manifests para análisis posterior.

---

## 2. Dataset

Se utiliza **RDD2022**, con 47,420 imágenes en total.

Dominios disponibles:

- China_Drone
- China_MotorBike
- Czech
- India
- Japan
- Norway
- United_States

Validación local realizada:

| Elemento | Cantidad |
|---|---:|
| Imágenes train | 38,385 |
| XML train | 38,385 |
| Imágenes test | 9,035 |
| Total imágenes | 47,420 |
| Objetos anotados | 65,712 |
| Imágenes train con D40 | 3,674 |
| Imágenes train sin D40 | 34,711 |

La prevalencia global de D40 en las imágenes etiquetadas es aproximadamente **9.57%**, por lo que el problema supervisado está desbalanceado.

Los datos crudos **no se suben a Git**.

Consulta [data/README.md](data/README.md) para instalar y organizar el dataset.

---

## 3. Estructura del repositorio

```text
Pothole-Detection-Guatemala/
│
├── configs/
│
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── RDD2022/              # NO se sube a Git
│   └── manifests/
│       ├── images.csv
│       ├── annotations.csv
│       └── splits.csv             # se generará después del EDA
│
├── docs/
│   └── TEAM_GUIDE.md
│
├── notebooks/
│   ├── 00_dataset_validation.ipynb
│   ├── 01_eda_rdd2022.ipynb
│   ├── 02_resnet.ipynb
│   ├── 03_vae.ipynb
│   └── 04_xai_evaluation.ipynb
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── data/
│   │   ├── validate_dataset.py
│   │   ├── build_manifest.py
│   │   └── make_splits.py
│   ├── models/
│   ├── evaluation/
│   └── xai/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 4. Preparación del entorno

### Python recomendado

Python 3.10.

### Crear entorno virtual

En PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Instalar dependencias

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Registrar kernel de Jupyter

```powershell
python -m ipykernel install --user --name pothole-dl --display-name "Python (Pothole DL)"
```

---

## 5. Pipeline de datos

El flujo oficial del proyecto es:

```text
RDD2022
   ↓
validate_dataset.py
   ↓
build_manifest.py
   ↓
images.csv + annotations.csv
   ↓
00_dataset_validation.ipynb
   ↓
01_eda_rdd2022.ipynb
   ↓
selección del dominio OOD
   ↓
make_splits.py
   ↓
splits.csv
   ↓
entrenamiento y evaluación
```

### Validar dataset

```powershell
python src\data\validate_dataset.py
```

### Regenerar manifests

```powershell
python src\data\build_manifest.py
```

Actualmente se esperan:

```text
images.csv       38,385 filas
annotations.csv  65,712 filas
```

---

## 6. Notebooks

### `00_dataset_validation.ipynb`

Verifica:

- existencia de dominios;
- conteos de train/test;
- integridad de manifests;
- clases;
- consistencia de D40.

### `01_eda_rdd2022.ipynb`

Analiza:

- imágenes por dominio;
- prevalencia de D40;
- desbalance global;
- distribución de clases;
- estado `Pothole D40 / Other damage / No annotated damage`;
- bounding boxes D40;
- coocurrencia;
- candidatos para test OOD.

**Los splits finales no deben decidirse antes de terminar el EDA.**

---

## 7. Estrategia para el desbalance

El problema binario tiene aproximadamente una relación de **9.45 negativas por cada positiva**.

Estrategia inicial:

1. Mantener `validation`, `test_id`, `test_ood` y Guatemala con distribución realista.
2. Usar `BCEWithLogitsLoss(pos_weight=...)` en entrenamiento.
3. Calcular `pos_weight` únicamente con el split `train`.
4. Aplicar data augmentation solo sobre entrenamiento.
5. Seleccionar el umbral de decisión usando validation.
6. Reportar prioritariamente:
   - Recall
   - Precision
   - F1-score
   - PR-AUC
   - Matriz de confusión
7. Accuracy será una métrica secundaria.

No se utilizará SMOTE sobre imágenes.

`WeightedRandomSampler` o focal loss podrán evaluarse como experimentos adicionales, pero no se combinarán todos los mecanismos de balanceo desde el inicio.

---

## 8. Experimentos principales

### ResNet18

Comparaciones previstas:

1. ResNet18 desde cero.
2. ResNet18 preentrenada + fine-tuning.
3. Mejor configuración anterior + augmentation.

Todos deben utilizar **exactamente los mismos splits**.

### VAE

El VAE se entrenará inicialmente con imágenes clasificadas como:

```text
has_any_damage == 0
```

Es decir, imágenes **sin daño anotado**.

La evaluación distinguirá:

- No annotated damage
- Other damage
- Pothole D40

Esto permitirá analizar si el VAE aprende a detectar potholes específicamente o anomalías viales en general.

### XAI

Grad-CAM se utilizará para:

- visualizar regiones relevantes;
- estudiar verdaderos/falsos positivos;
- detectar sesgos;
- comparar atención con bounding boxes D40.

---

## 9. Domain shift

El proyecto evaluará al menos tres escenarios:

```text
test_id
    imágenes de dominios representados durante entrenamiento

test_ood
    un dominio completo nunca utilizado en entrenamiento/validación

test_guatemala
    conjunto externo recolectado en Guatemala
```

La elección del país `test_ood` se realizará **después del EDA**.

`China_Drone` se analizará por separado antes de decidir si entra al experimento principal, porque introduce un cambio importante de perspectiva de captura además del cambio geográfico.

---

## 10. Dataset de Guatemala

El conjunto externo de Guatemala deberá mantenerse completamente fuera del entrenamiento principal.

Idealmente debe incluir:

```text
Guatemala/
├── pothole/
├── cracked_road/
└── normal/
```

Se recomienda capturar variedad de:

- iluminación;
- calles;
- tamaño de bache;
- cámaras;
- clima;
- superficies.

No depender exclusivamente de imágenes obtenidas de búsquedas web.

Antes de compartir fotos públicamente, revisar información sensible como ubicación GPS en EXIF, rostros o placas.

---

## 11. Reproducibilidad

Reglas del proyecto:

- usar `random seed = 42` cuando aplique;
- no cambiar splits individualmente;
- no guardar rutas absolutas personales en código o CSV;
- no subir imágenes RDD2022 a Git;
- no subir pesos grandes a Git;
- guardar parámetros y métricas de cada experimento;
- documentar cualquier cambio en preprocessing.

---

## 12. Equipo

| Integrante | Área principal |
|---|---|
| Vianka Castro | Data pipeline, validación, manifests, EDA y splits |
| Ricardo Arturo | ResNet18: scratch, transfer learning y augmentation |
| Diego López | VAE y detección de anomalías |
| Felipe Aguilar | Evaluación, métricas, latencia y Grad-CAM |

Consulta [docs/TEAM_GUIDE.md](docs/TEAM_GUIDE.md) antes de comenzar una rama de trabajo.

---

## 13. Estado actual

- [x] Dataset RDD2022 descargado.
- [x] Integridad estructural validada.
- [x] 38,385 pares train imagen/XML verificados.
- [x] `images.csv` generado.
- [x] `annotations.csv` generado.
- [x] Notebook 00 preparado.
- [x] Notebook 01 preparado.
- [ ] EDA final ejecutado/revisado.
- [ ] Dominio OOD seleccionado.
- [ ] `splits.csv` generado y congelado.
- [ ] Modelos ResNet18.
- [ ] VAE.
- [ ] Grad-CAM.
- [ ] Dataset externo Guatemala.
- [ ] Evaluación final.
