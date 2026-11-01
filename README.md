# Pothole Detection Guatemala

Proyecto de Deep Learning para estudiar la **detección de presencia de baches (potholes)** en imágenes viales y analizar qué tan bien un modelo entrenado con datos internacionales generaliza a imágenes de **Guatemala**.

El proyecto utiliza **RDD2022** como dataset principal y compara enfoques de:

- **Transfer Learning** con ResNet18 preentrenada.
- ResNet18 entrenada **desde cero**.
- **EfficientNet-B0 preentrenada**, segundo clasificador supervisado con el mismo protocolo que ResNet18.
- **Data augmentation** para reducir sensibilidad al cambio de dominio.
- **Variational Autoencoder (VAE)** para detección de anomalías.
- **XAI / Grad-CAM** para interpretar predicciones y errores.
- Evaluación **in-distribution**, **out-of-distribution (OOD)** y posteriormente en un conjunto externo de Guatemala.

EfficientNet ya fue entrenada con RDD2022 local verificado e integrada con evaluación, latencia y Grad-CAM reales.
Alcanzó F1 0.6444 en test interno, 0.0744 en Noruega y 0.2405 en China_Drone; omite 241 de 256 baches
noruegos y no está lista para despliegue. Ver [resultados, protocolo y comandos](docs/EFFICIENTNET.md).
Los resultados negativos del VAE se conservan, no se sustituyen por métricas sintéticas.

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

Consulta [data/DATA_README.md](data/DATA_README.md) para instalar y organizar el dataset.

**Entrega dataset_v1 (23 de octubre de 2026):** Norway queda reservado completo
como `test_ood`; China_Drone se reserva como `test_perspective` y no entra al
entrenamiento principal. Los demás dominios se dividen aproximadamente 70/15/15
en train/val/test_id, estratificando por dominio y estado de daño, con seed 42
y agrupación de duplicados exactos. Ver [decisiones del EDA](docs/DATASET_V1.md)
y [entrega al equipo](docs/HANDOFF.md). Guatemala sigue pendiente de recolección.

---

## 3. Estructura del repositorio

```text
Pothole-Detection-Guatemala/
│
├── configs/
│
├── data/
│   ├── DATA_README.md
│   ├── raw/
│   │   └── RDD2022/              # NO se sube a Git
│   └── manifests/
│       ├── images.csv
│       ├── annotations.csv
│       ├── splits.csv             # contrato congelado dataset_v1
│       ├── file_checksums.csv
│       └── dataset_v1.lock.json
│
├── docs/
│   ├── DATASET_V1.md
│   ├── HANDOFF.md
│   └── GUATEMALA_PROTOCOL.md
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
├── TEAM_GUIDE.md
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

### Usar la versión congelada

Ricardo, Diego y Felipe deben usar los CSV existentes sin regenerarlos:

```powershell
python src/data/make_splits.py --check
python src/data/share_dataset.py --verify
```

El primer comando verifica hashes e integridad del contrato; el segundo compara
todos los JPG/XML locales con las huellas de la copia validada.
`make_splits.py` sin argumentos crea la entrega solo si aún no existe;
rechaza sobrescribir una versión congelada. No ejecutar nuevamente
`build_manifest.py` o `audit_release.py` sobre dataset_v1: cualquier modificación
requiere preparar y acordar una versión nueva.

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

El EDA se cerró para dataset_v1; las conclusiones y limitaciones están en
[docs/DATASET_V1.md](docs/DATASET_V1.md).

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

Para dataset_v1 se eligió **Norway** como `test_ood`, completo y sin rebalanceo.

`China_Drone` queda completo como `test_perspective`: evaluación adicional por
cambio de perspectiva, fuera de train/val/test_id y del OOD geográfico principal.

---

## 10. Dataset de Guatemala

El conjunto externo de Guatemala está pendiente de recolección y deberá
mantenerse completamente fuera del entrenamiento principal. Ver el
[protocolo de recolección y evaluación](docs/GUATEMALA_PROTOCOL.md).

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

Consulta [TEAM_GUIDE.md](TEAM_GUIDE.md) y [docs/HANDOFF.md](docs/HANDOFF.md)
antes de comenzar una rama de trabajo.

---
