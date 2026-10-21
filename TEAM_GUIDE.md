# Guía de trabajo para el equipo

Esta guía explica **qué datos existen, cómo deben utilizarse y qué debe hacer cada integrante**.

Antes de empezar a entrenar modelos, leer este documento completo.

---

## 1. Regla principal

**Nadie debe construir su propio split del dataset.**

Todo el equipo debe usar:

```text
data/manifests/images.csv
data/manifests/annotations.csv
data/manifests/splits.csv
```

`splits.csv` todavía no está congelado. Mientras se termina el EDA, cada integrante puede preparar código y arquitectura, pero **no producir resultados finales comparables**.

---

## 2. Preparar el proyecto

Clonar el repositorio:

```powershell
git clone <URL_DEL_REPOSITORIO>
cd Pothole-Detection-Guatemala
```

Crear entorno:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instalar:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para Jupyter:

```powershell
python -m ipykernel install --user --name pothole-dl --display-name "Python (Pothole DL)"
```

---

## 3. Obtener RDD2022

RDD2022 no está dentro del repositorio.

Cada integrante que necesite cargar imágenes debe colocar su copia local bajo:

```text
data/raw/RDD2022/
```

Debe terminar existiendo algo equivalente a:

```text
data/raw/RDD2022/
├── China_Drone/
├── China_MotorBike/
├── Czech/
├── India/
├── Japan/
├── Norway/
└── United_States/
```

No renombrar imágenes ni XML.

No usar rutas absolutas como:

```text
C:\Users\Nombre\Downloads\...
```

dentro de código compartido.

Usar rutas relativas desde la raíz del proyecto.

---

## 4. Qué representan los manifests

### `images.csv`

Una fila = una imagen.

Para la tarea supervisada principal:

```text
target = has_D40
```

- `has_D40 = 1`: existe al menos un pothole D40.
- `has_D40 = 0`: no existe D40.

Una imagen `has_D40 = 0` puede contener grietas u otros daños.

### `annotations.csv`

Una fila = un objeto anotado.

Se utiliza para:

- analizar clases;
- tamaños de bounding boxes;
- comparar Grad-CAM con región real;
- estudiar errores.

---

## 5. No confundir clasificación con detección de objetos

La ResNet18 propuesta realiza:

> **clasificación de presencia de D40 a nivel de imagen**

No produce bounding boxes.

Por eso, en el informe debe evitarse afirmar que ResNet18 es un detector de objetos.

Grad-CAM podrá mostrar una región de atención, pero tampoco reemplaza un detector de objetos.

La formulación correcta será algo como:

> clasificación/detección de presencia de potholes en imágenes.

---

## 6. Desbalance

Estado actual:

```text
3,674 imágenes con D40
34,711 imágenes sin D40
```

No utilizar accuracy como única métrica.

Métricas prioritarias:

```text
precision
recall
F1-score
PR-AUC
confusion matrix
```

El `pos_weight` debe calcularse **solo con el split train**.

No balancear test ni OOD.

---

# 7. Trabajo por integrante

## Vianka — Data pipeline / EDA / splits

Branch sugerida:

```text
feature/data-pipeline
```

Responsabilidades:

- validar RDD2022;
- mantener `images.csv`;
- mantener `annotations.csv`;
- ejecutar EDA;
- analizar distribución D40;
- decidir junto al equipo el dominio OOD;
- construir `splits.csv`;
- documentar dataset Guatemala;
- congelar `dataset_v1`.

Archivos:

```text
src/data/validate_dataset.py
src/data/build_manifest.py
src/data/make_splits.py
notebooks/00_dataset_validation.ipynb
notebooks/01_eda_rdd2022.ipynb
data/manifests/*
```

### Criterio de finalización

Vianka termina la primera fase cuando:

```text
[ ] EDA revisado
[ ] China_Drone decidido
[ ] test_ood decidido
[ ] splits.csv generado
[ ] random seed documentado
[ ] cada sample_id tiene exactamente un split
[ ] README actualizado
```

---

## Ricardo — ResNet18

Branch sugerida:

```text
feature/resnet
```

Responsabilidades:

### Experimento 1

```text
ResNet18 desde cero
```

### Experimento 2

```text
ResNet18 preentrenada + fine-tuning
```

### Experimento 3

```text
mejor configuración + augmentation
```

Archivos sugeridos:

```text
src/models/resnet.py
src/data/dataset.py
src/models/train_resnet.py
notebooks/02_resnet.ipynb
```

### Reglas

Ricardo **no debe crear splits propios**.

Debe cargar:

```text
images.csv
splits.csv
```

y filtrar usando la columna `split`.

Target:

```python
target = row["has_D40"]
```

### Desbalance

Primer enfoque recomendado:

```python
torch.nn.BCEWithLogitsLoss(pos_weight=...)
```

`pos_weight` debe calcularse con datos `train` únicamente.

No aplicar augmentation en validation/test/OOD.

### Resultados mínimos

Por experimento guardar:

```text
loss train
loss validation
precision
recall
F1
PR-AUC
confusion matrix
latency
```

También guardar:

```text
random seed
learning rate
epochs
batch size
optimizer
scheduler
augmentation
checkpoint seleccionado
```

---

## Diego — VAE

Branch sugerida:

```text
feature/vae
```

Responsabilidades:

- implementar VAE convolucional;
- entrenarlo con imágenes candidatas a normalidad;
- definir reconstruction error;
- seleccionar threshold usando validation;
- evaluar domain shift.

Archivos sugeridos:

```text
src/models/vae.py
src/models/train_vae.py
notebooks/03_vae.ipynb
```

### Muy importante

No usar:

```text
has_D40 == 0
```

como sinónimo automático de normal.

Para el experimento inicial:

```text
has_any_damage == 0
```

es el grupo candidato a entrenamiento normal.

Evaluar separadamente:

```text
No annotated damage
Other damage
Pothole D40
```

Pregunta importante:

> ¿El VAE detecta específicamente potholes o simplemente cualquier tipo de daño vial como anomalía?

### Threshold

No seleccionar threshold usando test ni Guatemala.

Debe fijarse usando validation.

---

## Felipe — Evaluación y XAI

Branch sugerida:

```text
feature/xai-evaluation
```

Responsabilidades:

- funciones comunes de métricas;
- matrices de confusión;
- medición de latencia;
- Grad-CAM;
- análisis de errores;
- visualizaciones finales.

Archivos sugeridos:

```text
src/evaluation/metrics.py
src/evaluation/latency.py
src/evaluation/confusion.py
src/xai/gradcam.py
notebooks/04_xai_evaluation.ipynb
```

### Grad-CAM

Se aplicará al modelo ResNet18.

Casos mínimos:

```text
true positive
true negative
false positive
false negative
```

También analizar por dominio.

Cuando exista D40, `annotations.csv` permite recuperar sus bounding boxes para estudiar si la zona de atención coincide razonablemente con el pothole.

Grad-CAM se usa como explicación; no debe presentarse como segmentación exacta.

---

# 8. Cómo cargar los datos

Ejemplo común:

```python
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()

if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

images = pd.read_csv(
    PROJECT_ROOT / "data" / "manifests" / "images.csv"
)

splits = pd.read_csv(
    PROJECT_ROOT / "data" / "manifests" / "splits.csv"
)

df = images.merge(
    splits,
    on="sample_id",
    validate="one_to_one"
)
```

Después:

```python
train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]
test_id_df = df[df["split"] == "test_id"]
test_ood_df = df[df["split"] == "test_ood"]
```

No hacer:

```python
train_test_split(...)
```

en cada notebook.

---

# 9. Cómo construir la ruta de una imagen

`image_path` es relativa a:

```text
data/raw/RDD2022/
```

Entonces:

```python
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "RDD2022"

image_path = DATASET_ROOT / row["image_path"]
```

Esto permite que funcione en cualquier computadora.

---

# 10. Experimentos comparables

Todos los modelos deben respetar:

```text
mismos splits
mismo test_id
mismo test_ood
mismo conjunto Guatemala
mismas métricas
```

Si un integrante cambia datos o preprocessing que afecta comparabilidad, debe documentarlo.

---

# 11. China_Drone

No mezclarlo silenciosamente con el resto.

Las imágenes de drone tienen una perspectiva diferente.

Antes de generar `splits.csv` se tomará una decisión explícita:

```text
A. excluir China_Drone del experimento principal;
B. usarlo como experimento adicional de domain shift;
C. incluirlo bajo una justificación documentada.
```

Hasta entonces, conservarlo en los manifests.

---

# 12. Guatemala

Guatemala será evaluación externa.

No utilizar imágenes Guatemala para:

```text
training
selección de hiperparámetros
threshold tuning
early stopping
```

si queremos reportar generalización externa limpia.

Si posteriormente se realiza un segundo experimento de adaptación de dominio, ese experimento debe estar separado del baseline.

---

# 13. Git

Antes de trabajar:

```powershell
git switch main
git pull
git switch <tu-rama>
```

Antes de commit:

```powershell
git status
```

Nunca deben aparecer:

```text
data/raw/
.venv/
*.pt
*.pth
```

Ejemplo de commit:

```powershell
git add src notebooks
git commit -m "Implement ResNet training pipeline"
```

No hacer commits gigantes mezclando trabajo no relacionado.

---

# 14. Antes de hacer merge

Cada integrante debe comprobar:

```text
[ ] código ejecuta desde la raíz
[ ] no hay rutas absolutas
[ ] no hay dataset crudo en Git
[ ] no hay checkpoints grandes
[ ] se usan splits oficiales
[ ] métricas se guardan
[ ] seed documentado
[ ] notebook se puede ejecutar en orden
[ ] README/documentación actualizada si aplica
```

---

# 15. Orden de trabajo del equipo

```text
Vianka
EDA + splits
        ↓
dataset_v1 congelado
        ↓
┌────────────────┬────────────────┬────────────────────┐
│ Ricardo        │ Diego          │ Felipe             │
│ ResNet         │ VAE            │ Metrics / XAI      │
└────────────────┴────────────────┴────────────────────┘
        ↓
resultados comparables
        ↓
Guatemala
        ↓
domain shift
        ↓
Grad-CAM + análisis de errores
        ↓
informe final
```

Felipe puede desarrollar las utilidades generales antes de tener modelos entrenados.

Ricardo y Diego pueden preparar arquitectura y loaders antes de congelar splits, pero los resultados finales deben producirse después.

---

# 16. Qué no hacer

No:

```text
- mover o renombrar datos crudos sin actualizar pipeline;
- crear un split diferente por integrante;
- usar test para escoger hiperparámetros;
- balancear artificialmente test;
- usar Guatemala durante entrenamiento baseline;
- reportar solo accuracy;
- llamar "normal" a todo No-D40;
- presentar Grad-CAM como detector de bounding boxes;
- subir dataset o checkpoints pesados a Git.
```

---

## Dudas o cambios

Si una decisión afecta:

```text
target
split
preprocessing
dominio OOD
dataset Guatemala
métrica principal
```

debe discutirse antes de modificar el pipeline compartido.
