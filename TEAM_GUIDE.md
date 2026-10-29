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

`splits.csv` queda congelado como **dataset_v1** con seed 42. Norway se reserva
como `test_ood` y China_Drone como `test_perspective`. Leer las
[decisiones y limitaciones](docs/DATASET_V1.md) y la
[entrega e instalación](docs/HANDOFF.md). Los experimentos deben registrar esta versión.

---

## 2. Preparar el proyecto

Clonar el repositorio:

```powershell
git clone https://github.com/Vann06/Pothole-Detection-Guatemala.git
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

Para entrenar en GPU NVIDIA, instalar antes torch con CUDA:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Si Windows bloquea una DLL de pandas ("Una directiva de Control de aplicaciones bloqueó este archivo"),
instalar las versiones registradas en `dataset_v1.lock.json`:

```powershell
python -m pip install pandas==2.3.3 numpy==2.2.6 scikit-learn==1.7.2
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
[x] EDA revisado y conclusiones documentadas
[x] China_Drone: test_perspective, fuera del entrenamiento principal
[x] test_ood: Norway completo
[x] splits.csv generado y dataset_v1 protegido por hashes
[x] random seed = 42 documentado
[x] cada sample_id tiene exactamente un split
[x] README y guía de entrega actualizados
```

Decisiones técnicas tomadas para esta entrega a solicitud de Vianka; no
representan una aprobación previa de los demás integrantes. Cualquier cambio
posterior del dominio OOD requiere acuerdo y una versión distinta del dataset.
Guatemala: protocolo documentado, fotos y etiquetas todavía pendientes.

---

## Ricardo — ResNet18

Branch sugerida:

```text
feature/resnet
```

Estado: **pipeline implementado y probado con datos sintéticos; falta entrenar con RDD2022**
(pendiente de los ZIPs de dataset_v1).

Responsabilidades:

### Experimento 1

```text
ResNet18 desde cero                       configs/resnet/exp1_scratch.yaml
```

### Experimento 2

```text
ResNet18 preentrenada + fine-tuning       configs/resnet/exp2_pretrained.yaml
```

### Experimento 3

```text
mejor configuración + augmentation        configs/resnet/exp3_augment.yaml
```

### Experimento 4 (adicional, propuesto; requiere visto bueno del equipo)

```text
D40 principal + D00/D10/D20 auxiliares    configs/resnet/exp4_auxiliary.yaml
```

La red tiene 4 salidas (D40 primero) y la pérdida es
`BCE(D40) + 0.4 · promedio BCE(D00, D10, D20)`, cada clase con su `pos_weight` de train.
**El target del proyecto no cambia**: selección de checkpoint, threshold y métricas principales siguen
siendo solo de D40. Las grietas se reportan como PR-AUC auxiliar. Se compara contra exp2 (misma
configuración) para aislar el efecto de las tareas auxiliares.

Archivos:

```text
src/data/dataset.py            carga splits oficiales (verifica SHA256 del lock), Dataset, transforms
src/models/resnet.py           ResNet18 con 1 salida (D40) o 4 (D40 + auxiliares)
src/models/train_resnet.py     entrenamiento, selección, threshold, evaluación, latencia
configs/resnet/*.yaml          un archivo por experimento
tests/test_resnet_pipeline.py  pruebas con dataset sintético
notebooks/02_resnet.ipynb      solo lee resultados: curvas, tablas, matrices, domain shift
```

Ejecución (desde la raíz, con `.venv` activo):

```powershell
python src/data/share_dataset.py --verify
python -m src.models.train_resnet --config configs/resnet/exp2_pretrained.yaml --limit 500 --epochs 1
python -m src.models.train_resnet --config configs/resnet/exp2_pretrained.yaml
python -m src.models.train_resnet --config configs/resnet/exp1_scratch.yaml
python -m src.models.train_resnet --config configs/resnet/exp3_augment.yaml
python -m src.models.train_resnet --config configs/resnet/exp4_auxiliary.yaml
```

Si exp1 supera a exp2 en PR-AUC de **val**, copiar su configuración a exp3 (y a exp4) antes de correrlos.
Tiempo estimado en GTX 1650: ~7–9 min por época, ~8–10 h para los cuatro.

### Selección de modelo y threshold

```text
PR-AUC (D40) en val   → elegir checkpoint (cada época, early stopping) y comparar experimentos
F1 máximo (D40) en val → fijar el threshold, una sola vez, con el checkpoint ya elegido
threshold fijo         → precision, recall, F1 y confusion matrix en test_id, test_ood, test_perspective
```

PR-AUC no depende del threshold; su referencia de azar es la prevalencia
(~0.12 en val/test_id, ~0.03 en test_ood y test_perspective), por lo que debe leerse junto a ella.

### Salidas por experimento

```text
reports/experiments/resnet/<experiment_id>/config.yaml      configuración resuelta, pos_weight, versión y SHA256
reports/experiments/resnet/<experiment_id>/history.csv      loss train/val y PR-AUC val por época
reports/experiments/resnet/<experiment_id>/metrics.json     métricas por split, threshold, latencia
reports/experiments/resnet/<experiment_id>/predictions.csv  contrato de predicciones (abajo)
checkpoints/resnet/<experiment_id>/best.pt                  fuera de Git; compartir por Drive
```

Las corridas con `--limit` se guardan como `<experiment_id>_limitN` y el notebook las ignora.

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

Todo lo anterior queda registrado automáticamente en `config.yaml`, `history.csv` y `metrics.json`.

### Entregables para el equipo

- A Felipe: `best.pt` del modelo final (por Drive), su `predictions.csv` y el preprocessing de evaluación
  `build_transforms(train=False)` (resize 224×224 + normalización ImageNet), necesarios para Grad-CAM.
- Formato de `predictions.csv` (contrato de `docs/HANDOFF.md`):

```text
sample_id, split, y_true, y_score, threshold, experiment_id, dataset_version
```

`y_score` es siempre la probabilidad de D40. exp4 agrega al final `y_true_D00, y_score_D00, …, y_score_D20`
sin alterar las primeras siete columnas.

---

## Diego — VAE

Branch sugerida:

```text
feature/vae
```

Estado: **pipeline implementado, probado y entrenado (`vae_base`)**. Resultado: con esta configuración el
score de reconstrucción **no separa D40** (PR-AUC en val 0.10 con prevalencia 0.12); ver `notebooks/03_vae.ipynb`.

Responsabilidades:

- implementar VAE convolucional;
- entrenarlo con imágenes candidatas a normalidad;
- definir reconstruction error;
- seleccionar threshold usando validation;
- evaluar domain shift.

Archivos:

```text
src/models/vae.py               ConvVAE (128 -> 4 -> 128, salida sigmoide), pérdida y scores de anomalía
src/models/train_vae.py         entrenamiento, selección, threshold, evaluación, desglose, latencia
src/data/cache_images.py        caché PNG sin pérdida de las imágenes redimensionadas
src/evaluation/breakdown.py     desglose por grupo, país y clase de daño (sirve para cualquier modelo)
configs/vae/vae_base.yaml       configuración del experimento
tests/test_vae_model.py         pruebas del modelo, la pérdida y los scores
tests/test_vae_pipeline.py      pruebas del pipeline con dataset sintético
tests/test_cache_images.py      pruebas del caché
notebooks/03_vae.ipynb          solo lee resultados: curvas, grupos, países, clases, reconstrucciones
```

Ejecución (desde la raíz, con el entorno activo):

```powershell
python -m src.data.cache_images --size 128          # una sola vez; queda en data/raw/RDD2022_128 (fuera de Git)
python -m src.models.train_vae --config configs/vae/vae_base.yaml --limit 256 --epochs 3 --warmup-epochs 1
python -m src.models.train_vae --config configs/vae/vae_base.yaml
```

`warmup_epochs` no puede superar `epochs` (β nunca llegaría a 1); por eso la prueba corta usa `--warmup-epochs`.

### Diseño

```text
train                 split == train y has_any_damage == 0 (4,531 imágenes: solo Czech, India, Japan)
entrada               128×128 estirada (como ResNet), píxeles en [0, 1], sin normalización ImageNet
pérdida               suma por imagen de error cuadrático + β · KL, promedio del batch; β sube 0 -> 1 en warmup_epochs
checkpoint            menor pérdida con β = 1 en las imágenes sin daño de val (no usa etiquetas), tras el warmup
scores                global = error medio de la imagen; patch = máximo del error promediado en parches 16×16
threshold             F1 máximo de D40 en val, por score; score principal = mayor PR-AUC de D40 en val
```

Escala del score: error cuadrático por píxel en RGB [0, 1], promediado en canales; mayor = más anómalo.
En eval se usa z = μ, así que los scores son deterministas.

### Salidas por experimento

```text
reports/experiments/vae/<experiment_id>/config.yaml      configuración resuelta, versión y SHA256 de splits
reports/experiments/vae/<experiment_id>/history.csv      β, pérdida/reconstrucción/KL de train y val, PR-AUC val por época
reports/experiments/vae/<experiment_id>/metrics.json     métricas por split y score, desglose, thresholds, latencia, VRAM
reports/experiments/vae/<experiment_id>/predictions.csv  contrato de 7 columnas + y_score_global, y_score_patch, group, country
checkpoints/vae/<experiment_id>/best.pt                  fuera de Git; compartir por Drive
```

`y_score` y `threshold` del CSV corresponden al score principal. El desglose (`breakdown`) separa
`no_damage` / `other` / `pothole`, calcula AUROC entre cada par, métricas dentro de cada país y la tasa de
falsos positivos por clase de daño sin D40 (una imagen con varias clases cuenta en cada una).

### Limitaciones

- United_States y China_MotorBike no tienen imágenes sin daño: comparar dentro de cada país.
- A 128×128 un D40 de Norway mide ~4×4 px.
- `test_perspective` tiene solo 5 imágenes sin daño.
- El threshold de F1 máximo puede quedar degenerado (marcar casi todo) cuando el score no separa D40.

`src/evaluation/metrics.py`, `latency.py` y `breakdown.py`, y `src/utils/seed.py` ya existen y los usan
ResNet y el VAE; Felipe puede extenderlos.

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

### Comparación con ResNet

Para comparar en la misma tabla, entregar `predictions.csv` con el mismo contrato que ResNet
(`sample_id, split, y_true, y_score, threshold, experiment_id, dataset_version`), con `y_true = has_D40`
y `y_score` = error de reconstrucción (documentar su escala; mayor = más anómalo). Reportar PR-AUC de D40
y F1 con el threshold fijado en val, igual que ResNet.

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

### Insumos desde ResNet

- `reports/experiments/resnet/<exp>/predictions.csv`: permite recalcular métricas y separar TP/TN/FP/FN
  con la columna `threshold` (ya fijada en val).
- `best.pt` por Drive; cargar con `build_resnet18(pretrained=False, outputs=1)` (4 en exp4) y
  `torch.load(..., weights_only=True)["model"]`. Preprocessing: `build_transforms(train=False)`.
- Grad-CAM se calcula sobre la salida 0 (D40) en todos los experimentos.
- Para exp4, las columnas `y_true_D00/D10/D20` permiten verificar si los falsos positivos son grietas.

Las funciones de `src/evaluation/` pueden reemplazar más adelante el cálculo de métricas que hoy hace
`train_resnet.py` con sklearn, siempre que produzcan los mismos valores.

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
test_perspective_df = df[df["split"] == "test_perspective"]
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

Decisión dataset_v1: **B**, reservarlo completo como `test_perspective`, una
evaluación adicional de domain shift. No usarlo para training ni selección de
hiperparámetros. Las opciones consideradas fueron:

```text
A. excluir China_Drone del experimento principal;
B. usarlo como experimento adicional de domain shift;
C. incluirlo bajo una justificación documentada.
```

Se conserva en los manifests y cada muestra tiene su split explícito.

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
