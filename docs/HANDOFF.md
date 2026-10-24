# Entrega de datos a Ricardo, Diego y Felipe

## Qué compartir

El código, notebooks y `data/manifests/` se comparten por el repositorio:

```powershell
git clone https://github.com/Vann06/Pothole-Detection-Guatemala.git
cd Pothole-Detection-Guatemala
```

Los cambios de esta entrega deben publicarse en el repositorio antes de que
ellos hagan clone/pull; una modificación local no aparece automáticamente en GitHub.

Las imágenes no van a Git. Compartir por Drive una carpeta `dataset_v1` con
un ZIP por dominio y `packages.json`. Dar acceso de lectura a las cuentas del
equipo y pasarles el enlace a esa carpeta. No subir `.venv`, `.git`, archivos de
recuperación, pesos ni toda la carpeta del proyecto como un único ZIP.

Ricardo necesita las imágenes para entrenar y evaluar. Felipe puede comenzar
las funciones de métricas con los CSV, pero necesita las imágenes para Grad-CAM,
análisis visual y latencia de extremo a extremo; también necesitará el checkpoint
y preprocessing de Ricardo. Diego necesita imágenes para VAE. Pueden usar una
máquina/Colab compartida; en ese caso los datos deben estar accesibles allí,
sin descargar una copia en cada computadora.

## Preparar ZIPs (Vianka)

Desde la raíz, con el entorno activo:

```powershell
python src/data/make_splits.py --check
python src/data/share_dataset.py --package
```

Los siete ZIPs de esta entrega se preparan en `.dist/dataset_v1/`, con las rutas exactas de
`images.csv` y únicamente los JPG/XML etiquetados. El tamaño de imágenes es
aproximadamente 11.07 GB decimales, más XML y cabeceras ZIP. Reservar al menos
12 GB adicionales para crear los paquetes. Se usa ZIP64 sin recompresión de
JPG para evitar tiempo innecesario. No se incluyen las 9,035 imágenes del test
original sin etiquetas: **no son** `test_id` ni `test_ood`.

Subir los ZIPs y `packages.json` a Drive. Estos pasos preparan archivos locales;
no suben ni envían nada por sí mismos. No regenerar manifests al recibirlos.

## Instalar la copia (cada integrante)

1. Clonar/actualizar el código y configurar el entorno según README.
2. Descargar los ZIPs. Todos los dominios permiten reproducir el EDA y auditoría;
   China_Drone solo se evalúa como `test_perspective`.
3. Crear `data/raw/RDD2022/` y **extraer allí el contenido de cada ZIP**, sin
   agregar un nivel con el nombre del ZIP. Conservar incluso niveles repetidos
   como `Japan/Japan/train/`: los CSV guardan la estructura local validada.
4. Verificar el SHA256 del ZIP contra `packages.json`:

```powershell
Get-FileHash -Algorithm SHA256 .\RDD2022_Japan_dataset_v1.zip
```

5. Comprobar el contrato y los archivos extraídos:

```powershell
python src/data/make_splits.py --check
python src/data/share_dataset.py --verify
```

La copia reducida compartida sirve para entrenar/evaluar, pero el notebook 00
cuenta también el test original y espera la copia completa de 47,420 imágenes.
Para repetir esa validación de adquisición, obtener además el test original;
no hace falta para los experimentos definidos en dataset_v1.

## Contrato para modelos

Unir `images.csv` con `splits.csv` por `sample_id`, con `validate="one_to_one"`.
Usar exactamente `train`, `val`, `test_id`, `test_ood` y `test_perspective`.
No meter el último en entrenamiento. `target = has_D40`; para VAE el candidato
inicial es `split == 'train'` y `has_any_damage == 0`. Calcular pos_weight solo
en train. Umbrales y selección de checkpoints solo con val. Guardar la versión
`dataset_v1` y el SHA256 de `splits.csv` con cada experimento.

Los checkpoints y predicciones se entregarán aparte; acordar un CSV de
predicciones con `sample_id`, `split`, `y_true`, `y_score`, `threshold`,
`experiment_id` y `dataset_version`. `y_score` de ResNet será probabilidad D40;
para VAE documentar la escala del error y su threshold.
