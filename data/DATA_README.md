# Datos del proyecto

Esta carpeta contiene la configuración y los manifests utilizados por **Pothole Detection Guatemala**.

Los datos crudos de RDD2022 **no deben subirse al repositorio**.

---

## 1. Estructura esperada

```text
data/
├── README.md
│
├── raw/
│   └── RDD2022/
│       ├── China_Drone/
│       ├── China_MotorBike/
│       ├── Czech/
│       ├── India/
│       ├── Japan/
│       ├── Norway/
│       └── United_States/
│
└── manifests/
    ├── images.csv
    ├── annotations.csv
    └── splits.csv
```

Algunos ZIP originales pueden contener un nivel extra, por ejemplo:

```text
Japan/Japan/train/...
```

El código del proyecto busca archivos recursivamente y no depende de que todos los países tengan exactamente la misma profundidad.

---

## 2. RDD2022

La copia validada contiene:

| Dominio | Train IMG | Train XML | Test IMG |
|---|---:|---:|---:|
| China_Drone | 2,401 | 2,401 | 0 |
| China_MotorBike | 1,977 | 1,977 | 500 |
| Czech | 2,829 | 2,829 | 709 |
| India | 7,706 | 7,706 | 1,959 |
| Japan | 10,506 | 10,506 | 2,627 |
| Norway | 8,161 | 8,161 | 2,040 |
| United_States | 4,805 | 4,805 | 1,200 |
| **TOTAL** | **38,385** | **38,385** | **9,035** |

Total: **47,420 imágenes**.

Los XML de train contienen anotaciones Pascal VOC.

---

## 3. No subir datos crudos

`.gitignore` debe incluir al menos:

```gitignore
data/raw/
data/recovery/
*.zip
*.rar
*.7z
```

Nunca ejecutar un commit si `git status` muestra miles de JPG/XML.

---

## 4. `images.csv`

Una fila representa una imagen de `train`.

Columnas principales:

| Campo | Descripción |
|---|---|
| `sample_id` | Identificador único |
| `country` | Dominio de origen |
| `image_path` | Ruta relativa desde `data/raw/RDD2022` |
| `xml_path` | Ruta relativa del XML |
| `width` | Ancho |
| `height` | Alto |
| `depth` | Canales |
| `num_objects` | Total de objetos anotados |
| `num_D00` | Número de objetos D00 |
| `num_D10` | Número de objetos D10 |
| `num_D20` | Número de objetos D20 |
| `num_D40` | Número de objetos D40 |
| `has_D00` | 1 si aparece D00 |
| `has_D10` | 1 si aparece D10 |
| `has_D20` | 1 si aparece D20 |
| `has_D40` | 1 si aparece D40 |
| `has_any_damage` | 1 si existe al menos un objeto anotado |
| `is_empty_annotation` | 1 si el XML no tiene objetos |
| `has_other_label` | 1 si existen etiquetas fuera de D00/D10/D20/D40 |
| `other_labels` | Etiquetas adicionales separadas por `;` |

Cantidad esperada:

```text
38,385 filas
```

---

## 5. `annotations.csv`

Una fila representa **un objeto anotado**.

Columnas:

| Campo | Descripción |
|---|---|
| `sample_id` | Imagen a la que pertenece |
| `country` | Dominio |
| `object_index` | Índice del objeto en la imagen |
| `class` | Etiqueta original |
| `xmin`, `ymin`, `xmax`, `ymax` | Bounding box |
| `bbox_width` | Ancho del bbox |
| `bbox_height` | Alto del bbox |
| `bbox_area` | Área del bbox |
| `bbox_area_ratio` | Área bbox / área imagen |

Cantidad esperada:

```text
65,712 filas
```

---

## 6. Etiquetas encontradas

La copia validada contiene:

| Etiqueta | Objetos |
|---|---:|
| D00 | 26,016 |
| D01 | 179 |
| D0w0 | 1 |
| D10 | 11,830 |
| D11 | 45 |
| D20 | 10,617 |
| D40 | 6,544 |
| D43 | 793 |
| D44 | 5,057 |
| D50 | 3,581 |
| Repair | 1,046 |
| Block crack | 3 |

Las cuatro clases principales consideradas por el proyecto son:

```text
D00
D10
D20
D40
```

**No eliminar** las demás etiquetas del manifest.

---

## 7. Objetivo binario

Para ResNet18:

```text
target = has_D40
```

Entonces:

```text
has_D40 = 1
    la imagen contiene al menos un pothole D40

has_D40 = 0
    la imagen no contiene D40
```

Una imagen negativa puede contener otro tipo de daño vial.

No confundir:

```text
No D40
```

con:

```text
carretera sin daño
```

---

## 8. Desbalance

En train:

```text
Con D40:     3,674 imágenes
Sin D40:    34,711 imágenes
```

Prevalencia aproximada:

```text
9.57%
```

La estrategia de entrenamiento se calculará **después de generar los splits**.

No calcular `pos_weight` con test/OOD.

---

## 9. Datos para VAE

Para el VAE distinguimos:

### Pothole

```text
has_D40 == 1
```

### Other damage

```text
has_D40 == 0
has_any_damage == 1
```

### No annotated damage

```text
has_any_damage == 0
```

El conjunto inicial candidato para entrenar el VAE es:

```text
has_any_damage == 0
```

Debe describirse como **sin daño anotado**, no necesariamente como una afirmación absoluta de carretera perfecta.

---

## 10. Regenerar manifests

Desde la raíz:

```powershell
python src\data\validate_dataset.py
```

Luego:

```powershell
python src\data\build_manifest.py
```

Resultados esperados:

```text
images.csv       38,385
annotations.csv  65,712
```

---

## 11. `splits.csv`

Todavía no debe crearse manualmente.

Será generado por:

```text
src/data/make_splits.py
```

después de:

1. finalizar el EDA;
2. decidir qué hacer con `China_Drone`;
3. seleccionar un dominio completo como `test_ood`;
4. definir train/validation/test_id;
5. fijar `random seed = 42`.

Una vez generado, `splits.csv` será el **contrato oficial del equipo**.

Ningún integrante debe crear splits independientes.

---

## 12. Dataset de Guatemala

Se agregará como conjunto externo.

Recomendación de estructura:

```text
data/
└── external/
    └── Guatemala/
        ├── pothole/
        ├── cracked_road/
        └── normal/
```

Idealmente deberá incluir metadata separada, pero no información personal innecesaria.

No mezclar Guatemala con el entrenamiento principal si su propósito es medir generalización externa.
